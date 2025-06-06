import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- AI Configuration ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- Streamlit UI ---
st.set_page_config(page_title="AI Financial Statement Analyzer", layout="wide")
st.title("Ever AI - Enhanced Financial Statement Analyzer")
st.write(
    """
    **Upload your financial statements in CSV format.**  
    This tool will:
    - Analyze the data using AI (Gemini) for metrics and ratios
    - Perform robust financial due diligence with advanced metrics
    - Detect anomalies and outliers using multiple ML methods
    - Visualize key metrics, trends, and anomalies
    """
)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your financial statement (CSV)", type=["csv"])

def filter_regex_case_insensitive(df, pattern):
    """Helper to filter columns using regex, case-insensitive (avoids NDFrame.filter case= bug)."""
    import re
    cols = [col for col in df.columns if re.search(pattern, col, re.IGNORECASE)]
    return df[cols] if cols else pd.DataFrame()

def calculate_financial_ratios(df):
    ratios = {}
    try:
        # Use helper for regex, case-insensitive matching
        current_assets = filter_regex_case_insensitive(df, "current.*asset")
        total_assets = filter_regex_case_insensitive(df, "total.*asset")
        current_liabilities = filter_regex_case_insensitive(df, "current.*liab")
        total_liabilities = filter_regex_case_insensitive(df, "total.*liab")
        equity = filter_regex_case_insensitive(df, "total.*equity")
        revenue = filter_regex_case_insensitive(df, "revenue|sales")
        net_income = filter_regex_case_insensitive(df, "net.*income|profit")

        # Use first column if found
        current_assets = current_assets.iloc[:,0] if not current_assets.empty else np.nan
        total_assets = total_assets.iloc[:,0] if not total_assets.empty else np.nan
        current_liabilities = current_liabilities.iloc[:,0] if not current_liabilities.empty else np.nan
        total_liabilities = total_liabilities.iloc[:,0] if not total_liabilities.empty else np.nan
        equity = equity.iloc[:,0] if not equity.empty else np.nan
        revenue = revenue.iloc[:,0] if not revenue.empty else np.nan
        net_income = net_income.iloc[:,0] if not net_income.empty else np.nan

        # Calculating ratios
        ratios['Current Ratio'] = (current_assets/current_liabilities).round(2) if not (isinstance(current_assets, float) and np.isnan(current_assets)) and not (isinstance(current_liabilities, float) and np.isnan(current_liabilities)) else "N/A"
        ratios['Debt to Equity Ratio'] = (total_liabilities/equity).round(2) if not (isinstance(total_liabilities, float) and np.isnan(total_liabilities)) and not (isinstance(equity, float) and np.isnan(equity)) else "N/A"
        ratios['Return on Assets (ROA)'] = (net_income/total_assets).round(2) if not (isinstance(net_income, float) and np.isnan(net_income)) and not (isinstance(total_assets, float) and np.isnan(total_assets)) else "N/A"
        ratios['Return on Equity (ROE)'] = (net_income/equity).round(2) if not (isinstance(net_income, float) and np.isnan(net_income)) and not (isinstance(equity, float) and np.isnan(equity)) else "N/A"
        ratios['Net Profit Margin'] = (net_income/revenue).round(2) if not (isinstance(net_income, float) and np.isnan(net_income)) and not (isinstance(revenue, float) and np.isnan(revenue)) else "N/A"
        ratios['Equity Ratio'] = (equity/total_assets).round(2) if not (isinstance(equity, float) and np.isnan(equity)) and not (isinstance(total_assets, float) and np.isnan(total_assets)) else "N/A"
    except Exception as e:
        st.warning(f"Error calculating ratios: {e}")

    return ratios

def plot_metrics(df):
    st.subheader("Key Metrics Over Time")
    date_col = None
    for col in df.columns:
        if "date" in col.lower() or "year" in col.lower():
            date_col = col
            break

    if date_col:
        df_sorted = df.sort_values(by=date_col)
        numeric_cols = df_sorted.select_dtypes(include='number').columns
        metrics_to_plot = [c for c in numeric_cols if any(x in c.lower() for x in ['revenue', 'sales', 'income', 'profit', 'asset', 'liab', 'equity'])]
        for metric in metrics_to_plot:
            st.line_chart(df_sorted.set_index(date_col)[metric])
    else:
        st.info("No date or year column found for time series plots.")

def plot_correlation(df):
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation matrix.")

def anomaly_detection(df):
    st.subheader("ML-based Anomaly Detection (Isolation Forest, KMeans, PCA, Z-score)")
    numeric_df = df.select_dtypes(include='number').dropna(axis=1, how='all')
    anomaly_flags = pd.DataFrame(index=df.index)
    results = {}

    if numeric_df.shape[1] > 2 and numeric_df.shape[0] > 5:
        # Isolation Forest
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        anomaly_flags['iso'] = iso.fit_predict(numeric_df)
        results['Isolation Forest'] = (anomaly_flags['iso'] == -1).sum()

        # KMeans Distance Outlier
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        distances = np.min(kmeans.transform(X_scaled), axis=1)
        threshold = np.percentile(distances, 95)
        anomaly_flags['kmeans'] = (distances > threshold).astype(int)
        results['KMeans Outliers'] = anomaly_flags['kmeans'].sum()

        # PCA Outlier
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        z_scores = np.abs((X_pca - X_pca.mean(axis=0)) / X_pca.std(axis=0))
        anomaly_flags['pca'] = ((z_scores > 3).any(axis=1)).astype(int)
        results['PCA Outliers'] = anomaly_flags['pca'].sum()

        # Z-score anomaly detection per column
        zscore_cols = {}
        for col in numeric_df.columns:
            z = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std())
            zscore_cols[col] = (z > 3)
            anomaly_flags[f'zscore_{col}'] = (z > 3).astype(int)
        results['Z-score (any column)'] = int(any(flag.sum() > 0 for flag in zscore_cols.values()))

        # Show results
        st.write("Anomaly Detection Summary:")
        st.write(results)

        # Display rows flagged by any method
        any_anomaly = (anomaly_flags.sum(axis=1) > 0)
        outlier_rows = df[any_anomaly]
        st.write("Rows flagged as anomalies by any method:")
        st.dataframe(outlier_rows)
        st.markdown("**Visual: PCA Scatterplot of Outliers**")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=(anomaly_flags['iso']==-1), cmap='coolwarm', label='Isolation Forest Outlier')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        st.pyplot(fig)

        # More granular anomaly summary per row
        st.subheader("Granular Anomaly Flags Table")
        st.write(anomaly_flags)
    else:
        st.info("Not enough numeric data for robust anomaly detection.")

    return anomaly_flags

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        # --- Analysis: Financial Ratios ---
        st.header("Financial Ratio Analysis")
        ratios = calculate_financial_ratios(df)
        for name, value in ratios.items():
            st.write(f"**{name}:** {value}")

        # --- Visualizations ---
        plot_metrics(df)
        plot_correlation(df)

        # --- ML Anomaly Detection (Granular) ---
        anomaly_flags = anomaly_detection(df)

    except Exception as e:
        st.error(f"File Error: {e}")
else:
    st.info("Please upload a CSV file to begin analysis.")
