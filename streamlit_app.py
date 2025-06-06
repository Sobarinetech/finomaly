import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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
    - Analyze the data using AI (Gemini)
    - Perform robust financial due diligence with advanced metrics
    - Detect anomalies and outliers using multiple ML methods
    - Visualize key metrics, trends, and anomalies
    """
)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your financial statement (CSV)", type=["csv"])

def calculate_financial_ratios(df):
    ratios = {}
    try:
        # Common column name guesses for financial statements
        current_assets = df.filter(regex="current.*asset", axis=1, case=False).iloc[:,0] if not df.filter(regex="current.*asset", axis=1, case=False).empty else np.nan
        total_assets = df.filter(regex="total.*asset", axis=1, case=False).iloc[:,0] if not df.filter(regex="total.*asset", axis=1, case=False).empty else np.nan
        current_liabilities = df.filter(regex="current.*liab", axis=1, case=False).iloc[:,0] if not df.filter(regex="current.*liab", axis=1, case=False).empty else np.nan
        total_liabilities = df.filter(regex="total.*liab", axis=1, case=False).iloc[:,0] if not df.filter(regex="total.*liab", axis=1, case=False).empty else np.nan
        equity = df.filter(regex="total.*equity", axis=1, case=False).iloc[:,0] if not df.filter(regex="total.*equity", axis=1, case=False).empty else np.nan
        revenue = df.filter(regex="revenue|sales", axis=1, case=False).iloc[:,0] if not df.filter(regex="revenue|sales", axis=1, case=False).empty else np.nan
        net_income = df.filter(regex="net.*income|profit", axis=1, case=False).iloc[:,0] if not df.filter(regex="net.*income|profit", axis=1, case=False).empty else np.nan

        # Calculating ratios
        ratios['Current Ratio'] = (current_assets/current_liabilities).round(2) if not (np.isnan(current_assets).any() or np.isnan(current_liabilities).any()) else "N/A"
        ratios['Debt to Equity Ratio'] = (total_liabilities/equity).round(2) if not (np.isnan(total_liabilities).any() or np.isnan(equity).any()) else "N/A"
        ratios['Return on Assets (ROA)'] = (net_income/total_assets).round(2) if not (np.isnan(net_income).any() or np.isnan(total_assets).any()) else "N/A"
        ratios['Return on Equity (ROE)'] = (net_income/equity).round(2) if not (np.isnan(net_income).any() or np.isnan(equity).any()) else "N/A"
        ratios['Net Profit Margin'] = (net_income/revenue).round(2) if not (np.isnan(net_income).any() or np.isnan(revenue).any()) else "N/A"
        ratios['Equity Ratio'] = (equity/total_assets).round(2) if not (np.isnan(equity).any() or np.isnan(total_assets).any()) else "N/A"
    except Exception as e:
        st.warning(f"Error calculating ratios: {e}")

    return ratios

def plot_metrics(df):
    st.subheader("Key Metrics Over Time")
    date_col = None
    # Try to find a date column
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
    st.subheader("ML-based Anomaly Detection (Isolation Forest, KMeans, PCA)")
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

        # Show results
        outlier_rows = df[(anomaly_flags['iso'] == -1) | (anomaly_flags['kmeans'] == 1) | (anomaly_flags['pca'] == 1)]
        st.write("Anomaly Detection Summary:")
        st.write(results)
        st.write("Rows flagged as anomalies by any method:")
        st.dataframe(outlier_rows)
        st.markdown("**Visual: PCA Scatterplot of Outliers**")
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:,0], X_pca[:,1], c=(anomaly_flags['iso']==-1), cmap='coolwarm', label='Isolation Forest Outlier')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        st.pyplot(fig)
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

        # --- ML Anomaly Detection ---
        anomaly_flags = anomaly_detection(df)

        # --- Due Diligence AI Analysis ---
        st.header("AI Financial Due Diligence & Insights")
        default_prompt = (
            "You are a seasoned financial due diligence expert. "
            "Analyze the following financial statement data for risks, fraud indicators, financial health, key trends, and any anomalies. "
            "Report on liquidity, leverage, profitability, and solvency using all available metrics and ratios. "
            "Comment on trends, outliers, and any suspicious data points. "
            "Provide an overall assessment and any recommendations.\n\n"
            f"Financial Data (first 20 rows):\n{df.head(20).to_csv(index=False)}"
        )
        user_prompt = st.text_area(
            "Optional: Add specific questions or context for AI analysis.",
            "",
            help="You can ask about profitability, liquidity ratios, trends, unusual figures, potential fraud, etc."
        )
        ai_prompt = default_prompt + ('\n\nUser question: ' + user_prompt if user_prompt.strip() else '')

        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing with Gemini AI..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(ai_prompt)
                    st.write("## AI Due Diligence Report")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")

    except Exception as e:
        st.error(f"File Error: {e}")
else:
    st.info("Please upload a CSV file to begin analysis.")

# --- Prompt Playground (from provided code) ---
st.divider()
st.header("Gemini Prompt Playground")
st.write("Test the GenAI model with your own prompt below.")
prompt = st.text_input("Enter your prompt:", "Best alternatives to javascript?")
if st.button("Generate Response"):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        st.write("Response:")
        st.write(response.text)
    except Exception as e:
        st.error(f"Error: {e}")
