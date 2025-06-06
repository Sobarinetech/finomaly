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
import re
import tempfile
import os

# For PDF parsing
try:
    import pdfplumber
except ImportError:
    st.error("Please install pdfplumber: pip install pdfplumber")
    raise

# --- AI Configuration (can be removed if not using AI directly) ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="AI Financial Statement Analyzer", layout="wide")
st.title("Ever AI - Enhanced Financial Statement Analyzer")
st.write(
    """
    **Upload your financial statements in CSV or PDF (annual report) format.**  
    This tool will:
    - Analyze the data using advanced metrics
    - Detect anomalies and outliers using multiple ML methods
    - Visualize key metrics, trends, and anomalies
    - For PDFs, it will extract and process multiple statements if present (Balance Sheet, Income Statement, Cash Flow, etc.)
    """
)

def filter_regex_case_insensitive(df, pattern):
    """Helper to filter columns using regex, case-insensitive."""
    cols = [col for col in df.columns if re.search(pattern, col, re.IGNORECASE)]
    return df[cols] if cols else pd.DataFrame()

def calculate_financial_ratios(df):
    ratios = {}
    try:
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
        def _ratio(a, b):
            try:
                if isinstance(a, float) and np.isnan(a): return "N/A"
                if isinstance(b, float) and np.isnan(b): return "N/A"
                if (b == 0).any() if hasattr(b, 'any') else b == 0: return "N/A"
                return (a / b).round(2)
            except Exception:
                return "N/A"

        ratios['Current Ratio'] = _ratio(current_assets, current_liabilities)
        ratios['Debt to Equity Ratio'] = _ratio(total_liabilities, equity)
        ratios['Return on Assets (ROA)'] = _ratio(net_income, total_assets)
        ratios['Return on Equity (ROE)'] = _ratio(net_income, equity)
        ratios['Net Profit Margin'] = _ratio(net_income, revenue)
        ratios['Equity Ratio'] = _ratio(equity, total_assets)
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
        results['Isolation Forest'] = int((anomaly_flags['iso'] == -1).sum())

        # KMeans Distance Outlier
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        distances = np.min(kmeans.transform(X_scaled), axis=1)
        threshold = np.percentile(distances, 95)
        anomaly_flags['kmeans'] = (distances > threshold).astype(int)
        results['KMeans Outliers'] = int(anomaly_flags['kmeans'].sum())

        # PCA Outlier
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        z_scores = np.abs((X_pca - X_pca.mean(axis=0)) / X_pca.std(axis=0))
        anomaly_flags['pca'] = ((z_scores > 3).any(axis=1)).astype(int)
        results['PCA Outliers'] = int(anomaly_flags['pca'].sum())

        # Z-score anomaly detection per column
        zscore_cols = {}
        for col in numeric_df.columns:
            z = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std())
            zscore_cols[col] = (z > 3)
            anomaly_flags[f'zscore_{col}'] = (z > 3).astype(int)
        # Number of rows that are anomalous in any column
        results['Z-score (any column)'] = int(anomaly_flags[[c for c in anomaly_flags.columns if c.startswith('zscore_')]].any(axis=1).sum())

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

def extract_statements_from_pdf(pdf_path):
    """
    Extract tables for Balance Sheet, Income Statement, Cash Flow from an annual report PDF.
    Returns a dict of {statement_name: dataframe}
    """
    statement_patterns = {
        "balance_sheet": re.compile(r"balance\s*sheet", re.I),
        "income_statement": re.compile(r"(income\s*statement|statement\s*of\s*profit\s*and\s*loss)", re.I),
        "cash_flow": re.compile(r"cash\s*flow", re.I)
    }
    statements = {}
    with pdfplumber.open(pdf_path) as pdf:
        current_statement = None
        tables_buffer = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            for key, pattern in statement_patterns.items():
                if pattern.search(text):
                    current_statement = key
                    tables_buffer = []
                    break
            # If we're in a statement, try to extract tables
            if current_statement is not None:
                tables = page.extract_tables()
                for table in tables:
                    # Filter out silly tables
                    if table and len(table) > 2 and len(table[0]) > 2:
                        try:
                            df = pd.DataFrame(table)
                            # Try to use first row as header if it's text
                            if df.iloc[0].apply(lambda x: isinstance(x, str)).all():
                                df.columns = df.iloc[0]
                                df = df[1:]
                            # Clean empty columns
                            df = df.loc[:, ~df.columns.duplicated()]
                            df = df.dropna(axis=1, how='all')
                            tables_buffer.append(df)
                        except Exception:
                            continue
            # If we find a new statement or reach end, consolidate
            if current_statement is not None and tables_buffer:
                # Heuristic: if next page doesn't have the statement name, stop adding
                next_page_text = ""
                if page.page_number < len(pdf.pages):
                    next_page_text = pdf.pages[page.page_number].extract_text() or ""
                if not statement_patterns[current_statement].search(next_page_text):
                    # Merge all tables (row-wise)
                    combined = pd.concat(tables_buffer, axis=0, ignore_index=True)
                    statements[current_statement] = combined
                    current_statement = None
                    tables_buffer = []
    return statements

# --- File Upload (CSV or PDF) ---
uploaded_files = st.file_uploader(
    "Upload your financial statement(s) (CSV or PDF, multiple allowed)", 
    type=["csv", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"Processing: {uploaded_file.name}")

        # --- CSV ---
        if uploaded_file.name.lower().endswith('.csv'):
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

        # --- PDF ---
        elif uploaded_file.name.lower().endswith('.pdf'):
            # Save PDF to temp file for pdfplumber
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmp_pdf_path = tmpfile.name

            try:
                statements = extract_statements_from_pdf(tmp_pdf_path)
                if not statements:
                    st.warning("No recognizable statements found in PDF. Please ensure the PDF contains tables for Balance Sheet, Income Statement, or Cash Flow.")
                else:
                    for name, df in statements.items():
                        st.markdown(f"### Extracted Statement: {name.replace('_', ' ').title()}")
                        st.dataframe(df.head(20))

                        # Try to convert numeric columns
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",","").str.replace("$","").str.strip(), errors='ignore')

                        st.header(f"{name.replace('_', ' ').title()} - Financial Ratio Analysis")
                        ratios = calculate_financial_ratios(df)
                        for rname, value in ratios.items():
                            st.write(f"**{rname}:** {value}")

                        plot_metrics(df)
                        plot_correlation(df)
                        anomaly_flags = anomaly_detection(df)
            except Exception as e:
                st.error(f"PDF Extraction Error: {e}")
            finally:
                try:
                    os.unlink(tmp_pdf_path)
                except Exception:
                    pass
else:
    st.info("Please upload CSV or PDF file(s) to begin analysis.")
