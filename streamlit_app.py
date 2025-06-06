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

try:
    import pdfplumber
except ImportError:
    st.error("Please install pdfplumber: pip install pdfplumber")
    raise

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
    cols = [col for col in df.columns if re.search(pattern, str(col), re.IGNORECASE)]
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

        current_assets = current_assets.iloc[:, 0] if not current_assets.empty else np.nan
        total_assets = total_assets.iloc[:, 0] if not total_assets.empty else np.nan
        current_liabilities = current_liabilities.iloc[:, 0] if not current_liabilities.empty else np.nan
        total_liabilities = total_liabilities.iloc[:, 0] if not total_liabilities.empty else np.nan
        equity = equity.iloc[:, 0] if not equity.empty else np.nan
        revenue = revenue.iloc[:, 0] if not revenue.empty else np.nan
        net_income = net_income.iloc[:, 0] if not net_income.empty else np.nan

        def _ratio(a, b):
            try:
                if isinstance(a, float) and np.isnan(a): return "N/A"
                if isinstance(b, float) and np.isnan(b): return "N/A"
                if (b == 0).any() if hasattr(b, 'any') else b == 0: return "N/A"
                return (a / b).round(2) if hasattr(a, 'round') else round(a / b, 2)
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
        if isinstance(col, str) and ("date" in col.lower() or "year" in col.lower()):
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
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        anomaly_flags['iso'] = iso.fit_predict(numeric_df)
        results['Isolation Forest'] = int((anomaly_flags['iso'] == -1).sum())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        distances = np.min(kmeans.transform(X_scaled), axis=1)
        threshold = np.percentile(distances, 95)
        anomaly_flags['kmeans'] = (distances > threshold).astype(int)
        results['KMeans Outliers'] = int(anomaly_flags['kmeans'].sum())

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        z_scores = np.abs((X_pca - X_pca.mean(axis=0)) / X_pca.std(axis=0))
        anomaly_flags['pca'] = ((z_scores > 3).any(axis=1)).astype(int)
        results['PCA Outliers'] = int(anomaly_flags['pca'].sum())

        zscore_cols = {}
        for col in numeric_df.columns:
            z = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std())
            zscore_cols[col] = (z > 3)
            anomaly_flags[f'zscore_{col}'] = (z > 3).astype(int)
        results['Z-score (any column)'] = int(anomaly_flags[[c for c in anomaly_flags.columns if c.startswith('zscore_')]].any(axis=1).sum())

        st.write("Anomaly Detection Summary:")
        st.write(results)

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

        st.subheader("Granular Anomaly Flags Table")
        st.write(anomaly_flags)
    else:
        st.info("Not enough numeric data for robust anomaly detection.")

    return anomaly_flags

def clean_table_dataframe(df):
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, [not (str(c).strip() == "" or str(c).lower().startswith("unnamed")) for c in df.columns]]
    df.columns = [str(col) for col in df.columns]
    for c in df.columns:
        if str(c).strip().lower() in ['#', 'index', 'no', 's.no']:
            df = df.drop(columns=[c])
    df = df.loc[:, [not re.search(r'(crore|lakh|unit|thousand|million)', str(c), re.I) for c in df.columns]]
    return df

def extract_statement_blocks(text, keyword_patterns):
    blocks = {k: [] for k in keyword_patterns.keys()}
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        for key, regex in keyword_patterns.items():
            if regex.search(lines[idx]):
                start = idx
                end = start + 1
                while end < len(lines):
                    found_next = False
                    for other_key, other_re in keyword_patterns.items():
                        if other_key != key and other_re.search(lines[end]):
                            found_next = True
                            break
                    if found_next or (end - start > 50):
                        break
                    end += 1
                content = "\n".join(lines[start:end])
                if len(content.strip()) > 0:
                    blocks[key].append(content)
                idx = end
                break
        idx += 1
    return blocks

def pdf_statement_tables(pdf_path):
    statement_patterns = {
        "Balance Sheet": re.compile(r"balance\s*sheet", re.I),
        "Income Statement": re.compile(r"(income\s*statement|statement\s*of\s*profit\s*and\s*loss)", re.I),
        "Cash Flow": re.compile(r"cash\s*flow", re.I)
    }
    statement_tables = {k: [] for k in statement_patterns.keys()}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for key, regex in statement_patterns.items():
                if regex.search(text):
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 2 and len(table[0]) > 2:
                            df = pd.DataFrame(table)
                            for row_i in range(min(3, len(df))):
                                header = [str(x).strip() for x in df.iloc[row_i].values]
                                if sum([h != "" for h in header]) >= 2:
                                    df.columns = header
                                    df = df[row_i + 1:]
                                    break
                            df = clean_table_dataframe(df)
                            statement_tables[key].append(df)
    if not any(statement_tables.values()):
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages_text.append(page.extract_text() or "")
        full_text = "\n".join(pages_text)
        blocks = extract_statement_blocks(full_text, statement_patterns)
        for key, blocklist in blocks.items():
            for block in blocklist:
                try:
                    df = pd.read_fwf(pd.io.common.StringIO(block), header=None)
                    df = clean_table_dataframe(df)
                    if df.shape[1] > 2:
                        statement_tables[key].append(df)
                except Exception:
                    pass
    return statement_tables

uploaded_files = st.file_uploader(
    "Upload your financial statement(s) (CSV or PDF, multiple allowed)", 
    type=["csv", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"Processing: {uploaded_file.name}")

        if uploaded_file.name.lower().endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                st.header("Financial Ratio Analysis")
                ratios = calculate_financial_ratios(df)
                for name, value in ratios.items():
                    st.write(f"**{name}:** {value}")
                plot_metrics(df)
                plot_correlation(df)
                anomaly_flags = anomaly_detection(df)
            except Exception as e:
                st.error(f"File Error: {e}")

        elif uploaded_file.name.lower().endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmp_pdf_path = tmpfile.name

            try:
                statement_tables = pdf_statement_tables(tmp_pdf_path)
                if not any(statement_tables.values()):
                    st.warning("No recognizable statements found in PDF. Please ensure the PDF contains tables for Balance Sheet, Income Statement, or Cash Flow.")
                else:
                    for name, tables in statement_tables.items():
                        for i, df in enumerate(tables):
                            st.markdown(f"### Extracted Statement: {name} (Table {i+1})")
                            st.dataframe(df.head(20))
                            # Try to convert numeric columns (no FutureWarning)
                            for col in df.columns:
                                try:
                                    df[col] = pd.to_numeric(
                                        df[col].astype(str)
                                        .replace(",", "", regex=True)
                                        .replace("$", "", regex=True)
                                        .str.strip()
                                    )
                                except Exception:
                                    pass
                            st.header(f"{name} - Financial Ratio Analysis")
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
