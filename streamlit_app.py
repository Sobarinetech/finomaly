import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import zscore, iqr, kurtosis, skew, normaltest
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Configure Google Gemini API Key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="AI Financial Analyzer", layout="wide")
st.title("🧠 AI-Powered Financial Statement Analyzer")
st.markdown("""
Upload financial statements for deep anomaly detection, insights, and auto-generated audit reports.
""")

# Helper function to generate download link
def get_table_download_link(df, filename="audit_report.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Audit Report (CSV)</a>'

# Helper for audit summary PDF
def create_pdf_report(report_str, filename="Audit_Report.txt"):
    b = BytesIO()
    b.write(report_str.encode())
    b.seek(0)
    return b

# Sidebar navigation
analyze_mode = st.sidebar.selectbox(
    "Choose analysis type",
    [
        "AI Q&A",
        "Financial Due Diligence (AI-powered)",
        "Anomaly Detection & Audit (25+ Features)",
    ]
)

# 1. AI Q&A
if analyze_mode == "AI Q&A":
    st.header("💬 Chat with Financial AI")
    prompt = st.text_input("Enter your prompt (e.g., 'What is EBITDA?')", "")
    if st.button("Generate AI Response"):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            st.subheader("AI Response:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")

# 2. Financial Due Diligence
elif analyze_mode == "Financial Due Diligence (AI-powered)":
    st.header("🔎 Financial Due Diligence (AI-powered)")
    uploaded_file = st.file_uploader("Upload a financial statement (CSV or Excel)", type=['csv', 'xlsx'])
    user_question = st.text_area("Ask a due diligence question about your data:", "")
    if uploaded_file and user_question and st.button("Analyze with AI"):
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            preview = df.head(20).to_csv(index=False)
            ai_prompt = f"""You are an expert in financial due diligence. Analyze the following table and answer:\n{preview}\n\nQuestion: {user_question}\n\nAnalysis:"""
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(ai_prompt)
            st.subheader("AI Due Diligence Analysis:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")
    elif uploaded_file and not user_question:
        st.info("Please enter a due diligence question.")

# 3. Anomaly Detection (25+ Features) & Audit Report
elif analyze_mode == "Anomaly Detection & Audit (25+ Features)":
    st.header("🚨 Deep Anomaly Detection & Audit Insights")
    uploaded_file = st.file_uploader("Upload a financial statement (CSV or Excel)", type=['csv', 'xlsx'], key="anomaly")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # Numeric columns selection
            num_cols = df.select_dtypes(include='number').columns.tolist()
            if not num_cols:
                st.warning("No numeric columns found for anomaly detection.")
            else:
                selected_cols = st.multiselect("Select columns for anomaly detection", num_cols, default=num_cols)
                if st.button("Run Deep Anomaly Detection & Insights"):
                    X = df[selected_cols].copy()
                    X = X.fillna(X.mean()).replace([np.inf, -np.inf], 0)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # 1. Z-Score Outliers
                    z_scores = np.abs(zscore(X))
                    z_outliers = (z_scores > 3).any(axis=1)

                    # 2. IQR Method Outliers
                    iqr_scores = ((X < (X.quantile(0.25) - 1.5 * (X.quantile(0.75) - X.quantile(0.25)))) |
                                  (X > (X.quantile(0.75) + 1.5 * (X.quantile(0.75) - X.quantile(0.25))))).any(axis=1)

                    # 3. Isolation Forest
                    iforest = IsolationForest(n_estimators=200, contamination=0.05, random_state=0)
                    iso_preds = iforest.fit_predict(X_scaled)
                    iso_outliers = (iso_preds == -1)

                    # 4. Elliptic Envelope
                    try:
                        ee = EllipticEnvelope(contamination=0.05)
                        ee_preds = ee.fit_predict(X_scaled)
                        ee_outliers = (ee_preds == -1)
                    except Exception:
                        ee_outliers = np.zeros(X.shape[0], dtype=bool)

                    # 5. Local Outlier Factor
                    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
                    lof_preds = lof.fit_predict(X_scaled)
                    lof_outliers = (lof_preds == -1)

                    # 6. One-Class SVM
                    oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
                    svm_preds = oc_svm.fit_predict(X_scaled)
                    svm_outliers = (svm_preds == -1)

                    # 7. DBSCAN Clustering as Anomaly
                    db = DBSCAN(eps=2, min_samples=5).fit(X_scaled)
                    dbscan_outliers = (db.labels_ == -1)

                    # 8. Mahalanobis Distance
                    mean_vec = np.mean(X_scaled, axis=0)
                    cov_mat = np.cov(X_scaled, rowvar=False)
                    try:
                        inv_cov_mat = np.linalg.inv(cov_mat)
                        left = X_scaled - mean_vec
                        mahal = np.einsum('ij,jk,ik->i', left, inv_cov_mat, left)
                        mahal_outliers = (mahal > np.percentile(mahal, 95))
                    except Exception:
                        mahal_outliers = np.zeros(X.shape[0], dtype=bool)

                    # 9. Skewness/Kurtosis
                    skewness = skew(X)
                    kurt = kurtosis(X)
                    normaltests = [normaltest(X[col])[1] < 0.05 for col in X.columns]
                    
                    # 10. Constant values
                    constant_cols = [col for col in X.columns if X[col].nunique() == 1]

                    # 11. Zero variance
                    zero_var_cols = [col for col in X.columns if X[col].std() == 0]

                    # 12. High correlation
                    corr_matrix = X.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr_pairs = [(col, idx) for col in upper.columns for idx in upper.index if abs(upper.loc[idx, col]) > 0.95]

                    # 13. Duplicate Rows
                    duplicate_rows = df.duplicated().sum()

                    # 14. Negative Balances
                    negative_row = (X < 0).any(axis=1)

                    # 15. Large single transactions
                    large_single = ((X > (X.mean() + 4 * X.std())).any(axis=1))

                    # 16. Gaps in sequential numbers (if any column is sequential)
                    gap_dict = {}
                    for col in X.columns:
                        if pd.api.types.is_integer_dtype(X[col]):
                            diffs = np.diff(np.sort(X[col].unique()))
                            if (diffs > 1).any():
                                gap_dict[col] = np.where(diffs > 1)[0].tolist()

                    # 17. Outliers by MinMax scaling > 1 or < 0 (data errors)
                    mm = MinMaxScaler()
                    mm_data = mm.fit_transform(X)
                    mm_outliers = ((mm_data > 1) | (mm_data < 0)).any(axis=1)

                    # 18. Outliers by Robust Scaling > 3 or < -3
                    rb = RobustScaler()
                    rb_data = rb.fit_transform(X)
                    rb_outliers = ((rb_data > 3) | (rb_data < -3)).any(axis=1)

                    # 19. Sudden changes (delta > threshold)
                    sudden_change = pd.DataFrame(np.abs(X.diff())).max(axis=1) > (3 * X.std().mean())

                    # 20. Outliers by percentile (top/bottom 1%)
                    perc_outliers = ((X > X.quantile(0.99)) | (X < X.quantile(0.01))).any(axis=1)

                    # 21. Seasonal/periodic anomaly (rolling mean outliers)
                    roll_mean = X.rolling(window=5, min_periods=1).mean()
                    seasonal_outlier = (np.abs(X - roll_mean) > 4 * X.std().mean()).any(axis=1)

                    # 22. Outliers by MAD (Median Absolute Deviation)
                    med = X.median()
                    mad = (np.abs(X - med)).median()
                    mad_outlier = ((np.abs(X - med) / (mad + 1e-9)) > 6).any(axis=1)

                    # 23. Outliers near fiscal year end (use index if date provided)
                    fiscal_outlier = np.zeros(X.shape[0], dtype=bool)
                    if 'date' in df.columns:
                        fiscal_mask = pd.to_datetime(df['date']).dt.month.isin([12, 1])
                        fiscal_outlier = ((X[fiscal_mask] > X.mean() + 3 * X.std()).any(axis=1))

                    # 24. Repeated values (potential fraud)
                    repeated_val = (X.apply(lambda col: col.duplicated(keep=False))).any(axis=1)

                    # 25. Outliers by custom thresholds (user input)
                    custom_dict = {}
                    with st.expander("Set Custom Thresholds (Optional)"):
                        for col in selected_cols:
                            val = st.number_input(f"Max allowed for {col} (leave blank for none)", value=float("inf"))
                            if val != float("inf"):
                                custom_dict[col] = val
                    custom_thresh_outlier = np.zeros(X.shape[0], dtype=bool)
                    for col, thresh in custom_dict.items():
                        custom_thresh_outlier = custom_thresh_outlier | (X[col] > thresh)

                    # Combine all anomaly results
                    anomaly_matrix = np.vstack([
                        z_outliers, iqr_scores, iso_outliers, ee_outliers, lof_outliers,
                        svm_outliers, dbscan_outliers, mahal_outliers, negative_row, large_single,
                        mm_outliers, rb_outliers, sudden_change, perc_outliers, seasonal_outlier,
                        mad_outlier, fiscal_outlier, repeated_val, custom_thresh_outlier,
                    ])
                    anomaly_votes = anomaly_matrix.sum(axis=0)
                    # Mark as anomaly if detected by >=3 methods
                    anomaly_flag = (anomaly_votes >= 3).astype(int)

                    outlier_df = df.copy()
                    outlier_df['Anomaly_Score'] = anomaly_votes
                    outlier_df['Anomaly_Flag'] = np.where(anomaly_flag, "Anomaly", "Normal")

                    # Insights Generation
                    insights = []
                    insights.append(f"Total Rows: {df.shape[0]}")
                    insights.append(f"Total Detected Anomalies: {outlier_df['Anomaly_Flag'].value_counts().get('Anomaly', 0)}")
                    insights.append(f"Duplicate Rows: {duplicate_rows}")
                    if constant_cols:
                        insights.append(f"Columns with Constant Value: {constant_cols}")
                    if zero_var_cols:
                        insights.append(f"Columns with Zero Variance: {zero_var_cols}")
                    if high_corr_pairs:
                        insights.append(f"Highly Correlated Features (r>0.95): {high_corr_pairs}")
                    if gap_dict:
                        insights.append(f"Gaps in Sequential Columns: {gap_dict}")
                    for idx, col in enumerate(selected_cols):
                        insights.append(f"Skewness of {col}: {skewness[idx]:.2f}")
                        insights.append(f"Kurtosis of {col}: {kurt[idx]:.2f}")
                        if normaltests[idx]:
                            insights.append(f"{col} failed normality test (may indicate non-normal or anomalous distribution)")

                    # Visualizations
                    st.subheader("Summary Insights")
                    st.write("\n".join(insights))

                    st.subheader("Anomaly Results Table")
                    st.dataframe(outlier_df)

                    st.markdown(get_table_download_link(outlier_df), unsafe_allow_html=True)

                    # Audit Report Generation
                    audit_report = f"""
AI-Powered Financial Audit Report
=================================

Summary:
--------
{chr(10).join(insights)}

Detected Anomalies: {outlier_df['Anomaly_Flag'].value_counts().get('Anomaly', 0)} / {df.shape[0]} rows

Feature details:
----------------
- Z-Score Outliers: {z_outliers.sum()}
- IQR Outliers: {iqr_scores.sum()}
- Isolation Forest Outliers: {iso_outliers.sum()}
- Elliptic Envelope Outliers: {ee_outliers.sum()}
- Local Outlier Factor: {lof_outliers.sum()}
- One-Class SVM Outliers: {svm_outliers.sum()}
- DBSCAN Outliers: {dbscan_outliers.sum()}
- Mahalanobis Distance Outliers: {mahal_outliers.sum()}
- Negative Values: {negative_row.sum()}
- Large Single Transactions: {large_single.sum()}
- MinMax Outliers: {mm_outliers.sum()}
- Robust Outliers: {rb_outliers.sum()}
- Sudden Change: {sudden_change.sum()}
- Percentile Outliers: {perc_outliers.sum()}
- Rolling Mean Outliers: {seasonal_outlier.sum()}
- MAD Outliers: {mad_outlier.sum()}
- Fiscal Period Outliers: {fiscal_outlier.sum()}
- Repeated Value Outliers: {repeated_val.sum()}
- Custom Threshold Outliers: {custom_thresh_outlier.sum()}

End of Report

"""
                    st.subheader("Audit Report Download")
                    st.download_button(
                        label="Download Audit Report (TXT)",
                        data=create_pdf_report(audit_report),
                        file_name="Audit_Report.txt"
                    )

                    # Multi-feature anomaly visualizations
                    st.subheader("Visualizations")
                    col1, col2 = st.columns(2)

                    # 1. Heatmap of correlation
                    with col1:
                        st.markdown("**Correlation Heatmap**")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                        st.pyplot(fig)

                    # 2. Distribution for each selected column
                    with col2:
                        st.markdown("**Distributions with Anomalies Highlighted**")
                        for col in selected_cols[:3]:  # limit to 3 for display
                            fig, ax = plt.subplots()
                            sns.histplot(df[col], bins=30, kde=True, ax=ax, color="skyblue", label="All")
                            sns.histplot(df.loc[outlier_df['Anomaly_Flag']=="Anomaly", col], bins=30, color="red", ax=ax, label="Anomalies")
                            ax.legend()
                            ax.set_title(f"Distribution of {col} (Red: Anomalies)")
                            st.pyplot(fig)

                    # 3. Scatter of first two selected cols with anomalies
                    if len(selected_cols) >= 2:
                        fig, ax = plt.subplots()
                        ax.scatter(df[selected_cols[0]], df[selected_cols[1]], c=(outlier_df['Anomaly_Flag'] == "Anomaly").astype(int), cmap='coolwarm', label="Anomaly Flag")
                        ax.set_xlabel(selected_cols[0])
                        ax.set_ylabel(selected_cols[1])
                        ax.set_title(f"Scatter: {selected_cols[0]} vs {selected_cols[1]}")
                        st.pyplot(fig)

                    st.success("Audit completed. Download your audit report above.")

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built with Streamlit, Google Gemini, Scikit-learn, and Matplotlib. © 2025")
