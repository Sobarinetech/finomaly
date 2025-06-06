import streamlit as st
import google.generativeai as genai
import pandas as pd
from sklearn.ensemble import IsolationForest

# --- AI Configuration ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- Streamlit UI ---
st.set_page_config(page_title="AI Financial Statement Analyzer", layout="wide")
st.title("Ever AI - Financial Statement Analyzer")
st.write(
    "Upload your financial statements in CSV format. This tool will:\n"
    "- Analyze the data using AI (Gemini)\n"
    "- Perform financial due diligence\n"
    "- Check for anomalies using machine learning\n"
)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your financial statement (CSV)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        # --- ML Anomaly Detection ---
        st.subheader("ML-based Anomaly Detection")
        numeric_df = df.select_dtypes(include='number').dropna(axis=1, how='all')
        if numeric_df.shape[1] > 0 and numeric_df.shape[0] > 2:
            iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            anomalies = iso.fit_predict(numeric_df)
            df['Anomaly'] = anomalies
            anomaly_count = (anomalies == -1).sum()
            st.write(f"Detected **{anomaly_count}** anomalies (rows flagged as outliers):")
            st.dataframe(df[df['Anomaly'] == -1])
        else:
            st.info("Not enough numeric data for anomaly detection.")

        # --- Due Diligence AI Analysis ---
        st.subheader("AI Financial Due Diligence & Insights")
        default_prompt = (
            "You are a financial due diligence expert. "
            "Analyze the following financial statement data for potential risks, red flags, and overall health. "
            "Provide insights, key ratios, and any anomalies you notice.\n\n"
            f"Financial Data (CSV):\n{df.head(20).to_csv(index=False)}"
        )
        user_prompt = st.text_area(
            "Optional: Add specific questions or context for AI analysis.",
            "",
            help="You can ask about profitability, liquidity ratios, trends, unusual figures, etc."
        )
        ai_prompt = default_prompt + ('\n\nUser question: ' + user_prompt if user_prompt.strip() else '')

        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing with Gemini AI..."):
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash')
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
