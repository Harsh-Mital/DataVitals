import streamlit as st
import pandas as pd
import chardet  # for encoding detection

# Local imports
from data_quality import check_missing, check_duplicates, check_schema
from anomaly_detection import anomaly_scatter_plot, anomaly_box_plot
from report_generator import generate_pdf_report
from visual import missing_values_heatmap

st.set_page_config(page_title="📊 Automated Data Quality Tool", layout="wide")

st.title("📊 Automated Data Quality & Anomaly Detection Tool")

# -----------------------------
# Safe CSV Loader
# -----------------------------
def load_csv(uploaded_file):
    try:
        # Detect encoding first
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]

        # Reset pointer after reading
        uploaded_file.seek(0)

        # Read CSV with safe options
        df = pd.read_csv(
            uploaded_file,
            encoding=encoding,
            engine="python",          # more tolerant parser
            on_bad_lines="skip"       # skip malformed rows
        )
        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    data = load_csv(uploaded_file)
    if data is None:
        st.stop()

    st.write("### Data Preview", data.head())

    # -----------------------------
    # Data Quality Checks
    # -----------------------------
    missing = check_missing(data)
    duplicates = check_duplicates(data)
    schema = check_schema(data)

    # Normalize results
    if hasattr(missing, "to_dict"):   
        missing = missing.to_dict()
    if hasattr(duplicates, "to_dict"):  
        duplicates = duplicates.to_dict()
    if hasattr(schema, "to_dict"):
        schema = schema.to_dict()

    st.write("### Data Quality Checks")
    st.write("Missing Values:", missing)
    st.write("Duplicates:", duplicates)
    st.write("Schema Issues:", schema)

    # -----------------------------
    # Visualization: Missing Values
    # -----------------------------
    st.write("### Missing Values Heatmap")
    fig_missing = missing_values_heatmap(data)
    st.pyplot(fig_missing)

    # -----------------------------
    # Anomaly Detection
    # -----------------------------
    st.write("### Anomaly Detection Visuals")
    figs = [fig_missing]   # collect figs for PDF

    if len(data.select_dtypes(include="number").columns) >= 2:
        num_cols = data.select_dtypes(include="number").columns
        x_col = st.selectbox("X-axis column", num_cols)
        y_col = st.selectbox("Y-axis column", num_cols, index=1)

        # Simple anomaly detection: z-score > 3
        anomalies = data[(data[x_col] - data[x_col].mean()).abs() > 3*data[x_col].std()]

        # Scatter Plot
        fig1 = anomaly_scatter_plot(data, x_col, y_col, anomalies)
        st.pyplot(fig1)
        figs.append(fig1)

        # Box Plot
        fig2 = anomaly_box_plot(data, x_col, anomalies)
        st.pyplot(fig2)
        figs.append(fig2)

    # -----------------------------
    # PDF Report Generation
    # -----------------------------
    st.write("### Generate PDF Report")
    if st.button("Export Report"):
        summary = {
            "Missing Values": missing,
            "Duplicates": duplicates,
            "Schema Issues": schema
        }
        generate_pdf_report(summary, figs, "summary_report.pdf")

        st.success("✅ Report generated successfully!")
        with open("summary_report.pdf", "rb") as f:
            st.download_button(
                label="📥 Download Report",
                data=f,
                file_name="summary_report.pdf",
                mime="application/pdf"
            )
