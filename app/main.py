import streamlit as st
import pandas as pd
import chardet
import csv
from data_quality import missing_values_heatmap, data_quality_summary
from anomaly_detection import detect_anomalies, anomaly_scatter_plot, anomaly_box_plot
from report_generator import generate_report

st.title("📊 Data Quality & Anomaly Detection Tool")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    
    # --- Detect encoding ---
    raw_data = uploaded_file.read()
    encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
    uploaded_file.seek(0)
    
    # --- Detect delimiter ---
    sample = uploaded_file.read(2048).decode(encoding, errors="ignore")
    uploaded_file.seek(0)
    try:
        sep = csv.Sniffer().sniff(sample).delimiter
    except Exception:
        sep = ","
    
    # --- Read CSV robustly ---
    try:
        data = pd.read_csv(uploaded_file, encoding=encoding, sep=sep,
                           on_bad_lines="skip", engine="python")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file, encoding="latin1", sep=sep,
                           on_bad_lines="skip", engine="python")
    
    # --- Data Preview ---
    st.write("### Data Preview", data.head())
    
    # --- Data Quality Summary ---
    duplicated_count, schema = data_quality_summary(data)
    st.write(f"### Number of duplicated rows: {duplicated_count}")
    st.write("### Data Schema")
    st.dataframe(schema)
    
    # --- Missing Values Heatmap ---
    st.write("### Missing Values Heatmap")
    st.pyplot(missing_values_heatmap(data))
    
    # --- Detect anomalies ---
    anomalies = detect_anomalies(data)
    st.write("### Anomaly Detection")
    st.write(anomalies.head())
    
    # --- Scatter plot for numeric columns ---
    numeric_cols = data.select_dtypes(include="number").columns
    x_col = y_col = None
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Select X-axis for Scatter Plot", numeric_cols, index=0)
        y_col = st.selectbox("Select Y-axis for Scatter Plot", numeric_cols, index=1)
        st.pyplot(anomaly_scatter_plot(data, x_col, y_col, anomalies))
    
    # --- Box plots for numeric columns ---
    if len(numeric_cols) > 0:
        st.write("### Box Plots with Anomalies")
        for col in numeric_cols:
            st.pyplot(anomaly_box_plot(data, col, anomalies))
    
    # --- PDF Report ---
    if st.button("Download PDF Report"):
        figures = {"Missing Values Heatmap": missing_values_heatmap(data)}
        if x_col is not None and y_col is not None:
            figures["Anomaly Scatter Plot"] = anomaly_scatter_plot(data, x_col, y_col, anomalies)
        pdf_file = "summary_report.pdf"
        generate_report(data, figures, anomaly_box_plot, anomalies, pdf_file)
        with open(pdf_file, "rb") as f:
            st.download_button("Download Report", f, file_name=pdf_file)


#streamlit run app/main.py