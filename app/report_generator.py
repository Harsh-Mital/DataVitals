from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import matplotlib.pyplot as plt
import tempfile
import math

def generate_report(data, figures, anomaly_box_plot_func, anomalies, filename="summary_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # --- Title ---
    elements.append(Paragraph("Data Quality & Anomaly Detection Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # --- Dataset Info ---
    elements.append(Paragraph("<b>Dataset Information</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Rows: {data.shape[0]}", styles["Normal"]))
    elements.append(Paragraph(f"Columns: {data.shape[1]}", styles["Normal"]))
    duplicated_count = data.duplicated().sum()
    elements.append(Paragraph(f"Duplicated Rows: {duplicated_count}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # --- Schema ---
    elements.append(Paragraph("<b>Data Schema</b>", styles["Heading2"]))
    schema_data = [["Column", "Data Type", "Non-Null Count"]]
    for col in data.columns:
        schema_data.append([col, str(data[col].dtype), data[col].notnull().sum()])
    schema_table = Table(schema_data, colWidths=[150, 100, 100], hAlign='LEFT')
    schema_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elements.append(schema_table)
    elements.append(Spacer(1, 12))

    # --- Summary Statistics ---
    elements.append(Paragraph("<b>Summary Statistics</b>", styles["Heading2"]))
    summary_df = data.describe(include="all").fillna("").round(2)

    # --- Handle wide tables by splitting into chunks ---
    max_cols_per_table = 10  # adjust as needed
    total_cols = summary_df.shape[1]
    num_tables = math.ceil(total_cols / max_cols_per_table)

    for i in range(num_tables):
        start_col = i * max_cols_per_table
        end_col = min((i + 1) * max_cols_per_table, total_cols)
        sub_df = summary_df.iloc[:, start_col:end_col]

        table_data = [["Index"] + sub_df.columns.to_list()]
        for idx, row in sub_df.iterrows():
            table_data.append([str(idx)] + row.tolist())

        num_cols = len(table_data[0])
        page_width, _ = A4
        col_width = min((page_width - 40) / num_cols, 80)  # limit max width per column

        table = Table(table_data, colWidths=[col_width]*num_cols, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        if i < num_tables - 1:
            elements.append(PageBreak())

    # --- Figures (heatmap, scatter, etc.) ---
    for title, fig in figures.items():
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches="tight")
            elements.append(Image(tmpfile.name, width=400, height=300))
        elements.append(Spacer(1, 20))

    # --- Box plots for all numeric columns ---
    numeric_cols = data.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        elements.append(Paragraph("<b>Box Plots for Numeric Columns</b>", styles["Heading2"]))
        for col in numeric_cols:
            elements.append(Paragraph(f"Box Plot: {col}", styles["Normal"]))
            fig = anomaly_box_plot_func(data, col, anomalies)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig.savefig(tmpfile.name, bbox_inches="tight")
                elements.append(Image(tmpfile.name, width=350, height=250))
            plt.close(fig)
            elements.append(Spacer(1, 12))

    doc.build(elements)
    return filename
