from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt

def generate_pdf_report(summary, figs, filename="summary_report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    flowables = []

    # Add text summary
    for key, value in summary.items():
        flowables.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
        flowables.append(Spacer(1, 12))

    # Add charts
    for fig in figs:
        if fig is not None:
            tmpfile = f"{fig.get_label() or 'temp'}.png"
            fig.savefig(tmpfile, bbox_inches="tight")
            flowables.append(Image(tmpfile, width=400, height=250))
            flowables.append(Spacer(1, 12))

    doc.build(flowables)