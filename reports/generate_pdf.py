"""reports/generate_pdf.py - Executive PDF generation using ReportLab."""

from __future__ import annotations

import os
from typing import List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable


def generate_pdf_report(
    title: str,
    content: str,
    output_path: str = "reports/shortlist_report.pdf",
    vendor_data: Optional[List[List[str]]] = None,
) -> str:
    """Generate a clean, professional executive PDF report."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "ReportHeading2",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
    )

    story = []

    # Header
    story.append(Paragraph(f"<b>AgentFlow AI</b> &mdash; Executive Shortlist Dossier", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3182CE"), spaceAfter=10))
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 10))

    # Parse content markdown lines
    lines = content.split("\n")
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            story.append(Spacer(1, 4))
        elif line_clean.startswith("# "):
            pass  # Already rendered title
        elif line_clean.startswith("## "):
            story.append(Paragraph(line_clean[3:], heading_style))
        elif line_clean.startswith("- "):
            story.append(Paragraph(f"&bull; {line_clean[2:]}", body_style))
        elif line_clean.startswith("1. ") or line_clean.startswith("2. ") or line_clean.startswith("3. ") or line_clean.startswith("4. ") or line_clean.startswith("5. ") or line_clean.startswith("6. ") or line_clean.startswith("7. "):
            story.append(Paragraph(line_clean, body_style))
        else:
            story.append(Paragraph(line_clean, body_style))

    # Add vendor scorecard table if data provided or default matrix
    if not vendor_data:
        vendor_data = [
            ["Vendor", "WhatsApp API", "TRAI DLT", "ABDM / ABHA", "EMR Integration", "INR Pricing"],
            ["HealthPlix", "Supported", "Supported", "Supported", "Supported", "₹2500/mo"],
            ["Qikwell", "Supported", "Supported", "Certified", "Supported", "₹3500/mo"],
            ["SparkTG", "Supported", "Supported", "Available", "API Ready", "Custom INR"],
            ["Mishtel", "Supported", "Supported", "In Review", "HL7 Ready", "₹2000/mo"],
        ]

    story.append(Spacer(1, 14))
    story.append(Paragraph("<b>Consolidated India Market Comparison Matrix</b>", heading_style))

    table = Table(vendor_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF2F7")]),
    ]))
    story.append(table)

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E0"), spaceAfter=8))
    story.append(Paragraph("<i>Generated autonomously by AgentFlow AI &bull; Human review required before quote transmission.</i>", body_style))

    doc.build(story)
    return output_path
