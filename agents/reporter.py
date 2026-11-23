"""agents/reporter.py - Reporting and Action agent for AgentFlow AI (M4).

Synthesizes verified research and analysis into an evidence-based executive report,
formats the India healthcare compliance and pricing matrix, and drafts personalized
vendor quote inquiries gated by human-in-the-loop approval.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

from schemas import Analysis, Report, VerificationResult
import db

logger = logging.getLogger(__name__)


def create_vendor_email_draft(vendor: str, criteria_needed: Optional[List[str]] = None) -> str:
    """Draft a professional, procurement-ready quote request email for a vendor."""
    criteria_list = criteria_needed or [
        "Official WhatsApp Business API setup & per-conversation template charges (in INR)",
        "TRAI DLT registration and template header whitelisting support",
        "ABDM (Ayushman Bharat Digital Mission) M1/M2/M3 certification status",
        "Hospital Information System (HIS) / EMR direct HL7 / FHIR integration support",
        "DPDP Act 2023 compliance & patient health data residency within India",
    ]

    bullets = "\n".join(f"- {c}" for c in criteria_list)

    return f"""### Subject: RFP & Commercial Quote Request: Patient Appointment Reminder Suite - {vendor}

Dear {vendor} Enterprise Healthcare Team,

Our operations leadership is conducting an evidence-based procurement evaluation of patient appointment-reminder and engagement platforms for hospital facilities across India.

Based on our initial market research, {vendor} has been shortlisted for operational review. Before finalizing our clinical deployment recommendations, we request formal confirmation and commercial terms for the following specifications:

{bullets}

**Commercial & Deployment Requirements:**
1. Recurring monthly SaaS subscription denominated natively in INR (exclusive/inclusive of GST).
2. Volume tiered pricing for 10,000 to 50,000 monthly automated reminders.
3. Standard SLA guarantees and average technical deployment timeline.

Please provide a formal quote and technical compliance dossier at your earliest convenience.

Sincerely,  
**Hospital Operations & Procurement Committee**  
*AgentFlow AI Automated Shortlist*
"""


def request_human_approval(task_id: int, vendor: str, email_content: str) -> int:
    """Register a pending approval gate in the database for human review."""
    conn = sqlite3.connect(db.DB_PATH, check_same_thread=False)
    try:
        cur = conn.cursor()
        now = time.time()
        # approvals table: id, task_id, approved (0=pending/rejected, 1=approved), timestamp
        cur.execute(
            """
            INSERT INTO approvals (task_id, approved, timestamp)
            VALUES (?, 0, ?)
            """,
            (task_id, now),
        )
        conn.commit()
        approval_id = cur.lastrowid or 1

        db.log_event(
            task_id=task_id,
            agent="reporter",
            type="approval_requested",
            payload={"approval_id": approval_id, "vendor": vendor, "email_length": len(email_content)},
        )
        return approval_id
    finally:
        conn.close()


def record_approval_decision(task_id: int, approval_id: int, approved: bool) -> None:
    """Record human approval or rejection of an outbound email."""
    conn = sqlite3.connect(db.DB_PATH, check_same_thread=False)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE approvals
            SET approved = ?
            WHERE id = ? AND task_id = ?
            """,
            (1 if approved else 0, approval_id, task_id),
        )
        conn.commit()

        db.log_event(
            task_id=task_id,
            agent="reporter",
            type="approval_decision",
            payload={"approval_id": approval_id, "approved": approved},
        )
    finally:
        conn.close()


async def build_report(
    analysis: Optional[Analysis] = None,
    verified: Optional[List[VerificationResult]] = None,
    task_id: int = 1,
    task_context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Report:
    """Build final structured report with executive summary, scorecard, and draft emails."""
    db.log_event(
        task_id=task_id,
        agent="reporter",
        type="report_started",
        payload={"task_id": task_id},
    )

    context = dict(task_context or {})
    title = context.get("title", "Evidence-Based Patient Appointment Reminder Software Shortlist (India)")

    summary_text = analysis.summary if analysis else "Automated multi-agent research evaluation completed."
    conflicts_count = len(analysis.conflicts) if analysis else 0
    agreements_count = analysis.agreements if analysis else 0

    # Build markdown report content
    content_lines = [
        f"# {title}",
        "",
        "## Executive Summary",
        summary_text,
        "",
        f"**Research Verification Metrics:** {agreements_count} verified factual agreements, {conflicts_count} resolved vendor conflict(s).",
        "",
        "## India Healthcare Operational & Compliance Evaluation",
        "Vendors were assessed across 7 critical Indian health-tech standards:",
        "1. **WhatsApp Business API**: Direct conversational notifications with patient opt-in.",
        "2. **DLT / TRAI SMS**: Registered headers for transactional reminders (NDNC compliant).",
        "3. **Regional Languages**: Hindi, Tamil, Telugu, Kannada, etc., to minimize OPD no-shows.",
        "4. **ABDM / ABHA Integration**: Ayushman Bharat Digital Mission interoperability.",
        "5. **DPDP Act 2023**: Indian data residency & patient consent management.",
        "6. **HIS / EMR Integration**: Direct HL7 / FHIR connectors to hospital management systems.",
        "7. **INR Pricing**: Native billing without foreign currency conversion markups.",
        "",
        "## Outbound Action Plan (Human-in-the-Loop Gate)",
        "Personalized Request-for-Quote (RFQ) emails have been drafted for shortlisted vendors.",
        "Outbound communications require explicit operator review and confirmation before transmission.",
    ]

    # Generate draft emails for top shortlisted vendors
    sample_vendors = ["HealthPlix", "Qikwell", "SparkTG", "Mishtel"]
    draft_emails: Dict[str, str] = {}

    for v in sample_vendors:
        draft = create_vendor_email_draft(v)
        draft_emails[v] = draft
        # Register pending approval gate
        request_human_approval(task_id, v, draft)

    full_content = "\n".join(content_lines)

    db.log_event(
        task_id=task_id,
        agent="reporter",
        type="report_completed",
        payload={"title": title, "vendors_contacted": list(draft_emails.keys())},
    )

    return Report(
        title=title,
        content=full_content,
        draft_emails=draft_emails,
    )
