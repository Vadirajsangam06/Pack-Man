"""tests/test_reporter.py - Unit tests for M4 reporter, PDF generator, and HITL approval gate."""

from __future__ import annotations

import os
import sys
import sqlite3
import pytest

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import Analysis, Report, Conflict, VerificationResult
from agents.reporter import (
    build_report,
    create_vendor_email_draft,
    request_human_approval,
    record_approval_decision,
)
from reports.generate_pdf import generate_pdf_report
import db


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def clean_db(monkeypatch):
    """Ensure clean test environment."""
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)


# ===========================================================================
# 1. Report Synthesis Tests
# ===========================================================================

@pytest.mark.anyio
async def test_build_report_synthesis():
    """Verify build_report generates a complete structured Report object."""
    analysis = Analysis(
        summary="HealthPlix and Qikwell lead the appointment-reminder evaluation.",
        conflicts=[Conflict(vendor="mPower", attribute="pricing", values=["₹2000", "₹4000"], sources=["https://m.com"])],
        agreements=6,
    )

    report = await build_report(analysis=analysis, task_id=42)

    assert isinstance(report, Report)
    assert "Evidence-Based" in report.title
    assert "HealthPlix and Qikwell" in report.content
    assert "WhatsApp Business API" in report.content
    assert len(report.draft_emails) >= 3
    assert "HealthPlix" in report.draft_emails


def test_create_vendor_email_draft_content():
    """Verify vendor email draft contains procurement requirements and compliance bullets."""
    draft = create_vendor_email_draft("HealthPlix")

    assert "HealthPlix" in draft
    assert "WhatsApp Business API" in draft
    assert "TRAI DLT" in draft
    assert "ABDM" in draft
    assert "INR" in draft


# ===========================================================================
# 2. Human-in-the-Loop Approval Gate Tests
# ===========================================================================

def test_human_approval_gate_lifecycle():
    """Verify registration and recording of human approval decisions in SQLite."""
    task_id = 99
    vendor = "Qikwell"
    content = "Draft RFP content"

    # 1. Request approval -> creates pending record (approved=0)
    approval_id = request_human_approval(task_id, vendor, content)
    assert approval_id > 0

    conn = sqlite3.connect(db.DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("SELECT approved FROM approvals WHERE id = ?", (approval_id,))
    row = cur.fetchone()
    assert row is not None
    assert row[0] == 0  # Pending / not approved

    # 2. Record human approval -> updates record (approved=1)
    record_approval_decision(task_id, approval_id, approved=True)
    cur.execute("SELECT approved FROM approvals WHERE id = ?", (approval_id,))
    updated_row = cur.fetchone()
    assert updated_row[0] == 1  # Approved
    conn.close()


# ===========================================================================
# 3. ReportLab PDF Generation Tests
# ===========================================================================

def test_generate_pdf_report_file(tmp_path):
    """Verify ReportLab PDF generation builds a valid non-empty PDF file."""
    output_pdf = str(tmp_path / "test_report.pdf")
    title = "Test Healthcare Shortlist"
    content = "Executive summary text.\n- Bullet point 1\n- Bullet point 2"

    result_path = generate_pdf_report(title=title, content=content, output_path=output_pdf)

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 1000  # Valid PDF binary size
