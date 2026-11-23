"""tests/eval_run.py - Evaluation script for AgentFlow AI Analysis and Verification.

Can be run directly via:
    python tests/eval_run.py

Evaluates:
1. Demo Task 1: Hospital Patient Appointment-Reminder Software Evaluation.
2. Demo Task 2: Rural Clinic Telemedicine Tools Evaluation.
3. Edge Cases:
   - Empty claim list.
   - All claims unverifiable.
   - Single vendor evaluation.
   - Duplicate claims with identical values (agreements verification).
"""

from __future__ import annotations

import sys, os
from typing import List, Optional

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import Claim, VerificationResult
from agents.analysis import (
    analyze,
    build_feature_matrix,
    score_india_criteria,
    detect_conflicts_deterministic,
    generate_retry_list,
    calculate_metrics,
    AnalysisMetrics,
    RetryItem,
)
from agents.verifier import verify_claim

# ---------------------------------------------------------------------------
# Demo Task 1 Claims: Hospital Appointment-Reminder Software
# ---------------------------------------------------------------------------
HOSPITAL_CLAIMS = [
    Claim(
        vendor="HealthPlix",
        attribute="whatsapp_api",
        value="Supported",
        source_url="https://healthplix.com/features",
        snippet="Official WhatsApp Business API integration with automated template reminders."
    ),
    Claim(
        vendor="HealthPlix",
        attribute="dlt_sms",
        value="Supported",
        source_url="https://healthplix.com/compliance",
        snippet="Pre-approved TRAI DLT headers for patient transactional alerts."
    ),
    Claim(
        vendor="HealthPlix",
        attribute="abdm_abha",
        value="Supported",
        source_url="https://healthplix.com/abdm",
        snippet="ABDM M1, M2, and M3 certified for Ayushman Bharat digital records."
    ),
    Claim(
        vendor="HealthPlix",
        attribute="inr_pricing",
        value="Supported",
        source_url="https://healthplix.com/pricing",
        snippet="Transparent billing in INR starting at ₹2500 per month."
    ),
    Claim(
        vendor="HealthPlix",
        attribute="price_per_month",
        value="₹2500",
        source_url="https://healthplix.com/pricing",
        snippet="Starting subscription is ₹2500 per month with 5000 messages included."
    ),
    Claim(
        vendor="Qikwell",
        attribute="whatsapp_api",
        value="Supported",
        source_url="https://qikwell.com/reminders",
        snippet="Sends instant appointment confirmations and reminders over WhatsApp."
    ),
    Claim(
        vendor="Qikwell",
        attribute="emr_integration",
        value="Supported",
        source_url="https://qikwell.com/integrations",
        snippet="Direct HL7 / FHIR EMR integration with Apollo, Manipal, and Max HIS."
    ),
    Claim(
        vendor="Qikwell",
        attribute="price_per_month",
        value="₹3500",
        source_url="https://qikwell.com/pricing",
        snippet="Hospital OPD appointment reminder package starts at ₹3500 per month."
    ),
    Claim(
        vendor="mPower Health",
        attribute="price_per_month",
        value="₹2000",
        source_url="https://mpower.com/tier1",
        snippet="Basic clinic subscription: ₹2000 per month."
    ),
    Claim(
        vendor="mPower Health",
        attribute="price_per_month",
        value="₹4500",
        source_url="https://mpower.com/tier2",
        snippet="Hospital enterprise tier billed at ₹4500 per month."
    ),
]

# ---------------------------------------------------------------------------
# Demo Task 2 Claims: Rural Clinic Telemedicine Tools
# ---------------------------------------------------------------------------
TELEMED_CLAIMS = [
    Claim(
        vendor="eSanjeevani",
        attribute="regional_languages",
        value="Supported",
        source_url="https://esanjeevani.gov.in/languages",
        snippet="Available in 12 Indian regional languages including Hindi, Tamil, Telugu, and Kannada."
    ),
    Claim(
        vendor="eSanjeevani",
        attribute="abdm_abha",
        value="Supported",
        source_url="https://esanjeevani.gov.in/abdm",
        snippet="National telemedicine backbone integrated with ABHA health accounts."
    ),
    Claim(
        vendor="eSanjeevani",
        attribute="inr_pricing",
        value="Supported (Free for Govt clinics)",
        source_url="https://esanjeevani.gov.in/pricing",
        snippet="Completely free of charge for government and primary health centres (PHCs)."
    ),
    Claim(
        vendor="RuralHealth Connect",
        attribute="regional_languages",
        value="Supported",
        source_url="https://ruralhealth.org/features",
        snippet="Audio and video consults tailored for low-bandwidth 2G/3G connections in 6 languages."
    ),
    Claim(
        vendor="RuralHealth Connect",
        attribute="inr_pricing",
        value="Supported",
        source_url="https://ruralhealth.org/pricing",
        snippet="Rural clinic starter kit costs ₹1200 per clinic per month."
    ),
]


def run_eval(task_name: str, claims: List[Claim], verifications: Optional[List[VerificationResult]] = None) -> AnalysisMetrics:
    """Execute evaluation for a given task and print structured outputs."""
    print(f"\n{'='*70}")
    print(f"EVALUATION RUN: {task_name.upper()}")
    print(f"{'='*70}")

    # 1. Conflict detection
    conflicts, agreements = detect_conflicts_deterministic(claims)
    print(f"\n[1] DETERMINISTIC CONFLICT DETECTION:")
    print(f"    Total Claims: {len(claims)}")
    print(f"    Agreements:   {agreements}")
    print(f"    Conflicts:    {len(conflicts)}")
    for conf in conflicts:
        print(f"    - CONFLICT on '{conf.vendor}' attribute '{conf.attribute}': values={conf.values}")

    # 2. India Criteria Scorecard
    scorecard = score_india_criteria(claims)
    print(f"\n[2] INDIA HEALTHCARE CRITERIA SCORECARD:")
    for v, sc in scorecard.items():
        print(f"    - {v:20}: {sc['total_score']} ({sc['compliance_percentage']}%)")

    # 3. Feature & Pricing Matrix
    matrix = build_feature_matrix(claims)
    print(f"\n[3] FEATURE / PRICING MATRIX (SAMPLE ATTRIBUTES):")
    for v, attrs in matrix.items():
        price = attrs.get("price_per_month", "unknown")
        whatsapp = attrs.get("whatsapp_api", "unknown")
        abdm = attrs.get("abdm_abha", "unknown")
        print(f"    - {v:20} | price: {price:10} | WhatsApp: {whatsapp:10} | ABDM: {abdm:10}")

    # 4. Retry List
    retries = generate_retry_list(claims, verifications)
    print(f"\n[4] RETRY LIST (M1 RetryItem Format):")
    print(f"    Items needing retry: {len(retries)}")
    for r in retries:
        print(f"    - Retry: vendor='{r.vendor}', attribute='{r.attribute}', reason='{r.reason}'")

    # 5. Quantitative Metrics
    metrics = calculate_metrics(claims, conflicts, verifications, retries=retries)
    print(f"\n[5] QUANTITATIVE METRICS (For M4 Dashboard/Reports):")
    print(f"    - Total Claims:         {metrics.total_claims}")
    print(f"    - Verified Percentage:  {metrics.verified_percentage}%")
    print(f"    - Unverifiable Count:   {metrics.unverifiable_count}")
    print(f"    - Conflicts Found:      {metrics.conflicts_found}")
    print(f"    - Citations per Claim:  {metrics.citations_per_claim}")
    print(f"    - Retries Suggested:    {metrics.retries_suggested}")

    return metrics


def run_all_evals():
    """Run evaluation on both demo tasks and all hardened edge cases."""
    # Demo 1
    m1 = run_eval("Hospital Appointment-Reminder Software", HOSPITAL_CLAIMS)

    # Demo 2
    m2 = run_eval("Rural Clinic Telemedicine Tools", TELEMED_CLAIMS)

    # Edge Case 1: Empty claim list
    print(f"\n{'='*70}")
    print("EDGE CASE 1: EMPTY CLAIM LIST")
    print(f"{'='*70}")
    empty_metrics = run_eval("Empty Claim List", [])
    assert empty_metrics.total_claims == 0
    assert empty_metrics.verified_percentage == 0.0

    # Edge Case 2: All claims unverifiable
    print(f"\n{'='*70}")
    print("EDGE CASE 2: ALL CLAIMS UNVERIFIABLE")
    print(f"{'='*70}")
    unverifiable_claims = [
        Claim(vendor="PaywallVendor", attribute="pricing", value="₹1000", source_url="https://paywall.com", snippet="hidden")
    ]
    unverifiable_results = [
        VerificationResult(claim_id=0, status="unverifiable", details="403 Paywall")
    ]
    unv_metrics = run_eval("All Unverifiable", unverifiable_claims, unverifiable_results)
    assert unv_metrics.verified_percentage == 0.0
    assert unv_metrics.unverifiable_count == 1
    assert unv_metrics.retries_suggested == 1

    # Edge Case 3: Single vendor
    print(f"\n{'='*70}")
    print("EDGE CASE 3: SINGLE VENDOR")
    print(f"{'='*70}")
    single_vendor_claims = [
        Claim(vendor="SoloMed", attribute="whatsapp_api", value="Supported", source_url="https://solo.com", snippet="solo"),
        Claim(vendor="SoloMed", attribute="inr_pricing", value="Supported", source_url="https://solo.com", snippet="solo inr"),
    ]
    single_metrics = run_eval("Single Vendor", single_vendor_claims)
    assert single_metrics.total_claims == 2
    assert single_metrics.conflicts_found == 0

    # Edge Case 4: Duplicate claims with identical values (counting as agreements)
    print(f"\n{'='*70}")
    print("EDGE CASE 4: DUPLICATE CLAIMS WITH IDENTICAL VALUES (AGREEMENTS)")
    print(f"{'='*70}")
    dup_claims = [
        Claim(vendor="VendorX", attribute="whatsapp_api", value="Supported", source_url="https://source1.com", snippet="support whatsapp"),
        Claim(vendor="VendorX", attribute="whatsapp_api", value="Supported", source_url="https://source2.com", snippet="whatsapp supported"),
    ]
    dup_metrics = run_eval("Duplicate Claims Agreement", dup_claims)
    assert dup_metrics.conflicts_found == 0
    conflicts, agreements = detect_conflicts_deterministic(dup_claims)
    assert agreements == 2

    print(f"\n{'='*70}")
    print("ALL EVALUATIONS AND EDGE-CASE CHECKS COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_all_evals()
