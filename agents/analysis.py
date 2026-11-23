"""agents/analysis.py - Analysis agent for AgentFlow AI (M3).

Key capabilities:
1. Deterministic Conflict Detection:
   - Groups claims by vendor + attribute.
   - Flags discrepancies where the same vendor + attribute has conflicting values.
   - LLM adjudicates genuine conflicts.
   - If none, explicitly reports "0 conflicts, N agreements".
2. Feature & Pricing Matrix:
   - Tabulates all vendors against attributes with explicit 'unknown' cells.
3. India Healthcare Criteria Scoring:
   - Evaluates against: WhatsApp Business API, DLT/TRAI SMS, Regional Languages,
     ABDM/ABHA, DPDP Act 2023, HIS/EMR Integration, INR Pricing.
4. Retry List & Verification Metrics Generation:
   - Outputs 'needs retry' items in M1 format: (vendor, attribute, reason).
   - Generates structured metrics for M4 UI reporting:
     verified %, unverifiable count, conflicts found/resolved, citations per claim.
"""

from __future__ import annotations

import collections
import inspect
import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from schemas import Claim, Analysis, Conflict, VerificationResult
from orchestrator import RetryItem
import llm
import db

# Alias for backward compatibility
NeedsRetryItem = RetryItem

INDIA_CRITERIA_KEYS = [
    "whatsapp_api",
    "dlt_sms",
    "regional_languages",
    "abdm_abha",
    "dpdp_compliance",
    "emr_integration",
    "inr_pricing",
]

INDIA_CRITERIA_LABELS = {
    "whatsapp_api": "WhatsApp Business API",
    "dlt_sms": "DLT / TRAI SMS Compliance",
    "regional_languages": "Regional Languages Support",
    "abdm_abha": "ABDM / ABHA Integration",
    "dpdp_compliance": "DPDP Act 2023 Compliance",
    "emr_integration": "HIS / EMR Integration",
    "inr_pricing": "INR Pricing Transparent",
}


class AdjudicationResult(BaseModel):
    """LLM adjudication of a conflicting claim."""
    resolved_value: str = Field(..., description="The most accurate value or synthesis")
    explanation: str = Field(..., description="Explanation of why one value is preferred or how they reconcile")


class AnalysisMetrics(BaseModel):
    """Metrics object summarizing verification and claim quality for M4 reporting."""
    total_claims: int = 0
    verified_percentage: float = 0.0
    unverifiable_count: int = 0
    conflicts_found: int = 0
    conflicts_resolved: int = 0
    citations_per_claim: float = 0.0
    retries_suggested: int = 0





def normalize_val(val: str) -> str:
    """Normalize string value for comparison."""
    if not val:
        return ""
    return re.sub(r"\s+", " ", val).strip().lower()


def detect_conflicts_deterministic(claims: List[Claim]) -> Tuple[List[Conflict], int]:
    """Detect conflicts deterministically by grouping by (vendor, attribute).

    Returns
    -------
    (conflicts, agreement_count)
    """
    grouped: Dict[Tuple[str, str], List[Claim]] = collections.defaultdict(list)
    for c in claims:
        vendor_norm = c.vendor.strip()
        attr_norm = c.attribute.strip().lower()
        grouped[(vendor_norm, attr_norm)].append(c)

    conflicts: List[Conflict] = []
    agreement_count = 0

    for (vendor, attr), group in grouped.items():
        unique_vals = list({c.value.strip() for c in group})
        if len(unique_vals) > 1:
            sources = list({c.source_url for c in group})
            conflicts.append(
                Conflict(
                    vendor=vendor,
                    attribute=attr,
                    values=unique_vals,
                    sources=sources,
                )
            )
        else:
            agreement_count += len(group)

    return conflicts, agreement_count


async def adjudicate_conflicts_with_llm(
    conflicts: List[Conflict],
    claims: List[Claim],
) -> List[Dict[str, Any]]:
    """Adjudicate genuine conflicts using Gemini.

    Fallback: if LLM call fails, return both conflicting values with sources.
    """
    if not conflicts:
        return []

    adjudicated = []
    for conf in conflicts:
        matching_claims = [
            c for c in claims
            if c.vendor.lower() == conf.vendor.lower() and c.attribute.lower() == conf.attribute.lower()
        ]
        snippets = "\n".join(
            [f"- Value: '{c.value}' | Source: {c.source_url} | Snippet: \"{c.snippet}\"" for c in matching_claims]
        )

        prompt = (
            f"Adjudicate this factual conflict for vendor '{conf.vendor}' on attribute '{conf.attribute}':\n"
            f"{snippets}\n\n"
            "Determine the most accurate current value or reconcile the difference (e.g. different tiers/dates)."
        )

        try:
            decision = await llm.call_json(
                prompt=prompt,
                schema_model=AdjudicationResult,
                system="You are an expert fact adjudicator. Reconcile differences or select the most credible value."
            )
            adjudicated.append({
                "vendor": conf.vendor,
                "attribute": conf.attribute,
                "resolved_value": decision.resolved_value,
                "explanation": decision.explanation,
                "sources": [str(s) for s in conf.sources],
            })
        except Exception as exc:
            # Fallback per prompt rule: show both conflicting values with sources
            adjudicated.append({
                "vendor": conf.vendor,
                "attribute": conf.attribute,
                "resolved_value": " / ".join(conf.values),
                "explanation": f"Unresolved conflict (fallback mode): {exc}",
                "sources": [str(s) for s in conf.sources],
            })

    return adjudicated


def build_feature_matrix(claims: List[Claim]) -> Dict[str, Dict[str, str]]:
    """Build a vendor-by-attribute matrix with explicit 'unknown' cells."""
    vendors = sorted({c.vendor.strip() for c in claims})
    attributes = sorted({c.attribute.strip().lower() for c in claims})

    matrix: Dict[str, Dict[str, str]] = {v: {} for v in vendors}

    # Populate known cells
    for c in claims:
        v = c.vendor.strip()
        a = c.attribute.strip().lower()
        matrix[v][a] = c.value.strip()

    # Fill missing cells with 'unknown'
    for v in vendors:
        for a in attributes:
            if a not in matrix[v]:
                matrix[v][a] = "unknown"

    return matrix


def score_india_criteria(claims: List[Claim]) -> Dict[str, Dict[str, Any]]:
    """Score vendors against the 7 mandatory Indian healthcare criteria."""
    vendors = sorted({c.vendor.strip() for c in claims})
    scorecard: Dict[str, Dict[str, Any]] = {}

    for v in vendors:
        v_claims = [c for c in claims if c.vendor.strip().lower() == v.lower()]
        v_scores: Dict[str, str] = {}
        criteria_met = 0

        for key in INDIA_CRITERIA_KEYS:
            status = "unknown"
            for c in v_claims:
                attr = c.attribute.lower()
                val = c.value.lower()
                # Check for criteria matches
                if key in attr or (key == "inr_pricing" and ("price" in attr or "inr" in val or "₹" in val)):
                    if any(pos in val for pos in ["yes", "supported", "compliant", "verified", "true", "₹", "inr"]):
                        status = "Supported"
                        criteria_met += 1
                        break
                    elif any(neg in val for pos in ["no", "unsupported", "false"]):
                        status = "Not Supported"
                        break
            v_scores[key] = status

        scorecard[v] = {
            "scores": v_scores,
            "total_score": f"{criteria_met}/{len(INDIA_CRITERIA_KEYS)}",
            "compliance_percentage": round((criteria_met / len(INDIA_CRITERIA_KEYS)) * 100, 1),
        }

    return scorecard


def generate_retry_list(
    claims: List[Claim],
    verifications: Optional[List[VerificationResult]] = None,
) -> List[NeedsRetryItem]:
    """Generate list of items needing retry for M1 orchestrator consumption."""
    retry_items: List[NeedsRetryItem] = []
    seen = set()

    # Check for unverified or missing claims
    if verifications:
        for i, vr in enumerate(verifications):
            if vr.status in ("failed", "unverifiable") and i < len(claims):
                c = claims[i]
                key = (c.vendor.lower(), c.attribute.lower())
                if key not in seen:
                    seen.add(key)
                    retry_items.append(
                        NeedsRetryItem(
                            vendor=c.vendor,
                            attribute=c.attribute,
                            reason=vr.details or f"Verification status {vr.status}",
                        )
                    )

    # Check for claims with 'unknown' or unverified values
    for c in claims:
        if c.value.lower() in ("unknown", "unverified"):
            key = (c.vendor.lower(), c.attribute.lower())
            if key not in seen:
                seen.add(key)
                retry_items.append(
                    NeedsRetryItem(
                        vendor=c.vendor,
                        attribute=c.attribute,
                        reason="Value extracted as unknown/unverified during initial research",
                    )
                )

    return retry_items


def calculate_metrics(
    claims: List[Claim],
    conflicts: List[Conflict],
    verifications: Optional[List[VerificationResult]] = None,
    adjudicated: Optional[List[Dict[str, Any]]] = None,
    retries: Optional[List[NeedsRetryItem]] = None,
) -> AnalysisMetrics:
    """Calculate quantitative metrics for M4 UI reports."""
    total = len(claims)
    if total == 0:
        return AnalysisMetrics()

    v_count = 0
    u_count = 0
    if verifications:
        for v in verifications:
            if v.status == "verified":
                v_count += 1
            elif v.status == "unverifiable":
                u_count += 1
        v_pct = round((v_count / len(verifications)) * 100, 1)
    else:
        v_pct = 100.0

    sources_count = sum(1 for c in claims if str(c.source_url).startswith("http"))
    citations_per_claim = round(sources_count / total, 2)

    return AnalysisMetrics(
        total_claims=total,
        verified_percentage=v_pct,
        unverifiable_count=u_count,
        conflicts_found=len(conflicts),
        conflicts_resolved=len(adjudicated or []),
        citations_per_claim=citations_per_claim,
        retries_suggested=len(retries or []),
    )


async def analyze_async(
    claims: List[Claim],
    verifications: Optional[List[VerificationResult]] = None,
) -> Analysis:
    """Full asynchronous analysis pipeline."""
    # 1. Detect conflicts deterministically
    conflicts, agreements = detect_conflicts_deterministic(claims)

    # 2. Adjudicate genuine conflicts with LLM
    adjudicated = []
    if conflicts:
        adjudicated = await adjudicate_conflicts_with_llm(conflicts, claims)

    # 3. Build feature matrix and India scorecard
    matrix = build_feature_matrix(claims)
    scorecard = score_india_criteria(claims)

    # 4. Generate retry list and metrics
    retries = generate_retry_list(claims, verifications)
    metrics = calculate_metrics(claims, conflicts, verifications, adjudicated, retries)

    # 5. Build summary text
    if not conflicts:
        conflict_text = f"0 conflicts, {agreements} agreements."
    else:
        conflict_text = f"{len(conflicts)} conflict(s) detected and adjudicated ({len(adjudicated)} resolved)."

    summary_lines = [
        f"Analysis Summary across {len(claims)} claim(s): {conflict_text}",
        f"Verified: {metrics.verified_percentage}%, Unverifiable: {metrics.unverifiable_count}, Citations/claim: {metrics.citations_per_claim}.",
        "",
        "India Healthcare Compliance Scores:",
    ]
    for v, sc in scorecard.items():
        summary_lines.append(f"- {v}: {sc['total_score']} ({sc['compliance_percentage']}%)")

    summary_text = "\n".join(summary_lines)

    # Log analysis event
    db.log_event(
        task_id=0,
        agent="analysis",
        type="analysis_completed",
        payload={
            "total_claims": len(claims),
            "conflicts_count": len(conflicts),
            "agreements_count": agreements,
            "metrics": metrics.model_dump() if hasattr(metrics, "model_dump") else metrics.dict(),
            "retries_count": len(retries),
        },
    )

    return Analysis(
        summary=summary_text,
        conflicts=conflicts,
        agreements=agreements,
    )


def analyze(claims: List[Claim]) -> Analysis:
    """Contract-compliant synchronous entrypoint.

    Runs deterministic analysis and returns an Analysis model.
    """
    conflicts, agreements = detect_conflicts_deterministic(claims)
    matrix = build_feature_matrix(claims)
    scorecard = score_india_criteria(claims)

    if not conflicts:
        conflict_text = f"0 conflicts, {agreements} agreements."
    else:
        conflict_text = f"{len(conflicts)} conflict(s) detected."

    summary_lines = [
        f"Analysis Summary across {len(claims)} claim(s): {conflict_text}",
        "",
        "India Healthcare Compliance Scores:",
    ]
    for v, sc in scorecard.items():
        summary_lines.append(f"- {v}: {sc['total_score']} ({sc['compliance_percentage']}%)")

    return Analysis(
        summary="\n".join(summary_lines),
        conflicts=conflicts,
        agreements=agreements,
    )
