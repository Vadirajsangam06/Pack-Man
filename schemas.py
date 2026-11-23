# schemas.py
"""Pydantic data models for the AgentFlow AI system.

These models define the contract between the different module owners.
All models are simple, JSON‑serialisable, and include example fixtures in
their docstrings for quick reference and for unit‑tests.
"""

from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl, validator


class Claim(BaseModel):
    """A single factual statement extracted from a source.

    Example fixture (as dict)::
        {
            "vendor": "mPower Health",
            "attribute": "price_per_month",
            "value": "₹2500",
            "source_url": "https://example.com/mPower-pricing",
            "snippet": "Pricing starts at ₹2500 per month for up to 5000 patients."
        }
    """

    vendor: str = Field(..., description="Name of the vendor or product")
    attribute: str = Field(..., description="What property is being claimed")
    value: str = Field(..., description="The claimed value, as a raw string")
    source_url: HttpUrl = Field(..., description="URL of the page the claim came from")
    snippet: str = Field(..., description="Exact text excerpt supporting the claim")


class PlanStep(BaseModel):
    """One atomic step in a generated plan.

    Example::
        {"agent": "research_a", "action": "run_research", "args": {"task": "compare reminder apps", "vendor_focus": null}}
    """

    agent: Literal["research", "research_a", "research_b", "analysis", "verifier", "reporter", "manager"]
    action: str
    args: dict
    depends_on: List[int] = Field(default_factory=list, description="Indices of preceding steps this step depends on")


class Plan(BaseModel):
    """Validated plan emitted by the manager.

    Example::
        {
            "steps": [{"agent": "research_a", "action": "run_research", "args": {...}}],
            "assumptions": ["All vendors support WhatsApp Business API"],
            "success_criteria": "At least three vendors with verified pricing",
            "researcher_count": 2
        }
    """

    steps: List[PlanStep]
    assumptions: List[str] = Field(default_factory=list)
    success_criteria: str
    summary: Optional[str] = None
    researcher_count: int = Field(default=2, ge=1, le=6, description="Number of parallel researcher agents to spawn")


class Conflict(BaseModel):
    """Two claims that disagree on the same vendor attribute.

    Example::
        {"vendor": "mPower Health", "attribute": "price_per_month", "values": ["₹2500", "₹3000"], "sources": ["url1", "url2"]}
    """

    vendor: str
    attribute: str
    values: List[str]
    sources: List[HttpUrl]


class VerificationResult(BaseModel):
    """Result of the verification step for a claim.

    Example::
        {"claim_id": 42, "status": "verified", "details": "Snippet matches page content"}
    """

    claim_id: int
    status: Literal["verified", "unverifiable", "failed"]
    details: Optional[str] = None


class Event(BaseModel):
    """An event emitted by any agent for UI / audit logs.

    Example::
        {"task_id": 7, "agent": "research", "type": "search_completed", "payload": {"duration_s": 3.2}}
    """

    task_id: int
    agent: str
    type: str
    payload: dict
    timestamp: float = Field(default_factory=lambda: __import__("time").time())


class Analysis(BaseModel):
    """Aggregated analysis of a set of claims.

    Example::
        {"summary": "All vendors meet Indian data‑privacy requirements", "conflicts": [], "agreements": 5}
    """

    summary: str
    conflicts: List[Conflict] = Field(default_factory=list)
    agreements: int = 0


class Report(BaseModel):
    """Final report ready for UI rendering.

    Example::
        {"title": "Appointment‑reminder software shortlist", "content": "...markdown...", "draft_emails": {"mPower Health": "..."}}
    """

    title: str
    content: str
    draft_emails: dict = Field(default_factory=dict)
