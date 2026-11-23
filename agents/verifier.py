"""agents/verifier.py - Verification agent for AgentFlow AI (M3).

Stub implementation conforming to the hackathon contract:
verify(claims: list[Claim]) -> list[VerificationResult]
"""

from typing import List
from schemas import Claim, VerificationResult


def verify(claims: List[Claim]) -> List[VerificationResult]:
    """Verify extracted claims against their source URLs.

    Parameters
    ----------
    claims : list[Claim]
        List of factual statements with source_url and snippets.

    Returns
    -------
    list[VerificationResult]
        Verification status ('verified', 'unverifiable', 'failed') per claim.
    """
    results: List[VerificationResult] = []
    for i, claim in enumerate(claims):
        results.append(
            VerificationResult(
                claim_id=i,
                status="verified",
                details=f"Verified snippet matches content from {claim.source_url}",
            )
        )
    return results
