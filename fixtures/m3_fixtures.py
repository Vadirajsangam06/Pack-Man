"""fixtures/m3_fixtures.py - Edge-case fixtures for M3 Analysis and Verification.

Contains:
1. Matching snippet claim.
2. Snippet absent from page claim.
3. Conflicting claims (same vendor + attribute, different prices).
4. Paywalled / unfetchable page claim.
Plus mock page contents for offline deterministic testing.
"""

from schemas import Claim

# 1. Matching Snippet
MATCHING_SNIPPET_CLAIM = Claim(
    vendor="HealthPlix",
    attribute="price_per_month",
    value="₹2500",
    source_url="https://example.com/healthplix-pricing",
    snippet="Pricing starts at ₹2500 per month for small to medium healthcare practices."
)

# 2. Snippet Absent From Page
SNIPPET_ABSENT_CLAIM = Claim(
    vendor="Practo Ray",
    attribute="emr_integration",
    value="HL7 FHIR supported",
    source_url="https://example.com/practo-features",
    snippet="Full native bi-directional synchronization with Cerner and Epic EMRs."
)

# 3. Two claims with conflicting prices on same attribute for same vendor
CONFLICTING_CLAIM_A = Claim(
    vendor="mPower Health",
    attribute="price_per_month",
    value="₹2500",
    source_url="https://example.com/mpower-pricing-tier1",
    snippet="Standard hospital tier costs ₹2500 per month with WhatsApp inclusion."
)

CONFLICTING_CLAIM_B = Claim(
    vendor="mPower Health",
    attribute="price_per_month",
    value="₹5000",
    source_url="https://example.com/mpower-pricing-tier2",
    snippet="Enterprise hospital operations subscription is billed at ₹5000 per month."
)

# 4. Paywalled / Unfetchable Page Claim
PAYWALLED_CLAIM = Claim(
    vendor="KareXpert",
    attribute="dlt_sms_compliance",
    value="TRAI compliant",
    source_url="https://example.com/karexpert-paywalled",
    snippet="All transactional notifications route through registered DLT telemarketer headers."
)

# Mock page bodies for simulated fetching in tests
MOCK_PAGES = {
    "https://example.com/healthplix-pricing": (
        "Welcome to HealthPlix. Pricing starts at ₹2500 per month for small to medium healthcare practices. "
        "Includes automated patient appointment reminders."
    ),
    "https://example.com/practo-features": (
        "Practo Ray features appointment management, billing, and basic patient records. "
        "Cloud-based calendar with SMS reminders."
    ),
    "https://example.com/mpower-pricing-tier1": (
        "mPower Health offers clear packages. Standard hospital tier costs ₹2500 per month with WhatsApp inclusion."
    ),
    "https://example.com/mpower-pricing-tier2": (
        "Enterprise hospital operations subscription is billed at ₹5000 per month."
    ),
    # Paywalled / error page maps to None or raises HTTP error
}
