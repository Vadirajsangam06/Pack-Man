"""tests/test_integration.py - Integration test for Manager -> Orchestrator pipeline.

Tests the full end-to-end integration:
1. Manager creates a validated Plan DAG.
2. Orchestrator executes the Plan DAG via AgentRegistry.
3. If teammate agent modules (M2 research, M3 analysis/verifier, M4 reporter) are not yet on main,
   cleanly skips with a clear diagnostic explanation.
"""

import sys, os, importlib
import pytest

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import Plan, PlanStep, Claim, Analysis, VerificationResult, Report
from agents.manager import create_plan
from orchestrator import Orchestrator, AgentRegistry, RunResult
import db

@pytest.fixture
def anyio_backend():
    return "asyncio"

def check_missing_teammate_modules() -> list[str]:
    """Check which teammate modules are still missing from main."""
    required_modules = [
        ("research", "agents.research"),
        ("analysis", "agents.analysis"),
        ("verifier", "agents.verifier"),
        ("reporter", "agents.reporter"),
    ]
    missing = []
    for role, mod_name in required_modules:
        try:
            importlib.import_module(mod_name)
        except (ImportError, ModuleNotFoundError):
            missing.append(f"{role} ({mod_name})")
    return missing


@pytest.mark.anyio
async def test_manager_to_orchestrator_simulated_pipeline(monkeypatch):
    """End-to-end integration test: Manager generates plan -> Orchestrator executes steps."""
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)

    # Simulated plan from manager
    hospital_plan = Plan(
        steps=[
            PlanStep(agent="research_a", action="run_research", args={"track": "features"}, depends_on=[]),
            PlanStep(agent="research_b", action="run_research", args={"track": "pricing"}, depends_on=[]),
            PlanStep(agent="analysis", action="analyze", args={}, depends_on=[0, 1]),
            PlanStep(agent="verifier", action="verify", args={}, depends_on=[2]),
            PlanStep(agent="reporter", action="build_report", args={}, depends_on=[3]),
        ],
        success_criteria="Hospital appointment reminder shortlist",
        assumptions=["Hospital requires WhatsApp Business API and INR pricing"],
        researcher_count=2,
    )

    async def mock_call_json(prompt, schema_model, system=None, max_attempts=5):
        return hospital_plan

    import llm
    monkeypatch.setattr(llm, "call_json", mock_call_json)

    # 1. Manager generates the plan
    plan = await create_plan("Help an Indian hospital operations manager compare patient appointment-reminder software")
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 5

    # 2. Fake handlers simulating M2, M3, M4 stubs
    async def fake_research_a(**kwargs):
        return [Claim(vendor="Qikwell", attribute="whatsapp", value="supported", source_url="https://qikwell.com", snippet="WhatsApp API ready")]

    async def fake_research_b(**kwargs):
        return [Claim(vendor="Qikwell", attribute="pricing", value="₹3000/mo", source_url="https://qikwell.com", snippet="Starts at ₹3000/mo")]

    async def fake_analysis(claims=None, **kwargs):
        return Analysis(summary="All claims verified", conflicts=[], agreements=2)

    async def fake_verifier(claims=None, **kwargs):
        return [VerificationResult(claim_id=1, status="verified", details="verified")]

    async def fake_reporter(analysis=None, verified=None, **kwargs):
        return Report(title="Shortlist", content="Final Report", draft_emails={"Qikwell": "Quote request"})

    simulated_registry = AgentRegistry({
        "research_a": fake_research_a,
        "research_b": fake_research_b,
        "analysis": fake_analysis,
        "verifier": fake_verifier,
        "reporter": fake_reporter,
    })

    # 3. Orchestrator executes the manager's plan
    orch = Orchestrator(registry=simulated_registry)
    run_result = await orch.run(plan=plan, task_id=100)

    assert run_result.status == "completed"
    assert run_result.completed_count == 5
    assert isinstance(run_result.step_results[4].output, Report)


@pytest.mark.anyio
async def test_manager_to_orchestrator_real_registry():
    """Integration test attempting to run with the real AgentRegistry on main.

    Skips cleanly with an explicit explanation if any teammate module (M2/M3/M4)
    has not yet landed on main.
    """
    missing = check_missing_teammate_modules()
    if missing:
        pytest.skip(
            f"Teammate modules not yet landed on main: {', '.join(missing)}. "
            "Skipping real-registry integration test until M2, M3, and M4 stubs are merged."
        )

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set in environment for live real-registry integration test.")

    # If all teammate modules are present on main, execute the real pipeline
    plan = await create_plan("Hospital appointment reminder software")
    orch = Orchestrator(registry=AgentRegistry())
    result = await orch.run(plan=plan, task_id=200)
    assert result.status in ("completed", "partial")


@pytest.mark.anyio
async def test_end_to_end_pipeline_all_real_agents(monkeypatch):
    """End-to-end multi-agent pipeline using 100% REAL agent implementations.

    Tests the complete data flow across:
    Manager (Plan) -> Orchestrator (DAG execution) -> Research A & B (real agents)
    -> Analysis (real agent) -> Verifier (real agent) -> Reporter (real agent).
    """
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)

    # Mock web search and fetch so unit tests run 100% offline and deterministic
    async def mock_search(q, max_results=6):
        return [
            {"title": "HealthPlix", "url": "https://healthplix.com/solutions", "snippet": "HealthPlix appointment reminders"},
            {"title": "Qikwell", "url": "https://qikwell.com/solutions", "snippet": "Qikwell healthcare notifications"},
        ]

    async def mock_fetch(url):
        if "healthplix" in url:
            return (
                "HealthPlix automated appointment reminders support WhatsApp Business API. "
                "Compliant with TRAI DLT registration. Billed natively in INR starting at ₹2500 per month."
            )
        return (
            "Qikwell OPD appointment reminder software. ABDM and ABHA certified. "
            "Direct EMR integration with Apollo. Enterprise subscription is ₹3500 per month."
        )

    # Monkeypatch the default search & fetch used by research and verifier
    import agents.research
    import agents.verifier
    monkeypatch.setattr(agents.research, "default_search", mock_search)
    monkeypatch.setattr(agents.research, "default_fetch_url", mock_fetch)
    monkeypatch.setattr(agents.verifier, "_default_fetch", mock_fetch)

    # Valid DAG plan
    plan = Plan(
        steps=[
            PlanStep(agent="research_a", action="run_research_a", args={}, depends_on=[]),
            PlanStep(agent="research_b", action="run_research_b", args={}, depends_on=[]),
            PlanStep(agent="analysis", action="analyze", args={}, depends_on=[0, 1]),
            PlanStep(agent="verifier", action="verify", args={}, depends_on=[0, 1]),
            PlanStep(agent="reporter", action="build_report", args={}, depends_on=[2, 3]),
        ],
        success_criteria="Hospital appointment reminder shortlist",
        assumptions=["Hospital requires WhatsApp Business API and INR pricing"],
        researcher_count=2,
    )

    # Initialize Orchestrator with the real AgentRegistry
    registry = AgentRegistry()
    assert registry.has("research_a")
    assert registry.has("research_b")
    assert registry.has("analysis")
    assert registry.has("verifier")
    assert registry.has("reporter")

    orch = Orchestrator(registry=registry)
    run_result = await orch.run(
        plan=plan,
        task_id=999,
        task_context={"task": "Compare patient appointment-reminder software for Indian hospitals"}
    )

    # Verify overall pipeline outcome
    assert run_result.status == "completed"
    assert run_result.completed_count == 5

    # Verify step 0 & 1: Research A & B produced real Claim objects
    claims_a = run_result.step_results[0].output
    claims_b = run_result.step_results[1].output
    assert isinstance(claims_a, list) and len(claims_a) > 0
    assert isinstance(claims_b, list) and len(claims_b) > 0
    assert all(isinstance(c, Claim) for c in claims_a + claims_b)

    # Verify step 2: Analysis produced Analysis model
    analysis = run_result.step_results[2].output
    assert isinstance(analysis, Analysis)
    assert analysis.agreements > 0

    # Verify step 3: Verifier produced VerificationResults
    verifications = run_result.step_results[3].output
    assert isinstance(verifications, list) and len(verifications) > 0
    assert all(isinstance(v, VerificationResult) for v in verifications)

    # Verify step 4: Reporter produced Report with drafts
    report = run_result.step_results[4].output
    assert isinstance(report, Report)
    assert len(report.draft_emails) >= 2
    assert "HealthPlix" in report.draft_emails

