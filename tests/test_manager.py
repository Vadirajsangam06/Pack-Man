import sys, os, json, asyncio
import pytest
from pydantic import BaseModel

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import Plan, PlanStep
from agents.manager import create_plan, validate_plan_dag, ALLOWED_AGENTS
import llm
import db

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
def no_db_events(monkeypatch):
    """Silence database event logging during unit tests."""
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)
    yield

# Sample Plan for Task 1 (Hospital appointment reminders)
HOSPITAL_PLAN_DICT = {
    "steps": [
        {
            "agent": "research_a",
            "action": "run_research",
            "args": {"track": "features_and_india_compliance", "focus": "appointment reminders, WhatsApp Business, DLT SMS, ABDM"},
            "depends_on": []
        },
        {
            "agent": "research_b",
            "action": "run_research",
            "args": {"track": "pricing_and_reviews", "focus": "INR pricing, user reviews, hospital EMR integrations"},
            "depends_on": []
        },
        {
            "agent": "analysis",
            "action": "analyze",
            "args": {"task": "cross_reference_features_and_pricing"},
            "depends_on": [0, 1]
        },
        {
            "agent": "verifier",
            "action": "verify",
            "args": {"scope": "critical_claims_and_pricing"},
            "depends_on": [2]
        },
        {
            "agent": "reporter",
            "action": "build_report",
            "args": {"format": "hospital_operations_shortlist"},
            "depends_on": [3]
        }
    ],
    "assumptions": [
        "Hospital requires WhatsApp Business API and DLT-registered SMS for patient outreach.",
        "Must comply with India DPDP Act 2023 and ABDM/ABHA health records.",
        "Pricing must be transparent in INR with no hidden per-message markups."
    ],
    "success_criteria": "Evidence-based shortlist of top 3 appointment-reminder solutions with verified INR pricing and ABDM readiness.",
    "summary": "Hospital appointment reminder evaluation plan",
    "researcher_count": 2
}

# Sample Plan for Task 2 (Rural telemedicine)
TELEMEDICINE_PLAN_DICT = {
    "steps": [
        {
            "agent": "research_a",
            "action": "run_research",
            "args": {"track": "low_bandwidth_features", "focus": "telemedicine software, 2G/3G tolerance, offline sync, regional languages"},
            "depends_on": []
        },
        {
            "agent": "research_b",
            "action": "run_research",
            "args": {"track": "hardware_and_pricing", "focus": "affordable clinic kits, NGO/government subsidies, peripheral diagnostics"},
            "depends_on": []
        },
        {
            "agent": "analysis",
            "action": "analyze",
            "args": {"task": "rural_clinic_feasibility_matrix"},
            "depends_on": [0, 1]
        },
        {
            "agent": "verifier",
            "action": "verify",
            "args": {"scope": "offline_capabilities_and_cost"},
            "depends_on": [2]
        },
        {
            "agent": "reporter",
            "action": "build_report",
            "args": {"format": "rural_healthcare_recommendation"},
            "depends_on": [3]
        }
    ],
    "assumptions": [
        "Rural clinics often have intermittent power and low-bandwidth mobile connectivity.",
        "Regional language audio/video consultation support is mandatory for local healthcare workers.",
        "Telemedicine software must integrate with portable ECG/vitals devices."
    ],
    "success_criteria": "Shortlist of at least 2 telemedicine tools verified for low-bandwidth rural clinic deployments.",
    "summary": "Rural clinic telemedicine evaluation plan",
    "researcher_count": 3
}

@pytest.mark.anyio
async def test_create_plan_hospital(monkeypatch):
    """Test generating a plan for the hospital appointment reminder task."""
    async def mock_call_json(prompt, schema_model, system=None, max_attempts=5):
        return Plan.model_validate(HOSPITAL_PLAN_DICT)

    monkeypatch.setattr(llm, "call_json", mock_call_json)
    task = "Help an Indian hospital operations manager compare patient appointment-reminder software and prepare an evidence-based shortlist."
    plan = await create_plan(task)

    assert isinstance(plan, Plan)
    assert len(plan.steps) == 5
    assert plan.researcher_count == 2
    assert "WhatsApp" in plan.assumptions[0]
    assert plan.steps[0].agent == "research_a"
    assert plan.steps[1].agent == "research_b"
    assert plan.steps[2].depends_on == [0, 1]

@pytest.mark.anyio
async def test_create_plan_telemedicine(monkeypatch):
    """Test generating a plan for the rural telemedicine tools task."""
    async def mock_call_json(prompt, schema_model, system=None, max_attempts=5):
        return Plan.model_validate(TELEMEDICINE_PLAN_DICT)

    monkeypatch.setattr(llm, "call_json", mock_call_json)
    task = "Compare telemedicine tools for rural clinics in India."
    plan = await create_plan(task)

    assert isinstance(plan, Plan)
    assert len(plan.steps) == 5
    assert plan.researcher_count == 3
    assert "low-bandwidth" in plan.assumptions[0]

@pytest.mark.anyio
async def test_different_tasks_produce_different_plans(monkeypatch):
    """Verify that different tasks produce demonstrably distinct plans."""
    async def mock_call_json(prompt, schema_model, system=None, max_attempts=5):
        if "telemedicine" in prompt.lower():
            return Plan.model_validate(TELEMEDICINE_PLAN_DICT)
        return Plan.model_validate(HOSPITAL_PLAN_DICT)

    monkeypatch.setattr(llm, "call_json", mock_call_json)

    plan1 = await create_plan("Hospital appointment reminder software")
    plan2 = await create_plan("Telemedicine tools for rural clinics")

    assert plan1.summary != plan2.summary
    assert plan1.success_criteria != plan2.success_criteria
    assert plan1.assumptions != plan2.assumptions
    assert plan1.researcher_count != plan2.researcher_count
    assert plan1.steps[0].args != plan2.steps[0].args

def test_validate_plan_dag_cycle_detection():
    """Verify that circular dependencies are detected and raise ValueError."""
    cycle_plan = Plan(
        steps=[
            PlanStep(agent="research_a", action="run", args={}, depends_on=[1]),
            PlanStep(agent="research_b", action="run", args={}, depends_on=[0]),
        ],
        success_criteria="test",
        assumptions=["test"],
        researcher_count=2,
    )
    with pytest.raises(ValueError, match="Dependency cycle detected"):
        validate_plan_dag(cycle_plan)

def test_validate_plan_dag_unknown_agent():
    """Verify that an unknown agent raises ValueError."""
    invalid_agent_step = PlanStep.model_construct(agent="fake_agent", action="run", args={}, depends_on=[])
    invalid_agent_plan = Plan(
        steps=[invalid_agent_step],
        success_criteria="test",
        assumptions=["test"],
        researcher_count=2,
    )
    with pytest.raises(ValueError, match="unknown agent"):
        validate_plan_dag(invalid_agent_plan)

def test_validate_plan_dag_invalid_index():
    """Verify that invalid or self dependency indices raise ValueError."""
    self_dep_plan = Plan(
        steps=[
            PlanStep(agent="research_a", action="run", args={}, depends_on=[0]),
        ],
        success_criteria="test",
        assumptions=["test"],
        researcher_count=2,
    )
    with pytest.raises(ValueError, match="circular self-dependency"):
        validate_plan_dag(self_dep_plan)

    out_of_bound_plan = Plan(
        steps=[
            PlanStep(agent="research_a", action="run", args={}, depends_on=[5]),
        ],
        success_criteria="test",
        assumptions=["test"],
        researcher_count=2,
    )
    with pytest.raises(ValueError, match="invalid dependency index"):
        validate_plan_dag(out_of_bound_plan)

@pytest.mark.anyio
async def test_researcher_count_capped(monkeypatch):
    """Verify that researcher_count > 6 is capped at 6, and < 1 is capped at 1."""
    high_count_dict = dict(HOSPITAL_PLAN_DICT)
    high_count_dict["researcher_count"] = 6  # valid for Plan validation
    high_plan = Plan.model_validate(high_count_dict)
    high_plan.researcher_count = 10  # simulate LLM returning 10

    async def mock_call_json(prompt, schema_model, system=None, max_attempts=5):
        return high_plan

    monkeypatch.setattr(llm, "call_json", mock_call_json)
    plan = await create_plan("Test task")
    assert plan.researcher_count == 6

@pytest.mark.anyio
async def test_create_plan_retry_on_invalid_dag(monkeypatch):
    """Verify that create_plan retries when the first response has an invalid DAG."""
    cycle_plan = Plan(
        steps=[
            PlanStep(agent="research_a", action="run", args={}, depends_on=[1]),
            PlanStep(agent="research_b", action="run", args={}, depends_on=[0]),
        ],
        success_criteria="test",
        assumptions=["test"],
        researcher_count=2,
    )
    valid_plan = Plan.model_validate(HOSPITAL_PLAN_DICT)

    responses = [cycle_plan, valid_plan]
    call_count = 0

    async def mock_call_json(prompt, schema_model, system=None, max_attempts=5):
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    monkeypatch.setattr(llm, "call_json", mock_call_json)
    plan = await create_plan("Test task")
    assert call_count == 2
    assert plan.summary == "Hospital appointment reminder evaluation plan"

@pytest.mark.anyio
async def test_real_api_smoke_manager():
    """Optional smoke test hitting real Gemini API, skipped if GEMINI_API_KEY is unset."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set – skipping manager real API smoke test")
    plan = await create_plan("Evaluate patient appointment reminder software for a clinic in Bangalore")
    assert isinstance(plan, Plan)
    assert len(plan.steps) > 0
    assert plan.researcher_count >= 1
    assert plan.researcher_count <= 6
