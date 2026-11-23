import sys, os, time, asyncio
import pytest

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import Claim, VerificationResult
from orchestrator import RetryItem, RetryOutcome, run_retry_loop, AgentRegistry, Orchestrator
import db

@pytest.fixture
def anyio_backend():
    return "asyncio"

def test_retry_item_model():
    """Verify RetryItem model attributes and validation."""
    item = RetryItem(
        vendor="HealthPlix",
        attribute="pricing_inr",
        reason="Snippet missing explicit INR currency symbol."
    )
    assert item.vendor == "HealthPlix"
    assert item.attribute == "pricing_inr"
    assert "missing" in item.reason

@pytest.mark.anyio
async def test_retry_loop_success_round_1(monkeypatch):
    """Test retry succeeding on the first round."""
    logged_events = []
    monkeypatch.setattr(db, "log_event", lambda task_id, agent, type, payload: logged_events.append((type, payload)))

    async def fake_research(task=None, vendor_focus=None, **kwargs):
        return [
            Claim(
                vendor="HealthPlix",
                attribute="pricing_inr",
                value="₹2000/month",
                source_url="https://healthplix.com/pricing",
                snippet="Pricing is ₹2000 per month."
            )
        ]

    reg = AgentRegistry({"research": fake_research})
    items = [
        RetryItem(vendor="HealthPlix", attribute="pricing_inr", reason="Missing INR pricing")
    ]

    outcomes = await run_retry_loop(needs_retry=items, task_id=10, registry=reg, max_rounds_per_item=2)

    assert len(outcomes) == 1
    assert outcomes[0].status == "verified"
    assert outcomes[0].rounds_attempted == 1
    assert outcomes[0].claim.value == "₹2000/month"

    event_types = [e[0] for e in logged_events]
    assert "retry_round" in event_types
    assert "retry_resolved" in event_types

@pytest.mark.anyio
async def test_retry_loop_success_round_2(monkeypatch):
    """Test retry failing in round 1, then succeeding in round 2."""
    logged_events = []
    monkeypatch.setattr(db, "log_event", lambda task_id, agent, type, payload: logged_events.append((type, payload)))

    call_count = 0
    async def multi_round_research(task=None, vendor_focus=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return []  # Round 1 returns empty
        return [
            Claim(
                vendor="Practo Ray",
                attribute="abdm_integration",
                value="ABDM M1 & M2 certified",
                source_url="https://practo.com/abdm",
                snippet="Certified for ABDM milestone 1 and milestone 2."
            )
        ]

    reg = AgentRegistry({"research": multi_round_research})
    items = [
        RetryItem(vendor="Practo Ray", attribute="abdm_integration", reason="Unverified ABDM support")
    ]

    outcomes = await run_retry_loop(needs_retry=items, task_id=11, registry=reg, max_rounds_per_item=2)

    assert len(outcomes) == 1
    assert outcomes[0].status == "verified"
    assert outcomes[0].rounds_attempted == 2
    assert outcomes[0].claim.value == "ABDM M1 & M2 certified"

    # Check logged rounds
    round_events = [e for e in logged_events if e[0] == "retry_round"]
    assert len(round_events) == 2
    assert round_events[0][1]["round"] == 1
    assert round_events[1][1]["round"] == 2

@pytest.mark.anyio
async def test_retry_loop_exhausted_unverified(monkeypatch):
    """Test retry exhausting max rounds and marking the claim unverified."""
    logged_events = []
    monkeypatch.setattr(db, "log_event", lambda task_id, agent, type, payload: logged_events.append((type, payload)))

    async def empty_research(task=None, vendor_focus=None, **kwargs):
        return []

    reg = AgentRegistry({"research": empty_research})
    items = [
        RetryItem(vendor="UnknownVendor", attribute="dlt_sms", reason="No information found online")
    ]

    outcomes = await run_retry_loop(needs_retry=items, task_id=12, registry=reg, max_rounds_per_item=2)

    assert len(outcomes) == 1
    assert outcomes[0].status == "unverified"
    assert outcomes[0].rounds_attempted == 2
    assert outcomes[0].claim.value == "unverified"
    assert "Unverified after 2 retry rounds" in outcomes[0].claim.snippet

    event_types = [e[0] for e in logged_events]
    assert "retry_exhausted" in event_types

@pytest.mark.anyio
async def test_orchestrator_run_retries_method(monkeypatch):
    """Test calling run_retries via Orchestrator instance."""
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)

    async def fake_research(task=None, vendor_focus=None, **kwargs):
        return [
            Claim(
                vendor="Qikwell",
                attribute="emr_integration",
                value="HL7 FHIR supported",
                source_url="https://qikwell.com/integration",
                snippet="Integrates with major hospital EMRs via HL7 FHIR."
            )
        ]

    orch = Orchestrator(registry=AgentRegistry({"research": fake_research}))
    items = [RetryItem(vendor="Qikwell", attribute="emr_integration", reason="Check FHIR")]
    outcomes = await orch.run_retries(needs_retry=items, task_id=1)

    assert len(outcomes) == 1
    assert outcomes[0].status == "verified"
    assert outcomes[0].claim.value == "HL7 FHIR supported"

def test_db_task_history_save_and_load():
    """Verify task and run history lifecycle in db.py:
    1. Create task
    2. Record run start and end
    3. Log events
    4. List past runs with descriptions
    5. Load chronological run events
    6. List tasks
    """
    # 1. Create a task
    desc = "Compare patient reminder systems for Bangalore hospital"
    task_id = db.create_task(desc)
    assert isinstance(task_id, int)
    assert task_id > 0

    # 2. Record run
    run_id = db.record_run_start(task_id)
    assert isinstance(run_id, int)
    assert run_id > 0

    # 3. Log events
    db.log_event(task_id, agent="manager", type="plan_created", payload={"steps": 3})
    db.log_event(task_id, agent="research_a", type="step_started", payload={"action": "run_research"})
    db.log_event(task_id, agent="orchestrator", type="run_completed", payload={"status": "completed"})

    # Complete run
    db.record_run_end(run_id, status="completed")

    # 4. List runs
    runs = db.list_runs(limit=10)
    assert len(runs) > 0
    matching_run = next((r for r in runs if r["id"] == run_id), None)
    assert matching_run is not None
    assert matching_run["task_id"] == task_id
    assert matching_run["description"] == desc
    assert matching_run["status"] == "completed"
    assert matching_run["end_ts"] is not None

    # 5. Load run events
    events = db.load_run_events(task_id)
    assert len(events) >= 3
    event_types = [e["type"] for e in events]
    assert "plan_created" in event_types
    assert "step_started" in event_types
    assert "run_completed" in event_types
    # Verify parsed payload
    plan_event = next(e for e in events if e["type"] == "plan_created")
    assert plan_event["payload"]["steps"] == 3

    # 6. List tasks
    tasks = db.list_tasks(limit=10)
    matching_task = next((t for t in tasks if t["id"] == task_id), None)
    assert matching_task is not None
    assert matching_task["description"] == desc
