import sys, os, time, asyncio
import pytest

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import Plan, PlanStep, Claim, Analysis, VerificationResult, Report
from orchestrator import Orchestrator, AgentRegistry, StepResult, RunResult
import db

@pytest.fixture
def anyio_backend():
    return "asyncio"

# Sample fake callables for testing
async def fake_research_a(task=None, **kwargs):
    await asyncio.sleep(0.05)
    return [
        Claim(
            vendor="HealthRemind India",
            attribute="whatsapp_api",
            value="supported",
            source_url="https://example.com/healthremind",
            snippet="Supports official WhatsApp Business API with DLT templates."
        )
    ]

async def fake_research_b(task=None, **kwargs):
    await asyncio.sleep(0.05)
    return [
        Claim(
            vendor="HealthRemind India",
            attribute="pricing",
            value="₹3500/month",
            source_url="https://example.com/pricing",
            snippet="Standard hospital plan starts at ₹3500 per month."
        )
    ]

async def fake_analysis(claims=None, **kwargs):
    await asyncio.sleep(0.02)
    return Analysis(
        summary=f"Analyzed {len(claims or [])} claims across vendors.",
        conflicts=[],
        agreements=len(claims or [])
    )

async def fake_verifier(claims=None, **kwargs):
    await asyncio.sleep(0.02)
    return [
        VerificationResult(claim_id=1, status="verified", details="Matched snippet verbatim")
    ]

async def fake_reporter(analysis=None, verified=None, **kwargs):
    await asyncio.sleep(0.02)
    return Report(
        title="Hospital Reminder Shortlist",
        content="Evidence-based comparison report for hospital operations.",
        draft_emails={"HealthRemind India": "Dear Team, please provide an enterprise quote..."}
    )

@pytest.fixture
def test_registry():
    """Create an agent registry with fake callables."""
    registry = AgentRegistry({
        "research_a": fake_research_a,
        "research_b": fake_research_b,
        "analysis": fake_analysis,
        "verifier": fake_verifier,
        "reporter": fake_reporter,
    })
    return registry

@pytest.mark.anyio
async def test_orchestrator_full_pipeline(test_registry, monkeypatch):
    """Test end-to-end execution of a multi-step plan in dependency order."""
    logged_events = []
    monkeypatch.setattr(db, "log_event", lambda task_id, agent, type, payload: logged_events.append((type, agent)))

    plan = Plan(
        steps=[
            PlanStep(agent="research_a", action="run_research", args={}, depends_on=[]),
            PlanStep(agent="research_b", action="run_research", args={}, depends_on=[]),
            PlanStep(agent="analysis", action="analyze", args={}, depends_on=[0, 1]),
            PlanStep(agent="verifier", action="verify", args={}, depends_on=[2]),
            PlanStep(agent="reporter", action="build_report", args={}, depends_on=[3]),
        ],
        success_criteria="Shortlist produced",
        assumptions=["Hospital requires WhatsApp and INR pricing"],
        researcher_count=2,
    )

    orch = Orchestrator(registry=test_registry)
    result = await orch.run(plan=plan, task_id=42)

    assert result.status == "completed"
    assert result.completed_count == 5
    assert result.failed_count == 0
    assert result.skipped_count == 0

    # Verify report was generated
    report = result.step_results[4].output
    assert isinstance(report, Report)
    assert "HealthRemind India" in report.draft_emails

    # Verify event types logged
    event_types = [e[0] for e in logged_events]
    assert "run_started" in event_types
    assert "step_started" in event_types
    assert "step_completed" in event_types
    assert "run_completed" in event_types

@pytest.mark.anyio
async def test_orchestrator_parallel_execution(test_registry, monkeypatch):
    """Verify independent steps execute concurrently."""
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)

    start_times = {}
    async def timed_step_1(**kwargs):
        start_times[1] = time.time()
        await asyncio.sleep(0.1)
        return "done_1"

    async def timed_step_2(**kwargs):
        start_times[2] = time.time()
        await asyncio.sleep(0.1)
        return "done_2"

    reg = AgentRegistry({
        "research_a": timed_step_1,
        "research_b": timed_step_2,
    })

    plan = Plan(
        steps=[
            PlanStep(agent="research_a", action="run", args={}, depends_on=[]),
            PlanStep(agent="research_b", action="run", args={}, depends_on=[]),
        ],
        success_criteria="test",
        assumptions=[],
        researcher_count=2,
    )

    orch = Orchestrator(registry=reg)
    t0 = time.time()
    result = await orch.run(plan=plan, task_id=1)
    total_time = time.time() - t0

    assert result.status == "completed"
    # If sequential, it would take >= 0.2s. Parallel should take ~0.1s (< 0.18s)
    assert total_time < 0.18
    assert abs(start_times[1] - start_times[2]) < 0.05

@pytest.mark.anyio
async def test_orchestrator_step_timeout(monkeypatch):
    """Verify that a step exceeding its timeout fails gracefully and doesn't block."""
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)

    async def hanging_step(**kwargs):
        await asyncio.sleep(5.0)
        return "never"

    reg = AgentRegistry({
        "research_a": hanging_step,
    })

    plan = Plan(
        steps=[
            PlanStep(agent="research_a", action="run", args={}, depends_on=[]),
        ],
        success_criteria="test",
        assumptions=[],
        researcher_count=1,
    )

    orch = Orchestrator(registry=reg, default_step_timeout=0.1)
    result = await orch.run(plan=plan, task_id=1)

    assert result.status == "failed"
    assert result.failed_count == 1
    assert "timed out" in result.step_results[0].error.lower()

@pytest.mark.anyio
async def test_orchestrator_never_schedule_manager(test_registry, monkeypatch):
    """Verify that scheduling 'manager' as an execution step is rejected."""
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)

    plan = Plan(
        steps=[
            PlanStep(agent="manager", action="replan", args={}, depends_on=[]),
        ],
        success_criteria="test",
        assumptions=[],
        researcher_count=1,
    )

    orch = Orchestrator(registry=test_registry)
    result = await orch.run(plan=plan, task_id=1)

    assert result.status == "failed"
    assert result.step_results[0].status == "failed"
    assert "Disallowed: 'manager'" in result.step_results[0].error

@pytest.mark.anyio
async def test_orchestrator_resilience_to_failed_step(test_registry, monkeypatch):
    """Verify a failed step does not crash the run:
    - Step 0 (research_a) fails with an exception.
    - Step 1 (research_b) is independent and succeeds.
    - Step 2 depends on Step 0 and is skipped.
    """
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)

    async def failing_step(**kwargs):
        raise ConnectionError("Network unreachable")

    reg = AgentRegistry({
        "research_a": failing_step,
        "research_b": fake_research_b,
        "analysis": fake_analysis,
    })

    plan = Plan(
        steps=[
            # Step 0: fails
            PlanStep(agent="research_a", action="run", args={}, depends_on=[]),
            # Step 1: independent, should succeed
            PlanStep(agent="research_b", action="run", args={}, depends_on=[]),
            # Step 2: depends on 0, should be skipped
            PlanStep(agent="analysis", action="run", args={}, depends_on=[0]),
        ],
        success_criteria="test",
        assumptions=[],
        researcher_count=2,
    )

    orch = Orchestrator(registry=reg)
    result = await orch.run(plan=plan, task_id=1)

    assert result.status == "partial"
    assert result.step_results[0].status == "failed"
    assert "ConnectionError" in result.step_results[0].error
    assert result.step_results[1].status == "completed"
    assert result.step_results[2].status == "skipped"
    assert "dependencies failed" in result.step_results[2].error

def test_agent_registry_manager_rejection():
    """Verify that registering 'manager' directly in AgentRegistry raises ValueError."""
    reg = AgentRegistry()
    with pytest.raises(ValueError, match="Manager agent cannot be registered"):
        reg.register("manager", lambda: None)
