"""orchestrator.py - Generic Plan execution engine for AgentFlow AI.

Executes a validated Plan DAG:
- Resolves step dependencies dynamically.
- Runs independent steps concurrently using asyncio.gather.
- Enforces per-step timeouts.
- Disallows scheduling 'manager' as a plan step.
- Resilient failure handling: a failed step is marked and logged; remaining
  independent steps continue executing.
- Emits real-time progress events to SQLite via db.log_event.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from schemas import Plan, PlanStep, Claim
from pydantic import BaseModel, Field
import db


class RetryItem(BaseModel):
    """Item emitted by analysis/verification requiring targeted re-research."""
    vendor: str = Field(..., description="Name of the vendor")
    attribute: str = Field(..., description="Specific attribute requiring re-research")
    reason: str = Field(..., description="Explanation of why retry was triggered")


@dataclass
class RetryOutcome:
    """Outcome of a retry cycle for a specific vendor attribute."""
    vendor: str
    attribute: str
    status: str  # "verified" or "unverified"
    rounds_attempted: int
    claim: Claim
    reason: str


@dataclass
class StepResult:
    """Outcome of an individual plan step execution."""
    step_index: int
    agent: str
    action: str
    status: str  # "completed", "failed", "skipped"
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class RunResult:
    """Overall outcome of a plan execution run."""
    task_id: int
    status: str  # "completed", "partial", "failed"
    step_results: Dict[int, StepResult] = field(default_factory=dict)
    duration: float = 0.0

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self.step_results.values() if r.status == "completed")

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.step_results.values() if r.status == "failed")

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.step_results.values() if r.status == "skipped")


class AgentRegistry:
    """Registry mapping agent names to their executable callables."""

    def __init__(self, initial_mapping: Optional[Dict[str, Callable]] = None):
        self._registry: Dict[str, Callable] = dict(initial_mapping or {})
        if not self._registry:
            self._load_available_stubs()

    def _load_available_stubs(self) -> None:
        """Attempt to lazily load available agent modules if they exist."""
        try:
            from agents.research import run_research, run_research_a, run_research_b
            self._registry["research"] = run_research
            self._registry["research_a"] = run_research_a
            self._registry["research_b"] = run_research_b
        except (ImportError, AttributeError):
            pass

        try:
            from agents.analysis import analyze, analyze_async
            self._registry["analysis"] = analyze_async if callable(analyze_async) else analyze
        except (ImportError, AttributeError):
            pass

        try:
            from agents.verifier import verify
            self._registry["verifier"] = verify
        except (ImportError, AttributeError):
            pass

        try:
            from agents.reporter import build_report
            self._registry["reporter"] = build_report
        except (ImportError, AttributeError):
            pass

    def register(self, agent_name: str, handler: Callable) -> None:
        """Register or override a callable for a given agent name."""
        if agent_name == "manager":
            raise ValueError("Manager agent cannot be registered as an executable step worker.")
        self._registry[agent_name] = handler

    def get(self, agent_name: str) -> Optional[Callable]:
        return self._registry.get(agent_name)

    def has(self, agent_name: str) -> bool:
        return agent_name in self._registry


async def _execute_callable(
    func: Callable,
    step: PlanStep,
    dep_results: Dict[int, StepResult],
    task_context: Dict[str, Any],
) -> Any:
    """Invoke an agent callable with arguments from step.args and dependencies."""
    sig = inspect.signature(func)
    params = sig.parameters

    kwargs: Dict[str, Any] = dict(step.args)

    # Automatically resolve dependency outputs if the callable signature expects them
    dep_outputs = [dep_results[d].output for d in step.depends_on if d in dep_results and dep_results[d].status == "completed"]

    if "claims" in params and "claims" not in kwargs:
        # Collect claims from prior research steps (direct or transitive)
        aggregated_claims = []
        for out in dep_outputs:
            if isinstance(out, list):
                aggregated_claims.extend(out)
        if not aggregated_claims:
            for res in dep_results.values():
                if res.status == "completed" and isinstance(res.output, list):
                    for item in res.output:
                        if isinstance(item, Claim) or hasattr(item, "vendor"):
                            aggregated_claims.append(item)
        kwargs["claims"] = aggregated_claims

    if "analysis" in params and "analysis" not in kwargs:
        for out in dep_outputs:
            if hasattr(out, "conflicts") or hasattr(out, "summary"):
                kwargs["analysis"] = out
                break
        if "analysis" not in kwargs:
            for res in dep_results.values():
                if res.status == "completed" and (hasattr(res.output, "conflicts") or hasattr(res.output, "summary")):
                    kwargs["analysis"] = res.output
                    break

    if "verified" in params and "verified" not in kwargs:
        for out in dep_outputs:
            if isinstance(out, list) and out and hasattr(out[0], "claim_id"):
                kwargs["verified"] = out
                break
        if "verified" not in kwargs:
            for res in dep_results.values():
                if res.status == "completed" and isinstance(res.output, list) and res.output and hasattr(res.output[0], "claim_id"):
                    kwargs["verified"] = res.output
                    break

    if "task" in params and "task" not in kwargs:
        if "task" in task_context:
            kwargs["task"] = task_context["task"]

    # Filter kwargs to only those accepted by the callable, or pass all if **kwargs is present
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if not has_var_keyword:
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    # Handle async vs sync callables
    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    else:
        return await asyncio.to_thread(func, **kwargs)


class Orchestrator:
    """Generic DAG executor for multi-agent plans."""

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        default_step_timeout: float = 30.0,
    ):
        self.registry = registry or AgentRegistry()
        self.default_step_timeout = default_step_timeout

    async def execute_step(
        self,
        step_index: int,
        step: PlanStep,
        task_id: int,
        dep_results: Dict[int, StepResult],
        task_context: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> StepResult:
        """Execute a single plan step with timeout and error handling."""
        step_timeout = timeout or self.default_step_timeout
        start_time = time.time()

        # Rule: Never schedule 'manager' as a step
        if step.agent == "manager":
            err_msg = "Disallowed: 'manager' cannot be scheduled as an execution step."
            db.log_event(
                task_id=task_id,
                agent=step.agent,
                type="step_failed",
                payload={"step_index": step_index, "action": step.action, "error": err_msg},
            )
            return StepResult(
                step_index=step_index,
                agent=step.agent,
                action=step.action,
                status="failed",
                error=err_msg,
                duration=0.0,
            )

        handler = self.registry.get(step.agent)
        if not handler:
            err_msg = f"No handler registered for agent '{step.agent}'."
            db.log_event(
                task_id=task_id,
                agent=step.agent,
                type="step_failed",
                payload={"step_index": step_index, "action": step.action, "error": err_msg},
            )
            return StepResult(
                step_index=step_index,
                agent=step.agent,
                action=step.action,
                status="failed",
                error=err_msg,
                duration=0.0,
            )

        # Log step started
        db.log_event(
            task_id=task_id,
            agent=step.agent,
            type="step_started",
            payload={"step_index": step_index, "action": step.action, "args": step.args},
        )

        try:
            output = await asyncio.wait_for(
                _execute_callable(handler, step, dep_results, task_context),
                timeout=step_timeout,
            )
            duration = time.time() - start_time
            db.log_event(
                task_id=task_id,
                agent=step.agent,
                type="step_completed",
                payload={
                    "step_index": step_index,
                    "action": step.action,
                    "duration_s": round(duration, 3),
                },
            )
            return StepResult(
                step_index=step_index,
                agent=step.agent,
                action=step.action,
                status="completed",
                output=output,
                duration=duration,
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            err_msg = f"Step timed out after {step_timeout}s."
            db.log_event(
                task_id=task_id,
                agent=step.agent,
                type="step_failed",
                payload={"step_index": step_index, "error": err_msg, "duration_s": round(duration, 3)},
            )
            return StepResult(
                step_index=step_index,
                agent=step.agent,
                action=step.action,
                status="failed",
                error=err_msg,
                duration=duration,
            )

        except Exception as exc:
            duration = time.time() - start_time
            err_msg = f"{type(exc).__name__}: {str(exc)}"
            db.log_event(
                task_id=task_id,
                agent=step.agent,
                type="step_failed",
                payload={"step_index": step_index, "error": err_msg, "duration_s": round(duration, 3)},
            )
            return StepResult(
                step_index=step_index,
                agent=step.agent,
                action=step.action,
                status="failed",
                error=err_msg,
                duration=duration,
            )

    async def run(
        self,
        plan: Plan,
        task_id: int = 1,
        task_context: Optional[Dict[str, Any]] = None,
        timeout_per_step: Optional[float] = None,
    ) -> RunResult:
        """Execute a Plan in dependency order, running independent steps in parallel.

        Independent steps are dispatched in concurrent waves using asyncio.gather.
        Failed steps do not crash the run: any subsequent step whose dependencies
        fail is skipped, while independent branches continue executing.
        """
        run_start_time = time.time()
        context = dict(task_context or {})
        total_steps = len(plan.steps)

        step_results: Dict[int, StepResult] = {}
        pending_indices: Set[int] = set(range(total_steps))

        # Log run started
        db.log_event(
            task_id=task_id,
            agent="orchestrator",
            type="run_started",
            payload={
                "total_steps": total_steps,
                "assumptions": plan.assumptions,
                "success_criteria": plan.success_criteria,
            },
        )

        while pending_indices:
            # 1. Identify steps whose dependencies are all satisfied
            ready_to_run: List[int] = []
            steps_to_skip: List[int] = []

            for idx in sorted(pending_indices):
                step = plan.steps[idx]
                deps = step.depends_on

                # Check status of dependencies
                all_deps_finished = all(d in step_results for d in deps)
                if not all_deps_finished:
                    continue  # Still waiting on running or preceding steps

                # Check if any dependency failed or was skipped
                any_dep_failed = any(
                    step_results[d].status in ("failed", "skipped") for d in deps
                )

                if any_dep_failed:
                    steps_to_skip.append(idx)
                else:
                    ready_to_run.append(idx)

            # 2. Mark steps that cannot run due to failed dependencies as skipped
            for idx in steps_to_skip:
                step = plan.steps[idx]
                pending_indices.remove(idx)
                err_msg = "Skipped: one or more required dependencies failed."
                db.log_event(
                    task_id=task_id,
                    agent=step.agent,
                    type="step_skipped",
                    payload={"step_index": idx, "action": step.action, "reason": err_msg},
                )
                step_results[idx] = StepResult(
                    step_index=idx,
                    agent=step.agent,
                    action=step.action,
                    status="skipped",
                    error=err_msg,
                    duration=0.0,
                )

            # 3. If no steps are ready and no steps were skipped, we are in a deadlock/invalid state
            if not ready_to_run:
                if pending_indices:
                    # Mark remaining steps as skipped due to unresolved dependencies
                    for idx in sorted(pending_indices):
                        step = plan.steps[idx]
                        err_msg = "Skipped: unresolved dependency cycle or missing prior step."
                        db.log_event(
                            task_id=task_id,
                            agent=step.agent,
                            type="step_skipped",
                            payload={"step_index": idx, "action": step.action, "reason": err_msg},
                        )
                        step_results[idx] = StepResult(
                            step_index=idx,
                            agent=step.agent,
                            action=step.action,
                            status="skipped",
                            error=err_msg,
                            duration=0.0,
                        )
                    pending_indices.clear()
                break

            # 4. Dispatch all ready steps concurrently using asyncio.gather
            for idx in ready_to_run:
                pending_indices.remove(idx)

            tasks = [
                self.execute_step(
                    step_index=idx,
                    step=plan.steps[idx],
                    task_id=task_id,
                    dep_results=step_results,
                    task_context=context,
                    timeout=timeout_per_step,
                )
                for idx in ready_to_run
            ]

            results = await asyncio.gather(*tasks)
            for res in results:
                step_results[res.step_index] = res

        run_duration = time.time() - run_start_time

        # Determine overall run status
        completed = sum(1 for r in step_results.values() if r.status == "completed")
        failed = sum(1 for r in step_results.values() if r.status == "failed")
        skipped = sum(1 for r in step_results.values() if r.status == "skipped")

        if failed == 0 and skipped == 0 and completed == total_steps:
            overall_status = "completed"
        elif completed > 0:
            overall_status = "partial"
        else:
            overall_status = "failed"

        # Log run completed
        db.log_event(
            task_id=task_id,
            agent="orchestrator",
            type="run_completed",
            payload={
                "status": overall_status,
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
                "duration_s": round(run_duration, 3),
            },
        )

        return RunResult(
            task_id=task_id,
            status=overall_status,
            step_results=step_results,
            duration=run_duration,
        )

    async def run_retries(
        self,
        needs_retry: List[RetryItem],
        task_id: int = 1,
        max_rounds_per_item: int = 2,
        task_context: Optional[Dict[str, Any]] = None,
    ) -> List[RetryOutcome]:
        """Execute targeted retry rounds for items requiring re-research."""
        return await run_retry_loop(
            needs_retry=needs_retry,
            task_id=task_id,
            registry=self.registry,
            max_rounds_per_item=max_rounds_per_item,
            task_context=task_context,
        )


async def run_retry_loop(
    needs_retry: List[RetryItem],
    task_id: int,
    registry: Optional[AgentRegistry] = None,
    max_rounds_per_item: int = 2,
    task_context: Optional[Dict[str, Any]] = None,
) -> List[RetryOutcome]:
    """Execute targeted re-research loop for claims needing retry.

    Parameters
    ----------
    needs_retry : List[RetryItem]
        Items needing re-research (vendor, attribute, reason).
    task_id : int
        Current task ID.
    registry : Optional[AgentRegistry]
        Registry holding research and verifier handlers.
    max_rounds_per_item : int
        Maximum retry rounds per (vendor, attribute) pair (default 2).
    task_context : Optional[Dict[str, Any]]
        Optional context parameters.

    Returns
    -------
    List[RetryOutcome]
        Results of retries with verified claims or claims marked 'unverified'.
    """
    reg = registry or AgentRegistry()
    context = dict(task_context or {})
    outcomes: List[RetryOutcome] = []

    # Get research handler
    research_handler = reg.get("research") or reg.get("research_b") or reg.get("research_a")
    verifier_handler = reg.get("verifier")

    for item in needs_retry:
        rounds_attempted = 0
        resolved_claim: Optional[Claim] = None
        succeeded = False

        for round_num in range(1, max_rounds_per_item + 1):
            rounds_attempted = round_num

            # Log retry round started
            db.log_event(
                task_id=task_id,
                agent="orchestrator",
                type="retry_round",
                payload={
                    "vendor": item.vendor,
                    "attribute": item.attribute,
                    "reason": item.reason,
                    "round": round_num,
                    "max_rounds": max_rounds_per_item,
                },
            )

            if not research_handler:
                break

            try:
                # Build research arguments targeting the specific vendor & attribute
                task_desc = context.get("task", f"Re-research {item.vendor} {item.attribute}")
                sig = inspect.signature(research_handler)
                kwargs: Dict[str, Any] = {}
                if "task" in sig.parameters:
                    kwargs["task"] = task_desc
                if "vendor_focus" in sig.parameters:
                    kwargs["vendor_focus"] = item.vendor
                elif "vendor" in sig.parameters:
                    kwargs["vendor"] = item.vendor

                # Execute research call
                if inspect.iscoroutinefunction(research_handler):
                    claims_found = await research_handler(**kwargs)
                else:
                    claims_found = await asyncio.to_thread(research_handler, **kwargs)

                if isinstance(claims_found, list) and claims_found:
                    # Look for matching claim
                    candidate = None
                    for c in claims_found:
                        if isinstance(c, Claim) and c.vendor.lower() == item.vendor.lower():
                            if c.attribute.lower() == item.attribute.lower() or not candidate:
                                candidate = c

                    if candidate:
                        # If verifier is present, verify the candidate
                        is_verified = True
                        if verifier_handler:
                            v_sig = inspect.signature(verifier_handler)
                            v_kwargs = {"claims": [candidate]} if "claims" in v_sig.parameters else {}
                            if inspect.iscoroutinefunction(verifier_handler):
                                v_res = await verifier_handler(**v_kwargs)
                            else:
                                v_res = await asyncio.to_thread(verifier_handler, **v_kwargs)

                            if isinstance(v_res, list) and v_res:
                                is_verified = any(
                                    getattr(vr, "status", None) == "verified" for vr in v_res
                                )

                        if is_verified:
                            succeeded = True
                            resolved_claim = candidate
                            db.log_event(
                                task_id=task_id,
                                agent="orchestrator",
                                type="retry_resolved",
                                payload={
                                    "vendor": item.vendor,
                                    "attribute": item.attribute,
                                    "round": round_num,
                                    "value": candidate.value,
                                    "source_url": str(candidate.source_url),
                                },
                            )
                            break

            except Exception as exc:
                db.log_event(
                    task_id=task_id,
                    agent="orchestrator",
                    type="retry_round_error",
                    payload={
                        "vendor": item.vendor,
                        "attribute": item.attribute,
                        "round": round_num,
                        "error": str(exc),
                    },
                )

        if succeeded and resolved_claim:
            outcomes.append(
                RetryOutcome(
                    vendor=item.vendor,
                    attribute=item.attribute,
                    status="verified",
                    rounds_attempted=rounds_attempted,
                    claim=resolved_claim,
                    reason=item.reason,
                )
            )
        else:
            # Mark claim unverified as required by contract
            unverified_claim = Claim(
                vendor=item.vendor,
                attribute=item.attribute,
                value="unverified",
                source_url="https://unverified.internal/retry-exhausted",  # type: ignore
                snippet=f"Unverified after {rounds_attempted} retry rounds. Reason: {item.reason}",
            )
            db.log_event(
                task_id=task_id,
                agent="orchestrator",
                type="retry_exhausted",
                payload={
                    "vendor": item.vendor,
                    "attribute": item.attribute,
                    "reason": item.reason,
                    "rounds_attempted": rounds_attempted,
                    "status": "unverified",
                },
            )
            outcomes.append(
                RetryOutcome(
                    vendor=item.vendor,
                    attribute=item.attribute,
                    status="unverified",
                    rounds_attempted=rounds_attempted,
                    claim=unverified_claim,
                    reason=item.reason,
                )
            )

    return outcomes
