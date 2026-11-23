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

from schemas import Plan, PlanStep
import db


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
            from agents.research import run_research
            self._registry["research"] = run_research
            self._registry["research_a"] = run_research
            self._registry["research_b"] = run_research
        except (ImportError, AttributeError):
            pass

        try:
            from agents.analysis import analyze
            self._registry["analysis"] = analyze
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
        # Collect claims from prior research steps
        aggregated_claims = []
        for out in dep_outputs:
            if isinstance(out, list):
                aggregated_claims.extend(out)
        kwargs["claims"] = aggregated_claims

    if "analysis" in params and "analysis" not in kwargs:
        for out in dep_outputs:
            if hasattr(out, "conflicts") or hasattr(out, "summary"):
                kwargs["analysis"] = out
                break

    if "verified" in params and "verified" not in kwargs:
        for out in dep_outputs:
            if isinstance(out, list) and out and hasattr(out[0], "claim_id"):
                kwargs["verified"] = out
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
