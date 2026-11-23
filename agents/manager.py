"""agents/manager.py - Lead planning agent for AgentFlow AI.

Decomposes a user task into a validated, dependency-aware Plan.
- Chooses required agents (research_a, research_b, analysis, verifier, reporter).
- Determines dependency DAG with cycle detection.
- Decides on researcher_count (capped at 6).
- Logs events to db.
- Never names vendors in the plan structure; keeps market hints in assumptions.
"""

from typing import Set
from schemas import Plan, PlanStep
import llm
import db

ALLOWED_AGENTS: Set[str] = {
    "research",
    "research_a",
    "research_b",
    "analysis",
    "verifier",
    "reporter",
    "manager",
}

SYSTEM_PROMPT = """You are the Lead Planning Agent for AgentFlow AI, a multi-agent research-to-action platform.
Your job is to analyze the user's task and generate an optimal, structured execution Plan conforming to the Plan schema.

Rules:
1. Available agents to assign per step:
   - "research_a": Focuses on core features, technical capabilities, and Indian market criteria (e.g., WhatsApp Business API, DLT/TRAI SMS, ABDM/ABHA integration, DPDP Act 2023 compliance, regional language support, EMR integration).
   - "research_b": Focuses on pricing models (INR pricing), verified customer reviews, limitations, hidden costs, and alternative solutions.
   - "analysis": Consolidates and structures claims, detects conflicting information across sources.
   - "verifier": Re-fetches sources, verifies snippet evidence, and validates claim authenticity.
   - "reporter": Synthesizes verified findings, builds evidence-based recommendation matrix, and drafts vendor communications for human review.
2. Dependencies:
   - Each step specifies 'depends_on' as a list of 0-based indices of prior steps it requires before execution.
   - Independent steps should have empty depends_on so they can execute in parallel.
   - Never create circular dependencies or reference out-of-bound step indices.
3. No Hallucinated Vendors:
   - Do NOT invent or name specific vendors in the plan steps or action arguments. Vendor discovery happens during live research.
   - Provide market hints, domain context, search strategies, and compliance requirements strictly as plain text in the 'assumptions' list.
4. Researcher Count:
   - Decide how many parallel research tracks to spawn (researcher_count: integer between 1 and 6, typically 2 for dual track research_a and research_b).
"""

def validate_plan_dag(plan: Plan) -> None:
    """Validate that the plan forms a valid Directed Acyclic Graph (DAG) with known agents."""
    n = len(plan.steps)
    if n == 0:
        raise ValueError("Plan must contain at least one step.")

    # Validate agent names and dependency bounds
    for i, step in enumerate(plan.steps):
        if step.agent not in ALLOWED_AGENTS:
            raise ValueError(
                f"Step {i} uses unknown agent '{step.agent}'. Allowed agents: {sorted(ALLOWED_AGENTS)}"
            )
        for dep in step.depends_on:
            if not isinstance(dep, int) or dep < 0 or dep >= n:
                raise ValueError(
                    f"Step {i} has invalid dependency index {dep}. Must be an integer between 0 and {n - 1}."
                )
            if dep == i:
                raise ValueError(f"Step {i} has a circular self-dependency on itself (index {i}).")

    # Detect cycles using Kahn's algorithm (topological sort)
    adj = {i: [] for i in range(n)}
    in_degree = {i: 0 for i in range(n)}
    for i, step in enumerate(plan.steps):
        for dep in step.depends_on:
            adj[dep].append(i)
            in_degree[i] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    visited_count = 0
    while queue:
        u = queue.pop(0)
        visited_count += 1
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if visited_count < n:
        raise ValueError("Dependency cycle detected in plan steps.")


async def create_plan(task: str, max_retries: int = 2) -> Plan:
    """Generate a validated execution Plan for the given task using Gemini.

    Parameters
    ----------
    task : str
        Description of the objective (e.g. appointment reminder evaluation or rural telemedicine).
    max_retries : int, optional
        Number of retry attempts if LLM output fails DAG or agent validation.

    Returns
    -------
    Plan
        Validated plan ready for execution by the orchestrator.
    """
    base_prompt = (
        f"Task Description:\n{task}\n\n"
        "Generate a complete, tailored execution Plan for this task. "
        "Define steps, agent assignments, dependencies, success criteria, and domain assumptions. "
        "Set researcher_count between 1 and 6."
    )
    prompt = base_prompt

    for attempt in range(max_retries + 1):
        try:
            plan = await llm.call_json(prompt=prompt, schema_model=Plan, system=SYSTEM_PROMPT)

            # Cap researcher_count between 1 and 6
            if plan.researcher_count > 6:
                plan.researcher_count = 6
            elif plan.researcher_count < 1:
                plan.researcher_count = 1

            # Validate DAG and agent names
            validate_plan_dag(plan)

            # Log events for UI polling
            db.log_event(
                task_id=0,
                agent="manager",
                type="plan_created",
                payload={
                    "step_count": len(plan.steps),
                    "researcher_count": plan.researcher_count,
                    "success_criteria": plan.success_criteria,
                    "summary": plan.summary or "Plan generated successfully",
                },
            )
            db.log_event(
                task_id=0,
                agent="manager",
                type="plan_assumptions",
                payload={"assumptions": plan.assumptions},
            )
            return plan

        except Exception as e:
            if attempt < max_retries:
                prompt = (
                    f"{base_prompt}\n\n"
                    f"PREVIOUS PLAN FAILED VALIDATION:\n{e}\n"
                    "Please regenerate the JSON plan fixing this error. Ensure all agent names are valid, "
                    "no dependency cycles exist, and all dependency indices are within bounds (0 to N-1)."
                )
                db.log_event(
                    task_id=0,
                    agent="manager",
                    type="plan_validation_retry",
                    payload={"attempt": attempt + 1, "error": str(e)},
                )
            else:
                db.log_event(
                    task_id=0,
                    agent="manager",
                    type="plan_validation_failed",
                    payload={"error": str(e)},
                )
                raise
