from __future__ import annotations

from typing import Any, Awaitable, Callable

from .chat import run_chat
from .goals import classify_goal, plan_summary
from .intake import run_intake
from .intent import classify_intent
from .product_matching import run_product_matching
from .sector import run_sector_chat
from .session import strip_echoed_query
from .store import create_agent_run, update_agent_run
from .supplier_finder import run_buyer_finder, run_supplier_finder
from .trade_advisor import run_trade_advisor
from .commercial import human_intent_detail
from .types import AGENT_LABELS, AgentDeps, AgentOutcome, AgentStep

StepCallback = Callable[[AgentStep], Awaitable[None]]

_RUNNERS = {
    "chat": run_chat,
    "intake": run_intake,
    "sector_chat": run_sector_chat,
    "trade_advisor": run_trade_advisor,
    "buyer_finder": run_buyer_finder,
    "supplier_finder": run_supplier_finder,
    "product_matching": run_product_matching,
}


async def run_orchestrator(
    *,
    question: str,
    deps: AgentDeps,
    on_step: StepCallback | None = None,
) -> tuple[str | None, AgentOutcome]:
    steps: list[AgentStep] = []
    run_id = await create_agent_run(
        deps.supabase,
        question,
        account_id=getattr(deps, "account_id", None) or None,
    )

    async def emit(step: AgentStep) -> None:
        for index, existing in enumerate(steps):
            if existing.agent == step.agent and existing.key == step.key:
                steps[index] = step
                break
        else:
            steps.append(step)
        if on_step:
            await on_step(step)
        await update_agent_run(deps.supabase, run_id, steps=steps)

    route = AgentStep(
        key="intent",
        agent="orchestrator",
        label="Niyet analizi",
        status="running",
        detail="Sorgu türü sınıflandırılıyor.",
    )
    steps.append(route)
    await emit(route)

    intent, scores = classify_intent(question, session=deps.session)
    goal = classify_goal(question, deps.session)
    deps.goal = goal
    if deps.session is not None:
        deps.session.last_goal = goal
    selected = AGENT_LABELS[intent]
    route.status = "success"
    route.detail = human_intent_detail(intent, question)
    await emit(route)
    plan = AgentStep(
        key="plan",
        agent="orchestrator",
        label="Plan",
        status="success",
        detail=plan_summary(goal, question, deps.session),
    )
    steps.append(plan)
    await emit(plan)
    await update_agent_run(
        deps.supabase,
        run_id,
        intent=intent,
        selected_agent=intent,
        status="running",
        steps=steps,
    )

    dispatch = AgentStep(
        key="dispatch",
        agent="orchestrator",
        label=f"{selected} devreye alındı",
        status="success",
        detail=human_intent_detail(intent, question),
    )
    steps.append(dispatch)
    await emit(dispatch)

    try:
        advice, context, _agent_steps = await _RUNNERS[intent](question, deps, emit)
        advice = strip_echoed_query(question, advice)
    except Exception as exc:
        await update_agent_run(
            deps.supabase,
            run_id,
            status="error",
            intent=intent,
            selected_agent=intent,
            steps=steps,
            error_message=str(exc),
        )
        raise

    outcome = AgentOutcome(
        intent=intent,
        selected_agent=intent,
        advice=advice,
        context=context,
        steps=steps,
        scores=scores,
    )
    await update_agent_run(
        deps.supabase,
        run_id,
        status="success",
        intent=intent,
        selected_agent=intent,
        steps=steps,
        result={
            "advice": advice,
            "context_count": len(context),
            "scores": scores,
        },
    )
    return run_id, outcome


async def fail_run(supabase: Any, run_id: str | None, message: str) -> None:
    await update_agent_run(
        supabase,
        run_id,
        status="error",
        error_message=message,
    )
