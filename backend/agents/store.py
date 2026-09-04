from __future__ import annotations

from typing import Any

from supabase import AsyncClient

from .types import AgentStep


async def create_agent_run(supabase: AsyncClient, query: str) -> str | None:
    try:
        result = await (
            supabase.table("agent_runs")
            .insert({"query": query, "status": "running", "steps": []})
            .execute()
        )
        if result.data:
            return str(result.data[0]["id"])
    except Exception:
        return None
    return None


async def update_agent_run(
    supabase: AsyncClient,
    run_id: str | None,
    *,
    steps: list[AgentStep] | None = None,
    **fields: Any,
) -> None:
    if not run_id:
        return
    payload = dict(fields)
    if steps is not None:
        payload["steps"] = [step.to_dict() for step in steps]
    try:
        await supabase.table("agent_runs").update(payload).eq("id", run_id).execute()
    except Exception:
        return
