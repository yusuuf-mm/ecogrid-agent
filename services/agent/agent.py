"""services/agent/agent.py

LangChain orchestrator. Takes a natural language grid optimization
request and drives the three tools in the required order:

    1. tool_query_policies → safety buffer from policy doc
    2. tool_forecast_solar → 24-hour solar generation
    3. tool_optimize_grid  → LP solver call

Returns a fully-populated `OptimizationResponse` (from shared.contracts)
including the audit trail.

The agent does no arithmetic. It reads language, decides tool order,
extracts numbers from tool outputs, and synthesizes the final summary.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from loguru import logger

from services.agent.config import get_settings
from services.agent.prompts import SYSTEM_PROMPT
from services.agent.tools import (
    tool_forecast_solar,
    tool_optimize_grid,
    tool_query_policies,
)
from shared.contracts import (
    AuditTrail,
    OptimizationResponse,
    ScheduleHour,
    SolverObjective,
    SolverStatus,
    TaskStatus,
)


_DEFAULT_BATTERY_CAPACITY_KWH: float = 1000.0
_DEFAULT_MAX_CHARGE_RATE_KW: float = 250.0
_DEFAULT_INITIAL_SOC_KWH: float = 500.0
_FALLBACK_PRICE_KWH: list[float] = [0.05] * 24


def _build_agent() -> AgentExecutor:
    cfg = get_settings()
    llm = ChatOpenAI(
        model=cfg.OPENAI_MODEL,
        temperature=cfg.AGENT_TEMPERATURE,
        api_key=cfg.OPENAI_API_KEY,
    )
    tools = [tool_query_policies, tool_forecast_solar, tool_optimize_grid]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=cfg.AGENT_MAX_ITERATIONS,
        return_intermediate_steps=True,
        verbose=False,
    )


def _build_hourly_reasons(
    schedule: list[ScheduleHour],
    prices: list[float] | None,
    buffer: float,
    policy_doc_id: str | None,
    solar: list[float] | None,
) -> list[ScheduleHour]:
    """Attach natural-language `reason` strings to each schedule entry.

    The solver produces a mathematical schedule; the agent layer adds the
    explanation a human operator reads. Done here, deterministically, so
    the explanations are grounded in the actual numbers from the solver.
    """
    solar_peak = max(solar) if solar else 0.0
    enriched: list[ScheduleHour] = []

    for hour_entry in schedule:
        h = hour_entry.hour
        price = prices[h] if prices and h < len(prices) else 0.0
        charge = hour_entry.charge_kw
        discharge = hour_entry.discharge_kw
        solar_stored = hour_entry.solar_stored_kw

        if discharge > 0 and price > 0:
            reason = (
                f"Hour {h:02d}:00 — discharge {discharge:.1f} kW to grid at "
                f"${price:.3f}/kWh (est. revenue ${discharge * price:.2f})."
            )
        elif charge > 0 and solar_stored <= 0:
            reason = (
                f"Hour {h:02d}:00 — charge {charge:.1f} kW from grid at "
                f"${price:.3f}/kWh (cheap-window absorption)."
            )
        elif solar_stored > 0 or (solar and h < len(solar) and solar[h] > 50):
            reason = (
                f"Hour {h:02d}:00 — store {solar_stored:.1f} kW of solar "
                f"(daily peak {solar_peak:.1f} kW)."
            )
        elif charge > 0 or solar_stored > 0:
            reason = (
                f"Hour {h:02d}:00 — absorb low-cost or solar energy "
                f"({(charge + solar_stored):.1f} kW at ${price:.3f}/kWh)."
            )
        else:
            doc = policy_doc_id or "active policy"
            reason = (
                f"Hour {h:02d}:00 — hold reserve. Enforce "
                f"{buffer * 100:.0f}% SoC floor per {doc}."
            )
        enriched.append(hour_entry.model_copy(update={"reason": reason}))

    return enriched


def _parse_objective(value: str) -> SolverObjective:
    try:
        return SolverObjective(value)
    except ValueError:
        logger.warning("agent.run unknown_objective value={} defaulting", value)
        return SolverObjective.MAXIMIZE_PROFIT


def _summarize(
    solver_status: SolverStatus,
    reason: str | None,
    policy_doc_id: str | None,
    buffer: float,
) -> str:
    if solver_status == SolverStatus.INFEASIBLE:
        return (
            f"No feasible schedule for the requested day. "
            f"Policy {policy_doc_id or 'fallback'} requires a "
            f"{buffer * 100:.0f}% SoC reserve, which cannot be maintained "
            f"given the forecast inputs. {reason or ''}".strip()
        )
    if solver_status == SolverStatus.ERROR:
        return f"Solver error: {reason or 'unknown failure'}."
    return (
        f"Schedule produced under policy {policy_doc_id or 'fallback'} with a "
        f"{buffer * 100:.0f}% SoC reserve enforced."
    )


class GridOptimizationAgent:
    """End-to-end orchestrator. Public entry point is `run()`."""

    def __init__(self) -> None:
        cfg = get_settings()
        self.llm = ChatOpenAI(
            model=cfg.OPENAI_MODEL,
            temperature=cfg.AGENT_TEMPERATURE,
            api_key=cfg.OPENAI_API_KEY,
        )
        self.tools = [tool_query_policies, tool_forecast_solar, tool_optimize_grid]
        self.executor = _build_agent()

    def run(self, prompt: str, objective: str, date: str) -> OptimizationResponse:
        task_id = str(uuid.uuid4())
        started = time.perf_counter()
        logger.info(
            "agent.run.start task_id={} objective={} date={}", task_id, objective, date,
        )

        objective_enum = _parse_objective(objective)
        agent_input = (
            f"{prompt}\n\n"
            f"Objective: {objective_enum.value}\n"
            f"Target date: {date}\n"
            "Follow the three tool calls in the required order."
        )

        try:
            result = self.executor.invoke({"input": agent_input})
        except Exception as exc:  # noqa: BLE001 — any agent failure becomes a structured response
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("agent.run.executor_failed task_id={}", task_id)
            return OptimizationResponse(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                error=f"Agent executor failed: {exc}",
                audit=AuditTrail(
                    agent_tool_calls=[
                        {
                            "tool": "executor",
                            "input": {"prompt": prompt, "objective": objective, "date": date},
                            "output": None,
                            "duration_ms": elapsed_ms,
                            "error": str(exc),
                        }
                    ]
                ),
            )

        intermediate = result.get("intermediate_steps", []) or []
        audit_calls: list[dict[str, Any]] = []
        for step in intermediate:
            action, observation = step[0], step[1]
            tool_name = getattr(action, "tool", getattr(action, "type", "unknown"))
            tool_input = getattr(action, "tool_input", getattr(action, "input", None))
            audit_calls.append(
                {
                    "tool": tool_name,
                    "input": tool_input,
                    "output": observation,
                    "duration_ms": None,
                }
            )

        policy_doc_id: str | None = None
        policy_text: str | None = None
        constraint_injected: dict[str, Any] | None = None
        solar_forecast_used: list[float] | None = None
        market_prices_used: list[float] | None = None
        solver_status: SolverStatus = SolverStatus.ERROR
        solver_time_ms: float | None = None
        solver_reason: str | None = None
        safety_passed: bool = False
        total_profit_usd: float = 0.0
        carbon_saved_kg: float = 0.0
        raw_schedule: list[ScheduleHour] = []

        for call in audit_calls:
            tool_name = call["tool"]
            output = call["output"]
            if tool_name == "tool_query_policies" and isinstance(output, dict):
                policy_doc_id = output.get("doc_id")
                policy_text = output.get("policy_text")
                buf = output.get("min_battery_buffer")
                if buf is not None:
                    constraint_injected = {"min_battery_buffer": float(buf)}
            elif tool_name == "tool_forecast_solar" and isinstance(output, dict):
                solar_forecast_used = output.get("hourly_forecast_kw")
            elif tool_name == "tool_optimize_grid" and isinstance(output, dict):
                market_prices_used = (
                    output.get("market_prices_used") or market_prices_used
                )
                try:
                    solver_status = SolverStatus(output.get("status", "ERROR"))
                except ValueError:
                    solver_status = SolverStatus.ERROR
                solver_time_ms = output.get("solver_time_ms")
                solver_reason = output.get("reason")
                safety_passed = bool(output.get("safety_constraints_passed", False))
                total_profit_usd = float(output.get("total_profit_usd", 0.0))
                carbon_saved_kg = float(output.get("carbon_saved_kg", 0.0))
                raw_schedule = [
                    ScheduleHour.model_validate(h) for h in (output.get("schedule") or [])
                ]

        if market_prices_used is None:
            market_prices_used = _FALLBACK_PRICE_KWH

        buffer_float = (
            float(constraint_injected["min_battery_buffer"])
            if constraint_injected
            else 0.10
        )

        enriched_schedule = _build_hourly_reasons(
            schedule=raw_schedule,
            prices=market_prices_used,
            buffer=buffer_float,
            policy_doc_id=policy_doc_id,
            solar=solar_forecast_used,
        )

        task_status = (
            TaskStatus.SUCCESS
            if solver_status == SolverStatus.OPTIMAL
            else TaskStatus.FAILURE
        )
        summary = _summarize(solver_status, solver_reason, policy_doc_id, buffer_float)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "agent.run.done task_id={} status={} elapsed_ms={:.2f}",
            task_id, solver_status.value, elapsed_ms,
        )

        return OptimizationResponse(
            task_id=task_id,
            status=task_status,
            summary=summary,
            schedule=enriched_schedule,
            metrics={
                "total_profit_usd": total_profit_usd,
                "carbon_saved_kg": carbon_saved_kg,
                "safety_constraints_passed": safety_passed,
                "agent_elapsed_ms": elapsed_ms,
            },
            audit=AuditTrail(
                policy_doc_retrieved=policy_doc_id,
                policy_raw_text=policy_text,
                constraint_injected=constraint_injected,
                solar_forecast_used=solar_forecast_used,
                market_prices_used=market_prices_used,
                solver_status=solver_status.value,
                solver_time_ms=solver_time_ms,
                agent_tool_calls=audit_calls,
            ),
            error=solver_reason if solver_status != SolverStatus.OPTIMAL else None,
        )
