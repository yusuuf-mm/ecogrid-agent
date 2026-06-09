"""services/agent/agent.py

Gemini-native orchestrator. Takes a natural language grid optimization
request and drives the three tools in deterministic order:

    1. tool_query_policies → safety buffer from policy doc
    2. tool_forecast_solar → 24-hour solar generation
    3. tool_optimize_grid  → LP solver call

Returns a fully-populated `OptimizationResponse` (from shared.contracts)
including the audit trail.

Uses `google-genai` SDK natively with `response_schema` for structured
output enforcement — no string parsing, no LangChain, no risk of NoneType
splitting errors.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from google import genai
from google.genai import types as genai_types
from loguru import logger
from pydantic import BaseModel, Field

from services.agent.config import get_settings
from services.agent.prompts import PARSE_INTENT_PROMPT, SYNTHESIZE_SUMMARY_PROMPT
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


# ---------------------------------------------------------------------------
# response_schema models — guarantee structured I/O with the Gemini API.
# No manual JSON parsing, no string-split risk, no NoneType propagation.
# ---------------------------------------------------------------------------

class _AgentIntent(BaseModel):
    """Structured extraction of the user's natural-language request.
    Gemini populates this via response_schema — guaranteed valid on receipt.
    """
    policy_query: str = Field(
        description="Short operational-context phrase for policy retrieval, "
        "e.g. 'hospital reserve during heatwave' or 'critical infrastructure default'"
    )
    target_date: str = Field(
        description="ISO date string for the target day, e.g. '2025-07-15'"
    )


class _AgentSummary(BaseModel):
    """Structured explanation summary produced by Gemini after all tools
    have executed. Guaranteed to be a valid non-empty string.
    """
    summary: str = Field(
        description="One-sentence plain-language summary a grid operator can read aloud"
    )
    tool_call_count: int = Field(
        description="Number of tool invocations performed",
        ge=1, le=10,
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
    """End-to-end orchestrator. Public entry point is `run()`.

    Architecture:
      Phase 1 — Gemini parses user intent via response_schema (_AgentIntent).
      Phase 2 — Three tools execute deterministically in fixed order.
      Phase 3 — Gemini synthesises the final summary via response_schema (_AgentSummary).
      Final  — Python builds OptimizationResponse from structured data.

    No LangChain. No string-parsed JSON. No NoneType split errors.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self.client = genai.Client(api_key=cfg.GEMINI_API_KEY)
        self.model = cfg.GEMINI_MODEL
        self.temperature = cfg.AGENT_TEMPERATURE
        self.max_iterations = cfg.AGENT_MAX_ITERATIONS
        self.tools = [tool_query_policies, tool_forecast_solar, tool_optimize_grid]

    # ------------------------------------------------------------------
    # Internal: Gemini call with response_schema enforcement
    # ------------------------------------------------------------------

    def _generate_structured(
        self,
        schema: type[BaseModel],
        system_instruction: str,
        user_content: str,
    ) -> BaseModel:
        """Call Gemini with a Pydantic response_schema and return the
        validated instance. Raises on API error; caller handles."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=self.temperature,
            ),
        )
        if response.parsed is None:
            raise RuntimeError(
                f"Gemini returned None for response_schema={schema.__name__}: "
                f"{response.text!r}"
            )
        return response.parsed

    # ------------------------------------------------------------------
    # Phase 1: Parse user intent
    # ------------------------------------------------------------------

    def _parse_intent(self, prompt: str, objective: str, date: str) -> _AgentIntent:
        """Extract structured policy_query and target_date from the user's
        natural language prompt using Gemini's response_schema."""
        content = (
            f"User request: {prompt}\n"
            f"Objective: {objective}\n"
            f"Provided date: {date}\n\n"
            "Extract the policy query context and the target ISO date."
        )
        intent = self._generate_structured(
            schema=_AgentIntent,
            system_instruction=PARSE_INTENT_PROMPT,
            user_content=content,
        )
        logger.debug("agent.run.intent policy_query={} target_date={}", intent.policy_query, intent.target_date)
        return intent

    # ------------------------------------------------------------------
    # Phase 3: Synthesize final summary
    # ------------------------------------------------------------------

    def _synthesize_summary(
        self,
        solver_status: SolverStatus,
        policy_doc_id: str | None,
        buffer: float,
        solver_reason: str | None,
        num_tool_calls: int,
    ) -> str:
        """Use Gemini to produce a human-readable summary via response_schema."""
        status_line = f"solver_status={solver_status.value}"
        policy_line = f"policy={policy_doc_id or 'fallback'}, buffer={buffer * 100:.0f}%"
        reason_line = f"reason={solver_reason}" if solver_reason else "no_errors"

        content = (
            f"{status_line}\n{policy_line}\n{reason_line}\n"
            f"tool_calls={num_tool_calls}\n\n"
            "Produce a one-sentence plain-language summary a grid operator can read aloud."
        )
        result = self._generate_structured(
            schema=_AgentSummary,
            system_instruction=SYNTHESIZE_SUMMARY_PROMPT,
            user_content=content,
        )
        return result.summary

    # ------------------------------------------------------------------
    # Phase 2: Build solver input dict from tool outputs
    # ------------------------------------------------------------------

    @staticmethod
    def _build_solver_input(
        policy_result: dict[str, Any],
        solar_result: dict[str, Any],
        objective: SolverObjective,
    ) -> dict[str, Any]:
        """Assemble the dict passed to tool_optimize_grid from the outputs
        of the first two tools."""
        return {
            "solar_forecast_kw": solar_result.get("hourly_forecast_kw", [0.0] * 24),
            "market_prices_kwh": _FALLBACK_PRICE_KWH,
            "min_battery_buffer": float(policy_result.get("min_battery_buffer", 0.10)),
            "battery_capacity_kwh": _DEFAULT_BATTERY_CAPACITY_KWH,
            "max_charge_rate_kw": _DEFAULT_MAX_CHARGE_RATE_KW,
            "initial_soc_kwh": _DEFAULT_INITIAL_SOC_KWH,
            "objective": objective.value,
            "policy_doc_id": policy_result.get("doc_id"),
            "carbon_intensity_g_kwh": None,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, prompt: str, objective: str, date: str) -> OptimizationResponse:
        task_id = str(uuid.uuid4())
        started = time.perf_counter()
        logger.info(
            "agent.run.start task_id={} objective={} date={}", task_id, objective, date,
        )

        objective_enum = _parse_objective(objective)
        audit_calls: list[dict[str, Any]] = []

        # ---- Phase 1: Parse user intent via response_schema ----------------
        try:
            intent = self._parse_intent(prompt, objective, date)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("agent.run.intent_parse_failed task_id={}", task_id)
            return OptimizationResponse(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                error=f"Intent parsing failed: {exc}",
                audit=AuditTrail(
                    agent_tool_calls=[{
                        "tool": "gemini_intent_parser",
                        "input": {"prompt": prompt, "objective": objective, "date": date},
                        "output": None,
                        "duration_ms": elapsed_ms,
                        "error": str(exc),
                    }],
                ),
            )

        # ---- Phase 2: Execute three tools deterministically ----------------

        # Tool 1: Policy retrieval
        t0 = time.perf_counter()
        try:
            policy_result = tool_query_policies(query=intent.policy_query)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("agent.run.policy_tool_failed task_id={}", task_id)
            return OptimizationResponse(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                error=f"Policy tool failed: {exc}",
                audit=AuditTrail(
                    agent_tool_calls=[{
                        "tool": "tool_query_policies",
                        "input": {"query": intent.policy_query},
                        "output": None,
                        "duration_ms": (time.perf_counter() - t0) * 1000.0,
                        "error": str(exc),
                    }],
                ),
            )
        audit_calls.append({
            "tool": "tool_query_policies",
            "input": {"query": intent.policy_query},
            "output": policy_result,
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
        })

        # Tool 2: Solar forecast
        t1 = time.perf_counter()
        try:
            solar_result = tool_forecast_solar(date=intent.target_date)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("agent.run.solar_tool_failed task_id={}", task_id)
            return OptimizationResponse(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                error=f"Solar forecast tool failed: {exc}",
                audit=AuditTrail(
                    agent_tool_calls=audit_calls + [{
                        "tool": "tool_forecast_solar",
                        "input": {"date": intent.target_date},
                        "output": None,
                        "duration_ms": (time.perf_counter() - t1) * 1000.0,
                        "error": str(exc),
                    }],
                ),
            )
        audit_calls.append({
            "tool": "tool_forecast_solar",
            "input": {"date": intent.target_date},
            "output": solar_result,
            "duration_ms": (time.perf_counter() - t1) * 1000.0,
        })

        # Tool 3: LP solver
        solver_input = self._build_solver_input(policy_result, solar_result, objective_enum)
        t2 = time.perf_counter()
        try:
            solver_result = tool_optimize_grid(solver_input=solver_input)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("agent.run.solver_tool_failed task_id={}", task_id)
            return OptimizationResponse(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                error=f"Solver tool failed: {exc}",
                audit=AuditTrail(
                    agent_tool_calls=audit_calls + [{
                        "tool": "tool_optimize_grid",
                        "input": solver_input,
                        "output": None,
                        "duration_ms": (time.perf_counter() - t2) * 1000.0,
                        "error": str(exc),
                    }],
                ),
            )
        audit_calls.append({
            "tool": "tool_optimize_grid",
            "input": solver_input,
            "output": solver_result,
            "duration_ms": (time.perf_counter() - t2) * 1000.0,
        })

        # ---- Extract fields from structured solver output ------------------

        policy_doc_id: str | None = policy_result.get("doc_id")
        policy_text: str | None = policy_result.get("policy_text")
        buffer_float: float = float(policy_result.get("min_battery_buffer", 0.10))
        constraint_injected: dict[str, Any] | None = {"min_battery_buffer": buffer_float}

        solar_forecast_used: list[float] | None = solar_result.get("hourly_forecast_kw")
        market_prices_used: list[float] | None = solver_result.get("market_prices_used", _FALLBACK_PRICE_KWH)

        raw_status: str = solver_result.get("status", "ERROR")
        try:
            solver_status = SolverStatus(raw_status)
        except ValueError:
            solver_status = SolverStatus.ERROR
        solver_time_ms: float | None = solver_result.get("solver_time_ms")
        solver_reason: str | None = solver_result.get("reason")
        safety_passed: bool = bool(solver_result.get("safety_constraints_passed", False))
        total_profit_usd: float = float(solver_result.get("total_profit_usd", 0.0))
        carbon_saved_kg: float = float(solver_result.get("carbon_saved_kg", 0.0))
        raw_schedule: list[ScheduleHour] = [
            ScheduleHour.model_validate(h) for h in (solver_result.get("schedule") or [])
        ]

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

        # ---- Phase 3: Gemini synthesises the summary via response_schema ----
        try:
            summary = self._synthesize_summary(
                solver_status=solver_status,
                policy_doc_id=policy_doc_id,
                buffer=buffer_float,
                solver_reason=solver_reason,
                num_tool_calls=len(audit_calls),
            )
        except Exception:
            logger.warning("agent.run.summary_synthesis_failed task_id={}, using fallback", task_id)
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
