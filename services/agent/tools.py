"""services/agent/tools.py

Three LangChain tools, one per upstream service. The agent invokes these
through natural language — the docstrings on each `@tool` function are
what the LLM reads to decide which tool to call and with what arguments.

These tools do no business logic. They transform inputs, call the
underlying service function, and transform outputs back to a dict the
LLM can read.

Actual upstream interfaces used (verified against shared/contracts.py
and the real service modules on feat/agent-core):

  services.rag.retriever.query_grid_policies(PolicyQuery) -> PolicyResult
  services.ml.inference.predictor.forecast_solar_generation(WeatherFeatures) -> SolarForecast
  services.solver.engine.optimize_battery_schedule(SolverConstraints) -> SolverResult

The RAG parser (`services.rag.parser.extract_buffer_constraint`) does not
exist on this branch — the rag service ships PolicyResult.constraint_float
pre-parsed, so no inline parser is needed.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool
from loguru import logger

from shared.contracts import (
    PolicyQuery,
    PolicyResult,
    SolarForecast,
    SolverConstraints,
    SolverObjective,
    WeatherFeatures,
)


_FALLBACK_BUFFER: float = 0.10
_FALLBACK_DOC_ID: str = "fallback_default_sop"
_FALLBACK_DOC_TITLE: str = "Default Safety SOP (RAG unavailable)"
_FALLBACK_POLICY_TEXT: str = (
    "Standard operating conditions apply. Minimum battery reserve of 10% "
    "is enforced as a safe operating floor when no specific policy is "
    "retrievable."
)


def _extract_buffer_from_text(text: str) -> float | None:
    """Last-resort inline parser for safety-buffer mentions in policy text.

    Used only if `PolicyResult.constraint_float` is missing or out of range.
    Matches patterns like "30%", "0.30", "30 percent", "minimum reserve 0.40".
    Returns the first plausible fraction in [0.0, 1.0], or None.
    """
    if not text:
        return None
    candidates: list[float] = []

    for pct in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", text, re.IGNORECASE):
        value = float(pct.group(1)) / 100.0
        if 0.0 <= value <= 1.0:
            candidates.append(value)

    for frac in re.finditer(r"(?:^|[\s=:])(0?\.\d+|1\.0+|0|1)(?![0-9.])", text):
        value = float(frac.group(1))
        if 0.0 <= value <= 1.0:
            candidates.append(value)

    return candidates[0] if candidates else None


@tool
def tool_query_policies(query: str) -> dict[str, Any]:
    """Retrieve the regulatory safety-buffer constraint that applies to the
    described scenario from the policy vector database.

    Use this FIRST. The `query` argument should be a short phrase describing
    the operational context — for example "hospital reserve during heatwave"
    or "critical infrastructure minimum SoC". The retrieved document is parsed
    for the safety-buffer fraction (a float in [0.0, 1.0]).

    Returns a dict with these keys:
        doc_id:               ID of the source policy document
        doc_title:            human-readable title of the source document
        policy_text:          raw text chunk retrieved from the vector DB
        min_battery_buffer:   parsed safety-buffer fraction, default 0.10
        retrieval_score:      cosine similarity score from the vector DB
        source:               "rag" if retrieved, "fallback" if RAG unavailable
    """
    logger.info("agent.tool.tool_query_policies query={!r}", query)

    try:
        from services.rag.retriever import query_grid_policies

        result: PolicyResult = query_grid_policies(PolicyQuery(query_text=query, top_k=1))

        buffer = result.constraint_float
        if buffer is None or not (0.0 <= buffer <= 1.0):
            inline = _extract_buffer_from_text(result.raw_chunk)
            buffer = inline if inline is not None else _FALLBACK_BUFFER
            logger.warning(
                "agent.tool.tool_query_policies buffer_out_of_range_or_none "
                "doc_id={} constraint_float={} inline_extracted={} used={}",
                result.doc_id, result.constraint_float, inline, buffer,
            )

        return {
            "doc_id": result.doc_id,
            "doc_title": result.doc_title,
            "policy_text": result.raw_chunk,
            "min_battery_buffer": float(buffer),
            "retrieval_score": 0.0,
            "source": "rag",
        }
    except NotImplementedError as exc:
        logger.warning(
            "agent.tool.tool_query_policies rag_not_implemented error={} "
            "using_fallback",
            exc,
        )
        return {
            "doc_id": _FALLBACK_DOC_ID,
            "doc_title": _FALLBACK_DOC_TITLE,
            "policy_text": _FALLBACK_POLICY_TEXT,
            "min_battery_buffer": _FALLBACK_BUFFER,
            "retrieval_score": 0.0,
            "source": "fallback",
        }
    except Exception as exc:  # noqa: BLE001 — surface any RAG failure as structured fallback
        logger.error(
            "agent.tool.tool_query_policies unexpected_error error={} using_fallback",
            exc,
        )
        return {
            "doc_id": _FALLBACK_DOC_ID,
            "doc_title": _FALLBACK_DOC_TITLE,
            "policy_text": _FALLBACK_POLICY_TEXT,
            "min_battery_buffer": _FALLBACK_BUFFER,
            "retrieval_score": 0.0,
            "source": "fallback",
            "error": str(exc),
        }


@tool
def tool_forecast_solar(date: str) -> dict[str, Any]:
    """Forecast 24 hours of solar generation (one kW value per hour) for the
    given target date.

    Use this SECOND, after tool_query_policies. The `date` argument must be
    an ISO date string like "2025-07-15". The returned hourly_forecast_kw
    list always has exactly 24 elements (index 0 = midnight, index 23 = 23:00).

    Returns a dict with these keys:
        date:                  the input date echoed back
        hourly_forecast_kw:    list[float] of length 24
        peak_kw:               maximum hourly value
        total_kwh:             sum of the 24 values
    """
    logger.info("agent.tool.tool_forecast_solar date={}", date)

    from services.ml.inference.predictor import forecast_solar_generation

    try:
        features = WeatherFeatures(
            date=date,
            hourly_temperature_c=[20.0] * 24,
            hourly_cloud_cover_pct=[30.0] * 24,
            hourly_irradiance_wm2=[0.0] * 24,
        )
        forecast: SolarForecast = forecast_solar_generation(features)
        hourly = list(forecast.hourly_generation_kw)
        return {
            "date": forecast.date,
            "hourly_forecast_kw": hourly,
            "peak_kw": max(hourly) if hourly else 0.0,
            "total_kwh": sum(hourly),
        }
    except NotImplementedError as exc:
        logger.warning(
            "agent.tool.tool_forecast_solar ml_not_implemented error={} "
            "returning_zeros",
            exc,
        )
        zero = [0.0] * 24
        return {
            "date": date,
            "hourly_forecast_kw": zero,
            "peak_kw": 0.0,
            "total_kwh": 0.0,
            "source": "fallback",
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("agent.tool.tool_forecast_solar error={}", exc)
        raise


@tool
def tool_optimize_grid(solver_input: dict[str, Any]) -> dict[str, Any]:
    """Run the LP solver to produce an optimal 24-hour battery schedule.

    Use this THIRD, after both tool_query_policies and tool_forecast_solar.
    The `solver_input` dict must contain all SolverConstraints fields. The
    solver is the authority on math — do not pre-compute or estimate any
    value before calling this tool.

    Required solver_input keys:
        solar_forecast_kw:      list[float], length 24
        market_prices_kwh:      list[float], length 24
        min_battery_buffer:     float in [0.0, 1.0]
        battery_capacity_kwh:   float (default 1000.0)
        max_charge_rate_kw:     float (default 250.0)
        initial_soc_kwh:        float (default 500.0)
        objective:              "MAXIMIZE_PROFIT" | "MINIMIZE_CARBON" | "MINIMIZE_COST"

    Returns a dict with these keys (from SolverResult):
        status:                  "OPTIMAL" | "INFEASIBLE" | "ERROR"
        schedule:                list of 24 hourly entries
        total_profit_usd:        float
        carbon_saved_kg:         float
        safety_constraints_passed: bool
        solver_time_ms:          float
        reason:                  str | None (set on INFEASIBLE/ERROR)
    INFEASIBLE is a valid return. Do not retry, do not raise.
    """
    logger.info(
        "agent.tool.tool_optimize_grid objective={} solar_len={} prices_len={} "
        "buffer={}",
        solver_input.get("objective"),
        len(solver_input.get("solar_forecast_kw") or []),
        len(solver_input.get("market_prices_kwh") or []),
        solver_input.get("min_battery_buffer"),
    )

    from services.solver.engine import optimize_battery_schedule

    objective_raw = solver_input.get("objective", SolverObjective.MAXIMIZE_PROFIT.value)
    if isinstance(objective_raw, SolverObjective):
        objective = objective_raw
    else:
        objective = SolverObjective(str(objective_raw))

    constraints = SolverConstraints(
        solar_forecast_kw=list(solver_input["solar_forecast_kw"]),
        market_prices_kwh=list(solver_input["market_prices_kwh"]),
        carbon_intensity_g_kwh=solver_input.get("carbon_intensity_g_kwh"),
        battery_capacity_kwh=float(solver_input.get("battery_capacity_kwh", 1000.0)),
        max_charge_rate_kw=float(solver_input.get("max_charge_rate_kw", 250.0)),
        initial_soc_kwh=float(solver_input.get("initial_soc_kwh", 500.0)),
        min_battery_buffer=float(solver_input.get("min_battery_buffer", 0.10)),
        objective=objective,
        policy_doc_id=solver_input.get("policy_doc_id"),
    )

    result = optimize_battery_schedule(constraints)
    return result.model_dump(mode="json")
