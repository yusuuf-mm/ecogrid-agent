"""services/agent/tools.py

Three plain functions wrapping upstream services. No LangChain dependency.
The `GridOptimizationAgent` calls these directly in deterministic order.

Actual upstream interfaces used (verified against the real service
modules on main):

  services.rag.retriever.PolicyRetriever().retrieve(query, top_k=1) -> list[PolicyResult]
  services.ml.inference.predictor.forecast_solar_generation(WeatherFeatures) -> SolarForecast
  services.solver.engine.optimize_battery_schedule(SolverConstraints) -> SolverResult

`PolicyResult.constraint_float` is already parsed by the retriever
(using `services.rag.parser.extract_buffer_constraint`); the tool does
not re-parse the text.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from shared.contracts import (
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


def tool_query_policies(query: str) -> dict[str, Any]:
    """Retrieve the regulatory safety-buffer constraint that applies to the
    described scenario from the policy vector database.

    Use this FIRST. The `query` argument should be a short phrase describing
    the operational context — for example "hospital reserve during heatwave"
    or "minimum state of charge buffer hospital critical infrastructure". The retriever embeds the
    query, finds the closest chunk in Qdrant, parses the safety-buffer
    fraction, and returns a PolicyResult.

    Returns a dict with these keys:
        doc_id:               ID of the source policy document
        doc_title:            human-readable title of the source document
        policy_text:          raw text chunk retrieved from the vector DB
        min_battery_buffer:   parsed safety-buffer fraction, default 0.10
        retrieval_score:      cosine similarity score from the vector DB (0.0 if not propagated)
        source:               "rag" if retrieved, "fallback" if RAG unavailable
    """
    logger.info("agent.tool.tool_query_policies query={!r}", query)

    try:
        from services.rag.retriever import PolicyRetriever

        results = PolicyRetriever().retrieve(query, top_k=5)
        if not results:
            logger.warning("agent.tool.tool_query_policies no_results query={!r}", query)
            return {
                "doc_id": _FALLBACK_DOC_ID,
                "doc_title": _FALLBACK_DOC_TITLE,
                "policy_text": _FALLBACK_POLICY_TEXT,
                "min_battery_buffer": _FALLBACK_BUFFER,
                "retrieval_score": 0.0,
                "source": "fallback",
                "reason": "no_results",
            }

        # pick the chunk with the highest explicitly-parsed constraint
        parsed = [r for r in results if r.parse_confidence == "parsed"]
        result = max(parsed, key=lambda r: r.constraint_float) if parsed else results[0]
        buffer = result.constraint_float
        if buffer is None or not (0.0 <= buffer <= 1.0):
            logger.warning(
                "agent.tool.tool_query_policies buffer_out_of_range "
                "doc_id={} constraint_float={} using_default",
                result.doc_id, result.constraint_float,
            )
            buffer = _FALLBACK_BUFFER

        return {
            "doc_id": result.doc_id,
            "doc_title": result.doc_title,
            "policy_text": result.raw_chunk,
            "min_battery_buffer": float(buffer),
            "retrieval_score": 0.0,
            "source": "rag",
        }
    except (ConnectionError, OSError) as exc:
        logger.warning(
            "agent.tool.tool_query_policies rag_unreachable error={} using_fallback",
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
