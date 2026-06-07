"""
scripts/seed_market_prices.py

Generate 90 days of synthetic ERCOT-style Locational Marginal Prices and
insert them into the `market_prices` table in PostgreSQL.

Schema (matches CONTEXT.md and shared/contracts.py):
  timestamp     TIMESTAMPTZ
  price_kwh     FLOAT
  carbon_g_kwh  FLOAT
  node_id       VARCHAR

Behaviour:
  * Creates the table if it does not exist.
  * Writes rows in idempotent batches, skipping any (timestamp, node_id)
    pair that already exists (per-row ON CONFLICT DO NOTHING).
  * Default node: ERCOT_HOUSTON (Houston load zone).
  * Default history: last 90 days, hourly granularity.
  * Price pattern: low overnight ($0.03-0.06), afternoon peak
    ($0.15-0.30), occasional price spikes ($0.50+).
  * Carbon intensity pattern: roughly inverse to solar availability —
    higher overnight, lower midday.

Environment:
  POSTGRES_HOST      (default: localhost)
  POSTGRES_PORT      (default: 5432)
  POSTGRES_DB        (default: ecogrid)
  POSTGRES_USER      (default: ecogrid)
  POSTGRES_PASSWORD  (default: ecogrid)
  POSTGRES_URL       (optional, full DSN override)
  SEED_DAYS          (default: 90)
  SEED_NODE_ID       (default: ERCOT_HOUSTON)

Run locally:
  python scripts/seed_market_prices.py
Run inside docker:
  docker compose --profile seed run --rm ingest
"""
from __future__ import annotations

import asyncio
import math
import os
import random
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import asyncpg


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_prices (
    timestamp     TIMESTAMPTZ NOT NULL,
    price_kwh     DOUBLE PRECISION NOT NULL,
    carbon_g_kwh  DOUBLE PRECISION NOT NULL,
    node_id       VARCHAR(64) NOT NULL,
    PRIMARY KEY (timestamp, node_id)
);
CREATE INDEX IF NOT EXISTS idx_market_prices_node_ts
    ON market_prices (node_id, timestamp DESC);
"""


def _env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _dsn() -> str:
    if url := os.environ.get("POSTGRES_URL"):
        return url
    host = _env("POSTGRES_HOST", "localhost")
    port = _env("POSTGRES_PORT", "5432")
    db = _env("POSTGRES_DB", "ecogrid")
    user = _env("POSTGRES_USER", "ecogrid")
    pwd = _env("POSTGRES_PASSWORD", "ecogrid")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def _price_for_hour(hour: int, jitter: random.Random) -> float:
    """Synthesize a $/kWh LMP for a given hour of day.

    Curve (per hour 0..23):
      00-05  : 0.03 - 0.06  (overnight trough)
      06-10  : 0.05 - 0.10  (morning ramp)
      11-15  : 0.15 - 0.30  (afternoon peak; can spike >0.50)
      16-19  : 0.20 - 0.35  (evening peak)
      20-23  : 0.08 - 0.15  (wind-down)
    """
    if hour <= 5:
        base = 0.03 + 0.03 * (hour / 5.0)
    elif hour <= 10:
        base = 0.05 + 0.05 * ((hour - 6) / 4.0)
    elif hour <= 15:
        progress = (hour - 11) / 4.0
        base = 0.15 + 0.15 * progress
        if jitter.random() < 0.06:
            base += jitter.uniform(0.20, 0.45)
    elif hour <= 19:
        base = 0.20 + 0.15 * ((19 - hour) / 3.0)
        if jitter.random() < 0.04:
            base += jitter.uniform(0.15, 0.35)
    else:
        base = 0.15 - 0.07 * ((hour - 20) / 3.0)
    noise = jitter.uniform(-0.01, 0.01)
    return max(0.01, round(base + noise, 4))


def _carbon_for_hour(hour: int, jitter: random.Random) -> float:
    """gCO2/kWh — higher when grid relies on peakers (evenings), lower at midday
    when solar is contributing."""
    base = 380.0 + 120.0 * math.sin(((hour - 18) / 24.0) * 2.0 * math.pi)
    if 10 <= hour <= 15:
        base -= jitter.uniform(40.0, 90.0)
    return round(max(150.0, base + jitter.uniform(-15.0, 15.0)), 1)


def _generate_rows(
    days: int,
    node_id: str,
    seed: int = 42,
) -> AsyncIterator[tuple[datetime, float, float, str]]:
    jitter = random.Random(seed)
    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)
    ts = start
    total_hours = days * 24
    for _ in range(total_hours):
        price = _price_for_hour(ts.hour, jitter)
        carbon = _carbon_for_hour(ts.hour, jitter)
        yield (ts, price, carbon, node_id)
        ts += timedelta(hours=1)


async def _seed_async() -> int:
    days = int(_env("SEED_DAYS", "90"))
    node_id = _env("SEED_NODE_ID", "ERCOT_HOUSTON")
    dsn = _dsn()

    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(CREATE_TABLE_SQL)
        rows = list(_generate_rows(days, node_id))
        inserted = 0
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            await conn.executemany(
            """
            INSERT INTO market_prices (timestamp, price_kwh, carbon_g_kwh, node_id)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (timestamp, node_id) DO NOTHING
            """,
            batch,
        )
        inserted += len(batch)
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM market_prices WHERE node_id = $1", node_id
        )
        print(
            f"[seed_market_prices] node={node_id} requested={len(rows)} "
            f"newly_inserted={inserted} total_rows_for_node={total}"
        )
        return inserted
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(_seed_async())


if __name__ == "__main__":
    main()
