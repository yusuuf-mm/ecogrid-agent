"""
services/workers/celery_app.py

Celery application for EcoGrid-Agent.
Broker and result backend both run on Redis; URLs come from env vars so
docker-compose, local dev, and CI can each point at their own instance.

Branch: feat/backend-api
"""
from __future__ import annotations

import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

app = Celery(
    "ecogrid_worker",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["services.workers.tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
)
