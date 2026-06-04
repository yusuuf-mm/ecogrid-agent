"""
services/workers/celery_app.py

Celery app stub. Branch: feat/backend-api
Read CLAUDE.md and services/workers/CLAUDE.md before implementing.
"""
import os
from celery import Celery

# TODO: configure broker and backend from env vars
# See services/workers/CLAUDE.md for full config
app = Celery("ecogrid_worker")
