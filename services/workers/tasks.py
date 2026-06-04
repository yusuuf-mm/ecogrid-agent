"""
services/workers/tasks.py

Celery task stub. Branch: feat/backend-api
"""
from services.workers.celery_app import app


@app.task(bind=True, max_retries=1, default_retry_delay=30)
def run_optimization_pipeline(self, request_dict: dict) -> dict:
    """TODO: deserialize, run agent, serialize result."""
    raise NotImplementedError("Implement in feat/backend-api branch")
