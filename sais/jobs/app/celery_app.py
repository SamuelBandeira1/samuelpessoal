from __future__ import annotations

from celery import Celery

from jobs.app.config import settings

celery_app = Celery(
    "sais",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["jobs.app.tasks"],
)

celery_app.conf.timezone = settings.timezone
celery_app.conf.broker_connection_retry_on_startup = True
