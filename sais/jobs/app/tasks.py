from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from celery.utils.log import get_task_logger

from jobs.app.celery_app import celery_app

logger = get_task_logger(__name__)


@celery_app.task(name="jobs.send_reminder")
def send_reminder(appointment_id: str, eta: datetime | None = None) -> dict[str, Any]:
    """Send a reminder notification for an appointment."""

    eta = eta or datetime.utcnow()
    logger.info("Sending reminder for appointment %s at %s", appointment_id, eta.isoformat())
    return {"appointment_id": appointment_id, "scheduled_for": eta.isoformat()}


@celery_app.task(name="jobs.dispatch_message")
def dispatch_message(recipient: str, channel: str = "whatsapp") -> dict[str, Any]:
    """Dispatch a message to the configured channel."""

    logger.info("Dispatching %s message to %s", channel, recipient)
    return {"recipient": recipient, "channel": channel, "status": "queued"}


@celery_app.task(name="jobs.flag_no_show")
def flag_no_show(appointment_id: str) -> dict[str, Any]:
    """Mark an appointment as a potential no-show for follow-up."""

    follow_up = datetime.utcnow() + timedelta(hours=1)
    logger.warning(
        "Flagging appointment %s as potential no-show. Follow-up at %s",
        appointment_id,
        follow_up.isoformat(),
    )
    return {"appointment_id": appointment_id, "follow_up": follow_up.isoformat()}
