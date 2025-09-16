from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from celery.utils.log import get_task_logger

from jobs.app.celery_app import celery_app

logger = get_task_logger(__name__)


def _resolve_when(scheduled_start: str | None, delta: timedelta) -> datetime:
    """Return the timestamp when a reminder should be dispatched."""

    if scheduled_start:
        try:
            start = datetime.fromisoformat(scheduled_start)
            return start - delta
        except ValueError:
            logger.warning("Invalid scheduled_start %s", scheduled_start)
    return datetime.utcnow()


def _emit_reminder(
    *,
    appointment_id: str,
    window: str,
    scheduled_start: str | None,
    delta: timedelta,
) -> dict[str, Any]:
    eta = _resolve_when(scheduled_start, delta)
    logger.info(
        "Dispatching %s reminder for appointment %s at %s",
        window,
        appointment_id,
        eta.isoformat(),
    )
    return {
        "appointment_id": appointment_id,
        "reminder_window": window,
        "dispatch_at": eta.isoformat(),
    }


@celery_app.task(name="jobs.send_reminder_d2")
def send_reminder_d2(
    appointment_id: str, scheduled_start: str | None = None
) -> dict[str, Any]:
    """Send the D-2 reminder (two days before)."""

    return _emit_reminder(
        appointment_id=appointment_id,
        window="D-2",
        scheduled_start=scheduled_start,
        delta=timedelta(days=2),
    )


@celery_app.task(name="jobs.send_reminder_d1")
def send_reminder_d1(
    appointment_id: str, scheduled_start: str | None = None
) -> dict[str, Any]:
    """Send the D-1 reminder (one day before)."""

    return _emit_reminder(
        appointment_id=appointment_id,
        window="D-1",
        scheduled_start=scheduled_start,
        delta=timedelta(days=1),
    )


@celery_app.task(name="jobs.send_reminder_h2")
def send_reminder_h2(
    appointment_id: str, scheduled_start: str | None = None
) -> dict[str, Any]:
    """Send the H-2 reminder (two hours before)."""

    return _emit_reminder(
        appointment_id=appointment_id,
        window="H-2",
        scheduled_start=scheduled_start,
        delta=timedelta(hours=2),
    )


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
