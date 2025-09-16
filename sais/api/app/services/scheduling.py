"""Scheduling utilities for slot discovery and booking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import (
    Appointment,
    AppointmentOrigin,
    AppointmentStatus,
    Patient,
    Provider,
    ScheduleSlot,
    Service,
    SlotStatus,
    Tenant,
)

HOLD_DURATION = timedelta(seconds=30)
SLOT_SEARCH_LIMIT = 6
DEFAULT_BUFFER_MINUTES = 10


@dataclass
class SlotSummary:
    """Serialized view of a slot held for presentation."""

    slot_id: UUID
    provider_id: UUID
    service_id: UUID | None
    start_utc: datetime
    end_utc: datetime
    hold_expires_at: datetime | None
    status: SlotStatus

    def as_dict(self, tz: ZoneInfo) -> dict[str, str | UUID | None]:
        """Convert the slot summary into JSON-friendly values."""

        local_start = self.start_utc.astimezone(tz)
        local_end = self.end_utc.astimezone(tz)
        hold_value = (
            self.hold_expires_at.astimezone(tz).isoformat()
            if self.hold_expires_at
            else None
        )
        return {
            "slot_id": str(self.slot_id),
            "provider_id": str(self.provider_id),
            "service_id": str(self.service_id) if self.service_id else None,
            "start_ts": self.start_utc.isoformat(),
            "end_ts": self.end_utc.isoformat(),
            "start_local": local_start.isoformat(),
            "end_local": local_end.isoformat(),
            "hold_expires_at": hold_value,
            "status": self.status.value,
        }


def tenant_timezone(tenant: Tenant | None) -> ZoneInfo:
    """Return the tenant timezone, falling back to application default."""

    tz_name = (tenant.timezone if tenant else None) or settings.timezone
    try:
        return ZoneInfo(tz_name)
    except Exception:  # pragma: no cover - defensive fallback
        return ZoneInfo("UTC")


def ensure_utc(value: datetime) -> datetime:
    """Coerce a datetime into UTC timezone-aware form."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _day_bounds(target_date: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start = datetime.combine(target_date, time.min, tzinfo=tz)
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def release_expired_holds(db: Session, provider_id: UUID) -> int:
    """Release expired holds for the given provider."""

    now = datetime.now(timezone.utc)
    stmt = (
        update(ScheduleSlot)
        .where(
            ScheduleSlot.provider_id == provider_id,
            ScheduleSlot.status == SlotStatus.HOLD,
            ScheduleSlot.hold_expires_at.is_not(None),
            ScheduleSlot.hold_expires_at <= now,
        )
        .values(status=SlotStatus.FREE, hold_expires_at=None)
    )
    result = db.execute(stmt)
    return int(result.rowcount or 0)


def _blocking_slots(slots: Sequence[ScheduleSlot], now: datetime) -> list[ScheduleSlot]:
    blocked: list[ScheduleSlot] = []
    for slot in slots:
        if slot.status == SlotStatus.BLOCKED or slot.status == SlotStatus.BOOKED:
            blocked.append(slot)
        elif slot.status == SlotStatus.HOLD and slot.hold_expires_at:
            expires = ensure_utc(slot.hold_expires_at)
            if expires <= now:
                slot.status = SlotStatus.FREE
                slot.hold_expires_at = None
            else:
                blocked.append(slot)
    return blocked


def offer_slots(
    db: Session,
    *,
    tenant: Tenant,
    provider: Provider,
    service: Service,
    target_date: date,
    limit: int = SLOT_SEARCH_LIMIT,
    buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
) -> tuple[list[SlotSummary], ZoneInfo]:
    """Return up to ``limit`` slots held for the caller."""

    tz = tenant_timezone(tenant)
    start_utc, end_utc = _day_bounds(target_date, tz)
    now = datetime.now(timezone.utc)

    release_expired_holds(db, provider.id)

    stmt = (
        select(ScheduleSlot)
        .where(
            ScheduleSlot.provider_id == provider.id,
            ScheduleSlot.start_time >= start_utc,
            ScheduleSlot.start_time < end_utc,
        )
        .order_by(ScheduleSlot.start_time)
    )
    slots = db.execute(stmt).scalars().all()

    blocking = _blocking_slots(slots, now)
    held: list[SlotSummary] = []
    hold_until = now + HOLD_DURATION

    for slot in slots:
        if slot.status != SlotStatus.FREE:
            continue

        start = ensure_utc(slot.start_time)
        end = ensure_utc(slot.end_time)
        if (end - start).total_seconds() / 60 < service.duration_min:
            continue

        conflict = False
        for busy in blocking:
            if busy.id == slot.id:
                continue
            busy_start = ensure_utc(busy.start_time)
            busy_end = ensure_utc(busy.end_time)

            if busy_end <= start:
                gap = (start - busy_end).total_seconds() / 60
                if gap < buffer_minutes:
                    conflict = True
                    break
            elif end <= busy_start:
                gap = (busy_start - end).total_seconds() / 60
                if gap < buffer_minutes:
                    conflict = True
                    break
            else:
                conflict = True
                break

        if conflict:
            continue

        slot.status = SlotStatus.HOLD
        slot.hold_expires_at = hold_until
        blocking.append(slot)
        held.append(
            SlotSummary(
                slot_id=slot.id,
                provider_id=slot.provider_id,
                service_id=slot.service_id,
                start_utc=start,
                end_utc=end,
                hold_expires_at=hold_until,
                status=SlotStatus.HOLD,
            )
        )

        if len(held) >= limit:
            break

    db.flush()
    return held, tz


def get_slot_by_start(
    db: Session,
    *,
    provider_id: UUID,
    start_ts: datetime,
) -> ScheduleSlot | None:
    """Return the slot that starts at the provided timestamp, locking it."""

    statement = (
        select(ScheduleSlot)
        .where(
            ScheduleSlot.provider_id == provider_id,
            ScheduleSlot.start_time == ensure_utc(start_ts),
        )
        .with_for_update()
    )
    return db.execute(statement).scalars().first()


def book_slot(
    db: Session,
    *,
    tenant: Tenant,
    provider: Provider,
    patient: Patient,
    service: Service,
    slot: ScheduleSlot,
    origin: AppointmentOrigin,
    notes: str | None = None,
) -> Appointment:
    """Confirm a slot into an appointment."""

    now = datetime.now(timezone.utc)
    start = ensure_utc(slot.start_time)
    end = ensure_utc(slot.end_time)
    required_end = start + timedelta(minutes=service.duration_min)
    if required_end > end:
        raise ValueError("Slot duration is shorter than the service duration")

    if slot.status == SlotStatus.HOLD:
        if not slot.hold_expires_at or ensure_utc(slot.hold_expires_at) <= now:
            slot.status = SlotStatus.FREE
            slot.hold_expires_at = None
            raise ValueError("Slot hold expired")
    elif slot.status != SlotStatus.FREE:
        raise ValueError("Slot unavailable")

    slot.status = SlotStatus.BOOKED
    slot.hold_expires_at = None
    slot.service_id = service.id

    appointment = Appointment(
        tenant_id=tenant.id,
        patient_id=patient.id,
        provider_id=provider.id,
        service_id=service.id,
        schedule_slot_id=slot.id,
        status=AppointmentStatus.CONFIRMED,
        scheduled_start=start,
        scheduled_end=required_end,
        notes=notes,
        origin=origin,
    )
    db.add(appointment)
    db.flush()
    return appointment


def serialize_appointment(
    appointment: Appointment,
    *,
    tz: ZoneInfo,
) -> dict[str, str]:
    """Return a JSON-friendly representation of an appointment."""

    start_local = ensure_utc(appointment.scheduled_start).astimezone(tz)
    end_local = ensure_utc(appointment.scheduled_end).astimezone(tz)
    return {
        "id": str(appointment.id),
        "status": appointment.status.value,
        "origin": appointment.origin.value,
        "scheduled_start": ensure_utc(appointment.scheduled_start).isoformat()
        if appointment.scheduled_start
        else None,
        "scheduled_end": ensure_utc(appointment.scheduled_end).isoformat()
        if appointment.scheduled_end
        else None,
        "scheduled_start_local": start_local.isoformat(),
        "scheduled_end_local": end_local.isoformat(),
        "provider_id": str(appointment.provider_id),
        "patient_id": str(appointment.patient_id),
        "service_id": str(appointment.service_id) if appointment.service_id else None,
    }
