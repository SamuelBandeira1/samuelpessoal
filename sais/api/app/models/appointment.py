from __future__ import annotations

import enum
import uuid

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class AppointmentStatus(str, enum.Enum):
    """Possible statuses for an appointment lifecycle."""

    CONFIRMED = "CONFIRMED"
    RESCHEDULED = "RESCHEDULED"
    CANCELLED = "CANCELLED"
    NO_SHOW = "NO_SHOW"
    COMPLETED = "COMPLETED"


class AppointmentOrigin(str, enum.Enum):
    """Origin of an appointment booking."""

    WHATSAPP = "WHATSAPP"
    WEB = "WEB"
    STAFF = "STAFF"


class Appointment(Base, TimestampMixin):
    """Appointment between a patient and provider."""

    __tablename__ = "appointments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )
    provider_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("providers.id", ondelete="CASCADE"), index=True
    )
    service_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("services.id", ondelete="SET NULL"), nullable=True
    )
    schedule_slot_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("schedule_slots.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[AppointmentStatus] = mapped_column(
        Enum(AppointmentStatus, name="appointment_status"),
        default=AppointmentStatus.CONFIRMED,
        nullable=False,
    )
    scheduled_start: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)
    scheduled_end: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    origin: Mapped[AppointmentOrigin] = mapped_column(
        Enum(AppointmentOrigin, name="appointment_origin"),
        default=AppointmentOrigin.WHATSAPP,
        nullable=False,
    )
