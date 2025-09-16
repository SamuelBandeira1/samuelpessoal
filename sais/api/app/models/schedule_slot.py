from __future__ import annotations

import enum
import uuid

from sqlalchemy import DateTime, Enum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class SlotStatus(str, enum.Enum):
    """Possible states for a schedule slot."""

    FREE = "FREE"
    HOLD = "HOLD"
    BOOKED = "BOOKED"
    BLOCKED = "BLOCKED"


class ScheduleSlot(Base, TimestampMixin):
    """Bookable schedule slots for providers."""

    __tablename__ = "schedule_slots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    provider_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("providers.id", ondelete="CASCADE"), index=True
    )
    service_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("services.id", ondelete="SET NULL"), nullable=True
    )
    start_time: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    status: Mapped[SlotStatus] = mapped_column(
        Enum(SlotStatus, name="schedule_slot_status"),
        default=SlotStatus.FREE,
        nullable=False,
    )
    hold_expires_at: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)
