from __future__ import annotations

import uuid

from sqlalchemy import Date, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Patient(Base, TimestampMixin):
    """Patient entity scoped by tenant."""

    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    date_of_birth: Mapped[Date | None] = mapped_column(Date, nullable=True)
