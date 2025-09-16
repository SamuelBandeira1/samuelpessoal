from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Provider(Base, TimestampMixin):
    """Care provider (doctor, nurse, etc.)."""

    __tablename__ = "providers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    specialty: Mapped[str | None] = mapped_column(String(255), nullable=True)
