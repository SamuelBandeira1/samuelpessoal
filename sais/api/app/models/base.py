from datetime import datetime
from typing import Any

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base declarative class with default metadata convention."""

    metadata = MetaData(naming_convention=convention)


class TimestampMixin:
    """Mixin providing created and updated timestamps."""

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TenantScopedMixin:
    """Mixin ensuring multi-tenant awareness for models."""

    tenant_id: Mapped[Any]  # actual column declared per model to allow typing flexibility
