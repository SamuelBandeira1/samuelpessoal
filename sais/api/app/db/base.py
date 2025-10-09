"""Import SQLAlchemy models for Alembic's autogenerate feature."""

from app.models.base import Base
from app.models import (  # noqa: F401
    Appointment,
    AuditLog,
    MessageLog,
    Patient,
    Provider,
    ScheduleSlot,
    Service,
    Tenant,
)

__all__ = [
    "Base",
    "Appointment",
    "AuditLog",
    "MessageLog",
    "Patient",
    "Provider",
    "ScheduleSlot",
    "Service",
    "Tenant",
]
