from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.logging_utils import configure_logging, set_tenant_context
from app.models import Patient, Provider, ScheduleSlot, Service, SlotStatus, Tenant

logger = logging.getLogger(__name__)

SERVICE_CATALOG: list[tuple[str, int, int]] = [
    ("Avaliação Odontológica", 30, 15000),
    ("Limpeza Profissional", 45, 25000),
    ("Clareamento Dental", 60, 48000),
]

PROVIDERS: list[tuple[str, str]] = [
    ("Dra. Ana Costa", "Ortodontista"),
    ("Dr. Bruno Lima", "Implantodontista"),
]

PATIENTS: list[tuple[str, str, str]] = [
    ("Maria Silva", "maria.silva@example.com", "+5585987654321"),
    ("João Pereira", "joao.pereira@example.com", "+558593334455"),
]


def ensure_tenant(session) -> Tenant:
    tenant = (
        session.execute(
            select(Tenant).where(Tenant.name == "Clínica SAIS Demo")
        ).scalar_one_or_none()
    )
    if tenant:
        set_tenant_context(tenant.id)
        logger.info("tenant already present", extra={"tenant_id": str(tenant.id)})
        return tenant

    tenant = Tenant(name="Clínica SAIS Demo", timezone=settings.timezone)
    session.add(tenant)
    session.flush()
    set_tenant_context(tenant.id)
    logger.info("created tenant", extra={"tenant_id": str(tenant.id)})
    return tenant


def ensure_services(session, tenant: Tenant) -> list[Service]:
    created = 0
    services: list[Service] = []
    for name, duration, price in SERVICE_CATALOG:
        service = (
            session.execute(
                select(Service).where(
                    Service.tenant_id == tenant.id,
                    Service.name == name,
                )
            ).scalar_one_or_none()
        )
        if not service:
            service = Service(
                tenant_id=tenant.id,
                name=name,
                duration_min=duration,
                price_cents=price,
            )
            session.add(service)
            session.flush()
            created += 1
        services.append(service)

    logger.info(
        "ensured services",
        extra={"tenant_id": str(tenant.id), "created": created, "total": len(services)},
    )
    return services


def ensure_providers(session, tenant: Tenant) -> list[Provider]:
    created = 0
    providers: list[Provider] = []
    for name, specialty in PROVIDERS:
        provider = (
            session.execute(
                select(Provider).where(
                    Provider.tenant_id == tenant.id,
                    Provider.full_name == name,
                )
            ).scalar_one_or_none()
        )
        if not provider:
            provider = Provider(
                tenant_id=tenant.id,
                full_name=name,
                specialty=specialty,
            )
            session.add(provider)
            session.flush()
            created += 1
        providers.append(provider)

    logger.info(
        "ensured providers",
        extra={"tenant_id": str(tenant.id), "created": created, "total": len(providers)},
    )
    return providers


def ensure_patients(session, tenant: Tenant) -> list[Patient]:
    created = 0
    patients: list[Patient] = []
    for name, email, phone in PATIENTS:
        patient = (
            session.execute(
                select(Patient).where(
                    Patient.tenant_id == tenant.id,
                    Patient.full_name == name,
                )
            ).scalar_one_or_none()
        )
        if not patient:
            patient = Patient(
                tenant_id=tenant.id,
                full_name=name,
                email=email,
                phone_number=phone,
            )
            session.add(patient)
            session.flush()
            created += 1
        patients.append(patient)

    logger.info(
        "ensured patients",
        extra={"tenant_id": str(tenant.id), "created": created, "total": len(patients)},
    )
    return patients


def ensure_slots(session, tenant: Tenant, providers: Iterable[Provider]) -> None:
    tz = ZoneInfo(tenant.timezone or settings.timezone)
    today = datetime.now(tz).date()
    created = 0

    for provider in providers:
        for offset in range(5):
            day = today + timedelta(days=offset)
            for start_hour in range(9, 17):
                local_start = datetime.combine(day, time(hour=start_hour, minute=0), tzinfo=tz)
                start_utc = local_start.astimezone(timezone.utc)
                end_utc = start_utc + timedelta(minutes=45)
                existing = (
                    session.execute(
                        select(ScheduleSlot).where(
                            ScheduleSlot.provider_id == provider.id,
                            ScheduleSlot.start_time == start_utc,
                        )
                    ).scalar_one_or_none()
                )
                if existing:
                    continue
                slot = ScheduleSlot(
                    tenant_id=tenant.id,
                    provider_id=provider.id,
                    start_time=start_utc,
                    end_time=end_utc,
                    status=SlotStatus.FREE,
                )
                session.add(slot)
                created += 1

    logger.info(
        "ensured schedule slots",
        extra={"tenant_id": str(tenant.id), "created": created},
    )


def seed() -> None:
    configure_logging()
    logger.info("starting seed process")

    session = SessionLocal()
    try:
        tenant = ensure_tenant(session)
        set_tenant_context(tenant.id)
        ensure_services(session, tenant)
        providers = ensure_providers(session, tenant)
        ensure_patients(session, tenant)
        ensure_slots(session, tenant, providers)
        session.commit()
        logger.info("seed complete", extra={"tenant_id": str(tenant.id)})
    except Exception:
        session.rollback()
        logger.exception("seed failed")
        raise
    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    seed()
