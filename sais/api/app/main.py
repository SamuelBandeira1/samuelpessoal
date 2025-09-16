from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.db.session import get_db
from app.models import (
    Appointment,
    AppointmentOrigin,
    AppointmentStatus,
    MessageLog,
    Patient,
    Provider,
    ScheduleSlot,
    Service,
    SlotStatus,
    Tenant,
)
from app.services import (
    BotState,
    clear_context,
    get_context,
    get_state,
    record_last_interaction,
    send_interactive_buttons,
    send_template,
    send_text,
    set_context,
    set_state,
)
from app.services.bot_state import last_interaction_within
from app.services.scheduling import (
    book_slot,
    ensure_utc,
    get_slot_by_start,
    offer_slots,
    serialize_appointment,
    tenant_timezone,
)
from app.logging_utils import (
    configure_logging,
    get_current_tenant,
    get_request_id,
    set_tenant_context,
    _request_id_ctx_var,
    _tenant_id_ctx_var,
)

configure_logging()

app = FastAPI(title=settings.app_name, version="0.1.0")

logger = logging.getLogger(__name__)

WHATSAPP_SESSION_WINDOW = timedelta(hours=24)
MENU_PROMPT = "Como podemos ajudar? Escolha uma opção para continuar:"
MENU_BUTTONS = ["Agendar", "Reagendar", "Falar com atendente"]

REQUEST_COUNTER = Counter(
    "sais_api_requests_total",
    "Total number of processed HTTP requests.",
    ["method", "path", "status", "tenant"],
)
REQUEST_LATENCY = Histogram(
    "sais_api_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)


class SimpleRateLimiter:
    """In-memory rate limiter keyed by IP and tenant."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = max(1, limit)
        self.window_seconds = max(1, window_seconds)
        self._entries: dict[str, tuple[int, float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        async with self._lock:
            count, window_start = self._entries.get(key, (0, now))
            if now - window_start >= self.window_seconds:
                self._entries[key] = (1, now)
                return True
            if count >= self.limit:
                return False
            self._entries[key] = (count + 1, window_start)
            return True


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Populate request and tenant context for logging."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        tenant_hint = request.headers.get("X-Tenant-ID")

        request.state.request_id = request_id
        request_id_token = _request_id_ctx_var.set(request_id)
        tenant_token = _tenant_id_ctx_var.set(tenant_hint)

        try:
            response = await call_next(request)
        finally:
            _request_id_ctx_var.reset(request_id_token)
            _tenant_id_ctx_var.reset(tenant_token)

        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply a coarse rate limit per IP and tenant."""

    def __init__(self, app: FastAPI, limiter: SimpleRateLimiter) -> None:  # type: ignore[override]
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.method == "OPTIONS":
            return await call_next(request)

        client_host = request.client.host if request.client else "unknown"
        tenant_key = _tenant_id_ctx_var.get() or request.headers.get("X-Tenant-ID")
        tenant_value = tenant_key or "anonymous"
        rate_key = f"{client_host}:{tenant_value}"

        allowed = await self.limiter.allow(rate_key)
        if not allowed:
            logger.warning(
                "rate limit exceeded",
                extra={"client_ip": client_host, "tenant": tenant_value},
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
            )

        return await call_next(request)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Emit structured access logs and feed metrics."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        start_time = time.perf_counter()
        path = request.scope.get("root_path", "") + request.scope.get("path", request.url.path)
        method = request.method

        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start_time
            tenant_label = get_current_tenant()
            REQUEST_COUNTER.labels(method=method, path=path, status="500", tenant=tenant_label).inc()
            REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
            logger.exception(
                "request failed",
                extra={
                    "method": method,
                    "path": path,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )
            raise

        elapsed = time.perf_counter() - start_time
        status_code = response.status_code
        tenant_label = get_current_tenant()

        REQUEST_COUNTER.labels(
            method=method,
            path=path,
            status=str(status_code),
            tenant=tenant_label,
        ).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)

        logger.info(
            "request completed",
            extra={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": round(elapsed * 1000, 2),
            },
        )

        return response


rate_limiter = SimpleRateLimiter(
    settings.rate_limit_requests, settings.rate_limit_window_seconds
)


app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, limiter=rate_limiter)
app.add_middleware(AccessLogMiddleware)


class AppointmentCreate(BaseModel):
    patient_id: UUID
    provider_id: UUID
    service_id: UUID
    start_ts: datetime
    origin: AppointmentOrigin = AppointmentOrigin.WEB
    notes: str | None = None
    schedule_slot_id: UUID | None = None


class AppointmentUpdate(BaseModel):
    status: AppointmentStatus
    notes: str | None = None


def ensure_default_tenant(db: Session) -> UUID:
    """Return a tenant identifier, creating a placeholder if required."""

    tenant_id = settings.whatsapp_default_tenant_id
    if tenant_id:
        tenant = db.get(Tenant, tenant_id)
        if tenant:
            set_tenant_context(tenant.id)
            return tenant.id
        tenant = Tenant(
            id=tenant_id,
            name=settings.whatsapp_default_tenant_name,
            timezone=settings.timezone,
        )
        db.add(tenant)
        db.flush()
        logger.debug("Created default tenant %s for WhatsApp logs", tenant_id)
        set_tenant_context(tenant.id)
        return tenant.id

    tenant = db.execute(select(Tenant).limit(1)).scalar_one_or_none()
    if tenant:
        set_tenant_context(tenant.id)
        return tenant.id

    placeholder = Tenant(
        name=settings.whatsapp_default_tenant_name,
        timezone=settings.timezone,
    )
    db.add(placeholder)
    db.flush()
    logger.debug(
        "Created placeholder tenant %s for WhatsApp logs", placeholder.id
    )
    set_tenant_context(placeholder.id)
    return placeholder.id


def persist_message_log(
    db: Session,
    *,
    tenant_id: UUID,
    channel: str,
    recipient: str | None,
    payload: Any,
    metadata: dict[str, Any] | None,
    status: str,
    sent_at: datetime | None,
) -> MessageLog:
    """Persist a record in the message log table."""

    set_tenant_context(tenant_id)

    payload_value = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, default=str)
    )

    log_entry = MessageLog(
        tenant_id=tenant_id,
        channel=channel,
        recipient=recipient,
        payload=payload_value,
        metadata_json=metadata,
        status=status,
        sent_at=sent_at,
    )
    db.add(log_entry)
    db.flush()
    return log_entry


def log_outbound_message(
    db: Session,
    *,
    tenant_id: UUID,
    phone: str,
    request_payload: Any,
    response_payload: Any,
    response_type: str,
    session_active: bool,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist an outbound message in the log and return response metadata."""

    set_tenant_context(tenant_id)

    message_id = None
    if isinstance(response_payload, dict):
        message_id = (
            response_payload.get("key", {}).get("id")
            or response_payload.get("id")
            or response_payload.get("message", {}).get("key", {}).get("id")
        )

    metadata = {
        "direction": "outbound",
        "type": response_type,
        "session_active": session_active,
        "integration": "evolution_api",
    }
    if message_id:
        metadata["wa_message_id"] = message_id
    if metadata_extra:
        metadata.update(metadata_extra)

    now = datetime.now(timezone.utc)
    persist_message_log(
        db,
        tenant_id=tenant_id,
        channel="whatsapp",
        recipient=phone,
        payload={"request": request_payload, "response": response_payload},
        metadata=metadata,
        status="sent",
        sent_at=now,
    )
    record_last_interaction(phone, now)
    return {
        "message_id": message_id,
        "type": response_type,
        "session_active": session_active,
    }


def parse_timestamp(raw_timestamp: str | None) -> datetime:
    """Convert WhatsApp timestamps (seconds since epoch) to timezone-aware datetimes."""

    if not raw_timestamp:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromtimestamp(int(raw_timestamp), tz=timezone.utc)
    except (TypeError, ValueError):
        logger.debug("Invalid timestamp payload received: %s", raw_timestamp)
        return datetime.now(timezone.utc)


def jid_to_phone(jid: str | None) -> str | None:
    """Extract the numeric portion of a WhatsApp JID."""

    if not jid:
        return None
    phone = jid.split("@", 1)[0]
    digits = re.sub(r"\D", "", phone)
    return digits or None


def ensure_patient_profile(
    db: Session, *, tenant_id: UUID, phone_number: str, display_name: str | None
) -> Patient:
    """Return a patient for the phone number, creating a placeholder if needed."""

    stmt = select(Patient).where(
        Patient.tenant_id == tenant_id, Patient.phone_number == phone_number
    )
    patient = db.execute(stmt).scalars().first()
    if patient:
        return patient

    placeholder_name = display_name or f"Paciente {phone_number[-4:]}"
    patient = Patient(
        tenant_id=tenant_id,
        full_name=placeholder_name,
        phone_number=phone_number,
    )
    db.add(patient)
    db.flush()
    return patient


def service_options(db: Session, tenant_id: UUID) -> list[dict[str, Any]]:
    """Fetch the available services for the tenant."""

    stmt = select(Service).where(Service.tenant_id == tenant_id).order_by(Service.name)
    services = db.execute(stmt).scalars().all()
    return [
        {
            "id": str(service.id),
            "name": service.name,
            "duration_min": service.duration_min,
            "price_cents": service.price_cents,
        }
        for service in services
    ]


def primary_provider(db: Session, tenant_id: UUID) -> Provider | None:
    """Return the first provider for a tenant to drive the WhatsApp flow."""

    stmt = (
        select(Provider)
        .where(Provider.tenant_id == tenant_id)
        .order_by(Provider.full_name)
        .limit(1)
    )
    return db.execute(stmt).scalars().first()


def parse_service_choice(
    user_input: str, options: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    """Resolve the service based on numeric or textual input."""

    normalized = user_input.strip().casefold()
    if normalized.isdigit():
        index = int(normalized) - 1
        if 0 <= index < len(options):
            return options[index]

    for option in options:
        if option["name"].casefold() == normalized:
            return option
        if option["id"].casefold() == normalized:
            return option
    return None


def parse_date_choice(user_input: str) -> date | None:
    """Parse a YYYY-MM-DD formatted date."""

    try:
        return datetime.strptime(user_input.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def format_service_prompt(options: Sequence[dict[str, Any]]) -> str:
    """Generate the text prompt listing services."""

    if not options:
        return "No momento não encontramos serviços disponíveis."

    lines = [
        "Perfeito! Escolha um serviço digitando o número correspondente:",
    ]
    for index, option in enumerate(options, start=1):
        lines.append(
            f"{index}. {option['name']} ({option['duration_min']} min)"
        )
    return "\n".join(lines)


def format_slot_prompt(
    *,
    slots: Sequence[dict[str, Any]],
    target_date: date,
    tz: ZoneInfo,
    service_name: str,
) -> str:
    """Generate a textual prompt listing available slots."""

    if not slots:
        return (
            f"Não encontramos horários disponíveis para {service_name} em {target_date.isoformat()}."
            "\nEnvie outra data no formato YYYY-MM-DD para tentar novamente."
        )

    lines = [
        f"Horários disponíveis em {target_date.strftime('%d/%m/%Y')} para {service_name}:",
    ]
    for index, option in enumerate(slots, start=1):
        start_local = datetime.fromisoformat(option["start_local"]).astimezone(tz)
        lines.append(f"{index}. {start_local.strftime('%H:%M')}")
    lines.append("Responda com o número ou horário desejado (HH:MM).")
    return "\n".join(lines)


def parse_slot_choice(
    user_input: str, options: Sequence[dict[str, Any]], tz: ZoneInfo
) -> dict[str, Any] | None:
    """Select a slot based on numeric index or HH:MM time."""

    normalized = user_input.strip()
    if normalized.isdigit():
        index = int(normalized) - 1
        if 0 <= index < len(options):
            return options[index]

    try:
        requested_time = datetime.strptime(normalized, "%H:%M").time()
    except ValueError:
        return None

    for option in options:
        start_local = datetime.fromisoformat(option["start_local"]).astimezone(tz)
        if (
            start_local.hour == requested_time.hour
            and start_local.minute == requested_time.minute
        ):
            return option
    return None


def lock_slot_by_id(db: Session, slot_id: UUID) -> ScheduleSlot | None:
    """Fetch a slot by identifier with row-level locking."""

    stmt = select(ScheduleSlot).where(ScheduleSlot.id == slot_id).with_for_update()
    return db.execute(stmt).scalars().first()


def to_utc_from_tenant(dt_value: datetime, tenant: Tenant) -> datetime:
    """Convert an arbitrary datetime to UTC assuming tenant timezone when naive."""

    tz = tenant_timezone(tenant)
    if dt_value.tzinfo is None:
        localized = dt_value.replace(tzinfo=tz)
    else:
        localized = dt_value.astimezone(tz)
    return localized.astimezone(timezone.utc)

def normalize_evolution_message(
    raw_message: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert Evolution webhook payloads into a Graph-like message structure."""

    key = raw_message.get("key", {}) or {}
    message_body = raw_message.get("message", {}) or {}
    message_type = raw_message.get("messageType")
    remote_jid = key.get("remoteJid") or ""
    message_id = key.get("id") or raw_message.get("id")

    timestamp = raw_message.get("messageTimestamp")
    if timestamp is None:
        normalized_timestamp = datetime.now(timezone.utc)
    else:
        try:
            normalized_timestamp = datetime.fromtimestamp(
                int(timestamp), tz=timezone.utc
            )
        except (TypeError, ValueError):
            normalized_timestamp = datetime.now(timezone.utc)

    normalized_message: dict[str, Any] = {
        "id": message_id,
        "from": jid_to_phone(remote_jid),
        "timestamp": str(int(normalized_timestamp.timestamp())),
        "type": "text",
    }

    if "conversation" in message_body:
        normalized_message["type"] = "text"
        normalized_message["text"] = {"body": message_body.get("conversation", "")}
    elif "extendedTextMessage" in message_body:
        text_body = message_body.get("extendedTextMessage", {}).get("text", "")
        normalized_message["type"] = "text"
        normalized_message["text"] = {"body": text_body}
    elif "imageMessage" in message_body:
        normalized_message["type"] = "image"
        normalized_message["image"] = message_body.get("imageMessage", {})
    elif "videoMessage" in message_body:
        normalized_message["type"] = "video"
        normalized_message["video"] = message_body.get("videoMessage", {})
    elif "audioMessage" in message_body:
        normalized_message["type"] = "audio"
        normalized_message["audio"] = message_body.get("audioMessage", {})
    elif "documentMessage" in message_body:
        normalized_message["type"] = "document"
        normalized_message["document"] = message_body.get("documentMessage", {})
    elif "stickerMessage" in message_body:
        normalized_message["type"] = "sticker"
        normalized_message["sticker"] = message_body.get("stickerMessage", {})
    elif "locationMessage" in message_body:
        normalized_message["type"] = "location"
        normalized_message["location"] = message_body.get("locationMessage", {})
    elif "reactionMessage" in message_body:
        normalized_message["type"] = "reaction"
        normalized_message["reaction"] = message_body.get("reactionMessage", {})
    elif "contactMessage" in message_body:
        normalized_message["type"] = "contacts"
        normalized_message["contacts"] = [
            message_body.get("contactMessage", {})
        ]
    elif "contactsArrayMessage" in message_body:
        normalized_message["type"] = "contacts"
        normalized_message["contacts"] = message_body.get(
            "contactsArrayMessage", {}
        ).get("contacts", [])
    else:
        normalized_message["type"] = message_type or "unknown"

    metadata_extra = {
        "integration": "evolution_api",
        "message_type": message_type,
        "from_me": key.get("fromMe"),
        "remote_jid": remote_jid,
        "instance_id": raw_message.get("instanceId"),
        "push_name": raw_message.get("pushName"),
    }

    return normalized_message, metadata_extra


def extract_message_text(message: dict[str, Any]) -> str:
    """Return the human-readable text from a WhatsApp message payload."""

    message_type = message.get("type")
    if message_type == "text":
        return message.get("text", {}).get("body", "")
    if message_type == "interactive":
        interactive = message.get("interactive", {})
        if interactive.get("type") == "button_reply":
            return interactive.get("button_reply", {}).get("title", "")
        if interactive.get("type") == "list_reply":
            reply = interactive.get("list_reply", {})
            return reply.get("title") or reply.get("id", "")
    return ""


def handle_status_update(db: Session, status_payload: dict[str, Any]) -> None:
    """Apply delivery status updates to the message log."""

    message_id = (
        status_payload.get("keyId")
        or status_payload.get("id")
        or status_payload.get("message_id")
        or status_payload.get("messageId")
    )
    if not message_id:
        return

    stmt = select(MessageLog).where(
        MessageLog.metadata_json["wa_message_id"].astext == message_id
    )
    message_log = db.execute(stmt).scalars().first()
    if not message_log:
        logger.debug("Received status for unknown message id %s", message_id)
        return

    status_value = status_payload.get("status")
    if isinstance(status_value, str):
        message_log.status = status_value.lower()
    metadata = message_log.metadata_json or {}
    if "keyId" in status_payload:
        metadata.setdefault("integration", "evolution_api")
        metadata["status_origin"] = "evolution_api"
    history = metadata.setdefault("status_history", [])
    history.append(status_payload)
    message_log.metadata_json = metadata

    timestamp = status_payload.get("timestamp")
    if timestamp:
        try:
            message_log.sent_at = datetime.fromtimestamp(
                int(timestamp), tz=timezone.utc
            )
        except (TypeError, ValueError):
            logger.debug("Invalid status timestamp %s", timestamp)


def handle_inbound_message(
    db: Session,
    message: dict[str, Any],
    *,
    raw_payload: dict[str, Any] | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Process inbound WhatsApp messages, update state, and optionally reply."""

    phone = message.get("from")
    timestamp = parse_timestamp(message.get("timestamp"))
    tenant_id = ensure_default_tenant(db)

    metadata: dict[str, Any] = {
        "direction": "inbound",
        "wa_message_id": message.get("id"),
        "type": message.get("type"),
    }
    if metadata_extra:
        metadata.update({k: v for k, v in metadata_extra.items() if v is not None})
    persist_message_log(
        db,
        tenant_id=tenant_id,
        channel="whatsapp",
        recipient=phone,
        payload=raw_payload or message,
        metadata=metadata,
        status="received",
        sent_at=timestamp,
    )

    session_active = True
    if phone:
        session_active = last_interaction_within(
            phone, WHATSAPP_SESSION_WINDOW, timestamp
        )
        record_last_interaction(phone, timestamp)

    text_body = extract_message_text(message)
    normalized = text_body.strip().lower()

    state = get_state(phone) if phone else BotState.MENU_INICIAL
    context = get_context(phone) if phone else {}
    response_details: dict[str, Any] | None = None

    if normalized == "agendar" and phone:
        set_state(phone, BotState.AGENDAR)
        if not session_active:
            message_id, response_payload, request_payload = send_template(
                to=phone,
                template_name="confirmacao_consulta",
                variables=[
                    text_body or phone,
                    settings.app_name,
                    "em breve",
                ],
            )
            response_details = log_outbound_message(
                db,
                tenant_id=tenant_id,
                phone=phone,
                request_payload=request_payload,
                response_payload=response_payload,
                response_type="template",
                session_active=session_active,
                metadata_extra={"template": "confirmacao_consulta"},
            )
            return response_details

        provider = primary_provider(db, tenant_id)
        services = service_options(db, tenant_id)
        if not provider or not services:
            message_text = (
                "Estamos atualizando nossa agenda. Tente novamente em breve."
            )
            message_id, response_payload, request_payload = send_text(
                to=phone, body=message_text
            )
            response_details = log_outbound_message(
                db,
                tenant_id=tenant_id,
                phone=phone,
                request_payload=request_payload,
                response_payload=response_payload,
                response_type="text",
                session_active=session_active,
                metadata_extra={"stage": "unavailable"},
            )
            clear_context(phone)
            set_state(phone, BotState.MENU_INICIAL)
            return response_details

        display_name = None
        if metadata_extra:
            display_name = metadata_extra.get("push_name")
        patient = ensure_patient_profile(
            db,
            tenant_id=tenant_id,
            phone_number=phone,
            display_name=display_name,
        )
        context = {
            "stage": "service_selection",
            "services": services,
            "tenant_id": str(tenant_id),
            "provider_id": str(provider.id),
            "patient_id": str(patient.id),
        }
        set_context(phone, context)

        message_id, response_payload, request_payload = send_interactive_buttons(
            to=phone,
            body=MENU_PROMPT,
            buttons=MENU_BUTTONS,
        )
        log_outbound_message(
            db,
            tenant_id=tenant_id,
            phone=phone,
            request_payload=request_payload,
            response_payload=response_payload,
            response_type="interactive",
            session_active=session_active,
            metadata_extra={"buttons": MENU_BUTTONS, "stage": "menu"},
        )

        prompt_text = format_service_prompt(services)
        message_id, response_payload, request_payload = send_text(
            to=phone, body=prompt_text
        )
        response_details = log_outbound_message(
            db,
            tenant_id=tenant_id,
            phone=phone,
            request_payload=request_payload,
            response_payload=response_payload,
            response_type="text",
            session_active=session_active,
            metadata_extra={"stage": "service_selection"},
        )
        response_details["stage"] = "service_selection"
        return response_details
    elif normalized == "reagendar" and phone:
        set_state(phone, BotState.REMARCAR)
        clear_context(phone)
    elif normalized in {"humano", "falar com atendente"} and phone:
        set_state(phone, BotState.HUMANO)
        clear_context(phone)

    if phone and state == BotState.AGENDAR and context:
        stage = context.get("stage")
        tenant_key = context.get("tenant_id")
        tenant_uuid = UUID(tenant_key) if tenant_key else tenant_id
        tenant = db.get(Tenant, tenant_uuid)
        tz = tenant_timezone(tenant)

        if stage == "service_selection":
            options = context.get("services", [])
            selection = parse_service_choice(text_body, options)
            if not selection:
                message_text = (
                    "Não entendi a escolha. Responda com o número do serviço desejado."
                )
                message_id, response_payload, request_payload = send_text(
                    to=phone, body=message_text
                )
                response_details = log_outbound_message(
                    db,
                    tenant_id=tenant_id,
                    phone=phone,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    response_type="text",
                    session_active=session_active,
                    metadata_extra={"stage": "service_selection", "error": "invalid_choice"},
                )
                return response_details

            context["service_id"] = selection["id"]
            context["service_name"] = selection["name"]
            context["stage"] = "date_selection"
            set_context(phone, context)

            message_text = (
                f"Ótimo! Para {selection['name']}, informe a data desejada no formato YYYY-MM-DD."
            )
            message_id, response_payload, request_payload = send_text(
                to=phone, body=message_text
            )
            response_details = log_outbound_message(
                db,
                tenant_id=tenant_id,
                phone=phone,
                request_payload=request_payload,
                response_payload=response_payload,
                response_type="text",
                session_active=session_active,
                metadata_extra={"stage": "date_selection"},
            )
            response_details["stage"] = "date_selection"
            return response_details

        if stage == "date_selection":
            selected_date = parse_date_choice(text_body)
            if not selected_date:
                message_id, response_payload, request_payload = send_text(
                    to=phone,
                    body="Data inválida. Use o formato YYYY-MM-DD.",
                )
                response_details = log_outbound_message(
                    db,
                    tenant_id=tenant_id,
                    phone=phone,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    response_type="text",
                    session_active=session_active,
                    metadata_extra={"stage": "date_selection", "error": "invalid_format"},
                )
                return response_details

            provider_id = context.get("provider_id")
            service_id = context.get("service_id")
            provider = db.get(Provider, UUID(provider_id)) if provider_id else None
            service = db.get(Service, UUID(service_id)) if service_id else None
            if not provider or not service or not tenant:
                message_id, response_payload, request_payload = send_text(
                    to=phone,
                    body="Não foi possível encontrar a agenda. Por favor, tente novamente mais tarde.",
                )
                response_details = log_outbound_message(
                    db,
                    tenant_id=tenant_id,
                    phone=phone,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    response_type="text",
                    session_active=session_active,
                    metadata_extra={"stage": "date_selection", "error": "missing_entities"},
                )
                clear_context(phone)
                set_state(phone, BotState.MENU_INICIAL)
                return response_details

            held_slots, slot_tz = offer_slots(
                db,
                tenant=tenant,
                provider=provider,
                service=service,
                target_date=selected_date,
            )
            slot_options = [slot.as_dict(slot_tz) for slot in held_slots]
            context["slot_options"] = slot_options
            context["selected_date"] = selected_date.isoformat()
            context["timezone"] = getattr(slot_tz, "key", str(slot_tz))
            context["stage"] = "time_selection" if slot_options else "date_selection"
            set_context(phone, context)

            prompt = format_slot_prompt(
                slots=slot_options,
                target_date=selected_date,
                tz=slot_tz,
                service_name=service.name,
            )
            message_id, response_payload, request_payload = send_text(
                to=phone, body=prompt
            )
            response_details = log_outbound_message(
                db,
                tenant_id=tenant_id,
                phone=phone,
                request_payload=request_payload,
                response_payload=response_payload,
                response_type="text",
                session_active=session_active,
                metadata_extra={
                    "stage": context["stage"],
                    "slot_count": len(slot_options),
                },
            )
            response_details["stage"] = context["stage"]
            return response_details

        if stage == "time_selection":
            tz_name = context.get("timezone") or getattr(tz, "key", str(tz))
            try:
                slot_tz = ZoneInfo(tz_name)
            except Exception:  # pragma: no cover - fallback safety
                slot_tz = tz

            slot_options = context.get("slot_options", [])
            if not slot_options:
                context["stage"] = "date_selection"
                set_context(phone, context)
                message_id, response_payload, request_payload = send_text(
                    to=phone,
                    body="Os horários expiraram. Informe uma nova data para continuar.",
                )
                response_details = log_outbound_message(
                    db,
                    tenant_id=tenant_id,
                    phone=phone,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    response_type="text",
                    session_active=session_active,
                    metadata_extra={"stage": "date_selection", "error": "slots_expired"},
                )
                return response_details

            choice = parse_slot_choice(text_body, slot_options, slot_tz)
            if not choice:
                message_id, response_payload, request_payload = send_text(
                    to=phone,
                    body="Não entendi o horário selecionado. Escolha um dos horários listados.",
                )
                response_details = log_outbound_message(
                    db,
                    tenant_id=tenant_id,
                    phone=phone,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    response_type="text",
                    session_active=session_active,
                    metadata_extra={"stage": "time_selection", "error": "invalid_slot"},
                )
                return response_details

            provider = db.get(Provider, UUID(context.get("provider_id")))
            service = db.get(Service, UUID(context.get("service_id")))
            patient = db.get(Patient, UUID(context.get("patient_id")))
            if not provider or not service or not patient or not tenant:
                message_id, response_payload, request_payload = send_text(
                    to=phone,
                    body="Não foi possível concluir o agendamento. Tente novamente.",
                )
                response_details = log_outbound_message(
                    db,
                    tenant_id=tenant_id,
                    phone=phone,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    response_type="text",
                    session_active=session_active,
                    metadata_extra={"stage": "time_selection", "error": "missing_entities"},
                )
                clear_context(phone)
                set_state(phone, BotState.MENU_INICIAL)
                return response_details

            slot_id = UUID(choice["slot_id"])
            slot = lock_slot_by_id(db, slot_id)
            if not slot:
                message_id, response_payload, request_payload = send_text(
                    to=phone,
                    body="O horário escolhido não está mais disponível. Envie uma nova data.",
                )
                context["stage"] = "date_selection"
                set_context(phone, context)
                response_details = log_outbound_message(
                    db,
                    tenant_id=tenant_id,
                    phone=phone,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    response_type="text",
                    session_active=session_active,
                    metadata_extra={"stage": "date_selection", "error": "slot_missing"},
                )
                return response_details

            try:
                appointment = book_slot(
                    db,
                    tenant=tenant,
                    provider=provider,
                    patient=patient,
                    service=service,
                    slot=slot,
                    origin=AppointmentOrigin.WHATSAPP,
                )
            except ValueError:
                message_id, response_payload, request_payload = send_text(
                    to=phone,
                    body="O horário acabou de ser reservado por outra pessoa. Envie uma nova data.",
                )
                context["stage"] = "date_selection"
                set_context(phone, context)
                response_details = log_outbound_message(
                    db,
                    tenant_id=tenant_id,
                    phone=phone,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    response_type="text",
                    session_active=session_active,
                    metadata_extra={"stage": "date_selection", "error": "slot_conflict"},
                )
                return response_details

            confirmation_tz = slot_tz
            start_local = ensure_utc(appointment.scheduled_start).astimezone(
                confirmation_tz
            )
            message_text = (
                f"Consulta confirmada para {start_local.strftime('%d/%m/%Y às %H:%M')} com {service.name}."
            )
            message_id, response_payload, request_payload = send_text(
                to=phone, body=message_text
            )
            response_details = log_outbound_message(
                db,
                tenant_id=tenant_id,
                phone=phone,
                request_payload=request_payload,
                response_payload=response_payload,
                response_type="text",
                session_active=session_active,
                metadata_extra={
                    "stage": "completed",
                    "appointment_id": str(appointment.id),
                },
            )
            response_details["appointment"] = serialize_appointment(
                appointment, tz=confirmation_tz
            )
            clear_context(phone)
            set_state(phone, BotState.MENU_INICIAL)
            return response_details

    return response_details


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint used by infrastructure probes."""

    return {"status": "ok"}


@app.get("/api/v1/wa/webhook")
def whatsapp_webhook_verification(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
):
    """Handle the WhatsApp webhook verification handshake."""

    if (
        hub_mode == "subscribe"
        and hub_verify_token
        and hub_verify_token == settings.webhook_verify_token
    ):
        if not hub_challenge:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing hub.challenge",
            )
        return PlainTextResponse(content=hub_challenge)

    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


@app.post("/api/v1/wa/webhook")
def whatsapp_webhook(
    payload: dict[str, Any],
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Ingest WhatsApp events (messages and delivery statuses)."""

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload required",
        )

    processed: list[dict[str, Any]] = []

    event_name = payload.get("event")
    if event_name:
        event_key = str(event_name).upper()
        event_data = payload.get("data")
        items = event_data if isinstance(event_data, list) else [event_data]

        if event_key == "MESSAGES_UPSERT":
            for item in items:
                if not isinstance(item, dict):
                    continue
                normalized, extra_metadata = normalize_evolution_message(item)
                result = handle_inbound_message(
                    db,
                    normalized,
                    raw_payload=item,
                    metadata_extra=extra_metadata,
                )
                if result:
                    processed.append(result)
        elif event_key in {"MESSAGES_UPDATE"}:
            for status_update in items:
                if isinstance(status_update, dict):
                    handle_status_update(db, status_update)
        else:
            logger.debug("Unhandled Evolution webhook event: %s", event_name)

        return {"status": "ok", "processed": processed, "event": event_name}

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for message in value.get("messages", []) or []:
                result = handle_inbound_message(db, message)
                if result:
                    processed.append(result)
            for status_update in value.get("statuses", []) or []:
                handle_status_update(db, status_update)

    return {"status": "ok", "processed": processed}


@app.get("/api/v1/slots/search")
def search_slots(
    provider_id: UUID,
    service_id: UUID,
    date: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return available slots for a provider/service on a given date."""

    provider = db.get(Provider, provider_id)
    if not provider:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider not found")

    set_tenant_context(provider.tenant_id)

    service = db.get(Service, service_id)
    if not service:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Service not found")

    if service.tenant_id != provider.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider and service belong to different tenants",
        )

    tenant = db.get(Tenant, provider.tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    set_tenant_context(tenant.id)

    target_date = parse_date_choice(date)
    if not target_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD.",
        )

    held_slots, tz = offer_slots(
        db,
        tenant=tenant,
        provider=provider,
        service=service,
        target_date=target_date,
    )
    serialized = [slot.as_dict(tz) for slot in held_slots]
    for item in serialized:
        item["duration_min"] = service.duration_min
        item["price_cents"] = service.price_cents

    tz_name = getattr(tz, "key", str(tz))
    return {
        "provider_id": str(provider_id),
        "service_id": str(service_id),
        "date": target_date.isoformat(),
        "timezone": tz_name,
        "results": serialized,
    }


@app.get("/api/v1/appointments")
def list_appointments(
    tenant_id: UUID,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """List appointments for a tenant."""

    tenant = db.get(Tenant, tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    set_tenant_context(tenant.id)

    tz = tenant_timezone(tenant)
    stmt = (
        select(Appointment)
        .where(Appointment.tenant_id == tenant_id)
        .order_by(Appointment.scheduled_start)
    )
    appointments = db.execute(stmt).scalars().all()
    return {
        "tenant_id": str(tenant_id),
        "appointments": [serialize_appointment(appt, tz=tz) for appt in appointments],
    }


@app.post("/api/v1/appointments", status_code=status.HTTP_201_CREATED)
def create_appointment(
    payload: AppointmentCreate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Create a new appointment using a held slot."""

    provider = db.get(Provider, payload.provider_id)
    if not provider:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider not found")

    patient = db.get(Patient, payload.patient_id)
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    service = db.get(Service, payload.service_id)
    if not service:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Service not found")

    tenant = db.get(Tenant, provider.tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    set_tenant_context(tenant.id)

    if service.tenant_id != provider.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider and service belong to different tenants",
        )
    if patient.tenant_id != provider.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient is registered in another tenant",
        )

    start_utc = to_utc_from_tenant(payload.start_ts, tenant)

    slot: ScheduleSlot | None = None
    if payload.schedule_slot_id:
        slot = lock_slot_by_id(db, payload.schedule_slot_id)
        if not slot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schedule slot not found",
            )
        if slot.provider_id != provider.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Schedule slot belongs to another provider",
            )
    else:
        slot = get_slot_by_start(db, provider_id=provider.id, start_ts=start_utc)
        if not slot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No slot available at the requested time",
            )

    try:
        appointment = book_slot(
            db,
            tenant=tenant,
            provider=provider,
            patient=patient,
            service=service,
            slot=slot,
            origin=payload.origin,
            notes=payload.notes,
        )
    except ValueError as exc:  # slot no longer available
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    tz = tenant_timezone(tenant)
    return {"appointment": serialize_appointment(appointment, tz=tz)}


@app.patch("/api/v1/appointments/{appointment_id}")
def update_appointment_status(
    appointment_id: UUID,
    payload: AppointmentUpdate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Update the status of an appointment."""

    stmt = (
        select(Appointment)
        .where(Appointment.id == appointment_id)
        .with_for_update()
    )
    appointment = db.execute(stmt).scalars().first()
    if not appointment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Appointment not found",
        )

    appointment.status = payload.status
    if payload.notes is not None:
        appointment.notes = payload.notes

    tenant = db.get(Tenant, appointment.tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    set_tenant_context(tenant.id)

    release_statuses = {
        AppointmentStatus.CANCELLED,
        AppointmentStatus.NO_SHOW,
        AppointmentStatus.RESCHEDULED,
    }
    if payload.status in release_statuses and appointment.schedule_slot_id:
        slot = db.get(ScheduleSlot, appointment.schedule_slot_id)
        if slot:
            slot.status = SlotStatus.FREE
            slot.hold_expires_at = None

    tz = tenant_timezone(tenant)
    return {"appointment": serialize_appointment(appointment, tz=tz)}
