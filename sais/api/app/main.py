from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models import MessageLog, Tenant
from app.services import (
    BotState,
    record_last_interaction,
    send_interactive_buttons,
    send_template,
    set_state,
)
from app.services.bot_state import last_interaction_within

app = FastAPI(title=settings.app_name, version="0.1.0")

logger = logging.getLogger(__name__)

WHATSAPP_SESSION_WINDOW = timedelta(hours=24)
MENU_PROMPT = "Como podemos ajudar? Escolha uma opção para continuar:"
MENU_BUTTONS = ["Agendar", "Reagendar", "Falar com atendente"]


class AppointmentCreate(BaseModel):
    tenant_id: str
    patient_id: str
    provider_id: str
    service_id: str | None = None
    schedule_slot_id: str | None = None
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    notes: str | None = None


def ensure_default_tenant(db: Session) -> UUID:
    """Return a tenant identifier, creating a placeholder if required."""

    tenant_id = settings.whatsapp_default_tenant_id
    if tenant_id:
        tenant = db.get(Tenant, tenant_id)
        if tenant:
            return tenant.id
        tenant = Tenant(
            id=tenant_id,
            name=settings.whatsapp_default_tenant_name,
            timezone=settings.timezone,
        )
        db.add(tenant)
        db.flush()
        logger.debug("Created default tenant %s for WhatsApp logs", tenant_id)
        return tenant.id

    tenant = db.execute(select(Tenant).limit(1)).scalar_one_or_none()
    if tenant:
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
        metadata=metadata,
        status=status,
        sent_at=sent_at,
    )
    db.add(log_entry)
    db.flush()
    return log_entry


def parse_timestamp(raw_timestamp: str | None) -> datetime:
    """Convert WhatsApp timestamps (seconds since epoch) to timezone-aware datetimes."""

    if not raw_timestamp:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromtimestamp(int(raw_timestamp), tz=timezone.utc)
    except (TypeError, ValueError):
        logger.debug("Invalid timestamp payload received: %s", raw_timestamp)
        return datetime.now(timezone.utc)


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

    message_id = status_payload.get("id")
    if not message_id:
        return

    stmt = select(MessageLog).where(
        MessageLog.metadata["wa_message_id"].astext == message_id
    )
    message_log = db.execute(stmt).scalars().first()
    if not message_log:
        logger.debug("Received status for unknown message id %s", message_id)
        return

    message_log.status = status_payload.get("status", message_log.status)
    metadata = message_log.metadata or {}
    history = metadata.setdefault("status_history", [])
    history.append(status_payload)
    message_log.metadata = metadata

    timestamp = status_payload.get("timestamp")
    if timestamp:
        try:
            message_log.sent_at = datetime.fromtimestamp(
                int(timestamp), tz=timezone.utc
            )
        except (TypeError, ValueError):
            logger.debug("Invalid status timestamp %s", timestamp)


def handle_inbound_message(db: Session, message: dict[str, Any]) -> dict[str, Any] | None:
    """Process inbound WhatsApp messages, update state, and optionally reply."""

    phone = message.get("from")
    timestamp = parse_timestamp(message.get("timestamp"))
    tenant_id = ensure_default_tenant(db)

    metadata = {
        "direction": "inbound",
        "wa_message_id": message.get("id"),
        "type": message.get("type"),
    }
    persist_message_log(
        db,
        tenant_id=tenant_id,
        channel="whatsapp",
        recipient=phone,
        payload=message,
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

    response_details: dict[str, Any] | None = None

    if normalized == "agendar" and phone:
        set_state(phone, BotState.AGENDAR)
        if session_active:
            message_id, response_payload, request_payload = send_interactive_buttons(
                to=phone,
                body=MENU_PROMPT,
                buttons=MENU_BUTTONS,
            )
            response_type = "interactive"
        else:
            message_id, response_payload, request_payload = send_template(
                to=phone,
                template_name="confirmacao_consulta",
                variables=[
                    text_body or phone,
                    settings.app_name,
                    "em breve",
                ],
            )
            response_type = "template"

        outbound_metadata = {
            "direction": "outbound",
            "wa_message_id": message_id,
            "type": response_type,
            "session_active": session_active,
        }
        if response_type == "interactive":
            outbound_metadata["buttons"] = MENU_BUTTONS

        payload_to_store = {
            "request": request_payload,
            "response": response_payload,
        }
        persist_message_log(
            db,
            tenant_id=tenant_id,
            channel="whatsapp",
            recipient=phone,
            payload=payload_to_store,
            metadata=outbound_metadata,
            status="sent",
            sent_at=datetime.now(timezone.utc),
        )
        record_last_interaction(phone, datetime.now(timezone.utc))

        response_details = {
            "message_id": message_id,
            "type": response_type,
            "session_active": session_active,
        }
    elif normalized == "reagendar" and phone:
        set_state(phone, BotState.REMARCAR)
    elif normalized in {"humano", "falar com atendente"} and phone:
        set_state(phone, BotState.HUMANO)
    elif phone:
        set_state(phone, BotState.MENU_INICIAL)

    return response_details


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
        and hub_verify_token == settings.whatsapp_verify_token
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
    tenant_id: str,
    db: Session = Depends(get_db),
) -> dict[str, list[dict[str, Any]]]:
    """Return available slots for a given tenant (stub)."""

    # A real implementation would query the database using the SQLAlchemy session.
    _ = db  # suppress unused variable warnings
    return {
        "tenant_id": tenant_id,
        "results": [],
    }


@app.get("/api/v1/appointments")
def list_appointments(
    tenant_id: str,
    db: Session = Depends(get_db),
) -> dict[str, list[dict[str, Any]]]:
    """List appointments for a tenant (stub)."""

    _ = db
    return {"tenant_id": tenant_id, "appointments": []}


@app.post("/api/v1/appointments", status_code=status.HTTP_201_CREATED)
def create_appointment(
    payload: AppointmentCreate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Create a new appointment (stub)."""

    _ = db
    return {"appointment_id": "stub", "data": payload.model_dump()}
