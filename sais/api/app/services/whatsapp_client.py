"""Thin wrapper around the WhatsApp Cloud API."""

from __future__ import annotations

import logging
import uuid
from typing import Iterable, Sequence

import httpx

from app.core.config import settings
from app.services.whatsapp_templates import TEMPLATE_MOCKS

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=15.0, pool=5.0)


def _build_messages_url() -> str:
    base_url = settings.whatsapp_api_base_url.rstrip("/")
    phone_number_id = settings.whatsapp_phone_number_id
    if not phone_number_id:
        raise RuntimeError("WHATSAPP_PHONE_NUMBER_ID is not configured")
    return f"{base_url}/{phone_number_id}/messages"


def _mock_send(payload: dict) -> tuple[str, dict]:
    message_id = f"mocked-{uuid.uuid4()}"
    logger.debug("Mocking WhatsApp send with payload: %s", payload)
    mock_response = {
        "messages": [{"id": message_id}],
        "mocked": True,
        "payload": payload,
    }
    if payload.get("type") == "template":
        template_name = payload.get("template", {}).get("name", "")
        mock_response["template"] = TEMPLATE_MOCKS.get(template_name)
    return message_id, mock_response


def _dispatch(payload: dict) -> tuple[str, dict]:
    if settings.whatsapp_mock_mode:
        return _mock_send(payload)

    token = settings.whatsapp_token
    if not token:
        raise RuntimeError("WHATSAPP_TOKEN is not configured")

    url = _build_messages_url()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    messages = data.get("messages", [])
    if not messages:
        raise RuntimeError("WhatsApp API response did not include message identifiers")

    message_id = messages[0].get("id")
    if not message_id:
        raise RuntimeError("WhatsApp API response returned an empty message id")

    logger.debug("WhatsApp API responded with %s", data)
    return message_id, data


def send_template(
    to: str,
    template_name: str,
    variables: Iterable[str] | None = None,
    language_code: str = "pt_BR",
) -> tuple[str, dict, dict]:
    """Send a template-based WhatsApp message."""

    components = []
    if variables:
        components.append(
            {
                "type": "body",
                "parameters": [
                    {"type": "text", "text": str(value)} for value in variables
                ],
            }
        )

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
            "components": components,
        },
    }
    message_id, response = _dispatch(payload)
    return message_id, response, payload


def send_interactive_buttons(
    to: str, body: str, buttons: Sequence[str]
) -> tuple[str, dict, dict]:
    """Send an interactive message containing up to three quick-reply buttons."""

    if not buttons:
        raise ValueError("At least one button title must be provided")
    if len(buttons) > 3:
        raise ValueError("WhatsApp only supports up to 3 buttons in a message")

    action_buttons = []
    for index, title in enumerate(buttons, start=1):
        action_buttons.append(
            {
                "type": "reply",
                "reply": {
                    "id": f"option_{index}",
                    "title": title,
                },
            }
        )

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body},
            "action": {"buttons": action_buttons},
        },
    }
    message_id, response = _dispatch(payload)
    return message_id, response, payload
