"""Thin wrapper around the Evolution WhatsApp API."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Iterable, Sequence
from urllib.parse import quote

import httpx

from app.core.config import settings
from app.services.whatsapp_templates import TEMPLATE_MOCKS

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=15.0, pool=5.0)


def _build_evolution_url(action: str) -> str:
    base_url = settings.evolution_api_base_url.rstrip("/")
    instance_name = settings.evolution_instance_name
    if not instance_name:
        raise RuntimeError("EVOLUTION_INSTANCE_NAME is not configured")
    return f"{base_url}/message/{action}/{quote(instance_name)}"


def _mock_send(action: str, payload: dict) -> tuple[str, dict]:
    message_id = f"mocked-{uuid.uuid4()}"
    logger.debug("Mocking Evolution send (%s) with payload: %s", action, payload)
    phone = payload.get("number")
    normalized_phone = re.sub(r"\D", "", phone or "")
    mock_response = {
        "key": {
            "id": message_id,
            "remoteJid": f"{normalized_phone or '00000000000'}@mock",
        },
        "messageType": action,
        "mocked": True,
        "payload": payload,
    }
    if action == "sendTemplate":
        template_name = payload.get("name", "")
        mock_response["template"] = TEMPLATE_MOCKS.get(template_name)
    return message_id, mock_response


def _dispatch(action: str, payload: dict) -> tuple[str, dict]:
    if settings.whatsapp_mock_mode:
        return _mock_send(action, payload)

    api_key = settings.evolution_api_key
    if not api_key:
        raise RuntimeError("EVOLUTION_API_KEY is not configured")

    url = _build_evolution_url(action)
    headers = {
        "apikey": api_key,
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    message_id = (
        data.get("key", {}).get("id")
        or data.get("id")
        or data.get("message", {}).get("key", {}).get("id")
    )
    if not message_id:
        raise RuntimeError("Evolution API response did not include a message identifier")

    logger.debug("Evolution API responded with %s", data)
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
        "number": to,
        "name": template_name,
        "language": language_code,
        "components": components,
    }
    message_id, response = _dispatch("sendTemplate", payload)
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
                "id": f"option_{index}",
                "displayText": title,
            }
        )

    payload = {
        "number": to,
        "title": body,
        "buttons": action_buttons,
    }
    message_id, response = _dispatch("sendButtons", payload)
    return message_id, response, payload
