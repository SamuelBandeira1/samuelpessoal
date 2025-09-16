"""Conversation state management backed by Redis."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Final

import redis

from app.core.config import settings

_STATE_KEY_TEMPLATE: Final[str] = "sais:wa:state:{phone}"
_LAST_INTERACTION_KEY_TEMPLATE: Final[str] = "sais:wa:last_interaction:{phone}"
_CONTEXT_KEY_TEMPLATE: Final[str] = "sais:wa:context:{phone}"
_STATE_TTL_SECONDS: Final[int] = 60 * 60 * 24 * 7  # one week


class BotState(str, Enum):
    """Enumerated conversation states for the WhatsApp bot."""

    MENU_INICIAL = "MENU_INICIAL"
    AGENDAR = "AGENDAR"
    REMARCAR = "REMARCAR"
    HUMANO = "HUMANO"


def _get_client() -> redis.Redis:
    """Return a Redis client configured via application settings."""

    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def get_state(phone_e164: str) -> BotState:
    """Fetch the current state for a WhatsApp contact."""

    client = _get_client()
    raw_state = client.get(_STATE_KEY_TEMPLATE.format(phone=phone_e164))
    if not raw_state:
        return BotState.MENU_INICIAL
    try:
        return BotState(raw_state)
    except ValueError:
        return BotState.MENU_INICIAL


def set_state(phone_e164: str, state: BotState) -> None:
    """Persist the bot state for a contact."""

    client = _get_client()
    client.setex(
        _STATE_KEY_TEMPLATE.format(phone=phone_e164),
        _STATE_TTL_SECONDS,
        state.value,
    )


def record_last_interaction(phone_e164: str, timestamp: datetime) -> None:
    """Store the timestamp of the latest interaction for the contact."""

    client = _get_client()
    client.setex(
        _LAST_INTERACTION_KEY_TEMPLATE.format(phone=phone_e164),
        _STATE_TTL_SECONDS,
        timestamp.isoformat(),
    )


def get_context(phone_e164: str) -> dict[str, Any]:
    """Return the structured context stored for the contact."""

    client = _get_client()
    raw_value = client.get(_CONTEXT_KEY_TEMPLATE.format(phone=phone_e164))
    if not raw_value:
        return {}
    try:
        data = json.loads(raw_value)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return {}
    return {}


def set_context(phone_e164: str, context: dict[str, Any]) -> None:
    """Persist conversation context for the contact."""

    client = _get_client()
    client.setex(
        _CONTEXT_KEY_TEMPLATE.format(phone=phone_e164),
        _STATE_TTL_SECONDS,
        json.dumps(context, ensure_ascii=False),
    )


def clear_context(phone_e164: str) -> None:
    """Remove any stored context for the contact."""

    client = _get_client()
    client.delete(_CONTEXT_KEY_TEMPLATE.format(phone=phone_e164))


def last_interaction_within(
    phone_e164: str, delta: timedelta, reference: datetime
) -> bool:
    """Return whether the last interaction was within the provided time delta."""

    client = _get_client()
    raw_value = client.get(_LAST_INTERACTION_KEY_TEMPLATE.format(phone=phone_e164))
    if not raw_value:
        # Consider no history as within window to keep onboarding simple.
        return True

    try:
        last_seen = datetime.fromisoformat(raw_value)
    except ValueError:
        return False
    return reference - last_seen <= delta
