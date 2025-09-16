from __future__ import annotations

import contextvars
import json
import logging
import sys
from typing import Any
from uuid import UUID

_request_id_ctx_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
_tenant_id_ctx_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tenant_id", default=None
)


class RequestContextFilter(logging.Filter):
    """Inject request scoped context variables into log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple
        record.request_id = _request_id_ctx_var.get()
        record.tenant_id = _tenant_id_ctx_var.get()
        return True


class JSONLogFormatter(logging.Formatter):
    """Serialize log records as JSON with context metadata."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        super().__init__(datefmt="%Y-%m-%dT%H:%M:%S%z")

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": record.__dict__.get("request_id"),
            "tenant_id": record.__dict__.get("tenant_id"),
        }

        standard_attributes = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }

        for key, value in record.__dict__.items():
            if key in ("request_id", "tenant_id"):
                continue
            if key.startswith("_") or key in standard_attributes:
                continue
            log_entry[key] = value

        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for structured JSON output."""

    global _configured
    if _configured:  # pragma: no cover - defensive
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONLogFormatter())
    handler.addFilter(RequestContextFilter())

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = True

    _configured = True


def set_tenant_context(tenant_id: UUID | str | None) -> None:
    """Bind the tenant identifier to the current logging context."""

    if tenant_id is None:
        _tenant_id_ctx_var.set(None)
    elif isinstance(tenant_id, UUID):
        _tenant_id_ctx_var.set(str(tenant_id))
    else:
        _tenant_id_ctx_var.set(tenant_id)


def get_current_tenant() -> str:
    """Return the tenant id bound to the current context."""

    tenant = _tenant_id_ctx_var.get()
    return tenant or "anonymous"


def get_request_id() -> str:
    """Return the request id bound to the current context."""

    request_id = _request_id_ctx_var.get()
    return request_id or "unknown"


__all__ = [
    "configure_logging",
    "get_current_tenant",
    "get_request_id",
    "set_tenant_context",
    "_request_id_ctx_var",
    "_tenant_id_ctx_var",
]
