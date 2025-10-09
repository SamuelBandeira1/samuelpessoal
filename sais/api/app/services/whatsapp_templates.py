"""Mock WhatsApp templates for development environments."""

from __future__ import annotations

from typing import Any, Dict

TEMPLATE_MOCKS: Dict[str, Dict[str, Any]] = {
    "confirmacao_consulta": {
        "category": "TRANSACTIONAL",
        "language": "pt_BR",
        "body": "Olá {{1}}, sua consulta com {{2}} está confirmada para {{3}}.",
    },
    "lembrete_d1": {
        "category": "TRANSACTIONAL",
        "language": "pt_BR",
        "body": "Lembrete: você possui uma consulta em 1 dia. Responda 1 para confirmar.",
    },
    "pos_consulta_nps": {
        "category": "MARKETING",
        "language": "pt_BR",
        "body": "Como foi sua experiência? Responda de 0 a 10 para avaliarmos.",
    },
}

__all__ = ["TEMPLATE_MOCKS"]
