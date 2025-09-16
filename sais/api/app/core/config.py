from functools import lru_cache
from uuid import UUID

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "SAIS API"
    database_url: str = (
        "postgresql+psycopg2://sais:sais@db:5432/sais"  # pragma: allowlist secret
    )
    redis_url: str = "redis://redis:6379/0"
    timezone: str = "America/Fortaleza"
    cors_origins: list[str] = ["http://localhost:3000"]
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    evolution_api_base_url: str = "http://localhost:8080"
    evolution_instance_name: str = ""
    evolution_api_key: str = ""
    webhook_verify_token: str = ""
    whatsapp_mock_mode: bool = False
    whatsapp_default_tenant_id: UUID | None = None
    whatsapp_default_tenant_name: str = "Default WhatsApp Tenant"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()


settings = get_settings()
