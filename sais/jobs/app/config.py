from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Celery worker configuration."""

    redis_url: str = "redis://redis:6379/0"
    timezone: str = "America/Fortaleza"
    database_url: str = "postgresql+psycopg2://sais:sais@db:5432/sais"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
