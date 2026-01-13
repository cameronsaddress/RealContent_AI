"""
Configuration management using pydantic-settings.
Loads from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Database
    DATABASE_URL: str = "postgresql://n8n:n8n_password@postgres:5432/content_pipeline"

    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # API Keys
    OPENROUTER_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"
    HEYGEN_API_KEY: str = ""
    HEYGEN_AVATAR_ID: str = ""
    PEXELS_API_KEY: str = ""
    APIFY_API_KEY: str = ""
    BLOTATO_API_KEY: str = ""

    # Dropbox OAuth
    DROPBOX_APP_KEY: str = ""
    DROPBOX_APP_SECRET: str = ""
    DROPBOX_REFRESH_TOKEN: str = ""

    # Service URLs
    VIDEO_PROCESSOR_URL: str = "http://video-processor:8080"
    BACKEND_URL: str = "http://backend:8000"

    # Asset paths
    ASSETS_BASE_PATH: str = "/app/assets"

    # Pipeline settings
    HEYGEN_MAX_RETRIES: int = 60
    HEYGEN_POLL_INTERVAL: int = 30



@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
