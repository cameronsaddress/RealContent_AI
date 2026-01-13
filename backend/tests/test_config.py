"""Tests for configuration module."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_settings_defaults():
    """Test that settings have sensible defaults."""
    from config import Settings

    # Create settings without env file
    settings = Settings(
        _env_file=None,
        DATABASE_URL="postgresql://test:test@localhost/test"
    )

    assert settings.DATABASE_URL == "postgresql://test:test@localhost/test"
    assert settings.CELERY_BROKER_URL == "redis://redis:6379/0"
    assert settings.HEYGEN_MAX_RETRIES == 60
    assert settings.HEYGEN_POLL_INTERVAL == 30


def test_settings_env_override():
    """Test that environment variables override defaults."""
    os.environ["HEYGEN_MAX_RETRIES"] = "100"

    from importlib import reload
    import config
    reload(config)

    # Clear the lru_cache
    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.HEYGEN_MAX_RETRIES == 100

    # Cleanup
    del os.environ["HEYGEN_MAX_RETRIES"]
    config.get_settings.cache_clear()


def test_asset_paths():
    """Test asset path management."""
    from utils.paths import AssetPaths

    paths = AssetPaths("/tmp/test_assets")

    assert paths.voice_path(123) == paths.base / "audio" / "123_voice.mp3"
    assert paths.avatar_path(123) == paths.base / "avatar" / "123_avatar.mp4"
    assert paths.source_video_path(123) == paths.base / "videos" / "123_source.mp4"
    assert paths.combined_video_path(123) == paths.base / "output" / "123_combined.mp4"
    assert paths.final_video_path(123) == paths.base / "output" / "123_final.mp4"
    assert paths.srt_path(123) == paths.base / "captions" / "123_captions.srt"


def test_file_exists_check():
    """Test file existence check."""
    from utils.paths import AssetPaths
    from pathlib import Path
    import tempfile

    paths = AssetPaths("/tmp/test_assets")

    # Create a test file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(b"x" * 2000)  # 2KB file
        temp_path = Path(f.name)

    try:
        assert paths.file_exists_and_valid(temp_path, min_size=1000) is True
        assert paths.file_exists_and_valid(temp_path, min_size=5000) is False
        assert paths.file_exists_and_valid(Path("/nonexistent"), min_size=100) is False
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    test_settings_defaults()
    print("✓ test_settings_defaults passed")

    test_asset_paths()
    print("✓ test_asset_paths passed")

    test_file_exists_check()
    print("✓ test_file_exists_check passed")

    print("\nAll config tests passed!")
