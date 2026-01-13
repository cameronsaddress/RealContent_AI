"""
Tests for VoiceService - uses mocked responses, no real API calls.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from services.voice import (
    VoiceService,
    VoiceRequest,
    VoiceResult
)


class TestVoiceRequest:
    """Test VoiceRequest model."""

    def test_defaults(self):
        """Test default values."""
        request = VoiceRequest(script_id=22, text="Hello world")
        assert request.script_id == 22
        assert request.text == "Hello world"
        assert request.voice_id is None  # Will use default from settings
        assert request.model_id == "eleven_multilingual_v2"
        assert request.stability == 0.5
        assert request.similarity_boost == 0.75

    def test_custom_values(self):
        """Test custom values."""
        request = VoiceRequest(
            script_id=30,
            text="Custom voice text",
            voice_id="custom_voice_123",
            stability=0.7,
            similarity_boost=0.8,
            style=0.3,
            use_speaker_boost=False
        )
        assert request.voice_id == "custom_voice_123"
        assert request.stability == 0.7
        assert request.use_speaker_boost is False


class TestVoiceResult:
    """Test VoiceResult model."""

    def test_result(self):
        """Test result model."""
        result = VoiceResult(
            script_id=22,
            audio_path="/app/assets/audio/22_voice.mp3",
            duration_seconds=45.5,
            file_size_bytes=512000,
            voice_id="test_voice_id"
        )
        assert result.script_id == 22
        assert result.duration_seconds == 45.5
        assert result.file_size_bytes == 512000


class TestVoiceService:
    """Test VoiceService methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return VoiceService()

    def test_prepare_tts_text_basic(self, service):
        """Test basic text preparation."""
        text = "Hello  world.   This  is a test."
        result = service.prepare_tts_text(text)
        # Multiple spaces should be normalized
        assert "  " not in result

    def test_prepare_tts_text_pauses(self, service):
        """Test pause insertion."""
        text = "First sentence. Second sentence! Third sentence?"
        result = service.prepare_tts_text(text)
        # Should have pauses after sentence enders
        assert "..." in result

    def test_prepare_tts_text_cleanup(self, service):
        """Test cleanup of excessive pauses."""
        text = "Too.... many.... dots...."
        result = service.prepare_tts_text(text)
        # Should not have more than 3 dots in a row
        assert "...." not in result

    def test_voice_exists_false(self, service):
        """Test voice_exists returns False for missing file."""
        with patch('services.voice.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_paths.voice_path.return_value = mock_path

            result = service.voice_exists(999)
            assert result is False

    def test_voice_exists_empty_file(self, service):
        """Test voice_exists returns False for empty file."""
        with patch('services.voice.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_stat = MagicMock()
            mock_stat.st_size = 0
            mock_path.stat.return_value = mock_stat
            mock_paths.voice_path.return_value = mock_path

            result = service.voice_exists(999)
            assert result is False

    def test_voice_exists_true(self, service):
        """Test voice_exists returns True for valid file."""
        with patch('services.voice.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_stat = MagicMock()
            mock_stat.st_size = 512000
            mock_path.stat.return_value = mock_stat
            mock_paths.voice_path.return_value = mock_path

            result = service.voice_exists(22)
            assert result is True

    def test_get_voice_path_exists(self, service):
        """Test get_voice_path returns path when file exists."""
        with patch('services.voice.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_paths.voice_path.return_value = mock_path

            result = service.get_voice_path(22)
            assert result == mock_path

    def test_get_voice_path_not_exists(self, service):
        """Test get_voice_path returns None when file doesn't exist."""
        with patch('services.voice.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_paths.voice_path.return_value = mock_path

            result = service.get_voice_path(999)
            assert result is None


class TestVoiceServiceDuration:
    """Test duration-related methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return VoiceService()

    def test_get_audio_duration_mocked(self, service):
        """Test duration extraction with mocked ffprobe."""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "45.123456"
            mock_run.return_value = mock_result

            duration = service.get_audio_duration(Path("/fake/path.mp3"))

            assert duration == pytest.approx(45.123456)
            mock_run.assert_called_once()

    def test_get_audio_duration_error(self, service):
        """Test duration extraction handles errors."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("ffprobe not found")

            with pytest.raises(Exception):
                service.get_audio_duration(Path("/fake/path.mp3"))


class TestVoiceServiceAsync:
    """Test async methods with mocked HTTP responses."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return VoiceService()

    @pytest.mark.asyncio
    async def test_generate_voice_mocked(self, service):
        """Test voice generation with mocked response."""
        mock_response = MagicMock()
        mock_response.content = b"fake audio data" * 1000  # Fake MP3 bytes

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_voice_path = Path(tmpdir) / "22_voice.mp3"

            with patch.object(service, '_post', new_callable=AsyncMock) as mock_post, \
                 patch('services.voice.asset_paths') as mock_paths, \
                 patch.object(service, 'get_audio_duration', return_value=45.5):

                mock_post.return_value = mock_response
                mock_paths.voice_path.return_value = mock_voice_path

                request = VoiceRequest(
                    script_id=22,
                    text="Hello, this is a test voice generation."
                )
                result = await service.generate_voice(request)

                assert result.script_id == 22
                assert result.duration_seconds == 45.5
                assert result.file_size_bytes > 0
                assert mock_voice_path.exists()

                # Verify ElevenLabs API was called
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert "elevenlabs.io" in call_args[0][0]
                assert "text-to-speech" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_generate_voice_error_handling(self, service):
        """Test that generation errors are raised."""
        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("ElevenLabs API error")

            request = VoiceRequest(script_id=22, text="Test text")

            with pytest.raises(Exception) as exc_info:
                await service.generate_voice(request)

            assert "ElevenLabs API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_voices_mocked(self, service):
        """Test listing voices with mocked response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "voices": [
                {"voice_id": "voice1", "name": "Rachel"},
                {"voice_id": "voice2", "name": "Adam"}
            ]
        }

        with patch.object(service, '_get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await service.list_voices()

            assert "voices" in result
            assert len(result["voices"]) == 2

    @pytest.mark.asyncio
    async def test_get_voice_info_mocked(self, service):
        """Test getting voice info with mocked response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "voice_id": "custom_voice_123",
            "name": "Beth Clone",
            "category": "cloned"
        }

        with patch.object(service, '_get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await service.get_voice_info("custom_voice_123")

            assert result["voice_id"] == "custom_voice_123"
            assert result["name"] == "Beth Clone"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
