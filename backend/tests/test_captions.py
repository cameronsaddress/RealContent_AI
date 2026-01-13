"""
Tests for CaptionService - uses mocked responses, no real API calls.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from services.captions import (
    CaptionService,
    CaptionRequest,
    CaptionResult,
    CaptionSegment,
    TranscriptionResult
)


class TestCaptionSegment:
    """Test CaptionSegment model."""

    def test_segment(self):
        """Test segment model."""
        segment = CaptionSegment(
            start=5.5,
            end=10.2,
            text="Hello world"
        )
        assert segment.start == 5.5
        assert segment.end == 10.2
        assert segment.text == "Hello world"


class TestTranscriptionResult:
    """Test TranscriptionResult model."""

    def test_result(self):
        """Test result model."""
        result = TranscriptionResult(
            script_id=22,
            segments=[
                CaptionSegment(start=0.0, end=3.0, text="First segment"),
                CaptionSegment(start=3.0, end=6.0, text="Second segment")
            ],
            full_text="First segment Second segment",
            language="en",
            duration=6.0
        )
        assert len(result.segments) == 2
        assert result.duration == 6.0


class TestCaptionRequest:
    """Test CaptionRequest model."""

    def test_defaults(self):
        """Test default values."""
        request = CaptionRequest(
            script_id=22,
            video_path="/app/assets/output/22_combined.mp4"
        )
        assert request.font_name == "Arial"
        assert request.font_size == 24
        assert request.font_color == "white"
        assert request.position == "bottom"
        assert request.margin_v == 50

    def test_custom_values(self):
        """Test custom values."""
        request = CaptionRequest(
            script_id=22,
            video_path="/app/assets/output/22_combined.mp4",
            srt_path="/custom/path.srt",
            font_name="Roboto",
            font_size=32,
            position="center"
        )
        assert request.font_name == "Roboto"
        assert request.font_size == 32
        assert request.position == "center"


class TestCaptionResult:
    """Test CaptionResult model."""

    def test_success_result(self):
        """Test successful result."""
        result = CaptionResult(
            script_id=22,
            output_path="/app/assets/output/22_final.mp4",
            srt_path="/app/assets/captions/22.srt",
            duration_seconds=65.5,
            file_size_bytes=30000000,
            success=True
        )
        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        """Test failed result."""
        result = CaptionResult(
            script_id=22,
            output_path="/app/assets/output/22_final.mp4",
            srt_path="/app/assets/captions/22.srt",
            duration_seconds=0,
            file_size_bytes=0,
            success=False,
            error="Subtitle parsing failed"
        )
        assert result.success is False
        assert "parsing" in result.error


class TestCaptionService:
    """Test CaptionService methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return CaptionService()

    def test_format_srt_time(self, service):
        """Test SRT time formatting."""
        # 0 seconds
        assert service._format_srt_time(0) == "00:00:00,000"

        # 5.5 seconds
        assert service._format_srt_time(5.5) == "00:00:05,500"

        # 1 minute 30.123 seconds
        assert service._format_srt_time(90.123) == "00:01:30,123"

        # 1 hour 5 minutes 30.5 seconds
        assert service._format_srt_time(3930.5) == "01:05:30,500"

    def test_format_ass_time(self, service):
        """Test ASS time formatting."""
        # 0 seconds
        assert service._format_ass_time(0) == "0:00:00.00"

        # 5.5 seconds
        assert service._format_ass_time(5.5) == "0:00:05.50"

        # 1 minute 30.12 seconds
        assert service._format_ass_time(90.12) == "0:01:30.12"

    def test_parse_whisper_response(self, service):
        """Test parsing Whisper API response."""
        response = {
            "text": "Hello world. This is a test.",
            "language": "en",
            "duration": 5.5,
            "segments": [
                {"start": 0.0, "end": 2.5, "text": " Hello world."},
                {"start": 2.5, "end": 5.5, "text": " This is a test."}
            ]
        }

        result = service.parse_whisper_response(22, response)

        assert result.script_id == 22
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello world."
        assert result.segments[1].text == "This is a test."
        assert result.full_text == "Hello world. This is a test."
        assert result.duration == 5.5

    def test_generate_srt(self, service):
        """Test SRT file generation."""
        result = TranscriptionResult(
            script_id=22,
            segments=[
                CaptionSegment(start=0.0, end=3.0, text="First line"),
                CaptionSegment(start=3.0, end=6.0, text="Second line")
            ],
            full_text="First line Second line",
            duration=6.0
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.srt"

            srt_path = service.generate_srt(result, output_path)

            assert srt_path.exists()
            content = srt_path.read_text()

            assert "1" in content
            assert "00:00:00,000 --> 00:00:03,000" in content
            assert "First line" in content
            assert "2" in content
            assert "Second line" in content

    def test_generate_ass(self, service):
        """Test ASS file generation."""
        result = TranscriptionResult(
            script_id=22,
            segments=[
                CaptionSegment(start=0.0, end=2.0, text="Hello world")
            ],
            full_text="Hello world",
            duration=2.0
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.ass"

            ass_path = service.generate_ass(result, output_path)

            assert ass_path.exists()
            content = ass_path.read_text()

            assert "[Script Info]" in content
            assert "[V4+ Styles]" in content
            assert "[Events]" in content
            assert "Dialogue:" in content
            assert "\\k" in content  # Karaoke timing

    def test_build_burn_command(self, service):
        """Test FFmpeg burn command generation."""
        request = CaptionRequest(
            script_id=22,
            video_path="/app/assets/output/22_combined.mp4",
            srt_path="/app/assets/captions/22.srt",
            font_name="Arial",
            font_size=24
        )

        with patch('services.captions.asset_paths') as mock_paths:
            mock_paths.final_path.return_value = Path("/app/assets/output/22_final.mp4")

            cmd = service.build_burn_command(request)

            assert cmd[0] == "ffmpeg"
            assert "-y" in cmd
            assert "/app/assets/output/22_combined.mp4" in cmd
            assert "-vf" in cmd

            # Check filter contains subtitle info
            vf_idx = cmd.index("-vf") + 1
            assert "subtitles" in cmd[vf_idx]
            assert "Arial" in cmd[vf_idx]

    def test_build_ass_burn_command(self, service):
        """Test ASS burn command generation."""
        with patch('services.captions.asset_paths') as mock_paths:
            mock_paths.final_path.return_value = Path("/app/assets/output/22_final.mp4")
            mock_paths.ass_path.return_value = Path("/app/assets/captions/22.ass")

            cmd = service.build_ass_burn_command(22, "/app/assets/output/22_combined.mp4")

            assert cmd[0] == "ffmpeg"
            assert "-vf" in cmd

            vf_idx = cmd.index("-vf") + 1
            assert "ass=" in cmd[vf_idx]

    def test_srt_exists(self, service):
        """Test SRT existence check."""
        with patch('services.captions.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_stat = MagicMock()
            mock_stat.st_size = 1000
            mock_path.stat.return_value = mock_stat
            mock_paths.srt_path.return_value = mock_path

            assert service.srt_exists(22) is True

    def test_final_video_exists(self, service):
        """Test final video existence check."""
        with patch('services.captions.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_stat = MagicMock()
            mock_stat.st_size = 30000000
            mock_path.stat.return_value = mock_stat
            mock_paths.final_path.return_value = mock_path

            assert service.final_video_exists(22) is True


class TestCaptionServiceAsync:
    """Test async methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return CaptionService()

    @pytest.mark.asyncio
    async def test_transcribe_audio_mocked(self, service):
        """Test transcription with mocked Whisper response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "Hello world. This is a test.",
            "language": "en",
            "duration": 5.5,
            "segments": [
                {"start": 0.0, "end": 2.5, "text": " Hello world."},
                {"start": 2.5, "end": 5.5, "text": " This is a test."}
            ]
        }

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post, \
             patch('services.captions.asset_paths') as mock_paths, \
             patch('builtins.open', MagicMock()):

            mock_post.return_value = mock_response
            mock_audio_path = MagicMock()
            mock_audio_path.exists.return_value = True
            mock_paths.voice_path.return_value = mock_audio_path

            result = await service.transcribe_audio(22)

            assert result.script_id == 22
            assert len(result.segments) == 2
            assert result.duration == 5.5

    @pytest.mark.asyncio
    async def test_burn_captions_success(self, service):
        """Test successful caption burning."""
        request = CaptionRequest(
            script_id=22,
            video_path="/app/assets/output/22_combined.mp4"
        )

        with patch('services.captions.asset_paths') as mock_paths, \
             patch('subprocess.run') as mock_run, \
             patch.object(service, '_get_duration', return_value=65.5):

            mock_output_path = MagicMock()
            mock_output_path.parent.mkdir = MagicMock()
            mock_stat = MagicMock()
            mock_stat.st_size = 30000000
            mock_output_path.stat.return_value = mock_stat
            mock_paths.final_path.return_value = mock_output_path
            mock_paths.srt_path.return_value = Path("/app/assets/captions/22.srt")

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = await service.burn_captions(request)

            assert result.success is True
            assert result.duration_seconds == 65.5
            assert result.file_size_bytes == 30000000

    @pytest.mark.asyncio
    async def test_burn_captions_ffmpeg_error(self, service):
        """Test caption burning with FFmpeg error."""
        request = CaptionRequest(
            script_id=22,
            video_path="/app/assets/output/22_combined.mp4"
        )

        with patch('services.captions.asset_paths') as mock_paths, \
             patch('subprocess.run') as mock_run:

            mock_output_path = MagicMock()
            mock_output_path.parent.mkdir = MagicMock()
            mock_paths.final_path.return_value = mock_output_path
            mock_paths.srt_path.return_value = Path("/app/assets/captions/22.srt")

            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Subtitle parsing error"
            mock_run.return_value = mock_result

            result = await service.burn_captions(request)

            assert result.success is False
            assert "parsing" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
