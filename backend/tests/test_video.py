"""
Tests for VideoService - uses mocked responses, no real FFmpeg calls.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from services.video import (
    VideoService,
    VideoComposeRequest,
    VideoResult
)


class TestVideoComposeRequest:
    """Test VideoComposeRequest model."""

    def test_defaults(self):
        """Test default values."""
        request = VideoComposeRequest(script_id=22)
        assert request.script_id == 22
        assert request.greenscreen_enabled is True
        assert request.greenscreen_color == "#00FF00"
        assert request.output_width == 1080
        assert request.output_height == 1920

    def test_custom_values(self):
        """Test custom values."""
        request = VideoComposeRequest(
            script_id=30,
            avatar_path="/app/assets/avatar/30_avatar.mp4",
            source_video_path="/app/assets/videos/30_source.mp4",
            audio_path="/app/assets/audio/30_voice.mp3",
            greenscreen_enabled=True,
            greenscreen_color="#0000FF",
            output_width=1920,
            output_height=1080
        )
        assert request.avatar_path == "/app/assets/avatar/30_avatar.mp4"
        assert request.greenscreen_color == "#0000FF"


class TestVideoResult:
    """Test VideoResult model."""

    def test_success_result(self):
        """Test successful result."""
        result = VideoResult(
            script_id=22,
            output_path="/app/assets/combined/22_combined.mp4",
            duration_seconds=65.5,
            file_size_bytes=25000000,
            success=True
        )
        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        """Test failed result."""
        result = VideoResult(
            script_id=22,
            output_path="/app/assets/combined/22_combined.mp4",
            duration_seconds=0,
            file_size_bytes=0,
            success=False,
            error="FFmpeg encoding failed"
        )
        assert result.success is False
        assert "encoding" in result.error


class TestVideoService:
    """Test VideoService methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return VideoService()

    def test_build_chromakey_command(self, service):
        """Test chromakey command generation."""
        request = VideoComposeRequest(
            script_id=22,
            avatar_path="/app/assets/avatar/22_avatar.mp4",
            source_video_path="/app/assets/videos/22_source.mp4",
            audio_path="/app/assets/audio/22_voice.mp3",
            greenscreen_enabled=True,
            greenscreen_color="#00FF00"
        )

        with patch('services.video.asset_paths') as mock_paths:
            mock_paths.combined_path.return_value = Path("/app/assets/combined/22_combined.mp4")

            cmd = service.build_chromakey_command(request)

            # Verify command structure
            assert cmd[0] == "ffmpeg"
            assert "-y" in cmd
            assert "/app/assets/videos/22_source.mp4" in cmd
            assert "/app/assets/avatar/22_avatar.mp4" in cmd
            assert "/app/assets/audio/22_voice.mp3" in cmd
            assert "-filter_complex" in cmd

            # Check chromakey filter is in command
            filter_idx = cmd.index("-filter_complex") + 1
            assert "chromakey" in cmd[filter_idx]
            assert "00FF00" in cmd[filter_idx]

    def test_build_simple_compose_command(self, service):
        """Test simple compose command generation."""
        request = VideoComposeRequest(
            script_id=22,
            avatar_path="/app/assets/avatar/22_avatar.mp4",
            greenscreen_enabled=False
        )

        with patch('services.video.asset_paths') as mock_paths:
            mock_paths.combined_path.return_value = Path("/app/assets/combined/22_combined.mp4")

            cmd = service.build_simple_compose_command(request)

            assert cmd[0] == "ffmpeg"
            assert "-y" in cmd
            assert "/app/assets/avatar/22_avatar.mp4" in cmd
            assert "-filter_complex" not in cmd

    def test_build_simple_compose_with_separate_audio(self, service):
        """Test simple compose with separate audio track."""
        request = VideoComposeRequest(
            script_id=22,
            avatar_path="/app/assets/avatar/22_avatar.mp4",
            audio_path="/app/assets/audio/22_voice.mp3",
            greenscreen_enabled=False
        )

        with patch('services.video.asset_paths') as mock_paths:
            mock_paths.combined_path.return_value = Path("/app/assets/combined/22_combined.mp4")

            cmd = service.build_simple_compose_command(request)

            # Should have both video and audio inputs
            assert "/app/assets/avatar/22_avatar.mp4" in cmd
            assert "/app/assets/audio/22_voice.mp3" in cmd
            # Should map video from first input, audio from second
            assert "-map" in cmd

    def test_get_video_duration_mocked(self, service):
        """Test duration extraction with mocked ffprobe."""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "65.123456"
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            duration = service.get_video_duration(Path("/fake/path.mp4"))

            assert duration == pytest.approx(65.123456)

    def test_get_video_duration_error(self, service):
        """Test duration extraction handles errors gracefully."""
        import subprocess
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe", stderr="ffprobe not found")

            duration = service.get_video_duration(Path("/fake/path.mp4"))

            assert duration == 0.0  # Should return 0 on error

    def test_verify_video_exists_and_valid(self, service):
        """Test video verification with valid file."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_stat = MagicMock()
        mock_stat.st_size = 25000000
        mock_path.stat.return_value = mock_stat

        with patch.object(service, 'get_video_duration', return_value=65.5):
            result = service.verify_video(mock_path)
            assert result is True

    def test_verify_video_not_exists(self, service):
        """Test video verification with missing file."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False

        result = service.verify_video(mock_path)
        assert result is False

    def test_verify_video_empty_file(self, service):
        """Test video verification with empty file."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_stat = MagicMock()
        mock_stat.st_size = 0
        mock_path.stat.return_value = mock_stat

        result = service.verify_video(mock_path)
        assert result is False

    def test_verify_video_zero_duration(self, service):
        """Test video verification with zero duration."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_stat = MagicMock()
        mock_stat.st_size = 25000000
        mock_path.stat.return_value = mock_stat

        with patch.object(service, 'get_video_duration', return_value=0.0):
            result = service.verify_video(mock_path)
            assert result is False

    def test_combined_video_exists(self, service):
        """Test combined_video_exists check."""
        with patch('services.video.asset_paths') as mock_paths, \
             patch.object(service, 'verify_video', return_value=True):

            mock_paths.combined_path.return_value = Path("/app/assets/combined/22.mp4")

            result = service.combined_video_exists(22)
            assert result is True

    def test_get_combined_path_exists(self, service):
        """Test get_combined_path when file exists."""
        with patch('services.video.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_paths.combined_path.return_value = mock_path

            result = service.get_combined_path(22)
            assert result == mock_path

    def test_get_combined_path_not_exists(self, service):
        """Test get_combined_path when file doesn't exist."""
        with patch('services.video.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_paths.combined_path.return_value = mock_path

            result = service.get_combined_path(999)
            assert result is None


class TestVideoServiceAsync:
    """Test async methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return VideoService()

    @pytest.mark.asyncio
    async def test_compose_video_success(self, service):
        """Test successful video composition."""
        request = VideoComposeRequest(
            script_id=22,
            avatar_path="/app/assets/avatar/22_avatar.mp4",
            greenscreen_enabled=False
        )

        with patch('services.video.asset_paths') as mock_paths, \
             patch('subprocess.run') as mock_run, \
             patch.object(service, 'get_video_duration', return_value=65.5):

            mock_output_path = MagicMock()
            mock_output_path.parent.mkdir = MagicMock()
            mock_stat = MagicMock()
            mock_stat.st_size = 25000000
            mock_output_path.stat.return_value = mock_stat
            mock_paths.combined_path.return_value = mock_output_path

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = await service.compose_video(request)

            assert result.success is True
            assert result.duration_seconds == 65.5
            assert result.file_size_bytes == 25000000

    @pytest.mark.asyncio
    async def test_compose_video_ffmpeg_error(self, service):
        """Test video composition with FFmpeg error."""
        request = VideoComposeRequest(
            script_id=22,
            avatar_path="/app/assets/avatar/22_avatar.mp4",
            greenscreen_enabled=False
        )

        with patch('services.video.asset_paths') as mock_paths, \
             patch('subprocess.run') as mock_run:

            mock_output_path = MagicMock()
            mock_output_path.parent.mkdir = MagicMock()
            mock_paths.combined_path.return_value = mock_output_path

            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Error: Invalid input file"
            mock_run.return_value = mock_result

            result = await service.compose_video(request)

            assert result.success is False
            assert "Invalid input" in result.error

    @pytest.mark.asyncio
    async def test_compose_video_timeout(self, service):
        """Test video composition timeout."""
        request = VideoComposeRequest(
            script_id=22,
            avatar_path="/app/assets/avatar/22_avatar.mp4",
            greenscreen_enabled=False
        )

        with patch('services.video.asset_paths') as mock_paths, \
             patch('subprocess.run') as mock_run:

            mock_output_path = MagicMock()
            mock_output_path.parent.mkdir = MagicMock()
            mock_paths.combined_path.return_value = mock_output_path

            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 600)

            result = await service.compose_video(request)

            assert result.success is False
            assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_download_source_video(self, service):
        """Test source video download."""
        mock_response = MagicMock()
        mock_response.content = b"fake video data" * 10000

        with patch.object(service, '_get', new_callable=AsyncMock) as mock_get, \
             patch('services.video.asset_paths') as mock_paths, \
             patch.object(service, 'verify_video', return_value=True), \
             patch('builtins.open', MagicMock()):

            mock_get.return_value = mock_response
            mock_source_path = MagicMock()
            mock_source_path.parent.mkdir = MagicMock()
            mock_paths.source_video_path.return_value = mock_source_path

            result = await service.download_source_video(
                "https://example.com/video.mp4",
                22
            )

            mock_get.assert_called_once()
            assert result == mock_source_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
