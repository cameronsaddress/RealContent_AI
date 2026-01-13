"""
Tests for AvatarService - uses mocked responses, no real API calls.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from services.avatar import (
    AvatarService,
    AvatarRequest,
    AvatarStatus,
    AvatarResult,
    AvatarType
)


class TestAvatarRequest:
    """Test AvatarRequest model."""

    def test_defaults(self):
        """Test default values."""
        request = AvatarRequest(
            script_id=22,
            audio_url="https://example.com/audio.mp3"
        )
        assert request.script_id == 22
        assert request.avatar_id is None  # Will use settings default
        assert request.avatar_type == AvatarType.TALKING_PHOTO
        assert request.greenscreen_enabled is True
        assert request.greenscreen_color == "#00FF00"
        assert request.aspect_ratio == "9:16"
        assert request.test_mode is False

    def test_custom_values(self):
        """Test custom values."""
        request = AvatarRequest(
            script_id=30,
            audio_url="https://example.com/audio.mp3",
            avatar_id="custom_avatar_123",
            avatar_type=AvatarType.VIDEO_AVATAR,
            greenscreen_enabled=False,
            aspect_ratio="16:9",
            test_mode=True
        )
        assert request.avatar_id == "custom_avatar_123"
        assert request.avatar_type == AvatarType.VIDEO_AVATAR
        assert request.greenscreen_enabled is False
        assert request.test_mode is True


class TestAvatarStatus:
    """Test AvatarStatus model."""

    def test_pending_status(self):
        """Test pending status."""
        status = AvatarStatus(
            video_id="vid_123",
            status="processing",
            progress=0.5
        )
        assert status.status == "processing"
        assert status.video_url is None
        assert status.progress == 0.5

    def test_completed_status(self):
        """Test completed status."""
        status = AvatarStatus(
            video_id="vid_123",
            status="completed",
            video_url="https://heygen.com/video.mp4"
        )
        assert status.status == "completed"
        assert status.video_url is not None

    def test_failed_status(self):
        """Test failed status."""
        status = AvatarStatus(
            video_id="vid_123",
            status="failed",
            error="Audio processing error"
        )
        assert status.status == "failed"
        assert "Audio" in status.error


class TestAvatarResult:
    """Test AvatarResult model."""

    def test_result(self):
        """Test result model."""
        result = AvatarResult(
            script_id=22,
            video_id="vid_123",
            video_path="/app/assets/avatar/22_avatar.mp4",
            duration_seconds=65.5,
            file_size_bytes=15000000
        )
        assert result.script_id == 22
        assert result.duration_seconds == 65.5


class TestAvatarService:
    """Test AvatarService methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return AvatarService()

    def test_get_dimensions_portrait(self, service):
        """Test portrait dimensions."""
        dims = service._get_dimensions("9:16")
        assert dims["width"] == 1080
        assert dims["height"] == 1920

    def test_get_dimensions_landscape(self, service):
        """Test landscape dimensions."""
        dims = service._get_dimensions("16:9")
        assert dims["width"] == 1920
        assert dims["height"] == 1080

    def test_get_dimensions_square(self, service):
        """Test square dimensions."""
        dims = service._get_dimensions("1:1")
        assert dims["width"] == 1080
        assert dims["height"] == 1080

    def test_get_dimensions_default(self, service):
        """Test default dimensions for unknown aspect ratio."""
        dims = service._get_dimensions("4:3")
        # Should default to 9:16
        assert dims["width"] == 1080
        assert dims["height"] == 1920

    def test_avatar_exists_false(self, service):
        """Test avatar_exists returns False for missing file."""
        with patch('services.avatar.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_paths.avatar_path.return_value = mock_path

            result = service.avatar_exists(999)
            assert result is False

    def test_avatar_exists_true(self, service):
        """Test avatar_exists returns True for valid file."""
        with patch('services.avatar.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_stat = MagicMock()
            mock_stat.st_size = 15000000
            mock_path.stat.return_value = mock_stat
            mock_paths.avatar_path.return_value = mock_path

            result = service.avatar_exists(22)
            assert result is True

    def test_get_avatar_path_exists(self, service):
        """Test get_avatar_path returns path when file exists."""
        with patch('services.avatar.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_paths.avatar_path.return_value = mock_path

            result = service.get_avatar_path(22)
            assert result == mock_path

    def test_get_avatar_path_not_exists(self, service):
        """Test get_avatar_path returns None when file doesn't exist."""
        with patch('services.avatar.asset_paths') as mock_paths:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_paths.avatar_path.return_value = mock_path

            result = service.get_avatar_path(999)
            assert result is None


class TestAvatarServiceAsync:
    """Test async methods with mocked HTTP responses."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return AvatarService()

    @pytest.mark.asyncio
    async def test_create_video_talking_photo(self, service):
        """Test video creation with talking photo avatar."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"video_id": "vid_abc123"}
        }

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            request = AvatarRequest(
                script_id=22,
                audio_url="https://example.com/audio.mp3",
                avatar_type=AvatarType.TALKING_PHOTO
            )
            video_id = await service.create_video(request)

            assert video_id == "vid_abc123"
            mock_post.assert_called_once()

            # Verify payload structure
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert "video_inputs" in payload
            assert payload["video_inputs"][0]["character"]["type"] == "talking_photo"

    @pytest.mark.asyncio
    async def test_create_video_with_greenscreen(self, service):
        """Test video creation with greenscreen enabled."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"video_id": "vid_abc123"}
        }

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            request = AvatarRequest(
                script_id=22,
                audio_url="https://example.com/audio.mp3",
                greenscreen_enabled=True,
                greenscreen_color="#00FF00"
            )
            await service.create_video(request)

            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["video_inputs"][0]["background"]["type"] == "color"
            assert payload["video_inputs"][0]["background"]["value"] == "#00FF00"

    @pytest.mark.asyncio
    async def test_get_status_processing(self, service):
        """Test status check for processing video."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "status": "processing",
                "progress": 0.45
            }
        }

        with patch.object(service, '_get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            status = await service.get_status("vid_123")

            assert status.video_id == "vid_123"
            assert status.status == "processing"
            assert status.progress == 0.45

    @pytest.mark.asyncio
    async def test_get_status_completed(self, service):
        """Test status check for completed video."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "status": "completed",
                "video_url": "https://heygen.com/video/vid_123.mp4"
            }
        }

        with patch.object(service, '_get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            status = await service.get_status("vid_123")

            assert status.status == "completed"
            assert status.video_url == "https://heygen.com/video/vid_123.mp4"

    @pytest.mark.asyncio
    async def test_poll_until_complete_success(self, service):
        """Test polling until completion."""
        statuses = [
            AvatarStatus(video_id="vid_123", status="processing", progress=0.3),
            AvatarStatus(video_id="vid_123", status="processing", progress=0.7),
            AvatarStatus(video_id="vid_123", status="completed", video_url="https://example.com/video.mp4")
        ]
        call_count = 0

        async def mock_get_status(video_id):
            nonlocal call_count
            result = statuses[min(call_count, len(statuses) - 1)]
            call_count += 1
            return result

        with patch.object(service, 'get_status', side_effect=mock_get_status), \
             patch('asyncio.sleep', new_callable=AsyncMock):

            status = await service.poll_until_complete("vid_123", max_retries=10, poll_interval=1)

            assert status.status == "completed"
            assert status.video_url is not None

    @pytest.mark.asyncio
    async def test_poll_until_complete_failure(self, service):
        """Test polling with failure."""
        failed_status = AvatarStatus(
            video_id="vid_123",
            status="failed",
            error="Audio encoding failed"
        )

        with patch.object(service, 'get_status', new_callable=AsyncMock) as mock_status:
            mock_status.return_value = failed_status

            with pytest.raises(Exception) as exc_info:
                await service.poll_until_complete("vid_123", max_retries=1)

            assert "Audio encoding failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_poll_until_complete_timeout(self, service):
        """Test polling timeout."""
        processing_status = AvatarStatus(
            video_id="vid_123",
            status="processing",
            progress=0.5
        )

        with patch.object(service, 'get_status', new_callable=AsyncMock) as mock_status, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            mock_status.return_value = processing_status

            with pytest.raises(TimeoutError):
                await service.poll_until_complete("vid_123", max_retries=2, poll_interval=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
