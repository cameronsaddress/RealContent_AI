"""
Tests for PublisherService - uses mocked responses, no real API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.publisher import (
    PublisherService,
    PublishRequest,
    PublishResult,
    PublishPlatform
)


class TestPublishRequest:
    """Test PublishRequest model."""

    def test_defaults(self):
        """Test default values."""
        request = PublishRequest(
            script_id=22,
            video_url="https://storage.googleapis.com/bucket/video.mp4",
            caption="Check out this video!"
        )
        assert request.platforms == ["tiktok", "instagram", "youtube"]
        assert request.hashtags == []
        assert request.schedule_time is None

    def test_custom_values(self):
        """Test custom values."""
        request = PublishRequest(
            script_id=22,
            video_url="https://example.com/video.mp4",
            caption="Great content",
            platforms=["tiktok", "instagram"],
            hashtags=["realestate", "homes"],
            schedule_time="2025-01-15T10:00:00Z"
        )
        assert len(request.platforms) == 2
        assert len(request.hashtags) == 2
        assert request.schedule_time is not None


class TestPublishResult:
    """Test PublishResult model."""

    def test_success_result(self):
        """Test successful result."""
        result = PublishResult(
            script_id=22,
            platform="tiktok",
            success=True,
            post_id="post_123",
            post_url="https://tiktok.com/@user/video/123"
        )
        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        """Test failed result."""
        result = PublishResult(
            script_id=22,
            platform="instagram",
            success=False,
            error="Rate limit exceeded"
        )
        assert result.success is False
        assert "Rate limit" in result.error


class TestPublisherService:
    """Test PublisherService methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return PublisherService()

    def test_build_caption_no_hashtags(self, service):
        """Test caption without hashtags."""
        caption = service._build_caption("Check this out!", [])
        assert caption == "Check this out!"

    def test_build_caption_with_hashtags(self, service):
        """Test caption with hashtags."""
        caption = service._build_caption(
            "Check this out!",
            ["realestate", "homes", "luxury"]
        )
        assert "Check this out!" in caption
        assert "#realestate" in caption
        assert "#homes" in caption
        assert "#luxury" in caption
        assert "\n\n" in caption  # Hashtags on new line

    def test_prepare_publish_data_educational(self, service):
        """Test prepare_publish_data for educational pillar."""
        request = service.prepare_publish_data(
            script_id=22,
            video_url="https://example.com/video.mp4",
            script_text="DM me 'TIPS' for more!",
            pillar="educational_tips"
        )

        assert request.script_id == 22
        assert request.video_url == "https://example.com/video.mp4"
        assert "realestate" in request.hashtags
        assert "realestatetips" in request.hashtags
        assert len(request.platforms) == 3

    def test_prepare_publish_data_market_intelligence(self, service):
        """Test prepare_publish_data for market intelligence pillar."""
        request = service.prepare_publish_data(
            script_id=22,
            video_url="https://example.com/video.mp4",
            script_text="Market update!",
            pillar="market_intelligence"
        )

        assert "marketupdate" in request.hashtags
        assert "housingmarket" in request.hashtags

    def test_prepare_publish_data_lifestyle_local(self, service):
        """Test prepare_publish_data for lifestyle local pillar."""
        request = service.prepare_publish_data(
            script_id=22,
            video_url="https://example.com/video.mp4",
            script_text="Love this area!",
            pillar="lifestyle_local"
        )

        assert "idahorealestate" in request.hashtags
        assert "coeurdalene" in request.hashtags

    def test_prepare_publish_data_truncates_long_caption(self, service):
        """Test that long captions are truncated."""
        long_text = "A" * 3000  # Longer than Instagram limit

        request = service.prepare_publish_data(
            script_id=22,
            video_url="https://example.com/video.mp4",
            script_text=long_text,
            pillar="educational_tips"
        )

        assert len(request.caption) <= 2200


class TestPublisherServiceAsync:
    """Test async methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return PublisherService()

    @pytest.mark.asyncio
    async def test_publish_to_platform_success(self, service):
        """Test successful publishing to a platform."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "post_123",
            "url": "https://tiktok.com/@user/video/123"
        }

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            request = PublishRequest(
                script_id=22,
                video_url="https://example.com/video.mp4",
                caption="Great video!",
                platforms=["tiktok"]
            )

            result = await service._publish_to_platform(request, "tiktok")

            assert result.success is True
            assert result.post_id == "post_123"
            assert result.platform == "tiktok"

    @pytest.mark.asyncio
    async def test_publish_multiple_platforms(self, service):
        """Test publishing to multiple platforms."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "post_123",
            "url": "https://platform.com/video/123"
        }

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            request = PublishRequest(
                script_id=22,
                video_url="https://example.com/video.mp4",
                caption="Great video!",
                platforms=["tiktok", "instagram", "youtube"]
            )

            results = await service.publish(request)

            assert len(results) == 3
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_publish_partial_failure(self, service):
        """Test publishing with some platform failures."""
        call_count = 0

        async def mock_publish(request, platform):
            nonlocal call_count
            call_count += 1
            if platform == "instagram":
                raise Exception("Rate limit exceeded")
            return PublishResult(
                script_id=request.script_id,
                platform=platform,
                success=True,
                post_id=f"post_{call_count}"
            )

        with patch.object(service, '_publish_to_platform', side_effect=mock_publish):
            request = PublishRequest(
                script_id=22,
                video_url="https://example.com/video.mp4",
                caption="Great video!",
                platforms=["tiktok", "instagram", "youtube"]
            )

            results = await service.publish(request)

            assert len(results) == 3

            tiktok_result = next(r for r in results if r.platform == "tiktok")
            instagram_result = next(r for r in results if r.platform == "instagram")
            youtube_result = next(r for r in results if r.platform == "youtube")

            assert tiktok_result.success is True
            assert instagram_result.success is False
            assert "Rate limit" in instagram_result.error
            assert youtube_result.success is True

    @pytest.mark.asyncio
    async def test_get_post_status(self, service):
        """Test getting post status."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "post_123",
            "status": "published",
            "views": 1500
        }

        with patch.object(service, '_get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            status = await service.get_post_status("post_123")

            assert status["status"] == "published"
            assert status["views"] == 1500

    @pytest.mark.asyncio
    async def test_list_accounts(self, service):
        """Test listing connected accounts."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "accounts": [
                {"platform": "tiktok", "username": "realtor_beth"},
                {"platform": "instagram", "username": "beth_realty"}
            ]
        }

        with patch.object(service, '_get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            accounts = await service.list_accounts()

            assert len(accounts) == 2
            assert accounts[0]["platform"] == "tiktok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
