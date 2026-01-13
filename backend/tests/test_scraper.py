"""
Tests for ScraperService - uses mocked responses, no real API calls.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from services.scraper import (
    ScraperService,
    ScrapeParams,
    TrendItem,
    ScrapeResult
)


class TestScrapeParams:
    """Test ScrapeParams model."""

    def test_defaults(self):
        """Test default values."""
        params = ScrapeParams()
        assert params.niche == "real estate"
        assert params.platforms == ["tiktok", "instagram"]
        assert params.hashtags == ["realestate", "homebuying", "realtor"]
        assert params.results_per_platform == 20

    def test_custom_values(self):
        """Test custom values."""
        params = ScrapeParams(
            niche="luxury homes",
            platforms=["youtube"],
            hashtags=["luxuryhomes", "mansion"],
            results_per_platform=10
        )
        assert params.niche == "luxury homes"
        assert params.platforms == ["youtube"]
        assert params.hashtags == ["luxuryhomes", "mansion"]
        assert params.results_per_platform == 10


class TestTrendItem:
    """Test TrendItem model."""

    def test_minimal_item(self):
        """Test with minimal required fields."""
        item = TrendItem(
            platform="tiktok",
            url="https://tiktok.com/video/123",
            title="Test video"
        )
        assert item.platform == "tiktok"
        assert item.url == "https://tiktok.com/video/123"
        assert item.title == "Test video"
        assert item.views == 0
        assert item.likes == 0
        assert item.shares == 0
        assert item.author == ""
        assert item.hashtags == []
        assert item.viral_score is None
        assert item.pillar is None

    def test_full_item(self):
        """Test with all fields."""
        item = TrendItem(
            platform="instagram",
            url="https://instagram.com/reel/123",
            title="Amazing content",
            views=100000,
            likes=5000,
            shares=500,
            author="creator123",
            hashtags=["realestate", "homes"],
            viral_score=9,
            pillar="educational_tips",
            suggested_hook="You won't believe this...",
            why_viral="Great hook and relatable content"
        )
        assert item.views == 100000
        assert item.viral_score == 9
        assert item.pillar == "educational_tips"


class TestScrapeResult:
    """Test ScrapeResult model."""

    def test_success_result(self):
        """Test successful scrape result."""
        result = ScrapeResult(
            success=True,
            niche="real estate",
            scraped_at="2025-01-12T10:00:00",
            total_scraped=20,
            analyzed_count=10,
            saved_count=10,
            message="Scraped 20 trends, saved 10",
            trends=[{"platform": "tiktok", "url": "test", "title": "test"}]
        )
        assert result.success is True
        assert result.total_scraped == 20
        assert len(result.trends) == 1


class TestScraperService:
    """Test ScraperService methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return ScraperService()

    def test_parse_params(self, service):
        """Test parameter parsing."""
        raw = {
            "niche": "commercial",
            "platforms": ["tiktok"],
            "hashtags": ["commercial", "office"],
            "results_per_platform": 15
        }
        params = service.parse_params(raw)
        assert params.niche == "commercial"
        assert params.platforms == ["tiktok"]
        assert params.results_per_platform == 15

    def test_normalize_tiktok_item(self, service):
        """Test TikTok normalization."""
        raw_item = {
            "webVideoUrl": "https://tiktok.com/video/123",
            "text": "Great real estate tip! #realestate #homebuying",
            "playCount": 50000,
            "diggCount": 2000,
            "shareCount": 100,
            "authorMeta": {"name": "realtor_jane"},
            "hashtags": [{"name": "realestate"}, {"name": "homebuying"}]
        }
        item = service._normalize_tiktok_item(raw_item)
        assert item.platform == "tiktok"
        assert item.url == "https://tiktok.com/video/123"
        assert item.views == 50000
        assert item.likes == 2000
        assert item.shares == 100
        assert item.author == "realtor_jane"
        assert item.hashtags == ["realestate", "homebuying"]

    def test_normalize_instagram_item(self, service):
        """Test Instagram normalization."""
        raw_item = {
            "url": "https://instagram.com/reel/123",
            "caption": "Check out this amazing home! #luxuryhomes",
            "videoViewCount": 75000,
            "likesCount": 3000,
            "ownerUsername": "luxury_agent",
            "hashtags": ["luxuryhomes", "dreamhome"]
        }
        item = service._normalize_instagram_item(raw_item)
        assert item.platform == "instagram"
        assert item.url == "https://instagram.com/reel/123"
        assert item.views == 75000
        assert item.likes == 3000
        assert item.shares == 0  # Instagram doesn't expose shares
        assert item.author == "luxury_agent"
        assert item.hashtags == ["luxuryhomes", "dreamhome"]

    def test_normalize_youtube_item(self, service):
        """Test YouTube normalization."""
        raw_item = {
            "url": "https://youtube.com/shorts/abc123",
            "title": "Top 5 Home Buying Tips",
            "viewCount": 100000,
            "likes": 5000,
            "channelName": "RealEstateGuru"
        }
        item = service._normalize_youtube_item(raw_item)
        assert item.platform == "youtube"
        assert item.url == "https://youtube.com/shorts/abc123"
        assert item.title == "Top 5 Home Buying Tips"
        assert item.views == 100000
        assert item.likes == 5000
        assert item.author == "RealEstateGuru"
        assert item.hashtags == []  # YouTube doesn't use hashtags the same way

    def test_normalize_all_results_deduplication(self, service):
        """Test deduplication and sorting."""
        items = [
            TrendItem(platform="tiktok", url="https://example.com/1", title="Video 1", views=1000, likes=100),
            TrendItem(platform="tiktok", url="https://example.com/1", title="Video 1 duplicate", views=1000, likes=100),
            TrendItem(platform="instagram", url="https://example.com/2", title="Video 2", views=5000, likes=500),
            TrendItem(platform="youtube", url="https://example.com/3", title="Video 3", views=2000, likes=200),
        ]
        result = service.normalize_all_results(items)

        # Should have 3 unique items (1 duplicate removed)
        assert len(result) == 3

        # Should be sorted by engagement (views + likes) descending
        assert result[0].url == "https://example.com/2"  # 5500 engagement
        assert result[1].url == "https://example.com/3"  # 2200 engagement
        assert result[2].url == "https://example.com/1"  # 1100 engagement

    def test_normalize_all_results_empty_url(self, service):
        """Test that items with empty URLs are filtered."""
        items = [
            TrendItem(platform="tiktok", url="", title="No URL", views=1000, likes=100),
            TrendItem(platform="tiktok", url="https://example.com/valid", title="Valid", views=500, likes=50),
        ]
        result = service.normalize_all_results(items)
        assert len(result) == 1
        assert result[0].url == "https://example.com/valid"

    def test_format_response(self, service):
        """Test response formatting."""
        items = [
            TrendItem(
                platform="tiktok",
                url="https://example.com/1",
                title="Test",
                views=1000,
                likes=100,
                viral_score=8,
                pillar="educational_tips"
            )
        ]
        params = ScrapeParams(niche="real estate")
        result = service.format_response(items, params, saved_count=1)

        assert result.success is True
        assert result.niche == "real estate"
        assert result.total_scraped == 1
        assert result.analyzed_count == 1
        assert result.saved_count == 1
        assert "Scraped 1 trends" in result.message
        assert "saved 1" in result.message
        assert len(result.trends) == 1
        assert result.trends[0]["platform"] == "tiktok"


class TestScraperServiceAsync:
    """Test async methods with mocked HTTP responses."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return ScraperService()

    @pytest.mark.asyncio
    async def test_scrape_tiktok_mocked(self, service):
        """Test TikTok scraping with mocked response."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "webVideoUrl": "https://tiktok.com/video/1",
                "text": "Real estate tip #1",
                "playCount": 10000,
                "diggCount": 500,
                "shareCount": 50,
                "authorMeta": {"name": "agent1"},
                "hashtags": [{"name": "realestate"}]
            },
            {
                "webVideoUrl": "https://tiktok.com/video/2",
                "text": "Real estate tip #2",
                "playCount": 20000,
                "diggCount": 1000,
                "shareCount": 100,
                "authorMeta": {"name": "agent2"},
                "hashtags": [{"name": "homebuying"}]
            }
        ]

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await service._scrape_tiktok(["realestate"], limit=5)

            assert len(result) == 2
            assert result[0].platform == "tiktok"
            assert result[0].views == 10000
            assert result[1].views == 20000

            # Verify API was called correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "clockworks~free-tiktok-scraper" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_scrape_platforms_parallel(self, service):
        """Test parallel scraping of multiple platforms."""
        # Mock responses for each platform
        tiktok_response = MagicMock()
        tiktok_response.json.return_value = [
            {"webVideoUrl": "https://tiktok.com/1", "text": "TikTok", "playCount": 1000, "diggCount": 100, "shareCount": 10, "authorMeta": {"name": "t"}, "hashtags": []}
        ]

        instagram_response = MagicMock()
        instagram_response.json.return_value = [
            {"url": "https://instagram.com/1", "caption": "Instagram", "videoViewCount": 2000, "likesCount": 200, "ownerUsername": "i", "hashtags": []}
        ]

        async def mock_post(url, **kwargs):
            if "tiktok" in url:
                return tiktok_response
            elif "instagram" in url:
                return instagram_response
            raise ValueError(f"Unexpected URL: {url}")

        with patch.object(service, '_post', side_effect=mock_post):
            params = ScrapeParams(platforms=["tiktok", "instagram"], results_per_platform=5)
            result = await service.scrape_platforms(params)

            assert len(result) == 2
            platforms = {item.platform for item in result}
            assert "tiktok" in platforms
            assert "instagram" in platforms

    @pytest.mark.asyncio
    async def test_scrape_platforms_handles_errors(self, service):
        """Test that platform errors don't break entire scrape."""
        tiktok_response = MagicMock()
        tiktok_response.json.return_value = [
            {"webVideoUrl": "https://tiktok.com/1", "text": "TikTok", "playCount": 1000, "diggCount": 100, "shareCount": 10, "authorMeta": {"name": "t"}, "hashtags": []}
        ]

        async def mock_post(url, **kwargs):
            if "tiktok" in url:
                return tiktok_response
            elif "instagram" in url:
                raise Exception("Instagram API error")
            raise ValueError(f"Unexpected URL: {url}")

        with patch.object(service, '_post', side_effect=mock_post):
            params = ScrapeParams(platforms=["tiktok", "instagram"], results_per_platform=5)
            result = await service.scrape_platforms(params)

            # Should still get TikTok results despite Instagram failure
            assert len(result) == 1
            assert result[0].platform == "tiktok"

    @pytest.mark.asyncio
    async def test_analyze_trends_mocked(self, service):
        """Test LLM analysis with mocked response."""
        items = [
            TrendItem(platform="tiktok", url="https://example.com/1", title="Test video", views=10000, likes=500)
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "trends": [{
                            "viral_score": 8,
                            "pillar": "educational_tips",
                            "suggested_hook": "Want to know the secret?",
                            "why_viral": "Great hook and valuable content"
                        }]
                    })
                }
            }]
        }

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await service.analyze_trends(items, "real estate")

            assert len(result) == 1
            assert result[0].viral_score == 8
            assert result[0].pillar == "educational_tips"
            assert result[0].suggested_hook == "Want to know the secret?"

            # Verify OpenRouter was called
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "openrouter.ai" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_analyze_trends_handles_errors(self, service):
        """Test that LLM errors return items with default values."""
        items = [
            TrendItem(platform="tiktok", url="https://example.com/1", title="Test", views=1000, likes=100)
        ]

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("API error")

            result = await service.analyze_trends(items, "real estate")

            # Should return items with default values
            assert len(result) == 1
            assert result[0].viral_score == 7  # Default
            assert result[0].pillar == "educational_tips"  # Default

    @pytest.mark.asyncio
    async def test_analyze_trends_empty_list(self, service):
        """Test analysis with empty list."""
        result = await service.analyze_trends([], "real estate")
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
