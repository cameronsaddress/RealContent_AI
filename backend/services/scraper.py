"""
Scraper service for trend discovery using Apify and LLM analysis.

Uses KEYWORD SEARCH (not hashtags) to find the most viral videos
in your niche, sorted by views/engagement.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from .base import BaseService
from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)


class ScrapeParams(BaseModel):
    """Parameters for trend scraping."""
    niche: str = "real estate"
    platforms: List[str] = ["tiktok", "instagram", "youtube"]
    # Hashtags to search for viral content
    # Use popular niche hashtags that real viral videos use
    hashtags: List[str] = ["realtor", "realestate", "homebuying"]
    results_per_platform: int = 30
    # How many to analyze with LLM (costs money, so we limit)
    analyze_top_n: int = 50
    # Enable 2-phase discovery: first find trending hashtags, then scrape them
    discover_hashtags: bool = False
    # Seed keyword for hashtag discovery (e.g., "realtor")
    seed_keyword: str = ""


class DiscoveredHashtag(BaseModel):
    """A hashtag discovered from viral content."""
    tag: str
    post_count: int = 0
    view_count: int = 0
    frequency: int = 0  # How many top videos used this hashtag


class TrendItem(BaseModel):
    """Normalized trend item from any platform."""
    platform: str
    url: str
    title: str
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    author: str = ""
    author_followers: int = 0
    hashtags: List[str] = []
    duration_seconds: int = 0
    posted_at: Optional[str] = None

    # Added by LLM analysis
    viral_score: Optional[int] = None
    pillar: Optional[str] = None
    suggested_hook: Optional[str] = None
    why_viral: Optional[str] = None


class ScrapeResult(BaseModel):
    """Result of a scrape operation."""
    success: bool
    niche: str
    scraped_at: str
    total_scraped: int
    analyzed_count: int
    saved_count: int = 0
    message: str = ""
    trends: List[Dict[str, Any]] = []


class ScraperService(BaseService):
    """
    Service for scraping trends from social platforms and analyzing with LLM.

    Supports TWO modes:
    1. Direct hashtag search - scrape videos using provided hashtags
    2. Two-phase discovery - first discover what hashtags top videos use, then scrape

    Uses HASHTAG SEARCH to find viral videos:
    - TikTok: clockworks/tiktok-scraper (hashtag mode)
    - Instagram: apify/instagram-scraper (hashtag mode)
    - YouTube: streamers/youtube-scraper (search mode - uses hashtag as keyword)

    Results are sorted by views to get the MOST POPULAR content.
    """

    # Apify actor IDs for scraping (use ~ separator for API URLs)
    APIFY_TIKTOK_ACTOR = "clockworks~tiktok-hashtag-scraper"
    APIFY_INSTAGRAM_ACTOR = "apify~instagram-hashtag-scraper"  # Better for hashtag search
    APIFY_YOUTUBE_ACTOR = "h7iqpunk~youtube-search-scraper"  # Search-based scraper

    # Apify actor for hashtag discovery
    APIFY_TIKTOK_DISCOVER_ACTOR = "clockworks~tiktok-discover-scraper"

    def parse_params(self, raw_input: Dict[str, Any]) -> ScrapeParams:
        """Parse and validate scrape parameters."""
        # Support legacy 'search_queries' parameter by converting to hashtags
        if "search_queries" in raw_input and "hashtags" not in raw_input:
            raw_input["hashtags"] = raw_input.pop("search_queries")
        return ScrapeParams(**raw_input)

    async def discover_trending_hashtags(
        self,
        seed_keyword: str,
        limit: int = 20
    ) -> List[DiscoveredHashtag]:
        """
        PHASE 1: Discover what hashtags top viral videos are using.

        Searches for videos with the seed keyword, extracts their hashtags,
        and ranks by frequency and popularity.

        Args:
            seed_keyword: Base keyword to search (e.g., "realtor")
            limit: How many videos to analyze for hashtag extraction

        Returns:
            List of discovered hashtags sorted by usage frequency
        """
        logger.info(f"Discovering hashtags for seed keyword: {seed_keyword}")

        # Scrape TikTok videos using the seed keyword as a hashtag
        # This gets us real videos that use the term
        try:
            url = f"https://api.apify.com/v2/acts/{self.APIFY_TIKTOK_ACTOR}/run-sync-get-dataset-items"

            response = await self._post(
                url,
                params={"token": settings.APIFY_API_KEY},
                json={
                    "hashtags": [seed_keyword],
                    "resultsPerPage": limit
                },
                timeout=180.0
            )

            items = response.json()
            if not isinstance(items, list):
                logger.warning(f"Discovery returned unexpected format: {type(items)}")
                return []

            # Extract and count hashtags from all videos
            hashtag_stats: Dict[str, Dict[str, int]] = {}

            for item in items:
                video_views = item.get("playCount") or item.get("plays", 0)
                hashtags = item.get("hashtags", [])

                for tag_obj in hashtags:
                    if isinstance(tag_obj, dict):
                        tag = tag_obj.get("name", "").lower()
                    else:
                        tag = str(tag_obj).lower()

                    if not tag or tag == seed_keyword.lower():
                        continue

                    if tag not in hashtag_stats:
                        hashtag_stats[tag] = {"frequency": 0, "total_views": 0}

                    hashtag_stats[tag]["frequency"] += 1
                    hashtag_stats[tag]["total_views"] += video_views

            # Convert to DiscoveredHashtag objects and sort by frequency
            discovered = []
            for tag, stats in hashtag_stats.items():
                discovered.append(DiscoveredHashtag(
                    tag=tag,
                    frequency=stats["frequency"],
                    view_count=stats["total_views"]
                ))

            # Sort by frequency (most common first), then by total views
            discovered.sort(key=lambda x: (x.frequency, x.view_count), reverse=True)

            logger.info(f"Discovered {len(discovered)} unique hashtags from {len(items)} videos")
            logger.info(f"Top 5 hashtags: {[h.tag for h in discovered[:5]]}")

            return discovered

        except Exception as e:
            logger.error(f"Hashtag discovery failed: {e}")
            return []

    async def scrape_with_discovery(self, params: ScrapeParams) -> List[TrendItem]:
        """
        TWO-PHASE SCRAPING:
        1. Discover trending hashtags from seed keyword
        2. Use discovered hashtags to scrape more viral content

        Args:
            params: Scrape parameters with seed_keyword set

        Returns:
            List of trend items from discovered hashtags
        """
        if not params.seed_keyword:
            logger.warning("No seed keyword provided for discovery, using regular scrape")
            return await self.scrape_platforms(params)

        # Phase 1: Discover hashtags
        logger.info(f"Phase 1: Discovering hashtags from '{params.seed_keyword}'")
        discovered = await self.discover_trending_hashtags(
            params.seed_keyword,
            limit=30  # Analyze 30 videos for hashtag patterns
        )

        if not discovered:
            logger.warning(f"No hashtags discovered, falling back to seed keyword")
            params.hashtags = [params.seed_keyword]
        else:
            # Use top 5 most common hashtags (plus seed keyword)
            top_hashtags = [params.seed_keyword] + [h.tag for h in discovered[:5]]
            params.hashtags = list(dict.fromkeys(top_hashtags))[:5]  # Dedupe, limit to 5
            logger.info(f"Phase 2: Scraping with discovered hashtags: {params.hashtags}")

        # Phase 2: Scrape using discovered hashtags
        return await self.scrape_platforms(params)

    async def scrape_platforms(self, params: ScrapeParams) -> List[TrendItem]:
        """
        Scrape all requested platforms in parallel using HASHTAG SEARCH.

        Args:
            params: Scrape parameters with hashtags

        Returns:
            Combined list of trend items from all platforms, sorted by views
        """
        import asyncio

        tasks = []

        if "tiktok" in params.platforms:
            tasks.append(self._scrape_tiktok(params.hashtags, params.results_per_platform))

        if "instagram" in params.platforms:
            tasks.append(self._scrape_instagram(params.hashtags, params.results_per_platform))

        if "youtube" in params.platforms:
            tasks.append(self._scrape_youtube(params.hashtags, params.results_per_platform))

        if not tasks:
            logger.warning("No platforms selected for scraping")
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results, skipping errors
        all_items = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                platform = ["tiktok", "instagram", "youtube"][i] if i < 3 else "unknown"
                logger.error(f"{platform} scrape failed: {result}")
                continue
            all_items.extend(result)
            logger.info(f"Got {len(result)} items from platform")

        return all_items

    async def _scrape_tiktok(self, hashtags: List[str], limit: int) -> List[TrendItem]:
        """
        Scrape TikTok videos using HASHTAG SEARCH.

        Uses clockworks/tiktok-hashtag-scraper with hashtags parameter.
        Returns videos sorted by views (most viral first).
        """
        url = f"https://api.apify.com/v2/acts/{self.APIFY_TIKTOK_ACTOR}/run-sync-get-dataset-items"

        try:
            # Use hashtags parameter with token auth via query params
            response = await self._post(
                url,
                params={"token": settings.APIFY_API_KEY},
                json={
                    "hashtags": hashtags[:3],  # Limit to 3 hashtags to control costs
                    "resultsPerPage": limit
                },
                timeout=180.0
            )

            items = response.json()
            if not isinstance(items, list):
                logger.warning(f"TikTok returned unexpected format: {type(items)}")
                return []

            logger.info(f"TikTok hashtags {hashtags[:3]}: got {len(items)} results")

            # Normalize and return
            normalized = [self._normalize_tiktok_item(item) for item in items]
            # Sort by views descending
            normalized.sort(key=lambda x: x.views, reverse=True)
            return normalized[:limit]

        except Exception as e:
            logger.error(f"TikTok scrape failed: {e}")
            raise

    def _normalize_tiktok_item(self, item: Dict) -> TrendItem:
        """Normalize TikTok item to common format."""
        # Handle different response formats from TikTok scraper
        author_meta = item.get("authorMeta", {}) or {}

        # Convert createTime to string if it's an integer (timestamp)
        create_time = item.get("createTime", "")
        if isinstance(create_time, int):
            create_time = str(create_time)

        return TrendItem(
            platform="tiktok",
            url=item.get("webVideoUrl") or item.get("videoUrl") or "",
            title=item.get("text") or item.get("desc") or "",
            views=item.get("playCount") or item.get("plays", 0),
            likes=item.get("diggCount") or item.get("likes", 0),
            shares=item.get("shareCount") or item.get("shares", 0),
            comments=item.get("commentCount") or item.get("comments", 0),
            author=author_meta.get("name") or item.get("author", ""),
            author_followers=author_meta.get("fans", 0),
            hashtags=[h.get("name", "") for h in item.get("hashtags", []) if isinstance(h, dict)],
            duration_seconds=item.get("videoMeta", {}).get("duration", 0) if isinstance(item.get("videoMeta"), dict) else 0,
            posted_at=create_time
        )

    async def _scrape_instagram(self, hashtags: List[str], limit: int) -> List[TrendItem]:
        """
        Scrape Instagram Reels using HASHTAG SEARCH.

        Uses apify/instagram-hashtag-scraper with hashtags parameter.

        NOTE: Instagram hashtag scraping has limitations:
        - Returns "Recent" posts, not "Top" posts (no API access to Top)
        - View counts for Reels are not publicly available via API
        - We estimate engagement using likes * 100 as a proxy for reach
        - Filter to only include posts with decent engagement (100+ likes)
        """
        url = f"https://api.apify.com/v2/acts/{self.APIFY_INSTAGRAM_ACTOR}/run-sync-get-dataset-items"

        try:
            # Request more results to filter down to quality content
            response = await self._post(
                url,
                params={"token": settings.APIFY_API_KEY},
                json={
                    "hashtags": hashtags[:5],
                    "resultsLimit": limit * 5  # Get more to filter
                },
                timeout=180.0
            )

            items = response.json()
            if not isinstance(items, list):
                logger.warning(f"Instagram returned non-list: {str(items)[:200]}")
                return []

            # Log like distribution for debugging
            all_likes = [i.get("likesCount", 0) for i in items]
            if all_likes:
                logger.info(f"Instagram raw: {len(items)} items, likes range: {min(all_likes)}-{max(all_likes)}, median: {sorted(all_likes)[len(all_likes)//2]}")

            # Filter to ONLY video/reel content with minimum engagement
            # productType "reels" or "clips" = actual Reels
            # type "Video" = video posts
            # Require minimum 10 likes to filter complete spam (0-2 likes)
            video_items = [
                i for i in items
                if (i.get("type") == "Video" or i.get("productType") in ["reels", "clips"])
                and i.get("likesCount", 0) >= 10  # Lower threshold - hashtag search returns recent, not viral
            ]

            logger.info(f"Instagram: got {len(video_items)} reels from {len(items)} total (10+ likes, video only)")

            normalized = [self._normalize_instagram_item(item) for item in video_items]
            # Sort by likes (best proxy for virality since views aren't available)
            normalized.sort(key=lambda x: x.likes, reverse=True)
            return normalized[:limit]

        except Exception as e:
            logger.error(f"Instagram scrape failed: {e}")
            raise

    def _normalize_instagram_item(self, item: Dict) -> TrendItem:
        """Normalize Instagram item to common format (instagram-hashtag-scraper).

        NOTE: Instagram doesn't expose view counts via hashtag API.
        We estimate views as likes * 100 (typical engagement rate ~1%).
        """
        likes = item.get("likesCount", 0)
        # Estimate views: Instagram reels typically get 1-3% like rate
        # So views ≈ likes * 50-100. Using 100 as multiplier.
        estimated_views = item.get("videoViewCount") or (likes * 100)

        return TrendItem(
            platform="instagram",
            url=item.get("url") or f"https://www.instagram.com/p/{item.get('shortCode', '')}",
            title=item.get("caption") or "",
            views=estimated_views,
            likes=likes,
            shares=0,
            comments=item.get("commentsCount", 0),
            author=item.get("ownerUsername", ""),
            author_followers=0,
            hashtags=item.get("hashtags", []) if isinstance(item.get("hashtags"), list) else [],
            duration_seconds=0,
            posted_at=item.get("timestamp", "")
        )

    async def _scrape_youtube(self, hashtags: List[str], limit: int) -> List[TrendItem]:
        """
        Scrape YouTube Shorts using KEYWORD SEARCH.

        NOTE: YouTube scraping is currently disabled due to Apify actor availability issues.
        Returns empty list gracefully.
        """
        logger.warning("YouTube scraping is currently disabled - no working Apify actor available")
        return []  # Disabled for now - no reliable YouTube actor available

    def _normalize_youtube_item(self, item: Dict) -> TrendItem:
        """Normalize YouTube item to common format."""
        # Parse view count which might be string like "1.2M views"
        views = item.get("viewCount", 0)
        if isinstance(views, str):
            views = self._parse_count_string(views)

        likes = item.get("likes", 0)
        if isinstance(likes, str):
            likes = self._parse_count_string(likes)

        return TrendItem(
            platform="youtube",
            url=item.get("url") or item.get("videoUrl", ""),
            title=item.get("title", ""),
            views=views,
            likes=likes,
            shares=0,
            comments=item.get("commentsCount", 0),
            author=item.get("channelName") or item.get("channel", ""),
            author_followers=item.get("subscriberCount", 0),
            hashtags=[],  # YouTube doesn't expose hashtags easily
            duration_seconds=item.get("duration", 0),
            posted_at=item.get("date", "")
        )

    def _parse_count_string(self, count_str: str) -> int:
        """Parse strings like '1.2M' or '500K' to integers."""
        if not count_str or not isinstance(count_str, str):
            return 0

        count_str = count_str.lower().replace(",", "").replace(" ", "")
        count_str = count_str.replace("views", "").replace("likes", "").strip()

        try:
            if "m" in count_str:
                return int(float(count_str.replace("m", "")) * 1_000_000)
            elif "k" in count_str:
                return int(float(count_str.replace("k", "")) * 1_000)
            else:
                return int(float(count_str))
        except (ValueError, TypeError):
            return 0

    def normalize_all_results(self, items: List[TrendItem]) -> List[TrendItem]:
        """Deduplicate and sort results by VIEWS (most viral first)."""
        seen_urls = set()
        unique_items = []

        for item in items:
            if item.url and item.url not in seen_urls:
                seen_urls.add(item.url)
                unique_items.append(item)

        # Sort by VIEWS - this is what makes content "viral"
        unique_items.sort(key=lambda x: x.views, reverse=True)

        logger.info(f"Normalized {len(unique_items)} unique items, top video has {unique_items[0].views if unique_items else 0} views")

        return unique_items

    async def analyze_trends(self, items: List[TrendItem], niche: str, analyze_count: int = 50) -> List[TrendItem]:
        """
        Prepare trends for saving - NO LLM analysis during scrape.

        LLM analysis (hooks, scores, why_viral) happens when idea is APPROVED,
        not during scraping. This saves API costs.

        Args:
            items: List of trend items (already sorted by views)
            niche: Content niche for context
            analyze_count: Unused - kept for API compatibility

        Returns:
            ALL items with default values (no LLM call)
        """
        if not items:
            return []

        # Set defaults for ALL items - no LLM analysis during scrape
        for item in items:
            item.viral_score = item.viral_score or 0  # 0 = not yet analyzed
            item.pillar = item.pillar or ""  # Empty = not yet categorized
            item.suggested_hook = item.suggested_hook or ""
            item.why_viral = item.why_viral or ""

        logger.info(f"Prepared {len(items)} items for saving (LLM analysis deferred to approval)")
        return items

    async def analyze_single_idea(self, idea_id: int, title: str, url: str, platform: str, views: int, likes: int) -> dict:
        """
        Analyze a SINGLE content idea with LLM when it gets approved.

        Called from the approval endpoint, not during scraping.

        Args:
            idea_id: Content idea ID
            title: Video title/caption
            url: Source URL
            platform: Source platform
            views: View count
            likes: Like count

        Returns:
            Dict with viral_score, pillar, suggested_hook, why_viral
        """
        prompt = f"""Analyze this viral {platform} video for a real estate professional to recreate:

Title/Caption: {title}
URL: {url}
Views: {views:,}
Likes: {likes:,}

Your job:
1. Score how replicable this is for a real estate agent (viral_score 1-10)
2. Categorize by content pillar
3. Suggest a hook that captures what made this video work
4. Explain WHY it went viral

Content pillars:
- market_intelligence: Market data, trends, statistics
- educational_tips: How-to content, tips, advice
- lifestyle_local: Local area, community content
- brand_humanization: Personal stories, behind-the-scenes

Return JSON:
{{
  "viral_score": 8,
  "pillar": "educational_tips",
  "suggested_hook": "A compelling hook...",
  "why_viral": "1-2 sentences on why this went viral"
}}"""

        try:
            response = await self._post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "x-ai/grok-4.1-fast",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                },
                timeout=30.0
            )

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            analyzed = json.loads(content)

            logger.info(f"Analyzed idea {idea_id}: score={analyzed.get('viral_score')}, pillar={analyzed.get('pillar')}")
            return analyzed

        except Exception as e:
            logger.error(f"Failed to analyze idea {idea_id}: {e}")
            return {
                "viral_score": 7,
                "pillar": "educational_tips",
                "suggested_hook": "",
                "why_viral": ""
            }

    def format_response(
        self,
        items: List[TrendItem],
        params: ScrapeParams,
        saved_count: int = 0
    ) -> ScrapeResult:
        """Format final scrape response."""
        return ScrapeResult(
            success=True,
            niche=params.niche,
            scraped_at=datetime.now(timezone.utc).isoformat(),
            total_scraped=len(items),
            analyzed_count=len(items),
            saved_count=saved_count,
            message=f"Scraped {len(items)} viral trends" + (f", saved {saved_count}" if saved_count else ""),
            trends=[item.model_dump() for item in items]
        )

    async def save_trends_to_db(
        self,
        items: List[TrendItem],
        db_session
    ) -> int:
        """
        Save trends to database, skipping duplicates by URL.

        Args:
            items: Trend items to save
            db_session: Database session

        Returns:
            Number of NEW items saved (excludes duplicates)
        """
        from models import ContentIdea
        from sqlalchemy import select

        saved_count = 0
        skipped_duplicates = 0

        for item in items:
            try:
                # Check if URL already exists
                existing = await db_session.execute(
                    select(ContentIdea.id).where(ContentIdea.source_url == item.url)
                )
                if existing.scalar_one_or_none():
                    skipped_duplicates += 1
                    continue

                idea = ContentIdea(
                    source_url=item.url,
                    source_platform=item.platform,
                    original_text=item.title,
                    pillar=item.pillar or "educational_tips",
                    viral_score=item.viral_score or 5,
                    suggested_hook=item.suggested_hook or "",
                    why_viral=item.why_viral or "",
                    status="pending",
                    # Engagement metrics
                    views=item.views or 0,
                    likes=item.likes or 0,
                    shares=item.shares or 0,
                    comments=item.comments or 0,
                    author=item.author or "",
                    author_followers=item.author_followers or 0
                )
                db_session.add(idea)
                await db_session.commit()
                saved_count += 1

            except Exception as e:
                logger.error(f"Failed to save trend: {e}")
                await db_session.rollback()
                continue

        if skipped_duplicates > 0:
            logger.info(f"Skipped {skipped_duplicates} duplicate URLs")

        return saved_count
