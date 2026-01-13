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
    analyze_top_n: int = 20
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
    APIFY_INSTAGRAM_ACTOR = "apify~instagram-scraper"
    APIFY_YOUTUBE_ACTOR = "streamers~youtube-scraper"

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

        Uses apify/instagram-scraper in hashtag mode.
        """
        url = f"https://api.apify.com/v2/acts/{self.APIFY_INSTAGRAM_ACTOR}/run-sync-get-dataset-items"

        try:
            all_items = []

            for hashtag in hashtags[:3]:  # Limit to 3 hashtags
                response = await self._post(
                    url,
                    params={"token": settings.APIFY_API_KEY},
                    json={
                        "search": hashtag,
                        "searchType": "hashtag",
                        "resultsLimit": limit // len(hashtags[:3]),
                        "searchLimit": 1
                    },
                    timeout=180.0
                )

                items = response.json()
                if isinstance(items, list):
                    # Filter to only video/reel content
                    video_items = [i for i in items if i.get("type") == "Video" or i.get("isVideo")]
                    all_items.extend(video_items)
                    logger.info(f"Instagram #{hashtag}: got {len(video_items)} video results")

            normalized = [self._normalize_instagram_item(item) for item in all_items]
            normalized.sort(key=lambda x: x.views, reverse=True)
            return normalized[:limit]

        except Exception as e:
            logger.error(f"Instagram scrape failed: {e}")
            raise

    def _normalize_instagram_item(self, item: Dict) -> TrendItem:
        """Normalize Instagram item to common format."""
        return TrendItem(
            platform="instagram",
            url=item.get("url") or item.get("shortCode", ""),
            title=item.get("caption") or item.get("alt") or "",
            views=item.get("videoViewCount") or item.get("viewCount", 0),
            likes=item.get("likesCount") or item.get("likes", 0),
            shares=0,
            comments=item.get("commentsCount") or item.get("comments", 0),
            author=item.get("ownerUsername") or item.get("owner", {}).get("username", ""),
            author_followers=item.get("ownerFullName", 0) if isinstance(item.get("ownerFullName"), int) else 0,
            hashtags=item.get("hashtags", []) if isinstance(item.get("hashtags"), list) else [],
            duration_seconds=item.get("videoDuration", 0),
            posted_at=item.get("timestamp", "")
        )

    async def _scrape_youtube(self, hashtags: List[str], limit: int) -> List[TrendItem]:
        """
        Scrape YouTube Shorts using KEYWORD SEARCH.

        Uses streamers/youtube-scraper with hashtags as search keywords.
        """
        url = f"https://api.apify.com/v2/acts/{self.APIFY_YOUTUBE_ACTOR}/run-sync-get-dataset-items"

        try:
            # Convert hashtags to search keywords with "shorts"
            search_keywords = [f"{tag} shorts" for tag in hashtags[:3]]

            response = await self._post(
                url,
                params={"token": settings.APIFY_API_KEY},
                json={
                    "searchKeywords": search_keywords,
                    "maxResults": limit,
                    "maxResultsShorts": limit
                },
                timeout=180.0
            )

            items = response.json()
            if not isinstance(items, list):
                logger.warning(f"YouTube returned unexpected format: {type(items)}")
                return []

            logger.info(f"YouTube keywords {search_keywords}: got {len(items)} results")

            normalized = [self._normalize_youtube_item(item) for item in items]
            normalized.sort(key=lambda x: x.views, reverse=True)
            return normalized[:limit]

        except Exception as e:
            logger.error(f"YouTube scrape failed: {e}")
            raise

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

    async def analyze_trends(self, items: List[TrendItem], niche: str, analyze_count: int = 20) -> List[TrendItem]:
        """
        Analyze top trends using Grok LLM via OpenRouter.

        Args:
            items: List of trend items to analyze (already sorted by views)
            niche: Content niche for context
            analyze_count: How many top items to analyze

        Returns:
            Items with viral_score, pillar, suggested_hook, why_viral filled in
        """
        if not items:
            return []

        # Analyze top N items (they're already sorted by views)
        items_to_analyze = items[:analyze_count]

        # Build video data for prompt (can't have comments inside f-string)
        videos_data = json.dumps([{
            "platform": item.platform,
            "title": item.title[:200],
            "views": item.views,
            "likes": item.likes,
            "author": item.author,
            "url": item.url
        } for item in items_to_analyze], indent=2)

        prompt = f"""You are analyzing the TOP {len(items_to_analyze)} most-viewed {niche} videos from TikTok, Instagram, and YouTube.

These are PROVEN viral videos - they already have high view counts. Your job is to:
1. Score how replicable this content is for a {niche} professional (viral_score 1-10)
2. Categorize by content pillar
3. Suggest a hook that captures what made this video work
4. Explain WHY it went viral (what psychological triggers, trends, or techniques)

Content pillars:
- market_intelligence: Market data, trends, statistics, predictions
- educational_tips: How-to content, tips, tutorials, advice
- lifestyle_local: Local area features, community, lifestyle content
- brand_humanization: Personal stories, behind-the-scenes, day-in-the-life

Videos to analyze (sorted by views, highest first):
{videos_data}

Return JSON with "trends" array. Each item must have:
- url (string): The original URL
- viral_score (int 1-10): How easy to replicate successfully
- pillar (string): One of the 4 pillars above
- suggested_hook (string): A compelling hook for recreating this content
- why_viral (string): 1-2 sentences on why this specific video went viral"""

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
                timeout=90.0
            )

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            analyzed = json.loads(content)

            # Parse analyzed items
            analyzed_items = analyzed.get("trends", analyzed)
            if isinstance(analyzed_items, dict):
                analyzed_items = [analyzed_items]

            # Match analysis back to items by URL
            analysis_by_url = {a.get("url", ""): a for a in analyzed_items}

            for item in items_to_analyze:
                analysis = analysis_by_url.get(item.url, {})
                item.viral_score = analysis.get("viral_score", 7)
                item.pillar = analysis.get("pillar", "educational_tips")
                item.suggested_hook = analysis.get("suggested_hook", "")
                item.why_viral = analysis.get("why_viral", "")

            logger.info(f"Analyzed {len(items_to_analyze)} items with LLM")
            return items_to_analyze

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Return items with default values
            for item in items_to_analyze:
                item.viral_score = 7
                item.pillar = "educational_tips"
            return items_to_analyze

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
        Save analyzed trends to database.

        Args:
            items: Trend items to save
            db_session: Database session

        Returns:
            Number of items saved
        """
        from models import ContentIdea

        saved_count = 0

        for item in items:
            try:
                idea = ContentIdea(
                    source_url=item.url,
                    source_platform=item.platform,
                    original_text=item.title,
                    pillar=item.pillar or "educational_tips",
                    viral_score=item.viral_score or 7,
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

        return saved_count
