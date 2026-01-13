"""
Scrape Celery tasks - trend discovery.
Replaces the n8n scrape workflow.
"""

from celery import shared_task
from celery.utils.log import get_task_logger
from typing import Dict, Any, List, Optional

from config import settings

logger = get_task_logger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=120)
def run_scrape(
    self,
    params: Optional[Dict[str, Any]] = None,
    niche: str = "real estate",
    platforms: Optional[List[str]] = None,
    hashtags: Optional[List[str]] = None,
    results_per_platform: int = 30,
    analyze_top_n: int = 20,
    discover_hashtags: bool = False,
    seed_keyword: str = ""
) -> Dict[str, Any]:
    """
    Run trend scraping job using HASHTAG SEARCH.

    Supports TWO modes:
    1. Direct mode: Uses provided hashtags to search
    2. Discovery mode: First discovers what hashtags top videos use,
       then scrapes using those discovered hashtags

    Args:
        params: Optional dict with all parameters (for compatibility with main.py)
        niche: Content niche
        platforms: Platforms to scrape (default: tiktok, instagram, youtube)
        hashtags: Hashtags to search for (e.g., "realtor", "realestate")
        results_per_platform: Results per platform
        analyze_top_n: How many top items to analyze with LLM
        discover_hashtags: If True, first discover trending hashtags from seed_keyword
        seed_keyword: Base keyword for hashtag discovery (e.g., "realtor")

    Returns:
        Scrape results summary
    """
    # If params dict provided, extract values from it
    if isinstance(params, dict):
        niche = params.get("niche", niche)
        platforms = params.get("platforms", platforms)
        # Support both hashtags and legacy search_queries
        hashtags = params.get("hashtags", params.get("search_queries", hashtags))
        results_per_platform = params.get("results_per_platform", results_per_platform)
        analyze_top_n = params.get("analyze_top_n", analyze_top_n)
        discover_hashtags = params.get("discover_hashtags", discover_hashtags)
        seed_keyword = params.get("seed_keyword", seed_keyword)
        scrape_run_id = params.get("scrape_run_id")
    else:
        scrape_run_id = None

    import asyncio
    return asyncio.get_event_loop().run_until_complete(
        _run_scrape_async(
            niche, platforms, hashtags, results_per_platform, analyze_top_n,
            scrape_run_id, discover_hashtags, seed_keyword
        )
    )


async def _run_scrape_async(
    niche: str,
    platforms: Optional[List[str]],
    hashtags: Optional[List[str]],
    results_per_platform: int,
    analyze_top_n: int = 20,
    scrape_run_id: Optional[int] = None,
    discover_hashtags: bool = False,
    seed_keyword: str = ""
) -> Dict[str, Any]:
    """Async implementation of scraping using hashtag search."""
    from services.scraper import ScraperService, ScrapeParams
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import update
    from models import ScrapeRun, ScrapeRunStatus
    from datetime import datetime

    # Default values
    if platforms is None:
        platforms = ["tiktok", "instagram", "youtube"]
    if hashtags is None:
        hashtags = ["realtor", "realestate", "homebuying"]

    scraper = ScraperService()

    # Create database session
    engine = create_async_engine(settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"))
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        # Parse parameters
        params = ScrapeParams(
            niche=niche,
            platforms=platforms,
            hashtags=hashtags,
            results_per_platform=results_per_platform,
            analyze_top_n=analyze_top_n,
            discover_hashtags=discover_hashtags,
            seed_keyword=seed_keyword
        )

        logger.info(f"Starting scrape: {niche} from {platforms}")
        if discover_hashtags and seed_keyword:
            logger.info(f"Discovery mode enabled with seed keyword: {seed_keyword}")

        # Scrape platforms - use discovery mode if enabled
        if discover_hashtags and seed_keyword:
            items = await scraper.scrape_with_discovery(params)
        else:
            items = await scraper.scrape_platforms(params)
        logger.info(f"Scraped {len(items)} raw items")

        # Normalize and deduplicate
        items = scraper.normalize_all_results(items)
        logger.info(f"Normalized to {len(items)} unique items")

        # Analyze with LLM
        analyzed_items = await scraper.analyze_trends(items, niche)
        logger.info(f"Analyzed {len(analyzed_items)} items")

        # Save to database
        async with async_session() as session:
            saved_count = await scraper.save_trends_to_db(analyzed_items, session)

            # Update scrape run record if we have one
            if scrape_run_id:
                await session.execute(
                    update(ScrapeRun.__table__)
                    .where(ScrapeRun.id == scrape_run_id)
                    .values(
                        status=ScrapeRunStatus.completed,
                        results_count=saved_count,
                        completed_at=datetime.utcnow()
                    )
                )
                await session.commit()

        # Format response
        result = scraper.format_response(analyzed_items, params, saved_count)

        logger.info(f"Scrape complete: {result.total_scraped} scraped, {result.saved_count} saved")

        return result.model_dump()

    except Exception as e:
        logger.error(f"Scrape failed: {e}")

        # Update scrape run record with error
        if scrape_run_id:
            async with async_session() as session:
                await session.execute(
                    update(ScrapeRun.__table__)
                    .where(ScrapeRun.id == scrape_run_id)
                    .values(
                        status=ScrapeRunStatus.failed,
                        error_message=str(e),
                        completed_at=datetime.utcnow()
                    )
                )
                await session.commit()

        raise


@shared_task
def run_daily_scrape() -> Dict[str, Any]:
    """
    Daily scheduled scrape task.

    Runs at 6am UTC with default parameters.
    Uses HASHTAG SEARCH to find viral videos.

    Returns:
        Scrape results
    """
    return run_scrape(
        niche="real estate",
        platforms=["tiktok", "instagram", "youtube"],
        hashtags=["realtor", "realestate", "homebuying", "firsttimehomebuyer"],
        results_per_platform=30,
        analyze_top_n=20
    )


@shared_task
def scrape_with_preset(preset_id: int) -> Dict[str, Any]:
    """
    Scrape using a saved niche preset.

    Args:
        preset_id: Niche preset ID

    Returns:
        Scrape results
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(
        _scrape_with_preset_async(preset_id)
    )


async def _scrape_with_preset_async(preset_id: int) -> Dict[str, Any]:
    """Async implementation of preset scraping."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from models import NichePreset

    engine = create_async_engine(settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"))
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        result = await session.execute(
            NichePreset.__table__.select().where(NichePreset.id == preset_id)
        )
        preset = result.fetchone()

        if not preset:
            raise ValueError(f"Niche preset {preset_id} not found")

        preset_dict = dict(preset._mapping)

        return run_scrape(
            niche=preset_dict.get("name", "real estate"),
            platforms=["tiktok", "instagram", "youtube"],
            hashtags=preset_dict.get("hashtags", []),
            results_per_platform=30,
            analyze_top_n=20
        )
