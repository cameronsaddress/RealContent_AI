"""
Pipeline API endpoints - trigger and monitor video production.
Replaces n8n webhook endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class TriggerPipelineRequest(BaseModel):
    """Request to trigger the pipeline."""
    content_idea_id: Optional[int] = None


class TriggerPipelineResponse(BaseModel):
    """Response from pipeline trigger."""
    task_id: str
    content_idea_id: Optional[int]
    message: str


class PipelineStatusResponse(BaseModel):
    """Pipeline task status."""
    task_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/trigger", response_model=TriggerPipelineResponse)
async def trigger_pipeline(request: TriggerPipelineRequest):
    """
    Trigger the video production pipeline.

    If content_idea_id is provided, processes that specific idea.
    Otherwise, picks the next approved content idea.

    Replaces: POST /webhook/trigger-pipeline
    """
    from tasks import run_pipeline

    task = run_pipeline.delay(request.content_idea_id)

    return TriggerPipelineResponse(
        task_id=task.id,
        content_idea_id=request.content_idea_id,
        message="Pipeline queued for processing"
    )


@router.get("/status/{task_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(task_id: str):
    """
    Get status of a pipeline task.

    Args:
        task_id: Celery task ID from trigger response
    """
    from celery.result import AsyncResult
    from celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    response = PipelineStatusResponse(
        task_id=task_id,
        status=result.status
    )

    if result.ready():
        if result.successful():
            response.result = result.get()
        else:
            response.error = str(result.result)

    return response


@router.post("/trigger/{content_idea_id}", response_model=TriggerPipelineResponse)
async def trigger_specific_idea(content_idea_id: int):
    """
    Trigger pipeline for a specific content idea.

    Args:
        content_idea_id: Content idea ID to process
    """
    from tasks import process_specific_idea

    task = process_specific_idea.delay(content_idea_id)

    return TriggerPipelineResponse(
        task_id=task.id,
        content_idea_id=content_idea_id,
        message=f"Pipeline queued for content idea {content_idea_id}"
    )


@router.post("/check-pending")
async def check_pending_ideas():
    """
    Check for pending approved ideas and queue them.

    This can be called manually or runs automatically every 15 minutes.
    """
    from tasks import check_pending_ideas

    task = check_pending_ideas.delay()

    return {
        "task_id": task.id,
        "message": "Checking for pending ideas"
    }


class ScrapeRequest(BaseModel):
    """Request to trigger trend scraping."""
    niche: str = "real estate"
    platforms: List[str] = ["tiktok", "instagram"]
    hashtags: List[str] = ["realestate", "homebuying", "realtor"]
    results_per_platform: int = 20


class ScrapeResponse(BaseModel):
    """Response from scrape trigger."""
    task_id: str
    niche: str
    message: str


@router.post("/scrape", response_model=ScrapeResponse)
async def trigger_scrape(request: ScrapeRequest):
    """
    Trigger trend scraping.

    Replaces: POST /webhook/scrape-trends
    """
    from tasks import run_scrape

    task = run_scrape.delay(
        niche=request.niche,
        platforms=request.platforms,
        hashtags=request.hashtags,
        results_per_platform=request.results_per_platform
    )

    return ScrapeResponse(
        task_id=task.id,
        niche=request.niche,
        message=f"Scrape queued for {request.niche}"
    )


@router.get("/scrape/status/{task_id}")
async def get_scrape_status(task_id: str):
    """Get status of a scrape task."""
    from celery.result import AsyncResult
    from celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": result.status
    }

    if result.ready():
        if result.successful():
            response["result"] = result.get()
        else:
            response["error"] = str(result.result)

    return response


@router.post("/scrape/preset/{preset_id}")
async def trigger_scrape_with_preset(preset_id: int):
    """
    Trigger scraping using a saved niche preset.

    Args:
        preset_id: Niche preset ID
    """
    from tasks import scrape_with_preset

    task = scrape_with_preset.delay(preset_id)

    return {
        "task_id": task.id,
        "preset_id": preset_id,
        "message": f"Scrape queued with preset {preset_id}"
    }


# Stats and monitoring endpoints

@router.get("/stats")
async def get_pipeline_stats():
    """
    Get pipeline statistics.

    Returns counts of content at each stage.
    """
    from sqlalchemy import create_engine, text
    from config import settings

    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        # Content idea stats
        idea_stats = conn.execute(text("""
            SELECT status, COUNT(*) as count
            FROM content_ideas
            GROUP BY status
        """)).fetchall()

        # Script stats
        script_count = conn.execute(text("""
            SELECT COUNT(*) FROM scripts
        """)).scalar()

        # Asset stats
        asset_stats = conn.execute(text("""
            SELECT status, COUNT(*) as count
            FROM assets
            GROUP BY status
        """)).fetchall()

        # Published stats - count per platform based on non-null URLs
        published_count = conn.execute(text("SELECT COUNT(*) FROM published")).scalar()

        platform_counts = {}
        for platform in ['tiktok', 'ig', 'yt', 'linkedin', 'x', 'facebook', 'threads', 'pinterest']:
            count = conn.execute(text(f"SELECT COUNT(*) FROM published WHERE {platform}_url IS NOT NULL")).scalar()
            if count > 0:
                platform_counts[platform] = count

    return {
        "content_ideas": {row[0]: row[1] for row in idea_stats},
        "scripts_total": script_count,
        "assets": {row[0]: row[1] for row in asset_stats},
        "published_total": published_count,
        "published_by_platform": platform_counts
    }


@router.get("/overview")
async def get_pipeline_overview():
    """
    Get high-level pipeline overview.

    Used by the dashboard for quick status check.
    """
    from sqlalchemy import create_engine, text
    from config import settings

    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        pending = conn.execute(text(
            "SELECT COUNT(*) FROM content_ideas WHERE status = 'pending'"
        )).scalar()

        approved = conn.execute(text(
            "SELECT COUNT(*) FROM content_ideas WHERE status = 'approved'"
        )).scalar()

        processing = conn.execute(text(
            "SELECT COUNT(*) FROM content_ideas WHERE status IN ('script_generating', 'script_ready')"
        )).scalar()

        published = conn.execute(text(
            "SELECT COUNT(*) FROM content_ideas WHERE status = 'published'"
        )).scalar()

        recent_published = conn.execute(text("""
            SELECT p.id, p.tiktok_url, p.ig_url, p.published_at, a.script_id
            FROM published p
            JOIN assets a ON p.asset_id = a.id
            ORDER BY p.published_at DESC NULLS LAST
            LIMIT 5
        """)).fetchall()

    return {
        "counts": {
            "pending": pending,
            "approved": approved,
            "processing": processing,
            "published": published
        },
        "recent_published": [
            {
                "id": row[0],
                "tiktok_url": row[1],
                "ig_url": row[2],
                "published_at": str(row[3]) if row[3] else None,
                "script_id": row[4]
            }
            for row in recent_published
        ]
    }
