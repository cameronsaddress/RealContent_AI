"""
Publishing API Router - Queue management and distribution integration.
Handles publishing schedules, approval workflows, and content distribution.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

from models import (
    get_db, PublishingConfig, PublishingQueueItem, ViralClip, Influencer, ClipPersona
)
from utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/publishing", tags=["publishing"])


# === SCHEMAS ===

class PublishingConfigCreate(BaseModel):
    influencer_id: Optional[int] = None
    persona_id: Optional[int] = None
    blotato_account_id: str
    blotato_account_name: str
    platforms: List[str] = ["tiktok", "instagram_reels"]
    enabled: bool = False
    posts_per_day: int = 3
    posting_hours: List[int] = [9, 13, 18]
    posting_days: List[int] = [0, 1, 2, 3, 4, 5, 6]
    min_virality_score: float = 0.0
    clip_types_allowed: List[str] = ["educational", "market_update", "property_tour", "testimonial"]
    require_manual_approval: bool = True
    min_hours_between_posts: int = 4
    max_posts_per_platform_per_day: int = 2


class PublishingConfigUpdate(BaseModel):
    blotato_account_id: Optional[str] = None
    blotato_account_name: Optional[str] = None
    platforms: Optional[List[str]] = None
    enabled: Optional[bool] = None
    posts_per_day: Optional[int] = None
    posting_hours: Optional[List[int]] = None
    posting_days: Optional[List[int]] = None
    min_virality_score: Optional[float] = None
    clip_types_allowed: Optional[List[str]] = None
    require_manual_approval: Optional[bool] = None
    min_hours_between_posts: Optional[int] = None
    max_posts_per_platform_per_day: Optional[int] = None


class QueueItemApproval(BaseModel):
    approved: bool
    reason: Optional[str] = None


# === PUBLISHING CONFIG ENDPOINTS ===

@router.get("/configs")
def list_publishing_configs(db: Session = Depends(get_db)):
    """List all publishing configurations."""
    configs = db.query(PublishingConfig).all()
    return [
        {
            "id": c.id,
            "influencer_id": c.influencer_id,
            "influencer_name": c.influencer.name if c.influencer else None,
            "persona_id": c.persona_id,
            "persona_name": c.persona.name if c.persona else None,
            "blotato_account_id": c.blotato_account_id,
            "blotato_account_name": c.blotato_account_name,
            "platforms": c.platforms or [],
            "enabled": c.enabled or False,
            "posts_per_day": c.posts_per_day or 3,
            "posting_hours": c.posting_hours or [9, 13, 18],
            "posting_days": c.posting_days or [0, 1, 2, 3, 4, 5, 6],
            "min_virality_score": c.min_virality_score or 0.0,
            "clip_types_allowed": c.clip_types_allowed or [],
            "require_manual_approval": c.require_manual_approval if c.require_manual_approval is not None else True,
            "min_hours_between_posts": c.min_hours_between_posts or 4,
            "max_posts_per_platform_per_day": c.max_posts_per_platform_per_day or 2,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in configs
    ]


@router.get("/configs/{config_id}")
def get_publishing_config(config_id: int, db: Session = Depends(get_db)):
    """Get a single publishing configuration by ID."""
    config = db.query(PublishingConfig).filter(PublishingConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")

    return {
        "id": config.id,
        "influencer_id": config.influencer_id,
        "influencer_name": config.influencer.name if config.influencer else None,
        "persona_id": config.persona_id,
        "persona_name": config.persona.name if config.persona else None,
        "blotato_account_id": config.blotato_account_id,
        "blotato_account_name": config.blotato_account_name,
        "platforms": config.platforms or [],
        "enabled": config.enabled or False,
        "posts_per_day": config.posts_per_day or 3,
        "posting_hours": config.posting_hours or [9, 13, 18],
        "posting_days": config.posting_days or [0, 1, 2, 3, 4, 5, 6],
        "min_virality_score": config.min_virality_score or 0.0,
        "clip_types_allowed": config.clip_types_allowed or [],
        "require_manual_approval": config.require_manual_approval if config.require_manual_approval is not None else True,
        "min_hours_between_posts": config.min_hours_between_posts or 4,
        "max_posts_per_platform_per_day": config.max_posts_per_platform_per_day or 2,
    }


@router.post("/configs")
def create_publishing_config(config: PublishingConfigCreate, db: Session = Depends(get_db)):
    """Create a new publishing configuration."""
    new_config = PublishingConfig(
        influencer_id=config.influencer_id,
        persona_id=config.persona_id,
        blotato_account_id=config.blotato_account_id,
        blotato_account_name=config.blotato_account_name,
        platforms=config.platforms,
        enabled=config.enabled,
        posts_per_day=config.posts_per_day,
        posting_hours=config.posting_hours,
        posting_days=config.posting_days,
        min_virality_score=config.min_virality_score,
        clip_types_allowed=config.clip_types_allowed,
        require_manual_approval=config.require_manual_approval,
        min_hours_between_posts=config.min_hours_between_posts,
        max_posts_per_platform_per_day=config.max_posts_per_platform_per_day,
    )
    db.add(new_config)
    db.commit()
    db.refresh(new_config)

    logger.info(f"Created publishing config {new_config.id} for account {config.blotato_account_name}")
    return {"id": new_config.id, "status": "created"}


@router.put("/configs/{config_id}")
def update_publishing_config(config_id: int, config: PublishingConfigUpdate, db: Session = Depends(get_db)):
    """Update a publishing configuration."""
    existing = db.query(PublishingConfig).filter(PublishingConfig.id == config_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Config not found")

    update_data = config.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(existing, key, value)

    db.commit()
    logger.info(f"Updated publishing config {config_id}")
    return {"status": "updated", "id": config_id}


@router.delete("/configs/{config_id}")
def delete_publishing_config(config_id: int, db: Session = Depends(get_db)):
    """Delete a publishing configuration."""
    existing = db.query(PublishingConfig).filter(PublishingConfig.id == config_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Config not found")

    db.delete(existing)
    db.commit()
    logger.info(f"Deleted publishing config {config_id}")
    return {"status": "deleted", "id": config_id}


# === PUBLISHING QUEUE ENDPOINTS ===

@router.get("/queue")
def list_publishing_queue(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List publishing queue items."""
    query = db.query(PublishingQueueItem).join(ViralClip)

    if status:
        query = query.filter(PublishingQueueItem.status == status)

    items = query.order_by(
        PublishingQueueItem.scheduled_time.asc()
    ).limit(limit).all()

    result = []
    for item in items:
        clip = item.clip
        influencer_name = None
        thumbnail_url = None

        if clip and clip.source_video:
            if clip.source_video.influencer:
                influencer_name = clip.source_video.influencer.name
            thumbnail_url = clip.source_video.thumbnail_url

        result.append({
            "id": item.id,
            "clip_id": item.clip_id,
            "clip_title": clip.title if clip else None,
            "clip_type": clip.clip_type if clip else None,
            "thumbnail_url": thumbnail_url,
            "influencer_name": influencer_name,
            "scheduled_time": item.scheduled_time.isoformat() if item.scheduled_time else None,
            "target_platforms": item.target_platforms or [],
            "status": item.status,
            "requires_approval": item.requires_approval,
            "approved_at": item.approved_at.isoformat() if item.approved_at else None,
            "approved_by": item.approved_by,
            "published_at": item.published_at.isoformat() if item.published_at else None,
            "error_message": item.error_message,
            "blotato_post_ids": item.blotato_post_ids or {},
            "created_at": item.created_at.isoformat() if item.created_at else None,
        })

    return result


@router.get("/queue/{item_id}")
def get_queue_item(item_id: int, db: Session = Depends(get_db)):
    """Get a single queue item by ID."""
    item = db.query(PublishingQueueItem).filter(PublishingQueueItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")

    clip = item.clip
    influencer_name = None
    thumbnail_url = None

    if clip and clip.source_video:
        if clip.source_video.influencer:
            influencer_name = clip.source_video.influencer.name
        thumbnail_url = clip.source_video.thumbnail_url

    return {
        "id": item.id,
        "clip_id": item.clip_id,
        "clip_title": clip.title if clip else None,
        "clip_type": clip.clip_type if clip else None,
        "thumbnail_url": thumbnail_url,
        "influencer_name": influencer_name,
        "config_id": item.config_id,
        "scheduled_time": item.scheduled_time.isoformat() if item.scheduled_time else None,
        "priority": item.priority,
        "target_platforms": item.target_platforms or [],
        "status": item.status,
        "requires_approval": item.requires_approval,
        "approved_at": item.approved_at.isoformat() if item.approved_at else None,
        "approved_by": item.approved_by,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "error_message": item.error_message,
        "blotato_post_ids": item.blotato_post_ids or {},
    }


@router.put("/queue/{item_id}/approve")
def approve_queue_item(item_id: int, approval: QueueItemApproval, db: Session = Depends(get_db)):
    """Approve or reject a publishing queue item."""
    item = db.query(PublishingQueueItem).filter(PublishingQueueItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")

    if approval.approved:
        item.status = "approved"
        item.approved_at = datetime.utcnow()
        item.approved_by = "admin"  # TODO: Get from auth when implemented
        logger.info(f"Approved queue item {item_id} for clip {item.clip_id}")
    else:
        item.status = "skipped"
        if item.clip:
            item.clip.publishing_status = "skipped"
            item.clip.skip_reason = approval.reason or "Manually rejected"
        logger.info(f"Rejected queue item {item_id} for clip {item.clip_id}: {approval.reason}")

    db.commit()
    return {"status": item.status, "id": item_id}


@router.post("/queue/{item_id}/publish-now")
def publish_queue_item_now(item_id: int, db: Session = Depends(get_db)):
    """Immediately publish a queue item (bypasses schedule)."""
    item = db.query(PublishingQueueItem).filter(PublishingQueueItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")

    if item.status not in ["pending", "approved"]:
        raise HTTPException(status_code=400, detail=f"Cannot publish item with status '{item.status}'")

    item.status = "approved"
    item.scheduled_time = datetime.utcnow()
    item.priority = 100  # High priority
    db.commit()

    # Trigger immediate publishing via Celery
    from tasks.viral import publish_single_clip
    publish_single_clip.delay(item.id)

    logger.info(f"Triggered immediate publish for queue item {item_id}")
    return {"status": "publishing", "item_id": item.id}


@router.delete("/queue/{item_id}")
def remove_from_queue(item_id: int, db: Session = Depends(get_db)):
    """Remove an item from the publishing queue."""
    item = db.query(PublishingQueueItem).filter(PublishingQueueItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")

    # Reset clip status
    if item.clip:
        item.clip.publishing_status = "unpublished"

    db.delete(item)
    db.commit()
    logger.info(f"Removed queue item {item_id} from queue")
    return {"status": "removed", "id": item_id}


# === BLOTATO ACCOUNTS ===

@router.get("/blotato-accounts")
async def list_blotato_accounts():
    """
    Fetch available Blotato accounts.
    This calls Blotato API to get linked social accounts.
    """
    # TODO: Implement actual Blotato API call to list accounts
    # For now, return mock data for UI development
    return [
        {"id": "acc_realcontent", "name": "RealContent AI", "platforms": ["tiktok", "instagram"]},
        {"id": "acc_realestate", "name": "Real Estate Hub", "platforms": ["tiktok", "youtube", "instagram"]},
    ]


# === STATISTICS ===

@router.get("/stats")
def get_publishing_stats(db: Session = Depends(get_db)):
    """Get publishing statistics."""
    from sqlalchemy import func

    # Queue stats by status
    queue_stats = db.query(
        PublishingQueueItem.status,
        func.count(PublishingQueueItem.id)
    ).group_by(PublishingQueueItem.status).all()

    # Published today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    published_today = db.query(PublishingQueueItem).filter(
        PublishingQueueItem.status == "published",
        PublishingQueueItem.published_at >= today_start
    ).count()

    # Pending approval
    pending_approval = db.query(PublishingQueueItem).filter(
        PublishingQueueItem.status == "pending",
        PublishingQueueItem.requires_approval == True
    ).count()

    # Scheduled (approved but not yet published)
    scheduled = db.query(PublishingQueueItem).filter(
        PublishingQueueItem.status == "approved",
        PublishingQueueItem.scheduled_time > datetime.utcnow()
    ).count()

    # Failed in last 24h
    yesterday = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    failed_recent = db.query(PublishingQueueItem).filter(
        PublishingQueueItem.status == "failed",
        PublishingQueueItem.created_at >= yesterday
    ).count()

    return {
        "queue_by_status": {s: c for s, c in queue_stats},
        "published_today": published_today,
        "pending_approval": pending_approval,
        "scheduled": scheduled,
        "failed_recent": failed_recent,
    }


# === MANUAL QUEUE ADD ===

class ManualQueueAdd(BaseModel):
    clip_id: int
    config_id: int
    scheduled_time: Optional[datetime] = None
    target_platforms: Optional[List[str]] = None


@router.post("/queue")
def add_to_queue(request: ManualQueueAdd, db: Session = Depends(get_db)):
    """Manually add a clip to the publishing queue."""
    clip = db.query(ViralClip).filter(ViralClip.id == request.clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if clip.status != "ready":
        raise HTTPException(status_code=400, detail="Clip must be rendered (status='ready') to queue for publishing")

    config = db.query(PublishingConfig).filter(PublishingConfig.id == request.config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Publishing config not found")

    # Check if clip is already in queue
    existing = db.query(PublishingQueueItem).filter(
        PublishingQueueItem.clip_id == request.clip_id,
        PublishingQueueItem.status.in_(["pending", "approved"])
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Clip is already in the publishing queue")

    queue_item = PublishingQueueItem(
        clip_id=request.clip_id,
        config_id=request.config_id,
        scheduled_time=request.scheduled_time or datetime.utcnow(),
        target_platforms=request.target_platforms or config.platforms or [],
        requires_approval=config.require_manual_approval,
        status="pending" if config.require_manual_approval else "approved"
    )
    db.add(queue_item)

    clip.publishing_status = "queued"
    db.commit()
    db.refresh(queue_item)

    logger.info(f"Manually added clip {request.clip_id} to queue (item {queue_item.id})")
    return {"id": queue_item.id, "status": queue_item.status}
