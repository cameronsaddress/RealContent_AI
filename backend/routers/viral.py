from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import or_, func, case
from datetime import datetime
from typing import List, Optional
from datetime import datetime
import httpx
import os
from pathlib import Path

from models import (
    get_db, Influencer, InfluencerVideo, ViralClip, ClipPersona,
    InfluencerPlatformType, RenderTemplate, PublishingConfig, PublishingQueueItem
)
from pydantic import BaseModel
from utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/viral", tags=["viral"])

# Import Celery task
from tasks.viral import process_viral_clip_render, publish_single_clip

# --- Schemas ---
class InfluencerCreate(BaseModel):
    name: str
    platform: str
    channel_url: str
    persona_id: int

class VideoAnalyzeRequest(BaseModel):
    video_id: int
    persona_id: Optional[int] = None


# --- Auto-Mode Schemas ---
class AutoModeConfig(BaseModel):
    """Configuration for auto-mode processing of influencer videos."""
    auto_mode_enabled: bool
    fetch_frequency_hours: int = 24
    max_videos_per_fetch: int = 5
    auto_analyze_enabled: bool = True
    auto_render_enabled: bool = False
    auto_publish_enabled: bool = False


# --- Endpoints ---

@router.get("/influencers", response_model=List[dict])
def list_influencers(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    # Clamp limit to reasonable max
    limit = min(limit, 500)

    # Subquery for video counts per influencer
    video_count_subq = (
        db.query(
            InfluencerVideo.influencer_id,
            func.count(InfluencerVideo.id).label('video_count')
        )
        .group_by(InfluencerVideo.influencer_id)
        .subquery()
    )

    # Subquery for clip counts per influencer (total and ready)
    clip_count_subq = (
        db.query(
            InfluencerVideo.influencer_id,
            func.count(ViralClip.id).label('clip_count'),
            func.sum(case(
                (ViralClip.status.in_(['ready', 'completed']), 1),
                else_=0
            )).label('ready_clips')
        )
        .join(ViralClip, ViralClip.source_video_id == InfluencerVideo.id)
        .group_by(InfluencerVideo.influencer_id)
        .subquery()
    )

    # Main query with eager loading of videos for thumbnail extraction
    influencers = (
        db.query(Influencer)
        .options(selectinload(Influencer.videos))
        .outerjoin(video_count_subq, Influencer.id == video_count_subq.c.influencer_id)
        .outerjoin(clip_count_subq, Influencer.id == clip_count_subq.c.influencer_id)
        .add_columns(
            func.coalesce(video_count_subq.c.video_count, 0).label('video_count'),
            func.coalesce(clip_count_subq.c.clip_count, 0).label('clip_count'),
            func.coalesce(clip_count_subq.c.ready_clips, 0).label('ready_clips')
        )
        .offset(skip)
        .limit(limit)
        .all()
    )

    result = []
    for row in influencers:
        i = row[0]  # Influencer object
        video_count = row.video_count
        clip_count = row.clip_count
        ready_clips = row.ready_clips

        # Get thumbnail from most recent video with a thumbnail (already loaded)
        thumbnail_url = None
        for v in sorted(i.videos, key=lambda x: x.publication_date or x.created_at or datetime.min, reverse=True):
            if v.thumbnail_url:
                thumbnail_url = v.thumbnail_url
                break

        result.append({
            "id": i.id,
            "name": i.name,
            "platform": i.platform,
            "channel_url": i.channel_url,
            "profile_image_url": i.profile_image_url,
            "thumbnail_url": thumbnail_url,
            "video_count": video_count,
            "clip_count": clip_count,
            "ready_clips": ready_clips,
            "auto_mode_enabled": i.auto_mode_enabled or False,
            "auto_mode_enabled_at": i.auto_mode_enabled_at.isoformat() if i.auto_mode_enabled_at else None,
            "last_fetch_at": i.last_fetch_at.isoformat() if i.last_fetch_at else None,
            "fetch_frequency_hours": i.fetch_frequency_hours or 24
        })
    return result

@router.post("/influencers")
def create_influencer(inf: InfluencerCreate, db: Session = Depends(get_db)):
    # Basic logic, can be expanded
    db_inf = Influencer(
        name=inf.name,
        platform=inf.platform,
        channel_url=inf.channel_url,
        persona_id=inf.persona_id
    )
    db.add(db_inf)
    db.commit()
    db.refresh(db_inf)
    return db_inf


@router.delete("/influencers/{id}")
def delete_influencer(id: int, db: Session = Depends(get_db)):
    """Delete an influencer and cascade delete all their videos and clips."""
    inf = db.query(Influencer).filter(Influencer.id == id).first()
    if not inf:
        raise HTTPException(status_code=404, detail="Influencer not found")

    # Get counts for response
    video_count = db.query(InfluencerVideo).filter(InfluencerVideo.influencer_id == id).count()
    clip_count = db.query(ViralClip).join(InfluencerVideo).filter(
        InfluencerVideo.influencer_id == id
    ).count()

    # Delete clips first (foreign key constraint)
    db.query(ViralClip).filter(
        ViralClip.source_video_id.in_(
            db.query(InfluencerVideo.id).filter(InfluencerVideo.influencer_id == id)
        )
    ).delete(synchronize_session=False)

    # Delete videos
    db.query(InfluencerVideo).filter(InfluencerVideo.influencer_id == id).delete(synchronize_session=False)

    # Delete influencer
    influencer_name = inf.name
    db.delete(inf)
    db.commit()

    logger.info(f"Deleted influencer '{influencer_name}' (id={id}) with {video_count} videos and {clip_count} clips")

    return {
        "status": "deleted",
        "influencer_id": id,
        "influencer_name": influencer_name,
        "videos_deleted": video_count,
        "clips_deleted": clip_count
    }


# --- Auto-Mode Configuration Endpoints ---

@router.get("/influencers/{influencer_id}/auto-mode")
def get_influencer_auto_mode(influencer_id: int, db: Session = Depends(get_db)):
    """Get auto-mode configuration for an influencer."""
    influencer = db.query(Influencer).filter(Influencer.id == influencer_id).first()
    if not influencer:
        raise HTTPException(status_code=404, detail="Influencer not found")

    return {
        "auto_mode_enabled": influencer.auto_mode_enabled or False,
        "auto_mode_enabled_at": influencer.auto_mode_enabled_at.isoformat() if influencer.auto_mode_enabled_at else None,
        "fetch_frequency_hours": influencer.fetch_frequency_hours or 24,
        "max_videos_per_fetch": influencer.max_videos_per_fetch or 5,
        "auto_analyze_enabled": influencer.auto_analyze_enabled if influencer.auto_analyze_enabled is not None else True,
        "auto_render_enabled": influencer.auto_render_enabled or False,
        "auto_publish_enabled": influencer.auto_publish_enabled or False,
        "last_fetch_at": influencer.last_fetch_at.isoformat() if influencer.last_fetch_at else None
    }


@router.put("/influencers/{influencer_id}/auto-mode")
def configure_influencer_auto_mode(influencer_id: int, config: AutoModeConfig, db: Session = Depends(get_db)):
    """Enable/disable auto-mode and configure settings for an influencer."""
    influencer = db.query(Influencer).filter(Influencer.id == influencer_id).first()
    if not influencer:
        raise HTTPException(status_code=404, detail="Influencer not found")

    # Track when auto-mode was enabled (for NEW videos only constraint)
    was_enabled = influencer.auto_mode_enabled
    now_enabled = config.auto_mode_enabled

    if now_enabled and not was_enabled:
        # Just enabled - record timestamp for NEW videos constraint
        influencer.auto_mode_enabled_at = datetime.utcnow()
        logger.info(f"Auto-mode enabled for {influencer.name} at {influencer.auto_mode_enabled_at}")
    elif not now_enabled and was_enabled:
        # Disabled - clear the timestamp
        influencer.auto_mode_enabled_at = None
        logger.info(f"Auto-mode disabled for {influencer.name}")

    influencer.auto_mode_enabled = config.auto_mode_enabled
    influencer.fetch_frequency_hours = config.fetch_frequency_hours
    influencer.max_videos_per_fetch = config.max_videos_per_fetch
    influencer.auto_analyze_enabled = config.auto_analyze_enabled
    # Auto-render is always enabled when auto-mode is on
    influencer.auto_render_enabled = config.auto_mode_enabled
    influencer.auto_publish_enabled = config.auto_publish_enabled

    db.commit()

    return {
        "status": "updated",
        "influencer_id": influencer_id,
        "auto_mode_enabled": influencer.auto_mode_enabled,
        "auto_mode_enabled_at": influencer.auto_mode_enabled_at.isoformat() if influencer.auto_mode_enabled_at else None
    }


@router.post("/influencers/{id}/fetch")
async def fetch_videos(id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    inf = db.query(Influencer).filter(Influencer.id == id).first()
    if not inf:
        raise HTTPException(status_code=404, detail="Influencer not found")
    
    # Call Video Processor (Video Service)
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "http://video-processor:8080/fetch-channel",
                json={"url": inf.channel_url, "limit": 10, "platform": inf.platform},
                timeout=60.0
            )
            if resp.status_code == 200:
                data = resp.json()
                videos = data.get("videos", [])
                created_count = 0
                for v in videos:
                    # Check duplicate by platform_video_id (not URL - URLs can have different tracking params)
                    video_id = v.get("id")
                    exists = db.query(InfluencerVideo).filter(
                        InfluencerVideo.influencer_id == id,
                        InfluencerVideo.platform_video_id == video_id
                    ).first()
                    if not exists:
                        new_video = InfluencerVideo(
                            influencer_id=id,
                            platform_video_id=v.get("id"),
                            title=v.get("title"),
                            url=v.get("url"),
                            thumbnail_url=v.get("thumbnail"),
                            duration=v.get("duration"),
                            publication_date=datetime.strptime(v["upload_date"], "%Y%m%d") if v.get("upload_date") else None,
                            view_count=v.get("view_count", 0),
                            status="pending"
                        )
                        db.add(new_video)
                        created_count += 1
                db.commit()
                return {"message": f"Fetched {len(videos)} videos, added {created_count} new"}
            else:
                raise HTTPException(status_code=500, detail=f"Fetch error: {resp.text}")
        except Exception as e:
             raise HTTPException(status_code=500, detail=str(e))

@router.get("/videos/{id}/details")
def get_video_details(id: int, db: Session = Depends(get_db)):
    video = db.query(InfluencerVideo).filter(InfluencerVideo.id == id).first()
    if not video:
        raise HTTPException(status_code=404)
    return video

@router.get("/videos/{id}/download-progress")
async def get_download_progress(id: int, db: Session = Depends(get_db)):
    """Get download progress for a video being downloaded."""
    import json as json_mod
    video = db.query(InfluencerVideo).filter(InfluencerVideo.id == id).first()
    if not video:
        raise HTTPException(status_code=404)
    if video.status != "downloading":
        return {"status": video.status, "bytes_downloaded": 0}

    assets_base = os.environ.get("ASSETS_BASE_PATH", "/app/assets")
    final_path = os.path.join(assets_base, "videos", f"video_{id}.mp4")
    part_path = f"{final_path}.part"
    ytdl_path = f"{final_path}.ytdl"

    # Check if final file exists (download complete)
    if os.path.exists(final_path):
        return {"status": "complete", "bytes_downloaded": os.path.getsize(final_path)}

    # Check .part file
    if not os.path.exists(part_path):
        return {"status": "waiting", "bytes_downloaded": 0}

    size = os.path.getsize(part_path)

    # Try to get fragment info from .ytdl file
    fragment_info = None
    if os.path.exists(ytdl_path):
        try:
            with open(ytdl_path) as f:
                ytdl_data = json_mod.load(f)
            dl_info = ytdl_data.get("downloader", {})
            current_frag = dl_info.get("current_fragment", {}).get("index", 0)
            total_frags = dl_info.get("fragment_count", 0)
            if current_frag > 0:
                fragment_info = {"current": current_frag, "total": total_frags if total_frags > 0 else None}
        except Exception:
            pass

    return {"status": "downloading", "bytes_downloaded": size, "fragment_info": fragment_info}

@router.post("/viral-clips/{clip_id}/render")
async def render_viral_clip(clip_id: int, db: Session = Depends(get_db)):
    clip = db.query(ViralClip).filter(ViralClip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
        
    # Queue the Celery task
    clip.status = "queued for rendering"
    db.commit()
    
    process_viral_clip_render.delay(clip_id)
    
    return {"message": "Rendering started", "clip_id": clip_id}


class PublishClipRequest(BaseModel):
    platforms: Optional[List[str]] = None  # None = all configured platforms


@router.post("/viral-clips/{clip_id}/publish")
async def publish_clip_now(clip_id: int, body: PublishClipRequest = None, db: Session = Depends(get_db)):
    """Immediately publish a ready clip to Blotato accounts."""
    clip = db.query(ViralClip).filter(ViralClip.id == clip_id).first()
    if not clip or clip.status != "ready":
        raise HTTPException(status_code=404, detail="Clip not found or not ready")

    video = db.query(InfluencerVideo).filter(InfluencerVideo.id == clip.source_video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Source video not found")

    influencer = db.query(Influencer).filter(Influencer.id == video.influencer_id).first()
    if not influencer:
        raise HTTPException(status_code=404, detail="Influencer not found")

    configs = db.query(PublishingConfig).filter(
        or_(
            PublishingConfig.influencer_id == influencer.id,
            PublishingConfig.persona_id == influencer.persona_id,
        ),
        PublishingConfig.enabled == True,
    ).all()

    if not configs:
        raise HTTPException(status_code=400, detail="No active publishing configs for this influencer")

    requested_platforms = (body.platforms if body and body.platforms else None)
    queued = 0
    config_names = []

    for config in configs:
        # Filter platforms if specified
        config_platforms = config.platforms or ["tiktok"]
        if requested_platforms:
            platforms_to_use = [p for p in config_platforms if p in requested_platforms]
            if not platforms_to_use:
                continue
        else:
            platforms_to_use = config_platforms

        # Skip if already queued
        existing = db.query(PublishingQueueItem).filter(
            PublishingQueueItem.clip_id == clip.id,
            PublishingQueueItem.config_id == config.id,
            PublishingQueueItem.status.in_(["approved", "publishing", "published"]),
        ).first()
        if existing:
            continue

        item = PublishingQueueItem(
            clip_id=clip.id,
            config_id=config.id,
            target_platforms=platforms_to_use,
            status="approved",
            scheduled_time=datetime.utcnow(),
            requires_approval=False,
        )
        db.add(item)
        db.flush()
        publish_single_clip.delay(item.id)
        queued += 1
        config_names.append(config.blotato_account_name or str(config.blotato_account_id))

    if queued > 0:
        clip.publishing_status = "queued"
        db.commit()

    return {"queued": queued, "configs": config_names, "clip_id": clip_id}


@router.get("/publishing-configs/{influencer_id}")
async def get_publishing_configs(influencer_id: int, db: Session = Depends(get_db)):
    """Get publishing configs for an influencer."""
    influencer = db.query(Influencer).filter(Influencer.id == influencer_id).first()
    configs = db.query(PublishingConfig).filter(
        or_(
            PublishingConfig.influencer_id == influencer_id,
            PublishingConfig.persona_id == (influencer.persona_id if influencer else None),
        ),
        PublishingConfig.enabled == True,
    ).all()
    return [
        {
            "id": c.id,
            "blotato_account_id": c.blotato_account_id,
            "blotato_account_name": c.blotato_account_name,
            "platforms": c.platforms,
            "enabled": c.enabled,
        }
        for c in configs
    ]


@router.get("/settings/auto-publish")
async def get_auto_publish_setting(db: Session = Depends(get_db)):
    """Get the global auto-publish toggle state."""
    from models import SystemSettings
    setting = db.query(SystemSettings).filter(SystemSettings.key == "auto_publish_enabled").first()
    return {"enabled": setting.value if setting else False}


class AutoPublishToggle(BaseModel):
    enabled: bool

@router.put("/settings/auto-publish")
async def set_auto_publish_setting(body: AutoPublishToggle, db: Session = Depends(get_db)):
    """Set the global auto-publish toggle."""
    from models import SystemSettings
    setting = db.query(SystemSettings).filter(SystemSettings.key == "auto_publish_enabled").first()
    if setting:
        setting.value = body.enabled
    else:
        setting = SystemSettings(key="auto_publish_enabled", value=body.enabled)
        db.add(setting)
    db.commit()
    return {"enabled": body.enabled}


@router.post("/videos/{id}/analyze")
async def analyze_video(id: int, db: Session = Depends(get_db)):
    # This involves:
    # 1. Download (if not local)
    # 2. Transcribe (if not transcribed)
    # 3. Grok Analysis
    # We'll toggle status to 'analyzing' and trigger background task
    video = db.query(InfluencerVideo).filter(InfluencerVideo.id == id).first()
    if not video:
         raise HTTPException(status_code=404)
    
    def _video_file_exists(local_path: Optional[str]) -> bool:
        if not local_path:
            return False
        check_path = local_path
        if check_path.startswith("/downloads/"):
            check_path = check_path.replace("/downloads/", "/app/assets/videos/")
        return os.path.exists(check_path)

    file_exists = _video_file_exists(video.local_path)

    if video.transcript_json and file_exists:
        video.status = "transcribed"
        db.commit()
        from tasks.viral import process_viral_video_analyze
        process_viral_video_analyze.delay(id)
        return {"message": "Analysis pipeline started (Analyze only)"}

    if file_exists:
        video.status = "downloaded"
        db.commit()
        from tasks.viral import process_viral_video_transcribe
        process_viral_video_transcribe.delay(id)
        return {"message": "Analysis pipeline started (Transcribe -> Analyze)"}

    # video.status = "processing"
    # FIX: Do not overwrite 'transcribed' or 'downloaded' status,
    # otherwise the shortcut logic in tasks/viral.py will fail to detect them!
    if video.status not in ["transcribed", "downloaded"]:
        video.status = "processing"

    db.commit()

    # Trigger Celery pipeline (Download -> Transcribe -> Analyze)
    from tasks.viral import process_viral_video_download
    process_viral_video_download.delay(id)

    return {"message": "Analysis pipeline started (Download -> Transcribe -> Analyze)"}


@router.get("/influencers/{id}/videos", response_model=List[dict])
def list_influencer_videos(id: int, db: Session = Depends(get_db)):
    videos = db.query(InfluencerVideo).filter(InfluencerVideo.influencer_id == id).order_by(InfluencerVideo.created_at.desc()).limit(50).all()
    # Serialize simple
    return [{
        "id": v.id,
        "title": v.title,
        "url": v.url,
        "status": v.status,
        "thumbnail_url": v.thumbnail_url,
        "duration": v.duration,
        "error_message": v.error_message,
        "publication_date": v.publication_date,
        "created_at": v.created_at,
        "upload_date": v.publication_date,
        "processing_started_at": v.processing_started_at.isoformat() if v.processing_started_at else None,
        "clips": [{
            "id": c.id,
            "title": c.title,
            "status": c.status,
            "clip_type": c.clip_type,
            "edited_video_path": c.edited_video_path,
            "template_id": c.template_id,
            "recommended_template_id": c.recommended_template_id,
            "publishing_status": c.publishing_status
        } for c in v.clips]
    } for v in videos]
    
@router.get("/viral-clips", response_model=List[dict])
def list_viral_clips(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    # Clamp limit to reasonable max
    limit = min(limit, 500)

    # Join with InfluencerVideo and Influencer to get influencer name and id
    clips = db.query(
        ViralClip,
        Influencer.name.label("influencer_name"),
        Influencer.id.label("influencer_id")
    ).outerjoin(
        InfluencerVideo, ViralClip.source_video_id == InfluencerVideo.id
    ).outerjoin(
        Influencer, InfluencerVideo.influencer_id == Influencer.id
    ).order_by(ViralClip.created_at.desc()).offset(skip).limit(limit).all()

    return [{
        "id": c.ViralClip.id,
        "title": c.ViralClip.title,
        "hashtags": c.ViralClip.hashtags,
        "clip_type": c.ViralClip.clip_type,
        "status": c.ViralClip.status,
        "source_video_id": c.ViralClip.source_video_id,
        "edited_video_path": c.ViralClip.edited_video_path,
        "template_id": c.ViralClip.template_id,
        "recommended_template_id": c.ViralClip.recommended_template_id,
        "render_metadata": c.ViralClip.render_metadata,
        "publishing_status": c.ViralClip.publishing_status,
        "influencer_name": c.influencer_name,
        "influencer_id": c.influencer_id,
        "created_at": c.ViralClip.created_at.isoformat() if c.ViralClip.created_at else None,
        "updated_at": c.ViralClip.updated_at.isoformat() if c.ViralClip.updated_at else None
    } for c in clips]


from fastapi.responses import FileResponse

# Valid status values for viral clips (used by webhook callbacks)
VALID_CLIP_STATUSES = [
    "pending",
    "queued for rendering",
    "rendering",
    "ready",
    "published",
    "error"
]

# Status prefixes allowed for progress updates (video-processor sends "Processing: ..." updates)
VALID_STATUS_PREFIXES = ["Processing:", "queued ("]

def is_valid_clip_status(status: str) -> bool:
    """Check if status is valid - either in whitelist or has valid prefix."""
    if status in VALID_CLIP_STATUSES:
        return True
    for prefix in VALID_STATUS_PREFIXES:
        if status.startswith(prefix):
            return True
    return False

class StatusUpdate(BaseModel):
    status: str
    error_message: Optional[str] = None

@router.put("/viral-clips/{id}/status")
def update_viral_clip_status(id: int, status_update: StatusUpdate, db: Session = Depends(get_db)):
    # Validate status value
    if not is_valid_clip_status(status_update.status):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of {VALID_CLIP_STATUSES} or start with {VALID_STATUS_PREFIXES}"
        )

    clip = db.query(ViralClip).filter(ViralClip.id == id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    clip.status = status_update.status
    if status_update.error_message:
        clip.error_message = status_update.error_message
    db.commit()
    return {"status": "updated", "clip_id": id, "new_status": status_update.status}

@router.get("/file/{filename}")
async def get_viral_file(filename: str, request: Request):
    """
    Stream video file with HTTP Range request support.
    Enables seeking and progressive loading like YouTube.
    """
    # Security: basic check to prevent traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = Path(f"/app/assets/output/{filename}")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    # Default: stream entire file
    start = 0
    end = file_size - 1

    if range_header:
        # Parse range header: "bytes=0-1000" or "bytes=0-"
        range_str = range_header.replace("bytes=", "")
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)  # Clamp to file size

    content_length = end - start + 1

    def stream_file():
        """Generator to stream file chunks."""
        chunk_size = 1024 * 1024  # 1MB chunks
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "video/mp4",
    }

    status_code = 206 if range_header else 200
    return StreamingResponse(stream_file(), status_code=status_code, headers=headers)

@router.get("/music")
def list_music():
    music_dir = Path("/app/assets/music")
    if not music_dir.exists():
        return []
    files = []
    for f in music_dir.glob("*"):
        if f.suffix.lower() in ['.mp3', '.wav']:
             files.append({"name": f.name, "size": f.stat().st_size})
    return files

@router.get("/music/{filename}")
def get_music_file(filename: str):
    if ".." in filename or "/" in filename:
         raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = f"/app/assets/music/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)

# --- Font Management (proxy to video-processor) ---

@router.get("/fonts")
async def list_fonts():
    """List installed custom fonts from video-processor."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get("http://video-processor:8080/fonts")
        return resp.json()

@router.get("/fonts/{filename}/file")
async def serve_font_file(filename: str):
    """Proxy font file from video-processor for browser loading."""
    from fastapi.responses import Response

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"http://video-processor:8080/fonts/{filename}/file")
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Font not found")

        # Determine MIME type
        mime_type = "font/ttf" if filename.endswith(".ttf") else "font/otf"

        return Response(
            content=resp.content,
            media_type=mime_type,
            headers={"Access-Control-Allow-Origin": "*"}
        )

@router.delete("/fonts/{filename}")
async def delete_font(filename: str):
    """Delete a custom font from video-processor."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.delete(f"http://video-processor:8080/fonts/{filename}")
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.json().get("detail", "Failed"))
        return resp.json()

class GoogleFontDownload(BaseModel):
    font_name: str

@router.post("/fonts/google")
async def download_google_font(req: GoogleFontDownload):
    """Download a font from Google Fonts via video-processor."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "http://video-processor:8080/fonts/google",
            json={"font_name": req.font_name}
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.json().get("detail", "Failed"))
        return resp.json()

# --- Render Templates ---

class TemplateCreate(BaseModel):
    name: str
    description: Optional[str] = None
    broll_categories: List[str] = []
    broll_enabled: bool = True
    effect_settings: dict = {}
    keywords: List[str] = []
    is_default: bool = False

class TemplateUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    broll_categories: Optional[List[str]] = None
    broll_enabled: Optional[bool] = None
    effect_settings: Optional[dict] = None
    keywords: Optional[List[str]] = None
    is_default: Optional[bool] = None

@router.get("/templates", response_model=List[dict])
def list_templates(db: Session = Depends(get_db)):
    """List all render templates."""
    templates = db.query(RenderTemplate).order_by(RenderTemplate.sort_order).all()
    return [{
        "id": t.id,
        "name": t.name,
        "description": t.description,
        "broll_categories": t.broll_categories or [],
        "broll_enabled": t.broll_enabled,
        "effect_settings": t.effect_settings or {},
        "keywords": t.keywords or [],
        "is_default": t.is_default,
        "sort_order": t.sort_order
    } for t in templates]

@router.get("/templates/{id}")
def get_template(id: int, db: Session = Depends(get_db)):
    """Get a single template by ID."""
    t = db.query(RenderTemplate).filter(RenderTemplate.id == id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "id": t.id,
        "name": t.name,
        "description": t.description,
        "broll_categories": t.broll_categories or [],
        "broll_enabled": t.broll_enabled,
        "effect_settings": t.effect_settings or {},
        "keywords": t.keywords or [],
        "is_default": t.is_default,
        "sort_order": t.sort_order
    }

@router.post("/templates")
def create_template(template: TemplateCreate, db: Session = Depends(get_db)):
    """Create a new render template."""
    # If setting as default, unset other defaults
    if template.is_default:
        db.query(RenderTemplate).update({"is_default": False})

    # Get max sort_order
    max_order = db.query(RenderTemplate).count()

    new_template = RenderTemplate(
        name=template.name,
        description=template.description,
        broll_categories=template.broll_categories,
        broll_enabled=template.broll_enabled,
        effect_settings=template.effect_settings,
        keywords=template.keywords,
        is_default=template.is_default,
        sort_order=max_order + 1
    )
    db.add(new_template)
    db.commit()
    db.refresh(new_template)
    return {"id": new_template.id, "name": new_template.name}

@router.put("/templates/{id}")
def update_template(id: int, template: TemplateUpdate, db: Session = Depends(get_db)):
    """Update a render template."""
    t = db.query(RenderTemplate).filter(RenderTemplate.id == id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")

    # If setting as default, unset other defaults
    if template.is_default:
        db.query(RenderTemplate).filter(RenderTemplate.id != id).update({"is_default": False})

    if template.name is not None:
        t.name = template.name
    if template.description is not None:
        t.description = template.description
    if template.broll_categories is not None:
        t.broll_categories = template.broll_categories
    if template.broll_enabled is not None:
        t.broll_enabled = template.broll_enabled
    if template.effect_settings is not None:
        t.effect_settings = template.effect_settings
    if template.keywords is not None:
        t.keywords = template.keywords
    if template.is_default is not None:
        t.is_default = template.is_default

    db.commit()
    return {"id": t.id, "name": t.name, "updated": True}

@router.delete("/templates/{id}")
def delete_template(id: int, db: Session = Depends(get_db)):
    """Delete a render template."""
    t = db.query(RenderTemplate).filter(RenderTemplate.id == id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")

    db.delete(t)
    db.commit()
    return {"deleted": True, "id": id}

@router.put("/viral-clips/{clip_id}/template")
def set_clip_template(clip_id: int, template_id: int, db: Session = Depends(get_db)):
    """Set or change the template for a viral clip."""
    clip = db.query(ViralClip).filter(ViralClip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if template_id > 0:
        template = db.query(RenderTemplate).filter(RenderTemplate.id == template_id).first()
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        clip.template_id = template_id
    else:
        clip.template_id = None  # Clear template

    db.commit()
    return {"clip_id": clip_id, "template_id": clip.template_id}


# =============================================================================
# B-ROLL MANAGEMENT ENDPOINTS
# =============================================================================

class BRollUploadRequest(BaseModel):
    youtube_url: str
    clip_duration: float = 2.0  # Duration of each clip in seconds

# In-memory job tracking (for simplicity - could use Redis for persistence)
broll_upload_jobs = {}
BROLL_JOB_TTL_SECONDS = 3600  # Clean up jobs older than 1 hour

def cleanup_old_broll_jobs():
    """Remove jobs older than TTL to prevent memory leaks."""
    now = datetime.now()
    expired_keys = [
        job_id for job_id, job in broll_upload_jobs.items()
        if (now - job.get("created_at", now)).total_seconds() > BROLL_JOB_TTL_SECONDS
    ]
    for job_id in expired_keys:
        del broll_upload_jobs[job_id]
    if expired_keys:
        logger.debug(f"Cleaned up {len(expired_keys)} expired B-roll jobs")

@router.get("/broll")
def list_broll_clips(
    limit: int = 50,
    offset: int = 0,
    category: str = None
):
    """
    List B-roll clips with pagination and optional category filter.
    Returns clip info from metadata.json if available.

    Query params:
    - limit: max clips to return (default 50)
    - offset: starting position (default 0)
    - category: filter by category (optional, 'untagged' for untagged clips)
    """
    import json
    broll_dir = Path("/app/assets/broll")
    metadata_path = broll_dir / "metadata.json"

    # Load metadata if exists
    metadata = None
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except:
            pass

    # Build full clips list
    all_clips = []
    if metadata and "clips" in metadata:
        for clip_data in metadata["clips"]:
            all_clips.append({
                "filename": clip_data.get("filename"),
                "caption": clip_data.get("caption", ""),
                "categories": clip_data.get("categories", []),
                "duration": clip_data.get("duration", 2.0),
                "tagged": True
            })
    else:
        # No metadata - just list files
        if broll_dir.exists():
            for f in broll_dir.glob("*.mp4"):
                if not f.name.startswith("pexels_"):
                    all_clips.append({
                        "filename": f.name,
                        "caption": "",
                        "categories": [],
                        "duration": 2.0,
                        "tagged": False
                    })

    # Get category counts (always from full list)
    category_counts = {}
    if metadata and "category_counts" in metadata:
        category_counts = metadata["category_counts"]

    # Apply category filter
    if category:
        if category == "untagged":
            filtered_clips = [c for c in all_clips if not c.get("categories")]
        else:
            filtered_clips = [c for c in all_clips if category in c.get("categories", [])]
    else:
        filtered_clips = all_clips

    total_filtered = len(filtered_clips)

    # Apply pagination
    paginated_clips = filtered_clips[offset:offset + limit]

    return {
        "clips": paginated_clips,
        "metadata": {
            "total_clips": len(all_clips),
            "filtered_clips": total_filtered,
            "tagged_clips": sum(1 for c in all_clips if c.get("tagged")),
            "category_counts": category_counts,
            "generated_at": metadata.get("generated_at") if metadata else None
        },
        "pagination": {
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_filtered
        }
    }

@router.post("/broll/upload-youtube")
async def upload_youtube_broll(request: BRollUploadRequest, background_tasks: BackgroundTasks):
    """
    Upload a YouTube video and split it into B-roll clips.

    Process:
    1. Download video via video-processor
    2. Split into 2-second clips
    3. Tag each clip with AI (BLIP + CLIP)
    4. Add to metadata.json

    Returns job_id to track progress.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]

    # Clean up old jobs to prevent memory leaks
    cleanup_old_broll_jobs()

    # Initialize job status
    broll_upload_jobs[job_id] = {
        "status": "queued",
        "youtube_url": request.youtube_url,
        "clip_duration": request.clip_duration,
        "progress": 0,
        "message": "Job queued",
        "clips_created": 0,
        "created_at": datetime.now()
    }

    # Run in background
    background_tasks.add_task(
        process_broll_upload,
        job_id,
        request.youtube_url,
        request.clip_duration
    )

    return {"job_id": job_id, "status": "queued"}

@router.get("/broll/status/{job_id}")
def get_broll_upload_status(job_id: str):
    """Get the status of a B-roll upload job."""
    # Clean up old jobs on status check
    cleanup_old_broll_jobs()

    if job_id not in broll_upload_jobs:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    # Return job without exposing internal created_at field
    job = broll_upload_jobs[job_id].copy()
    job.pop("created_at", None)
    return job

@router.get("/broll/usage")
def get_broll_usage_stats():
    """
    Get B-roll usage statistics showing which clips have been used per category.
    Helps track round-robin usage and see when clips were last used.
    """
    import json
    from pathlib import Path

    broll_dir = Path("/app/assets/broll")
    used_clips_path = broll_dir / "used_clips.json"
    metadata_path = broll_dir / "metadata.json"

    # Load usage data
    usage_data = {}
    if used_clips_path.exists():
        try:
            with open(used_clips_path, "r") as f:
                usage_data = json.load(f)
        except (IOError, json.JSONDecodeError):
            pass

    # Load metadata for total counts
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except (IOError, json.JSONDecodeError):
            pass

    category_index = metadata.get("category_index", {})

    # Build stats per category
    stats = {}
    for category, clips_used in usage_data.items():
        total_in_category = len(category_index.get(category, []))
        used_count = len(clips_used) if isinstance(clips_used, dict) else 0

        # Get most recent and oldest used times
        timestamps = []
        if isinstance(clips_used, dict):
            timestamps = list(clips_used.values())
            timestamps.sort()

        stats[category] = {
            "total_clips": total_in_category,
            "used_clips": used_count,
            "remaining": max(0, total_in_category - used_count),
            "usage_percent": round((used_count / total_in_category * 100), 1) if total_in_category > 0 else 0,
            "oldest_used": timestamps[0] if timestamps else None,
            "newest_used": timestamps[-1] if timestamps else None,
            "exhausted": used_count >= total_in_category and total_in_category > 0
        }

    # Add categories that haven't been used yet
    for category in category_index.keys():
        if category not in stats:
            stats[category] = {
                "total_clips": len(category_index[category]),
                "used_clips": 0,
                "remaining": len(category_index[category]),
                "usage_percent": 0,
                "oldest_used": None,
                "newest_used": None,
                "exhausted": False
            }

    # Sort by usage percentage descending
    sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1]["usage_percent"], reverse=True))

    return {
        "categories": sorted_stats,
        "summary": {
            "total_categories": len(sorted_stats),
            "categories_with_usage": len([c for c in sorted_stats.values() if c["used_clips"] > 0]),
            "exhausted_categories": len([c for c in sorted_stats.values() if c["exhausted"]])
        }
    }

@router.post("/broll/reset-usage")
def reset_broll_usage(category: str = None):
    """
    Reset B-roll usage tracking. Optionally reset a single category.
    """
    import json
    from pathlib import Path

    broll_dir = Path("/app/assets/broll")
    used_clips_path = broll_dir / "used_clips.json"

    if category:
        # Reset single category
        usage_data = {}
        if used_clips_path.exists():
            try:
                with open(used_clips_path, "r") as f:
                    usage_data = json.load(f)
            except (IOError, json.JSONDecodeError):
                pass

        if category in usage_data:
            del usage_data[category]
            with open(used_clips_path, "w") as f:
                json.dump(usage_data, f, indent=2)
            return {"message": f"Reset usage for category '{category}'"}
        else:
            return {"message": f"Category '{category}' had no usage to reset"}
    else:
        # Reset all
        with open(used_clips_path, "w") as f:
            json.dump({}, f)
        return {"message": "Reset all B-roll usage tracking"}

@router.post("/broll/retag")
async def retag_all_broll(background_tasks: BackgroundTasks):
    """
    Re-run AI tagging on all B-roll clips.
    This regenerates metadata.json with fresh BLIP captions and CLIP categories.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]

    # Clean up old jobs to prevent memory leaks
    cleanup_old_broll_jobs()

    broll_upload_jobs[job_id] = {
        "status": "queued",
        "task": "retag",
        "progress": 0,
        "message": "Retagging job queued",
        "created_at": datetime.now()
    }

    background_tasks.add_task(process_broll_retag, job_id)

    return {"job_id": job_id, "status": "queued", "message": "Retagging started"}


async def process_broll_upload(job_id: str, youtube_url: str, clip_duration: float):
    """Background task to download and process YouTube video into B-roll clips."""
    import subprocess
    import json

    broll_dir = Path("/app/assets/broll")
    broll_dir.mkdir(parents=True, exist_ok=True)

    try:
        broll_upload_jobs[job_id]["status"] = "downloading"
        broll_upload_jobs[job_id]["message"] = "Downloading video..."

        # Step 1: Download video via video-processor
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                "http://video-processor:8080/download",
                json={
                    "url": youtube_url,
                    "filename": f"broll_source_{job_id}",
                    "format": "mp4"
                }
            )
            if response.status_code != 200:
                raise Exception(f"Download failed: {response.text}")

            download_result = response.json()
            source_path = download_result.get("path", "").replace("/downloads/", "/app/assets/videos/")

        broll_upload_jobs[job_id]["status"] = "splitting"
        broll_upload_jobs[job_id]["message"] = "Splitting into clips..."
        broll_upload_jobs[job_id]["progress"] = 20

        # Step 2: Get video duration
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", source_path
        ], capture_output=True, text=True)
        total_duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

        if total_duration < clip_duration:
            raise Exception("Video too short")

        # Step 3: Split into clips
        num_clips = int(total_duration / clip_duration)
        clips_created = []

        # Generate safe base name from URL
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', youtube_url.split('/')[-1])[:20]

        for i in range(min(num_clips, 500)):  # Max 500 clips per video
            start_time = i * clip_duration
            output_name = f"{safe_name}_{i:03d}.mp4"
            output_path = broll_dir / output_name

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", source_path,
                "-t", str(clip_duration),
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                "-c:v", "libx264", "-preset", "fast",
                "-an",  # No audio for B-roll
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and output_path.exists():
                clips_created.append(output_name)

            # Update progress
            progress = 20 + int(60 * (i + 1) / num_clips)
            broll_upload_jobs[job_id]["progress"] = progress
            broll_upload_jobs[job_id]["clips_created"] = len(clips_created)

        broll_upload_jobs[job_id]["status"] = "tagging"
        broll_upload_jobs[job_id]["message"] = f"AI tagging {len(clips_created)} clips..."
        broll_upload_jobs[job_id]["progress"] = 80

        # Step 4: Run AI tagging via video-processor
        # This calls the tag_broll.py script
        async with httpx.AsyncClient(timeout=1800.0) as client:  # 30 min for tagging
            response = await client.post(
                "http://video-processor:8080/tag-broll",
                json={"limit": len(clips_created)}
            )
            # Tagging is optional - don't fail if it doesn't work

        # Clean up source video
        if Path(source_path).exists():
            Path(source_path).unlink()

        broll_upload_jobs[job_id]["status"] = "completed"
        broll_upload_jobs[job_id]["message"] = f"Created {len(clips_created)} B-roll clips"
        broll_upload_jobs[job_id]["progress"] = 100
        broll_upload_jobs[job_id]["clips_created"] = len(clips_created)

    except Exception as e:
        broll_upload_jobs[job_id]["status"] = "error"
        broll_upload_jobs[job_id]["message"] = str(e)


async def process_broll_retag(job_id: str):
    """Background task to retag all B-roll clips."""
    try:
        broll_upload_jobs[job_id]["status"] = "running"
        broll_upload_jobs[job_id]["message"] = "Running AI tagging..."

        async with httpx.AsyncClient(timeout=3600.0) as client:  # 60 min
            response = await client.post(
                "http://video-processor:8080/tag-broll",
                json={"force": True}
            )

            if response.status_code == 200:
                result = response.json()
                broll_upload_jobs[job_id]["status"] = "completed"
                broll_upload_jobs[job_id]["message"] = f"Tagged {result.get('clips_tagged', 0)} clips"
            else:
                raise Exception(f"Tagging failed: {response.text}")

    except Exception as e:
        broll_upload_jobs[job_id]["status"] = "error"
        broll_upload_jobs[job_id]["message"] = str(e)


@router.get("/broll/file/{filename}")
def get_broll_file(filename: str):
    """Serve a B-roll video file."""
    broll_dir = Path("/app/assets/broll").resolve()
    # Resolve the full path to catch any traversal attempts (including URL-encoded)
    file_path = (broll_dir / filename).resolve()

    # Security: verify resolved path is within broll directory
    if not str(file_path).startswith(str(broll_dir) + "/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=file_path.name  # Use resolved filename
    )


@router.delete("/broll/{filename}")
def delete_broll_clip(filename: str):
    """Delete a B-roll clip and remove from metadata."""
    import json

    broll_dir = Path("/app/assets/broll").resolve()
    file_path = (broll_dir / filename).resolve()
    metadata_path = broll_dir / "metadata.json"

    # Security: verify resolved path is within broll directory
    if not str(file_path).startswith(str(broll_dir) + "/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Delete the file
    if file_path.exists():
        file_path.unlink()

    # Update metadata.json to remove the clip
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Remove clip from clips list
            if "clips" in metadata:
                metadata["clips"] = [c for c in metadata["clips"] if c.get("filename") != filename]
                metadata["total_clips"] = len(metadata["clips"])

            # Rebuild category index
            category_index = {}
            for clip in metadata.get("clips", []):
                for cat in clip.get("categories", []):
                    if cat not in category_index:
                        category_index[cat] = []
                    category_index[cat].append(clip["filename"])

            metadata["category_index"] = category_index
            metadata["category_counts"] = {k: len(v) for k, v in category_index.items()}

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating metadata after delete: {e}")

    return {"success": True, "deleted": filename}


# --- Effects API ---

class EffectsOverride(BaseModel):
    """User override for Grok director effects on a clip."""
    color_grade: Optional[str] = None
    camera_shake: Optional[dict] = None
    speed_ramps: Optional[list] = None
    retro_glow: Optional[float] = None
    temporal_trail: Optional[dict] = None
    wave_displacement: Optional[dict] = None
    caption_style: Optional[str] = None
    beat_sync: Optional[bool] = None
    audio_saturation: Optional[bool] = None
    transition: Optional[str] = None
    pulse_intensity: Optional[float] = None
    vhs_intensity: Optional[float] = None
    datamosh_segments: Optional[list] = None
    pixel_sort_segments: Optional[list] = None

@router.get("/effects-catalog")
def get_effects_catalog():
    """Return available effects catalog for the frontend override UI."""
    from services.clip_analyzer import EFFECT_CATALOG, VALID_EFFECT_KEYS

    return {
        "catalog_text": EFFECT_CATALOG,
        "valid_keys": list(VALID_EFFECT_KEYS),
        "color_grades": [
            {"id": "kodak_warm", "label": "Kodak Warm", "description": "Golden cinema"},
            {"id": "teal_orange", "label": "Teal Orange", "description": "Hollywood blockbuster"},
            {"id": "film_noir", "label": "Film Noir", "description": "Deep shadows"},
            {"id": "bleach_bypass", "label": "Bleach Bypass", "description": "Desaturated grit"},
            {"id": "cross_process", "label": "Cross Process", "description": "Surreal colors"},
            {"id": "golden_hour", "label": "Golden Hour", "description": "Warm amber"},
            {"id": "cold_chrome", "label": "Cold Chrome", "description": "Steel blue"},
            {"id": "vintage_tobacco", "label": "Vintage Tobacco", "description": "Aged warm"},
            {"id": "vibrant", "label": "Vibrant", "description": "Enhanced saturation"},
            {"id": "bw", "label": "Black & White", "description": "Full B&W"},
        ],
        "caption_styles": [
            {"id": "standard", "label": "Standard"},
            {"id": "pop_scale", "label": "Pop Scale"},
            {"id": "shake", "label": "Shake"},
            {"id": "blur_reveal", "label": "Blur Reveal"},
        ],
        "transitions": [
            {"id": "pixelize", "label": "Pixelize"},
            {"id": "radial", "label": "Radial"},
            {"id": "dissolve", "label": "Dissolve"},
            {"id": "slideleft", "label": "Slide Left"},
            {"id": "fadeblack", "label": "Fade Black"},
            {"id": "wiperight", "label": "Wipe Right"},
        ],
    }

@router.put("/viral-clips/{clip_id}/effects")
def update_clip_effects(clip_id: int, effects: EffectsOverride, db: Session = Depends(get_db)):
    """Save user effect overrides for a clip. Overrides Grok director choices."""
    clip = db.query(ViralClip).filter(ViralClip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Update render_metadata with user overrides
    metadata = clip.render_metadata or {}
    user_effects = {k: v for k, v in effects.dict().items() if v is not None}

    if user_effects:
        metadata["director_effects"] = {
            **(metadata.get("director_effects", {})),
            **user_effects
        }
        clip.render_metadata = metadata
        # Reset status so clip can be re-rendered with new effects
        if clip.status == "ready":
            clip.status = "pending"
        db.commit()

    return {
        "id": clip.id,
        "director_effects": metadata.get("director_effects", {}),
        "status": clip.status
    }


# --- B-Roll Tagging ---
@router.post("/broll/tag")
async def tag_broll():
    """Trigger AI tagging of local B-Roll clips on the GPU. Uses BLIP + CLIP models."""
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                "http://video-processor:8080/tag-broll",
                json={"force": False, "limit": 0}
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Tagging timed out (>10min)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/broll/tag-force")
async def tag_broll_force():
    """Force re-tag ALL local B-Roll clips (overwrites existing metadata)."""
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                "http://video-processor:8080/tag-broll",
                json={"force": True, "limit": 0}
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Tagging timed out (>10min)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
