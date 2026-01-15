from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional
from datetime import datetime
import httpx
import os
from pathlib import Path

from models import (
    get_db, Influencer, InfluencerVideo, ViralClip, ClipPersona,
    InfluencerPlatformType
)
from pydantic import BaseModel

router = APIRouter(prefix="/api/viral", tags=["viral"])

# Import Celery task
from tasks.viral import process_viral_clip_render

# --- Schemas ---
class InfluencerCreate(BaseModel):
    name: str
    platform: str
    channel_url: str
    persona_id: int

class VideoAnalyzeRequest(BaseModel):
    video_id: int
    persona_id: Optional[int] = None

# --- Endpoints ---

@router.get("/influencers", response_model=List[dict])
def list_influencers(db: Session = Depends(get_db)):
    influencers = db.query(Influencer).all()
    return [{"id": i.id, "name": i.name, "platform": i.platform, "channel_url": i.channel_url} for i in influencers]

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
                    # Check duplicate
                    exists = db.query(InfluencerVideo).filter(
                        InfluencerVideo.influencer_id == id,
                        InfluencerVideo.url == v["url"]
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
            "edited_video_path": c.edited_video_path
        } for c in v.clips]
    } for v in videos]
    
@router.get("/viral-clips", response_model=List[dict])
def list_viral_clips(db: Session = Depends(get_db)):
    clips = db.query(ViralClip).order_by(ViralClip.created_at.desc()).limit(50).all()
    return [{
        "id": c.id,
        "title": c.title,
        "hashtags": c.hashtags,
        "clip_type": c.clip_type,
        "status": c.status,
        "source_video_id": c.source_video_id,
        "edited_video_path": c.edited_video_path
    } for c in clips]


from fastapi.responses import FileResponse

class StatusUpdate(BaseModel):
    status: str

@router.put("/viral-clips/{id}/status")
def update_viral_clip_status(id: int, status_update: StatusUpdate, db: Session = Depends(get_db)):
    clip = db.query(ViralClip).filter(ViralClip.id == id).first()
    if not clip:
        raise HTTPException(status_code=404)
        
    # Prevent overwriting final states unless explicitly requested?
    # For now, just trust the worker.
    clip.status = status_update.status
    db.commit()
    return {"status": "updated"}

@router.get("/file/{filename}")
def get_viral_file(filename: str):
    # Security: basic check to prevent traversal
    if ".." in filename or "/" in filename:
         raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = f"/app/assets/output/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Return with proper media type for video streaming
    return FileResponse(file_path, media_type="video/mp4")

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
