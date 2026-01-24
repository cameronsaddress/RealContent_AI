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
    InfluencerPlatformType, RenderTemplate
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
            "recommended_template_id": c.recommended_template_id
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
        "edited_video_path": c.edited_video_path,
        "template_id": c.template_id,
        "recommended_template_id": c.recommended_template_id,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None
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

@router.get("/broll")
def list_broll_clips():
    """
    List all B-roll clips with their metadata and categories.
    Returns clip info from metadata.json if available.
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

    # Build response
    clips = []
    if metadata and "clips" in metadata:
        for clip_data in metadata["clips"]:
            clips.append({
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
                    clips.append({
                        "filename": f.name,
                        "caption": "",
                        "categories": [],
                        "duration": 2.0,
                        "tagged": False
                    })

    # Get category counts
    category_counts = {}
    if metadata and "category_counts" in metadata:
        category_counts = metadata["category_counts"]

    return {
        "clips": clips,
        "metadata": {
            "total_clips": len(clips),
            "tagged_clips": sum(1 for c in clips if c.get("tagged")),
            "category_counts": category_counts,
            "generated_at": metadata.get("generated_at") if metadata else None
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

    # Initialize job status
    broll_upload_jobs[job_id] = {
        "status": "queued",
        "youtube_url": request.youtube_url,
        "clip_duration": request.clip_duration,
        "progress": 0,
        "message": "Job queued",
        "clips_created": 0
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
    if job_id not in broll_upload_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return broll_upload_jobs[job_id]

@router.post("/broll/retag")
async def retag_all_broll(background_tasks: BackgroundTasks):
    """
    Re-run AI tagging on all B-roll clips.
    This regenerates metadata.json with fresh BLIP captions and CLIP categories.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]

    broll_upload_jobs[job_id] = {
        "status": "queued",
        "task": "retag",
        "progress": 0,
        "message": "Retagging job queued"
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
    broll_dir = Path("/app/assets/broll")
    file_path = broll_dir / filename

    # Security: prevent directory traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename
    )


@router.delete("/broll/{filename}")
def delete_broll_clip(filename: str):
    """Delete a B-roll clip and remove from metadata."""
    import json

    broll_dir = Path("/app/assets/broll")
    file_path = broll_dir / filename
    metadata_path = broll_dir / "metadata.json"

    # Security: prevent directory traversal
    if ".." in filename or "/" in filename:
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
