"""
Celery tasks for the Viral Clip Factory pipeline.
Handles downloading, transcribing, and analyzing influencer videos.
"""
import os
import asyncio
from typing import Dict, Any

# Keywords for pulse effect
TRAD_TRIGGER_WORDS = ["god", "jesus", "christ", "church", "faith", "pray", "win", "fight", "war", "strength", "power", "glory", "honor", "tradition", "men", "women", "family", "nation", "west", "save", "revive", "build", "create", "beauty", "truth", "justice", "freedom", "liberty", "order", "chaos", "evil", "good", "money", "rich", "wealth", "hustle", "grind", "success", "victory", "champion", "king", "lord", "spirit", "holy", "bible", "cross", "love", "hate", "fear", "death", "life", "soul", "mind", "heart", "body", "blood", "sweat", "tears", "pain", "gain", "gym", "train", "work", "hard", "focus", "discipline"]

from celery_app import celery_app
from models import InfluencerVideo, ViralClip, SessionLocal, LLMSettings
from services.clip_analyzer import ClipAnalyzerService
import httpx
from utils.logging import get_logger

logger = get_logger(__name__)

VIDEO_PROCESSOR_URL = "http://video-processor:8080" # internal docker network

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def _download_video_async(video_id: int):
    """Async implementation of download logic"""
    db = SessionLocal()
    try:
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if not video:
            logger.error(f"Video {video_id} not found")
            return

        video.status = "downloading"
        db.commit()

        async with httpx.AsyncClient(timeout=600.0) as client:
            # Call video-processor to download
            response = await client.post(
                f"{VIDEO_PROCESSOR_URL}/download",
                json={"url": video.url, "format": "mp4"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Download failed: {response.text}")
                
            data = response.json()
            if not data.get("success"):
                 raise Exception(f"Download reported failure: {data}")

            # Update DB with local path
            video.local_path = data["path"] # internal container path /downloads/...
            video.duration = data.get("duration", video.duration)
            video.status = "downloaded"
            db.commit()
            
            # Chain to Next Step: Transcribe
            process_viral_video_transcribe.delay(video_id)

    except Exception as e:
        logger.error(f"Error downloading video {video_id}: {e}")
        video.status = "error"
        video.error_message = str(e)
        db.commit()
    finally:
        db.close()

@celery_app.task
def process_viral_video_download(video_id: int):
    """Entry point: Download video"""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_download_video_async(video_id))

async def _transcribe_video_async(video_id: int):
    db = SessionLocal()
    try:
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if not video or not video.local_path:
             raise Exception("Video not ready for transcription")

        video.status = "transcribing (whisper)"
        db.commit()

        # Check if we need to extract audio first? 
        # Whisper can usually handle video files directly via ffmpeg binding, 
        # but our video-processor main.py endpoint expects 'audio_path'.
        # However, the dockerfile installs ffmpeg, and openai-whisper usually uses ffmpeg to load audio from video.
        # Let's pass the video path directly to transcribe-whisper.
        
        async with httpx.AsyncClient(timeout=1200.0) as client: # Long timeout for whisper
            response = await client.post(
                f"{VIDEO_PROCESSOR_URL}/transcribe-whisper",
                json={"audio_path": video.local_path, "output_format": "json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Transcription failed: {response.text}")

            result = response.json()
            video.transcript_json = result # Save full JSON with timestamps
            video.status = "transcribed"
            db.commit()
            
            # Chain to Next Step: Analyze
            process_viral_video_analyze.delay(video_id)

    except Exception as e:
        logger.error(f"Error transcribing {video_id}: {e}")
        video.status = "error"
        video.error_message = str(e)
        db.commit()
    finally:
        db.close()

@celery_app.task
def process_viral_video_transcribe(video_id: int):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_transcribe_video_async(video_id))

async def _analyze_video_async(video_id: int):
    db = SessionLocal()
    try:
        analyzer = ClipAnalyzerService() 
        logger.info(f"Starting analysis for video {video_id}")
        await analyzer.analyze_video(video_id, db)
        logger.info(f"Analysis complete for video {video_id}")
        # Next steps (Rendering) - Automated
        # Must re-query or refresh to see newly created clips from analyzer
        db.expire_all() # Ensure we get fresh data
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if video and video.clips:
            total_clips = len(video.clips)
            for i, clip in enumerate(video.clips):
                video.status = f"rendering clip ({i+1}/{total_clips})"
                db.commit()
                logger.info(f"Auto-rendering clip {clip.id} ({i+1}/{total_clips})")
                await _render_clip_async(clip.id)
                
            video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
            video.status = "completed"
            db.commit()
            
    except Exception as e:
        logger.error(f"Analysis or Rendering failed for {video_id}: {e}")
        db = SessionLocal() # Re-open just in case
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if video:
            video.status = "error"
            video.error_message = f"Process Error: {str(e)}"
            db.commit()
    finally:
        db.close()

@celery_app.task
def process_viral_video_analyze(video_id: int):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_analyze_video_async(video_id))


async def _render_clip_async(clip_id: int):
    db = SessionLocal()
    try:
        clip = db.query(ViralClip).filter(ViralClip.id == clip_id).first()
        if not clip: return
        
        # Get source video info including transcript
        video = clip.source_video
        
        clip.status = "rendering"
        db.commit()
        
        # Filter transcript segments for this clip time range
        # We assume video.transcript_json is set
        full_segments = video.transcript_json.get("segments", [])
        clip_segments = []
        for s in full_segments:
            # Include if overlaps with clip range
            if s["start"] < clip.end_time and s["end"] > clip.start_time:
                clip_segments.append({
                    "start": s["start"], 
                    "end": s["end"], 
                    "text": s["text"],
                    "words": s.get("words", [])
                })

        # Extract trigger words for pulse effect from clip_segments
        trigger_words = []
        for s in clip_segments:
            text_lower = s["text"].lower()
            # Check for any trigger word match
            for w in TRAD_TRIGGER_WORDS:
                if w in text_lower:
                    # If we have precise word timestamps, search inside
                    found_precise = False
                    if s.get("words"):
                        for word_obj in s["words"]:
                             if w in word_obj["word"].lower():
                                  trigger_words.append({"start": word_obj["start"], "end": word_obj["end"], "word": w})
                                  found_precise = True
                    
                    # If no word timestamps or not found in words (fuzzy match), use segment range
                    if not found_precise:
                         trigger_words.append({"start": s["start"], "end": s["end"], "word": w})
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            payload = {
                "video_path": video.local_path,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "transcript_segments": clip_segments,
                "style_preset": "trad_west", # Could come from ClipPersona
                "output_filename": f"clip_{clip.id}_{video.id}.mp4",
                "outro_path": "/assets/outro.mp4",
                "trigger_words": trigger_words,
                "channel_handle": db.query(LLMSettings).filter(LLMSettings.key == "VIRAL_CHANNEL_HANDLE").first().value if db.query(LLMSettings).filter(LLMSettings.key == "VIRAL_CHANNEL_HANDLE").first() else "TheRealClipFactory"
            }
            
            response = await client.post(
                f"{VIDEO_PROCESSOR_URL}/render-viral-clip",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Render failed: {response.text}")
                
            data = response.json()
            clip.edited_video_path = data["path"]
            clip.status = "ready"
            db.commit()
            
    except Exception as e:
        logger.error(f"Error rendering clip {clip_id}: {e}")
        clip = db.query(ViralClip).filter(ViralClip.id == clip_id).first()
        if clip:
            clip.status = "error"
            clip.error_message = str(e)
            db.commit()
    finally:
        db.close()

@celery_app.task
def process_viral_clip_render(clip_id: int):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_render_clip_async(clip_id))
