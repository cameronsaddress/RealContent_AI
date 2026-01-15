"""
Celery tasks for the Viral Clip Factory pipeline.
Handles downloading, transcribing, and analyzing influencer videos.
"""
import os
import asyncio
from datetime import datetime
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

def _map_video_path(video_path: str) -> str:
    if video_path and video_path.startswith("/downloads/"):
        return video_path.replace("/downloads/", "/app/assets/videos/")
    return video_path

def _video_file_exists(video_path: str) -> bool:
    if not video_path:
        return False
    return os.path.exists(_map_video_path(video_path))

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

        # --- SHORTCUT LOGIC ---
        if _video_file_exists(video.local_path):
            if video.transcript_json:
                logger.info(f"Video {video_id} transcript + file present. Skipping to Analysis.")
                process_viral_video_analyze.delay(video_id)
                return
            logger.info(f"Video {video_id} file present. Skipping to Transcription.")
            process_viral_video_transcribe.delay(video_id)
            return

        if video.local_path:
            logger.info(f"Video {video_id} local_path set but file missing. Proceeding to Download.")
        # ----------------------

        video.status = "downloading"
        db.commit()

        async with httpx.AsyncClient(timeout=5400.0) as client:
            # Call video-processor to download
            # PASS STABLE FILENAME to allow yt-dlp to RESUME .part files if they exist!
            # Otherwise random UUIDs are used and we lose progress.
            response = await client.post(
                f"{VIDEO_PROCESSOR_URL}/download",
                json={
                    "url": video.url, 
                    "format": "mp4",
                    "filename": f"video_{video_id}" # Stable ID for resume support
                }
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

        # --- SHORTCUT LOGIC ---
        # Trust the data: If transcript exists, SKIP. status doesn't matter (might be 'downloaded' from repair).
        if video.transcript_json:
            logger.info(f"Video {video_id} already has transcript. Skipping to Analysis.")
            process_viral_video_analyze.delay(video_id)
            return
        # ----------------------

        video.status = "transcribing (whisper)"
        video.processing_started_at = datetime.utcnow()  # For progress tracking
        db.commit()

        # Check if we need to extract audio first? 
        # Whisper can usually handle video files directly via ffmpeg binding, 
        # but our video-processor main.py endpoint expects 'audio_path'.
        # However, the dockerfile installs ffmpeg, and openai-whisper usually uses ffmpeg to load audio from video.
        # Let's pass the video path directly to transcribe-whisper.
        
        async with httpx.AsyncClient(timeout=5400.0) as client: # 90min timeout for long videos
            response = await client.post(
                f"{VIDEO_PROCESSOR_URL}/transcribe-whisper",
                json={"audio_path": video.local_path, "output_format": "json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Transcription failed: {response.text}")

            result = response.json()
            
            # --- SAVE RAW TRANSCRIPT FIRST (Safety Checkpoint) ---
            # This ensures we don't lose the expensive GPU work if the intro logic crashes
            video.transcript_json = result 
            video.status = "transcribed_raw"
            db.commit()
            logger.info(f"Raw transcript saved for video {video_id}")

            # --- Intro Skipping Logic for specific creators ---
            try:
                # For Nicholas J Fuentes, skip the "intro" loop until he says "Good evening! This is America First"
                channel_id = (video.influencer.channel_id or "").lower()
                influencer_name = (video.influencer.name or "").lower()
                
                if video.influencer and ("nicholasjfuentes" in channel_id or "nicholas" in influencer_name):
                    logger.info(f"Checking for intro skip for influencer: {video.influencer.name}")
                    segments = result.get("segments", [])
                    # Find LAST occurrence of the intro phrase
                    last_cutoff_time = 0.0
                    found_match = False
                    
                    for i, seg in enumerate(segments):
                        text = (seg.get("text") or "").lower()
                        
                        # Check current segment
                        # Primary Trigger: "America First" + context
                        match_primary = "america first" in text and ("good evening" in text or "watching" in text)
                        
                        # Secondary Trigger: "We have a great show for you" (explicit start marker)
                        match_secondary = "we have a great show for you" in text

                        # Check previous segment context for Primary Trigger
                        match_sequence = False
                        if i > 0:
                            prev_text = (segments[i-1].get("text") or "").lower()
                            if "good evening" in prev_text and "america" in text:
                                match_sequence = True

                        if match_primary or match_secondary or match_sequence:
                            cutoff_candidate = seg["end"]
                            
                            # Look ahead for name drop extension
                            if i + 1 < len(segments):
                                next_seg = segments[i+1]
                                next_text = (next_seg.get("text") or "").lower()
                                if "nicholas" in next_text:
                                    cutoff_candidate = next_seg["end"]
                            
                            last_cutoff_time = cutoff_candidate
                            found_match = True
                            logger.info(f"Found intro candidate at {cutoff_candidate}s")
                    
                    if found_match:
                        logger.info(f"Final Nick Fuentes intro cleanup cutoff: {last_cutoff_time}s")
                        
                        # Filter transcript to exclude everything before cutoff
                        new_segments = []
                        for seg in segments:
                            if seg["end"] > last_cutoff_time:
                                new_segments.append(seg)
                        
                        if new_segments:
                            result["segments"] = new_segments
                            # Update with filtered result
                            video.transcript_json = result
                            logger.info(f"Intro skipping applied. Removed segments before {last_cutoff_time}s.")
                        else:
                            logger.warning("Intro skipping removed ALL segments! Reverting...")
                    else:
                        # Fallback Rule: If no intro phrase found, assume Video starts at 0:00
                        logger.info("No 'Nick Fuentes' intro sequence found in transcript. Treating video start as actual beginning (0:00).")
            except Exception as logic_err:
                logger.error(f"Intro Logic Failed (continuing with raw transcript): {logic_err}")
                # Do not re-raise; keep the raw transcript we already saved
            
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
            # AUTO-RENDER DISABLED per user request
            # total_clips = len(video.clips)
            # for i, clip in enumerate(video.clips):
            #     video.status = f"rendering clip ({i+1}/{total_clips})"
            #     db.commit()
            #     logger.info(f"Auto-rendering clip {clip.id} ({i+1}/{total_clips})")
            #     await _render_clip_async(clip.id)
                
            video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
            video.status = "analyzed" # Ready for manual render
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
        clip_start = clip.start_time
        clip_end = clip.end_time
        for s in full_segments:
            # Include if overlaps with clip range
            if s["start"] < clip_end and s["end"] > clip_start:
                seg_start = max(s["start"], clip_start)
                seg_end = min(s["end"], clip_end)
                words = s.get("words", [])
                if words:
                    filtered_words = []
                    for word_obj in words:
                        w_start = word_obj.get("start")
                        w_end = word_obj.get("end")
                        if w_start is None or w_end is None:
                            continue
                        if w_start < clip_end and w_end > clip_start:
                            w_start = max(w_start, clip_start)
                            w_end = min(w_end, clip_end)
                            if w_end <= w_start:
                                continue
                            trimmed_word = dict(word_obj)
                            trimmed_word["start"] = w_start
                            trimmed_word["end"] = w_end
                            filtered_words.append(trimmed_word)
                    if not filtered_words:
                        continue
                    seg_start = filtered_words[0]["start"]
                    seg_end = filtered_words[-1]["end"]
                    words = filtered_words
                clip_segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "text": s["text"],
                    "words": words,
                })

        # Extract trigger words for pulse effect from clip_segments
        trigger_words = []
        
        # 1. Grok Triggers (Priority)
        if clip.render_metadata and clip.render_metadata.get("trigger_words"):
            trigger_words = clip.render_metadata.get("trigger_words")
            logger.info(f"Using {len(trigger_words)} Grok-detected High Intensity Triggers")
        else:
            # 2. Fallback to Hardcoded Word List Search
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

        # Get font settings from LLM settings
        font_random_setting = db.query(LLMSettings).filter(LLMSettings.key == "VIRAL_FONT_RANDOM").first()
        font_setting = db.query(LLMSettings).filter(LLMSettings.key == "VIRAL_CAPTION_FONT").first()

        # Default to random if not set or if explicitly set to random
        if not font_random_setting or font_random_setting.value != 'false':
            selected_font = "random"
        else:
            selected_font = font_setting.value if font_setting else "Honk"

        async with httpx.AsyncClient(timeout=3600.0) as client:  # 60min for extraction + per-clip transcription
            payload = {
                "video_path": video.local_path,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "transcript_segments": clip_segments,
                "style_preset": "trad_west", # Could come from ClipPersona
                "font": selected_font,
                "output_filename": f"clip_{clip.id}_{video.id}.mp4",
                "outro_path": "/assets/outro.mp4",
                "trigger_words": trigger_words,
                "channel_handle": db.query(LLMSettings).filter(LLMSettings.key == "VIRAL_CHANNEL_HANDLE").first().value if db.query(LLMSettings).filter(LLMSettings.key == "VIRAL_CHANNEL_HANDLE").first() else "TheRealClipFactory",
                "status_webhook_url": f"http://backend:8000/api/viral/viral-clips/{clip.id}/status"
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
