"""
Celery tasks for the Viral Clip Factory pipeline.
Handles downloading, transcribing, and analyzing influencer videos.
"""
import os
import random
import asyncio
import redis
from datetime import datetime
from typing import Dict, Any

# Keywords for pulse effect
TRAD_TRIGGER_WORDS = ["god", "jesus", "christ", "church", "faith", "pray", "win", "fight", "war", "strength", "power", "glory", "honor", "tradition", "men", "women", "family", "nation", "west", "save", "revive", "build", "create", "beauty", "truth", "justice", "freedom", "liberty", "order", "chaos", "evil", "good", "money", "rich", "wealth", "hustle", "grind", "success", "victory", "champion", "king", "lord", "spirit", "holy", "bible", "cross", "love", "hate", "fear", "death", "life", "soul", "mind", "heart", "body", "blood", "sweat", "tears", "pain", "gain", "gym", "train", "work", "hard", "focus", "discipline"]

from celery_app import celery_app
from models import (
    InfluencerVideo, ViralClip, SessionLocal, LLMSettings, RenderTemplate,
    Influencer, PublishingConfig, PublishingQueueItem
)
from services.clip_analyzer import ClipAnalyzerService
import httpx
from sqlalchemy import or_
from datetime import timedelta
import json as _json
from utils.logging import get_logger

logger = get_logger(__name__)

# ============ WHISPER GPU QUEUE (Redis Lock) ============
# Only one Whisper transcription runs at a time to prevent GPU contention
_redis_client = None
WHISPER_LOCK_KEY = "whisper_transcription_lock"
WHISPER_LOCK_TIMEOUT = 7200  # 2 hours max lock (for very long videos)
WHISPER_QUEUE_RETRY_DELAY = 30  # Seconds to wait before retrying if locked

def get_redis_client():
    """Get or create Redis client for distributed locking."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
    return _redis_client

def acquire_whisper_lock(video_id: int) -> bool:
    """Try to acquire the Whisper GPU lock. Returns True if acquired."""
    r = get_redis_client()
    # SET NX (only if not exists) with expiry
    acquired = r.set(WHISPER_LOCK_KEY, str(video_id), nx=True, ex=WHISPER_LOCK_TIMEOUT)
    return acquired is True

def release_whisper_lock(video_id: int):
    """Release the Whisper GPU lock if we hold it."""
    r = get_redis_client()
    # Only delete if we own the lock (check value matches our video_id)
    current = r.get(WHISPER_LOCK_KEY)
    if current == str(video_id):
        r.delete(WHISPER_LOCK_KEY)
        logger.info(f"Released Whisper lock for video {video_id}")

def get_whisper_lock_holder() -> str:
    """Get the video_id currently holding the Whisper lock, or None."""
    r = get_redis_client()
    return r.get(WHISPER_LOCK_KEY)
# ========================================================

# ============ RENDER GPU QUEUE (Redis Semaphore) ============
# Limit concurrent renders to prevent GPU memory exhaustion
RENDER_SEMAPHORE_KEY = "render_gpu_semaphore"
MAX_CONCURRENT_RENDERS = 3
RENDER_LOCK_TIMEOUT = 1800  # 30 min max per render
RENDER_QUEUE_RETRY_DELAY = 15  # Seconds to wait before retrying if full

def acquire_render_slot(clip_id: int) -> bool:
    """Try to acquire a render slot. Returns True if acquired."""
    r = get_redis_client()
    # Get current render count
    current_renders = r.scard(RENDER_SEMAPHORE_KEY) or 0
    if current_renders < MAX_CONCURRENT_RENDERS:
        # Add this clip to the active renders set with expiry
        r.sadd(RENDER_SEMAPHORE_KEY, str(clip_id))
        # Set per-clip timeout key for cleanup
        r.setex(f"render_active:{clip_id}", RENDER_LOCK_TIMEOUT, "1")
        logger.info(f"Acquired render slot for clip {clip_id} ({current_renders + 1}/{MAX_CONCURRENT_RENDERS})")
        return True
    return False

def release_render_slot(clip_id: int):
    """Release a render slot."""
    r = get_redis_client()
    r.srem(RENDER_SEMAPHORE_KEY, str(clip_id))
    r.delete(f"render_active:{clip_id}")
    current = r.scard(RENDER_SEMAPHORE_KEY) or 0
    logger.info(f"Released render slot for clip {clip_id} ({current}/{MAX_CONCURRENT_RENDERS} active)")

def get_active_renders() -> list:
    """Get list of clip IDs currently rendering."""
    r = get_redis_client()
    return list(r.smembers(RENDER_SEMAPHORE_KEY))

def cleanup_stale_renders():
    """Remove render slots where the timeout key has expired."""
    r = get_redis_client()
    active = r.smembers(RENDER_SEMAPHORE_KEY) or set()
    for clip_id in active:
        if not r.exists(f"render_active:{clip_id}"):
            r.srem(RENDER_SEMAPHORE_KEY, clip_id)
            logger.info(f"Cleaned up stale render slot for clip {clip_id}")
# ============================================================

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

@celery_app.task(bind=True, max_retries=None)
def process_viral_video_transcribe(self, video_id: int):
    """
    Transcribe video using Whisper. Uses Redis lock to ensure only one
    transcription runs at a time (GPU queue). If locked, requeues with delay.
    """
    db = SessionLocal()
    try:
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if not video:
            logger.error(f"Video {video_id} not found")
            return

        # Shortcut: If already transcribed, skip to analysis
        if video.transcript_json:
            logger.info(f"Video {video_id} already has transcript. Skipping to Analysis.")
            process_viral_video_analyze.delay(video_id)
            return

        # Try to acquire the Whisper GPU lock
        if acquire_whisper_lock(video_id):
            logger.info(f"Acquired Whisper lock for video {video_id}")
            db.close()  # Close before long-running operation
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(_transcribe_video_async(video_id))
            finally:
                release_whisper_lock(video_id)
        else:
            # Lock is held by another video - queue this one
            lock_holder = get_whisper_lock_holder()
            logger.info(f"Whisper GPU busy (video {lock_holder}). Queueing video {video_id} for retry in {WHISPER_QUEUE_RETRY_DELAY}s")

            # Update status to show queued state
            video.status = f"queued (waiting for video {lock_holder})"
            db.commit()

            # Requeue with delay
            raise self.retry(countdown=WHISPER_QUEUE_RETRY_DELAY)
    except self.MaxRetriesExceededError:
        # This shouldn't happen with max_retries=None, but handle gracefully
        logger.error(f"Video {video_id} exceeded max retries for Whisper queue")
    finally:
        if db.is_active:
            db.close()

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


def merge_director_effects(template_settings: dict, director_effects: dict) -> dict:
    """
    Merge template effect_settings with Grok director effects.
    Director choices override template conflicts.
    Maps creative names to technical params for the video-processor.

    Returns merged effect_settings dict ready for the render payload.
    """
    merged = dict(template_settings) if template_settings else {}

    if not director_effects:
        return merged

    # Color grade: director LUT overrides template color preset
    if director_effects.get("color_grade"):
        grade = director_effects["color_grade"]
        if grade == "bw":
            merged["color_grade"] = "bw"
            merged["saturation"] = 0
        elif grade == "vibrant":
            merged["color_preset"] = "vibrant"
        else:
            # LUT-based grade (kodak_warm, teal_orange, etc.)
            merged["lut_file"] = grade
            # Remove conflicting eq-based grade
            merged.pop("color_preset", None)

    # Camera shake
    if director_effects.get("camera_shake"):
        shake = director_effects["camera_shake"]
        if isinstance(shake, dict):
            merged["camera_shake"] = shake
        else:
            merged["camera_shake"] = {"intensity": 8, "frequency": 2.0}

    # Retro glow
    if director_effects.get("retro_glow"):
        val = director_effects["retro_glow"]
        if isinstance(val, (int, float)):
            merged["retro_glow"] = val
        else:
            merged["retro_glow"] = 0.3

    # Temporal trail
    if director_effects.get("temporal_trail"):
        val = director_effects["temporal_trail"]
        if isinstance(val, dict):
            merged["temporal_trail"] = val
        else:
            merged["temporal_trail"] = True

    # Wave displacement
    if director_effects.get("wave_displacement"):
        val = director_effects["wave_displacement"]
        if isinstance(val, dict):
            merged["wave_displacement"] = val
        else:
            merged["wave_displacement"] = True

    # Heavy VHS
    if director_effects.get("heavy_vhs"):
        val = director_effects["heavy_vhs"]
        if isinstance(val, (int, float)):
            merged["vhs_intensity"] = val
        else:
            merged["vhs_intensity"] = 1.3

    # VHS intensity (explicit)
    if "vhs_intensity" in director_effects:
        merged["vhs_intensity"] = float(director_effects["vhs_intensity"])

    # Pulse intensity
    if "pulse_intensity" in director_effects:
        merged["pulse_intensity"] = float(director_effects["pulse_intensity"])

    # Beat sync
    if director_effects.get("beat_sync"):
        merged["beat_sync"] = True

    # Audio saturation
    if director_effects.get("audio_saturation"):
        merged["audio_saturation"] = True

    # Caption style
    if director_effects.get("caption_style"):
        merged["caption_style"] = director_effects["caption_style"]

    # B-roll transition
    if director_effects.get("transition"):
        merged["transition"] = director_effects["transition"]

    # Rare effects: datamosh and pixel sort (max 3 segments, max 2s each)
    if director_effects.get("datamosh_segments"):
        segs = director_effects["datamosh_segments"]
        if isinstance(segs, list):
            merged["datamosh_segments"] = segs[:3]

    if director_effects.get("pixel_sort_segments"):
        segs = director_effects["pixel_sort_segments"]
        if isinstance(segs, list):
            merged["pixel_sort_segments"] = segs[:3]

    # BGM mood (dark, triumphant, aggressive, melancholic, hype)
    if director_effects.get("bgm_mood"):
        merged["bgm_mood"] = director_effects["bgm_mood"]

    logger.info(f"Merged effects: template={list(template_settings.keys()) if template_settings else []}, "
                f"director={list(director_effects.keys())}, result keys={list(merged.keys())}")
    return merged


# ============ TWO-PASS B-ROLL: Grok Timestamp Selector ============
# 1. Fetch YouTube transcripts for each topic_broll query
# 2. Send speaker's clip context + YouTube transcripts to Grok
# 3. Grok picks exact timestamps from each YouTube video
# 4. Return seek_times for precise clip extraction


# B-Roll is ONLY sourced from fresh YouTube/Rumble downloads matching the clip's spoken content.
# No local stock footage is used.

async def _select_broll_timestamps(
    speaker_context: str,
    topic_queries: list,
    clip_id: int,
    video_pub_date: str = "",
) -> list:
    """Two-pass B-Roll: fetch YouTube transcripts, let Grok pick exact moments.

    Returns list of dicts: [{"query": str, "query_hash": str, "seek_time": float, "reason": str}]
    """
    from config import settings

    if not topic_queries or not settings.OPENROUTER_API_KEY:
        return []

    # Step 1: Fetch transcripts + face-free segments for each query
    transcripts = []
    async with httpx.AsyncClient(timeout=90.0) as client:
        for query in topic_queries[:3]:  # Max 3 queries
            try:
                resp = await client.post(
                    f"{VIDEO_PROCESSOR_URL}/fetch-youtube-transcript",
                    json={"query": query, "clip_id": clip_id, "video_pub_date": video_pub_date}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("transcript") and len(data["transcript"]) > 0:
                        face_free = data.get("face_free_segments", [])
                        transcripts.append({
                            "query": query,
                            "query_hash": data.get("query_hash", ""),
                            "title": data.get("title", ""),
                            "channel": data.get("channel", ""),
                            "duration": data.get("duration", 0),
                            "transcript": data["transcript"][:300],
                            "face_free_segments": face_free,
                        })
                        logger.info(f"B-Roll transcript fetched: '{data.get('title', '')}' "
                                    f"({len(data['transcript'])} segs, {len(face_free)} face-free)")
                    else:
                        logger.warning(f"B-Roll transcript empty for query: {query}")
                else:
                    logger.warning(f"B-Roll transcript fetch failed ({resp.status_code}): {query}")
            except Exception as e:
                logger.warning(f"B-Roll transcript fetch error for '{query}': {e}")

    if not transcripts:
        logger.warning("No YouTube transcripts fetched, falling back to keyword matching")
        return []

    # Step 2: Build Grok prompt with face-free segments + context
    # Strategy: Show Grok ONLY the face-free segments (actual footage, not talking heads)
    # plus full transcript for context understanding
    video_sections = []
    has_face_free = False
    for i, t in enumerate(transcripts):
        face_free = t.get("face_free_segments", [])

        if face_free:
            has_face_free = True
            # Show face-free segments (these are the FOOTAGE moments)
            ff_lines = []
            for ff in face_free:
                ff_text = ff.get("text", "").strip()
                if ff_text:
                    ff_lines.append(f"  [{ff['start']:.0f}s-{ff['end']:.0f}s] FOOTAGE: {ff_text[:150]}")
                else:
                    ff_lines.append(f"  [{ff['start']:.0f}s-{ff['end']:.0f}s] FOOTAGE: (no narration)")
            footage_text = "\n".join(ff_lines[:30])

            # Also show full transcript for context
            context_lines = []
            for seg in t["transcript"][:50]:
                context_lines.append(f"  [{seg['start']:.0f}s] {seg['text']}")
            context_text = "\n".join(context_lines)

            video_sections.append(
                f"--- Video {i+1}: \"{t['title']}\" ({t['duration']:.0f}s) ---\n"
                f"Query: \"{t['query']}\"\n\n"
                f"AVAILABLE FOOTAGE SEGMENTS (no talking head visible - actual event footage):\n"
                f"{footage_text}\n\n"
                f"FULL TRANSCRIPT (for understanding what the video covers):\n"
                f"{context_text}\n"
            )
        else:
            # No face-free data - fall back to full transcript
            lines = []
            for seg in t["transcript"][:100]:
                lines.append(f"  [{seg['start']:.0f}s] {seg['text']}")
            transcript_text = "\n".join(lines)
            video_sections.append(
                f"--- Video {i+1}: \"{t['title']}\" ({t['duration']:.0f}s) ---\n"
                f"Query: \"{t['query']}\"\n"
                f"(No face detection data - pick timestamps likely to show footage)\n"
                f"{transcript_text}\n"
            )

    all_videos = "\n\n".join(video_sections)

    # Build constraint text based on whether we have face-free data
    if has_face_free:
        constraint_text = (
            "CRITICAL: You MUST pick timestamps ONLY from within the 'AVAILABLE FOOTAGE SEGMENTS' time ranges.\n"
            "These segments have been verified to show ACTUAL FOOTAGE (not a talking head/anchor/pundit).\n"
            "Pick the exact second within each footage range that best matches the speaker's words.\n"
            "DO NOT pick timestamps outside these ranges - those show people talking at a desk."
        )
    else:
        constraint_text = (
            "Pick timestamps that are likely to show ACTUAL EVENT FOOTAGE rather than talking heads.\n"
            "Avoid timestamps where the transcript sounds like commentary/reaction.\n"
            "Prefer moments described with action verbs or showing locations/events."
        )

    prompt = f"""You are a B-Roll editor selecting EXACT timestamps from YouTube news videos to overlay on a speaker's clip.
The goal is to show the ACTUAL EVENT being discussed - real footage of the incident/person/situation.

THE SPEAKER IS SAYING:
{speaker_context}

---

{all_videos}

---

{constraint_text}

TASK: Pick 6-10 EXACT timestamps that show footage matching what the speaker is discussing.

For each pick, choose a timestamp that would VISUALLY show:
- The actual EVENT, PERSON, or LOCATION being discussed
- Real footage (bodycam, surveillance, crowd shots, buildings, vehicles)
- NOT a news anchor, pundit, or podcast host sitting at a desk

Return ONLY valid JSON:
{{
  "picks": [
    {{
      "video_index": 0,
      "seek_time": 45.2,
      "visual_description": "Protesters blocking vehicle at intersection",
      "reason": "Matches speaker discussing the vehicle confrontation"
    }}
  ]
}}
"""

    # Step 3: Call Grok
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "x-ai/grok-4.1-fast",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.5
                },
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error(f"Grok B-Roll selector error: {data['error']}")
                return []

            content = data["choices"][0]["message"]["content"]
            import json as _json
            result = _json.loads(content)
            picks = result.get("picks", [])

            # Map picks back to query info
            broll_selections = []
            for pick in picks:
                vid_idx = pick.get("video_index", 0)
                if vid_idx < len(transcripts):
                    t = transcripts[vid_idx]
                    seek = pick.get("seek_time", 0)
                    # Clamp seek_time to valid range
                    if t["duration"] > 0:
                        seek = min(seek, t["duration"] - 5.0)
                    seek = max(2.0, seek)
                    broll_selections.append({
                        "query": t["query"],
                        "query_hash": t["query_hash"],
                        "seek_time": seek,
                        "reason": pick.get("reason", ""),
                        "visual_description": pick.get("visual_description", ""),
                    })

            logger.info(f"Grok B-Roll selector: {len(broll_selections)} timestamps picked from {len(transcripts)} videos")
            for sel in broll_selections:
                vis = sel.get('visual_description', '')
                logger.info(f"  -> t={sel['seek_time']:.1f}s from '{sel['query']}': {vis or sel['reason']}")

            return broll_selections

    except Exception as e:
        logger.error(f"Grok B-Roll selector failed: {e}")
        return []

# ===================================================================


async def _render_clip_async(clip_id: int):
    db = SessionLocal()
    try:
        clip = db.query(ViralClip).filter(ViralClip.id == clip_id).first()
        if not clip: return
        
        # Get source video info including transcript
        video = clip.source_video

        # Format publication date for YouTube date filtering (YYYYMMDD)
        video_pub_date_str = ""
        if video.publication_date:
            try:
                video_pub_date_str = video.publication_date.strftime("%Y%m%d")
            except Exception:
                pass

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

        # --- B-ROLL MONTAGE PREPARATION ---
        climax_time = None
        broll_paths = []
        broll_duration = 0

        # Extract climax_time from clip (either from column or render_metadata)
        if clip.climax_time:
            climax_time = clip.climax_time
        elif clip.render_metadata and clip.render_metadata.get("climax_time"):
            climax_time = clip.render_metadata.get("climax_time")

        # Validate climax_time is within clip bounds
        clip_duration = clip.end_time - clip.start_time
        if climax_time is not None:
            if climax_time < clip.start_time or climax_time > clip.end_time:
                # Invalid climax - default to 40% through clip (leaves 60% for B-Roll)
                climax_time = clip.start_time + (clip_duration * 0.4)
                logger.warning(f"Climax time outside clip bounds, reset to {climax_time:.1f}s (40% of clip)")

        # Get template settings (for B-roll categories and effect settings like B&W)
        template = None
        template_effect_settings = {}
        if clip.template_id:
            template = db.query(RenderTemplate).filter(RenderTemplate.id == clip.template_id).first()
            if template:
                template_effect_settings = template.effect_settings or {}
                logger.info(f"Using template '{template.name}' with effect settings: {template_effect_settings}")

        # Merge template + director effects (director overrides template)
        director_effects = (clip.render_metadata or {}).get("director_effects", {})
        effect_settings = merge_director_effects(template_effect_settings, director_effects)

        # Check if B-roll is enabled (template setting takes priority)
        broll_enabled = True
        if template and template.broll_enabled is False:
            broll_enabled = False
            logger.info(f"B-roll disabled by template '{template.name}'")
        elif clip.render_metadata and clip.render_metadata.get("broll_enabled") is False:
            broll_enabled = False

        if climax_time and broll_enabled:
            # Calculate B-roll duration: from climax to 2 seconds before clip end (for outro)
            climax_relative = climax_time - clip.start_time  # Convert to clip-relative time
            broll_duration = (clip_duration - 2) - climax_relative  # Leave 2s for outro

            if broll_duration >= 3:  # Only fetch B-roll if we have at least 3 seconds
                render_meta = clip.render_metadata or {}
                has_topic_broll = bool(render_meta.get("topic_broll"))

                if has_topic_broll:
                    logger.info(f"Clip has topic_broll queries - will use fresh YouTube footage only")
                    broll_paths = []
                else:
                    # No topic_broll queries - skip B-Roll entirely rather than use inaccurate stock footage
                    logger.info(f"No topic_broll queries for this clip - skipping B-Roll (no stock footage)")
                    broll_paths = []
                    broll_duration = 0
            else:
                logger.info(f"B-roll duration too short ({broll_duration:.1f}s < 3s), skipping montage")
                broll_duration = 0

        # --- HYBRID B-ROLL SYSTEM: Mix local thematic + YouTube event footage ---
        # New system: broll_insertions specifies source per insertion point
        # Fallback: topic_broll queries for older clips without broll_insertions
        render_meta = clip.render_metadata or {}
        broll_insertions = render_meta.get("broll_insertions", [])
        topic_broll_queries = render_meta.get("topic_broll", [])
        topic_broll_keywords = render_meta.get("topic_broll_keywords", [])

        # --- NEW HYBRID SYSTEM: Process broll_insertions if available ---
        if broll_insertions:
            logger.info(f"Hybrid B-roll: {len(broll_insertions)} insertion points specified")
            hybrid_paths = []

            # Group YouTube queries for batch processing
            youtube_insertions = [i for i in broll_insertions if i.get("source") == "youtube"]
            local_insertions = [i for i in broll_insertions if i.get("source") == "local"]
            logger.info(f"Hybrid B-roll: {len(youtube_insertions)} YouTube + {len(local_insertions)} local insertions")

            # Collect unique YouTube queries
            unique_queries = list(dict.fromkeys([i.get("query", "") for i in youtube_insertions if i.get("query")]))

            # Build speaker context for YouTube selection
            speaker_lines = []
            for seg in clip_segments:
                speaker_lines.append(f"[{seg['start']:.1f}s] {seg.get('text', '')}")
            speaker_context = "\n".join(speaker_lines)

            # Fetch YouTube clips via two-pass selection
            youtube_clip_map = {}  # query -> list of clip paths
            if unique_queries:
                broll_selections = await _select_broll_timestamps(
                    speaker_context=speaker_context,
                    topic_queries=unique_queries,
                    clip_id=clip.id,
                    video_pub_date=video_pub_date_str,
                )
                for sel in broll_selections or []:
                    try:
                        query = sel.get("query", "")
                        clip_index = len(youtube_clip_map.get(query, []))
                        resp = httpx.post(
                            f"{VIDEO_PROCESSOR_URL}/download-event-broll",
                            json={
                                "query": query,
                                "clip_id": clip.id,
                                "index": len(hybrid_paths) + clip_index,
                                "keywords": [],
                                "seek_time": sel.get("seek_time", 0),
                                "video_pub_date": video_pub_date_str,
                            },
                            timeout=180.0
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            clip_path = data.get("clip_path")
                            if clip_path:
                                if query not in youtube_clip_map:
                                    youtube_clip_map[query] = []
                                youtube_clip_map[query].append(clip_path)
                                logger.info(f"Hybrid B-roll: YouTube clip -> {clip_path}")
                    except Exception as e:
                        logger.warning(f"Hybrid B-roll: YouTube fetch failed: {e}")

            # Fetch local clips by category (blacklist firefighters)
            local_clip_map = {}  # category -> list of clip paths
            BLACKLISTED_PREFIXES = ["fire_firemen"]  # Remove firefighter footage

            if local_insertions:
                from services.pexels import PexelsService
                pexels = PexelsService()
                unique_categories = list(dict.fromkeys([i.get("category", "") for i in local_insertions if i.get("category")]))

                for category in unique_categories:
                    try:
                        # Get multiple clips for this category (may need several)
                        needed = sum(1 for i in local_insertions if i.get("category") == category)
                        clips = pexels.get_broll_clips(categories=[category], count=needed + 2)
                        # Filter out blacklisted clips and convert paths
                        filtered_clips = []
                        for p in clips:
                            p = p.replace("/app/assets/broll/", "/broll/")
                            # Skip blacklisted prefixes (e.g., firefighters)
                            basename = os.path.basename(p).lower()
                            if not any(basename.startswith(bl) for bl in BLACKLISTED_PREFIXES):
                                filtered_clips.append(p)
                        local_clip_map[category] = filtered_clips
                        logger.info(f"Hybrid B-roll: {len(filtered_clips)} local clips for '{category}'")
                    except Exception as e:
                        logger.warning(f"Hybrid B-roll: Local fetch for '{category}' failed: {e}")
                        local_clip_map[category] = []

            # Build final broll_paths in insertion order
            youtube_usage = {q: 0 for q in youtube_clip_map}
            local_usage = {c: 0 for c in local_clip_map}

            for ins in sorted(broll_insertions, key=lambda x: x.get("time", 0)):
                source = ins.get("source", "")
                if source == "youtube":
                    query = ins.get("query", "")
                    if query in youtube_clip_map and youtube_usage[query] < len(youtube_clip_map[query]):
                        hybrid_paths.append(youtube_clip_map[query][youtube_usage[query]])
                        youtube_usage[query] += 1
                elif source == "local":
                    category = ins.get("category", "")
                    if category in local_clip_map and local_usage[category] < len(local_clip_map[category]):
                        hybrid_paths.append(local_clip_map[category][local_usage[category]])
                        local_usage[category] += 1

            if hybrid_paths:
                broll_paths = hybrid_paths
                logger.info(f"Hybrid B-roll: Final {len(broll_paths)} clips ({sum(1 for p in broll_paths if '/event_' in p)} YouTube, {sum(1 for p in broll_paths if '/event_' not in p)} local)")
            else:
                logger.warning("Hybrid B-roll: No clips obtained")
                broll_paths = []
                broll_duration = 0

        # --- LEGACY FALLBACK: topic_broll queries (for older clips without broll_insertions) ---
        elif topic_broll_queries:
            logger.info(f"Topic B-roll (two-pass): {len(topic_broll_queries)} queries: {topic_broll_queries}")

            # Build speaker context from clip transcript (what Nick is saying)
            speaker_lines = []
            for seg in clip_segments:
                speaker_lines.append(f"[{seg['start']:.1f}s] {seg.get('text', '')}")
            speaker_context = "\n".join(speaker_lines)

            # Also include context BEFORE the clip (previous 60s of transcript)
            pre_context_lines = []
            for s in full_segments:
                if s["end"] > clip_start - 60 and s["end"] <= clip_start:
                    pre_context_lines.append(f"[{s['start']:.1f}s] {s.get('text', '')}")
            if pre_context_lines:
                speaker_context = (
                    "=== CONTEXT BEFORE CLIP (what was said in the 60 seconds prior) ===\n"
                    + "\n".join(pre_context_lines[-20:])  # Last 20 lines of pre-context
                    + "\n\n=== THE CLIP ITSELF (what the speaker is saying NOW) ===\n"
                    + speaker_context
                )

            # TWO-PASS: Let Grok pick exact timestamps from YouTube transcripts
            broll_selections = await _select_broll_timestamps(
                speaker_context=speaker_context,
                topic_queries=topic_broll_queries,
                clip_id=clip.id,
                video_pub_date=video_pub_date_str,
            )

            topic_clips = []
            if broll_selections:
                # Extract clips at Grok-directed timestamps
                for sel in broll_selections:
                    try:
                        clip_index = len(topic_clips)
                        logger.info(f"Topic B-roll: Extracting at t={sel['seek_time']:.1f}s from '{sel['query']}' ({sel['reason']})")
                        resp = httpx.post(
                            f"{VIDEO_PROCESSOR_URL}/download-event-broll",
                            json={
                                "query": sel["query"],
                                "clip_id": clip.id,
                                "index": clip_index,
                                "keywords": [],  # Not needed - using seek_time
                                "seek_time": sel["seek_time"],
                                "video_pub_date": video_pub_date_str,
                            },
                            timeout=180.0
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            clip_path = data.get("clip_path")
                            if clip_path:
                                topic_clips.append(clip_path)
                                logger.info(f"Topic B-roll: Got Grok-directed clip -> {clip_path}")
                        else:
                            logger.warning(f"Topic B-roll: {resp.status_code} for seek={sel['seek_time']:.1f}s")
                    except Exception as e:
                        logger.warning(f"Topic B-roll extraction failed: {e}")
            else:
                # FALLBACK: keyword matching (old approach) if Grok selector fails
                logger.info(f"Topic B-roll: Grok selector returned nothing, falling back to keyword matching")
                if topic_broll_keywords:
                    logger.info(f"Topic B-roll keywords for fallback: {topic_broll_keywords}")
                for idx, query in enumerate(topic_broll_queries[:3]):
                    if not query or not isinstance(query, str):
                        continue
                    for sub_idx in range(3):
                        try:
                            clip_index = len(topic_clips)
                            resp = httpx.post(
                                f"{VIDEO_PROCESSOR_URL}/download-event-broll",
                                json={
                                    "query": query,
                                    "clip_id": clip.id,
                                    "index": clip_index,
                                    "keywords": topic_broll_keywords,
                                    "video_pub_date": video_pub_date_str,
                                },
                                timeout=180.0
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                clip_path = data.get("clip_path")
                                if clip_path:
                                    topic_clips.append(clip_path)
                            else:
                                break
                        except Exception as e:
                            logger.warning(f"Topic B-roll fallback failed for '{query}': {e}")
                            break

            # Use ONLY fresh topic clips from YouTube - no local stock footage
            if topic_clips:
                broll_paths = topic_clips
                logger.info(f"Topic B-roll: Using {len(topic_clips)} fresh YouTube clips (no stock footage)")
            else:
                # No topic clips obtained - skip B-Roll entirely
                logger.warning("Topic B-roll: No clips obtained from YouTube - no B-Roll for this clip")
                broll_paths = []
                broll_duration = 0

            # If we have topic clips but broll_duration ended up invalid, recalculate
            if broll_paths and broll_duration <= 0:
                broll_duration = max(10.0, clip_duration * 0.60)
                climax_time = clip.start_time + (clip_duration * 0.40)
                logger.info(f"Topic B-roll: Recalculated broll_duration={broll_duration:.1f}s (climax at {clip_duration * 0.40:.1f}s relative)")

        # Adjust datamosh/pixel_sort segment timestamps: absolute -> clip-relative
        for seg_key in ("datamosh_segments", "pixel_sort_segments"):
            segs = effect_settings.get(seg_key, [])
            if isinstance(segs, list) and segs:
                adjusted_segs = []
                for seg in segs[:3]:
                    if isinstance(seg, dict):
                        adj = dict(seg)
                        adj["start"] = max(0, adj.get("start", 0) - clip.start_time)
                        adj["end"] = max(0, adj.get("end", 0) - clip.start_time)
                        if adj["end"] > adj["start"]:
                            adjusted_segs.append(adj)
                effect_settings[seg_key] = adjusted_segs

        caption_style = effect_settings.get("caption_style", "standard")
        broll_transition_type = effect_settings.get("transition")

        # Extract cold open hook data from render_metadata
        hook_phrase = render_meta.get("hook_phrase")
        hook_timestamp = render_meta.get("hook_timestamp")
        visual_hook_time = render_meta.get("visual_hook_time")

        # Extract emotional arc data for dynamic intensity ramp
        setup_end = render_meta.get("setup_end")
        escalation_peak = render_meta.get("escalation_peak")
        quotable_line = render_meta.get("quotable_line")

        # Extract dramatic pauses for effect hold
        key_pauses = render_meta.get("key_pauses", [])

        # Extract caption pacing data
        rapid_fire_sections = render_meta.get("rapid_fire_sections", [])
        question_moments = render_meta.get("question_moments", [])

        async with httpx.AsyncClient(timeout=3600.0) as client:  # 60min for extraction + per-clip transcription
            payload = {
                "video_path": video.local_path,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "transcript_segments": clip_segments,
                "style_preset": "trad_west",
                "font": selected_font,
                "output_filename": f"clip_{clip.id}_{video.id}.mp4",
                "outro_path": "/assets/outro.mp4",
                "trigger_words": trigger_words,
                "channel_handle": db.query(LLMSettings).filter(LLMSettings.key == "VIRAL_CHANNEL_HANDLE").first().value if db.query(LLMSettings).filter(LLMSettings.key == "VIRAL_CHANNEL_HANDLE").first() else "TheRealClipFactory",
                "status_webhook_url": f"http://backend:8000/api/viral/viral-clips/{clip.id}/status",
                # B-roll montage parameters
                "climax_time": climax_time,
                "broll_paths": broll_paths,
                "broll_duration": broll_duration if broll_paths else 0,
                # Merged template + director effect settings
                "effect_settings": effect_settings,
                # Grok director: per-clip effect overrides
                "caption_style": caption_style,
                "broll_transition_type": broll_transition_type,
                # Cold open hook for retention optimization
                "hook_phrase": hook_phrase,
                "hook_timestamp": hook_timestamp,
                "visual_hook_time": visual_hook_time,
                "hook_enabled": True,  # Enable by default, can be controlled via template
                # Emotional arc for dynamic intensity ramp
                "setup_end": setup_end,
                "escalation_peak": escalation_peak,
                "quotable_line": quotable_line,
                # Dramatic pauses for effect hold
                "key_pauses": key_pauses,
                # Title card at start
                "title_card_text": clip.title,
                "title_card_enabled": False,  # Disabled by default, can be enabled via template
                # Caption pacing
                "rapid_fire_sections": rapid_fire_sections,
                "question_moments": question_moments,
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
            # Store effect failures in render_metadata for UI visibility
            effect_failures = data.get("effect_failures", [])
            if effect_failures:
                logger.warning(f"Clip {clip_id}: {len(effect_failures)} effect(s) failed: {effect_failures}")
                metadata = dict(clip.render_metadata or {})
                metadata["effect_failures"] = effect_failures
                clip.render_metadata = metadata
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(clip, "render_metadata")
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

@celery_app.task(bind=True, max_retries=None)
def process_viral_clip_render(self, clip_id: int):
    """
    Render a viral clip. Uses Redis semaphore to limit concurrent renders.
    If all render slots are full, requeues with delay.
    """
    db = SessionLocal()
    try:
        clip = db.query(ViralClip).filter(ViralClip.id == clip_id).first()
        if not clip:
            logger.error(f"Clip {clip_id} not found")
            return

        # Clean up any stale render slots first
        cleanup_stale_renders()

        # Try to acquire a render slot
        if acquire_render_slot(clip_id):
            try:
                clip.status = "rendering"
                db.commit()
                db.close()  # Close before long-running operation

                loop = asyncio.get_event_loop()
                loop.run_until_complete(_render_clip_async(clip_id))
            finally:
                release_render_slot(clip_id)
        else:
            # All render slots are full - queue this one
            active = get_active_renders()
            logger.info(f"Render queue full ({len(active)}/{MAX_CONCURRENT_RENDERS}). Queueing clip {clip_id} for retry in {RENDER_QUEUE_RETRY_DELAY}s")

            # Update status to show queued state with position info
            clip.status = f"queued ({len(active)}/{MAX_CONCURRENT_RENDERS} rendering)"
            db.commit()

            # Requeue with delay
            raise self.retry(countdown=RENDER_QUEUE_RETRY_DELAY)
    except self.MaxRetriesExceededError:
        logger.error(f"Clip {clip_id} exceeded max retries for render queue")
    finally:
        if db.is_active:
            db.close()


# ==================== AUTO-MODE TASKS ====================
# These tasks are scheduled via Celery Beat to automate the viral clip factory.

@celery_app.task(name="tasks.viral.auto_fetch_influencer_videos")
def auto_fetch_influencer_videos():
    """
    Celery Beat task - runs every hour.
    Checks all influencers with auto_mode_enabled=True and fetches new videos
    if fetch_frequency_hours has elapsed since last_fetch_at.
    """
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        fetched_count = 0

        # Find influencers due for fetch
        influencers = db.query(Influencer).filter(
            Influencer.auto_mode_enabled == True
        ).all()

        for influencer in influencers:
            # Check if fetch is due
            if influencer.last_fetch_at:
                hours_since_fetch = (now - influencer.last_fetch_at).total_seconds() / 3600
                if hours_since_fetch < influencer.fetch_frequency_hours:
                    logger.debug(f"[AutoFetch] Skipping {influencer.name} - last fetched {hours_since_fetch:.1f}h ago (interval: {influencer.fetch_frequency_hours}h)")
                    continue

            logger.info(f"[AutoFetch] Checking {influencer.name} for new videos...")

            # Chain to download/analyze if auto settings enabled
            fetch_and_process_new_videos.delay(
                influencer_id=influencer.id,
                max_videos=influencer.max_videos_per_fetch or 5,
                auto_analyze=influencer.auto_analyze_enabled,
                auto_render=influencer.auto_render_enabled
            )

            # Update last_fetch_at
            influencer.last_fetch_at = now
            db.commit()
            fetched_count += 1

        logger.info(f"[AutoFetch] Triggered fetch for {fetched_count} influencers")
        return {"influencers_fetched": fetched_count}

    except Exception as e:
        logger.error(f"[AutoFetch] Error: {e}")
        return {"error": str(e)}
    finally:
        db.close()


@celery_app.task(name="tasks.viral.fetch_and_process_new_videos")
def fetch_and_process_new_videos(influencer_id: int, max_videos: int, auto_analyze: bool, auto_render: bool):
    """
    Fetch latest videos from channel, then optionally trigger analysis pipeline.
    CRITICAL: Only processes videos published AFTER auto_mode_enabled_at timestamp.
    """
    db = SessionLocal()
    try:
        influencer = db.query(Influencer).filter(Influencer.id == influencer_id).first()
        if not influencer:
            logger.error(f"[AutoFetch] Influencer {influencer_id} not found")
            return {"error": "Influencer not found"}

        # 1. Fetch video metadata from channel
        try:
            response = httpx.post(
                f"{VIDEO_PROCESSOR_URL}/fetch-channel",
                json={"channel_url": influencer.channel_url, "platform": influencer.platform.value},
                timeout=120.0
            )
            if response.status_code != 200:
                raise Exception(f"Fetch failed: {response.text}")
            videos_data = response.json().get("videos", [])[:max_videos]
        except Exception as e:
            logger.error(f"[AutoFetch] Channel fetch error for {influencer.name}: {e}")
            return {"error": str(e)}

        # 2. Filter and create InfluencerVideo records for NEW videos only
        new_video_ids = []
        auto_mode_cutoff = influencer.auto_mode_enabled_at

        for video_data in videos_data:
            # Check if video already exists
            existing = db.query(InfluencerVideo).filter_by(
                platform_video_id=video_data.get("id")
            ).first()

            if existing:
                continue

            # Parse publication date
            pub_date_str = video_data.get("publication_date") or video_data.get("upload_date")
            pub_date = None
            if pub_date_str:
                try:
                    # Try multiple formats
                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y%m%d"]:
                        try:
                            pub_date = datetime.strptime(pub_date_str[:19], fmt)
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass

            # CRITICAL: Only process videos published AFTER auto_mode_enabled_at
            if auto_mode_cutoff and pub_date:
                if pub_date < auto_mode_cutoff:
                    logger.info(f"[AutoFetch] Skipping old video '{video_data.get('title', '')[:50]}' "
                               f"(published {pub_date} < auto_mode enabled {auto_mode_cutoff})")
                    continue

            # Create new video record
            video = InfluencerVideo(
                influencer_id=influencer_id,
                platform_video_id=video_data.get("id"),
                title=video_data.get("title"),
                url=video_data.get("url") or video_data.get("webpage_url"),
                thumbnail_url=video_data.get("thumbnail"),
                description=video_data.get("description"),
                publication_date=pub_date,
                duration=video_data.get("duration"),
                view_count=video_data.get("view_count", 0),
                status="pending"
            )
            db.add(video)
            db.flush()
            new_video_ids.append(video.id)
            logger.info(f"[AutoFetch] New video: {video.title[:60]}...")

        db.commit()

        # 3. If auto_analyze enabled, start the pipeline for each new video
        if auto_analyze and new_video_ids:
            for video_id in new_video_ids:
                # Chain: download -> transcribe -> analyze -> (optional) auto-render
                logger.info(f"[AutoFetch] Auto-analyzing video {video_id}")
                process_viral_video_download.delay(video_id)

        logger.info(f"[AutoFetch] {influencer.name}: {len(new_video_ids)} new videos found")
        return {"new_videos": len(new_video_ids), "video_ids": new_video_ids}

    except Exception as e:
        logger.error(f"[AutoFetch] Error processing {influencer_id}: {e}")
        return {"error": str(e)}
    finally:
        db.close()


@celery_app.task(name="tasks.viral.auto_render_pending_clips")
def auto_render_pending_clips():
    """
    Celery Beat task - runs every 30 minutes.
    Finds clips that are pending render from auto-mode influencers and queues them.
    Respects GPU semaphore (existing logic handles queueing).
    """
    db = SessionLocal()
    try:
        # Find pending clips where influencer has auto_render_enabled
        clips = db.query(ViralClip).join(
            InfluencerVideo, ViralClip.source_video_id == InfluencerVideo.id
        ).join(
            Influencer, InfluencerVideo.influencer_id == Influencer.id
        ).filter(
            Influencer.auto_render_enabled == True,
            ViralClip.status == "pending"
        ).limit(10).all()  # Batch limit to prevent queue explosion

        queued_count = 0
        for clip in clips:
            logger.info(f"[AutoRender] Queuing clip {clip.id}: {clip.title[:50]}...")
            clip.status = "queued for rendering"
            db.commit()
            process_viral_clip_render.delay(clip.id)
            queued_count += 1

        if queued_count > 0:
            logger.info(f"[AutoRender] Queued {queued_count} clips for rendering")
        return {"queued": queued_count}

    except Exception as e:
        logger.error(f"[AutoRender] Error: {e}")
        return {"error": str(e)}
    finally:
        db.close()


@celery_app.task(name="tasks.viral.auto_queue_for_publishing")
def auto_queue_for_publishing():
    """
    Celery Beat task - runs every hour.
    Finds ready clips from auto-publish influencers and adds to publishing queue.
    """
    db = SessionLocal()
    try:
        # Find ready clips not yet queued
        clips = db.query(ViralClip).join(
            InfluencerVideo, ViralClip.source_video_id == InfluencerVideo.id
        ).join(
            Influencer, InfluencerVideo.influencer_id == Influencer.id
        ).filter(
            Influencer.auto_publish_enabled == True,
            ViralClip.status == "ready",
            ViralClip.publishing_status == "unpublished"
        ).all()

        queued_count = 0
        for clip in clips:
            # Find publishing config for this influencer/persona
            config = db.query(PublishingConfig).filter(
                or_(
                    PublishingConfig.influencer_id == clip.source_video.influencer_id,
                    PublishingConfig.persona_id == clip.source_video.influencer.persona_id
                ),
                PublishingConfig.enabled == True
            ).first()

            if not config:
                logger.debug(f"[AutoPublish] No enabled config for clip {clip.id}")
                continue

            # Check virality threshold
            if clip.virality_score and clip.virality_score < config.min_virality_score:
                clip.publishing_status = "skipped"
                clip.skip_reason = f"Virality score {clip.virality_score:.2f} below threshold {config.min_virality_score:.2f}"
                db.commit()
                continue

            # Check clip type allowed
            allowed_types = config.clip_types_allowed or []
            if clip.clip_type and clip.clip_type not in allowed_types:
                clip.publishing_status = "skipped"
                clip.skip_reason = f"Clip type '{clip.clip_type}' not in allowed types"
                db.commit()
                continue

            # Calculate next available posting slot
            scheduled_time = _calculate_next_posting_slot(db, config)

            # Add to queue
            queue_item = PublishingQueueItem(
                clip_id=clip.id,
                config_id=config.id,
                scheduled_time=scheduled_time,
                target_platforms=config.platforms or [],
                requires_approval=config.require_manual_approval,
                status="pending" if config.require_manual_approval else "approved"
            )
            db.add(queue_item)

            clip.publishing_status = "queued"
            db.commit()
            queued_count += 1
            logger.info(f"[AutoPublish] Queued clip {clip.id} for {scheduled_time}")

        if queued_count > 0:
            logger.info(f"[AutoPublish] Queued {queued_count} clips for publishing")
        return {"queued": queued_count}

    except Exception as e:
        logger.error(f"[AutoPublish] Error: {e}")
        return {"error": str(e)}
    finally:
        db.close()


def _calculate_next_posting_slot(db, config: PublishingConfig) -> datetime:
    """
    Find the next available posting time based on config rules and existing queue.
    Implements best practices for engagement:
    - Respects posting_hours (optimal times)
    - Respects min_hours_between_posts (avoid spam)
    - Respects posts_per_day limit
    """
    now = datetime.utcnow()

    # Get recent posts for this config
    recent_posts = db.query(PublishingQueueItem).filter(
        PublishingQueueItem.config_id == config.id,
        PublishingQueueItem.status.in_(["approved", "publishing", "published"]),
        PublishingQueueItem.scheduled_time >= now - timedelta(days=1)
    ).order_by(PublishingQueueItem.scheduled_time.desc()).all()

    posting_hours = config.posting_hours or [9, 13, 18]
    posting_days = config.posting_days or [0, 1, 2, 3, 4, 5, 6]
    posts_per_day = config.posts_per_day or 3
    min_hours_between = config.min_hours_between_posts or 4

    # Find next valid slot
    candidate = now
    attempts = 0
    max_attempts = 200  # Safety limit

    while attempts < max_attempts:
        attempts += 1

        # Check if this day is allowed
        if candidate.weekday() not in posting_days:
            candidate = (candidate + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            continue

        # Check posts_per_day limit
        day_start = candidate.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        posts_on_day = len([p for p in recent_posts
                          if p.scheduled_time and day_start <= p.scheduled_time < day_end])
        if posts_on_day >= posts_per_day:
            candidate = (candidate + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            continue

        # Find next valid hour
        for hour in sorted(posting_hours):
            slot = candidate.replace(hour=hour, minute=0, second=0, microsecond=0)
            if slot <= now:
                continue

            # Check min_hours_between_posts
            too_close = False
            for post in recent_posts:
                if post.scheduled_time:
                    delta = abs((slot - post.scheduled_time).total_seconds())
                    if delta < min_hours_between * 3600:
                        too_close = True
                        break

            if not too_close:
                return slot

        # No valid hour today, try tomorrow
        candidate = (candidate + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

        # Safety: don't schedule more than 7 days out
        if (candidate - now).days > 7:
            return now + timedelta(hours=1)

    # Fallback
    return now + timedelta(hours=1)


@celery_app.task(name="tasks.viral.process_publishing_queue")
def process_publishing_queue():
    """
    Celery Beat task - runs every 15 minutes.
    Publishes approved clips whose scheduled_time has arrived.
    """
    db = SessionLocal()
    try:
        now = datetime.utcnow()

        # Find items ready to publish
        items = db.query(PublishingQueueItem).filter(
            PublishingQueueItem.status == "approved",
            PublishingQueueItem.scheduled_time <= now
        ).order_by(PublishingQueueItem.priority.desc()).limit(5).all()

        published_count = 0
        for item in items:
            logger.info(f"[PublishQueue] Publishing clip {item.clip_id}...")
            publish_single_clip.delay(item.id)
            published_count += 1

        if published_count > 0:
            logger.info(f"[PublishQueue] Started publishing {published_count} items")
        return {"publishing": published_count}

    except Exception as e:
        logger.error(f"[PublishQueue] Error: {e}")
        return {"error": str(e)}
    finally:
        db.close()


@celery_app.task(name="tasks.viral.publish_single_clip", bind=True, max_retries=3)
def publish_single_clip(self, queue_item_id: int):
    """
    Publish a single clip to all target platforms via Blotato.
    """
    db = SessionLocal()
    try:
        item = db.query(PublishingQueueItem).filter(PublishingQueueItem.id == queue_item_id).first()
        if not item:
            logger.error(f"[Publish] Queue item {queue_item_id} not found")
            return {"error": "Queue item not found"}

        clip = db.query(ViralClip).filter(ViralClip.id == item.clip_id).first()
        config = db.query(PublishingConfig).filter(PublishingConfig.id == item.config_id).first()

        if not clip or not config:
            logger.error(f"[Publish] Clip or config not found for queue item {queue_item_id}")
            return {"error": "Clip or config not found"}

        item.status = "publishing"
        db.commit()

        # Prepare content
        video_path = clip.edited_video_path
        if video_path:
            video_path = video_path.replace("/outputs/", "/app/assets/output/")

        # Call Blotato API for each platform
        # TODO: Integrate with actual Blotato/Publisher service
        post_ids = {}
        target_platforms = item.target_platforms or []

        for platform in target_platforms:
            try:
                # Placeholder for Blotato integration
                # result = publisher_service.publish(
                #     account_id=config.blotato_account_id,
                #     platform=platform,
                #     video_path=video_path,
                #     title=clip.title,
                #     description=clip.description,
                #     hashtags=clip.hashtags
                # )
                # post_ids[platform] = result.get("post_id")

                # For now, log and mark as simulated
                logger.info(f"[Publish] Would publish clip {clip.id} to {platform}")
                post_ids[platform] = f"simulated_{platform}_{clip.id}"

            except Exception as e:
                logger.error(f"[Publish] Failed {platform}: {e}")
                post_ids[platform] = f"error: {str(e)}"

        # Update records
        item.blotato_post_ids = post_ids
        item.published_at = datetime.utcnow()
        item.status = "published" if any("error" not in str(v) for v in post_ids.values()) else "failed"

        clip.blotato_post_id = _json.dumps(post_ids)
        clip.published_at = datetime.utcnow()
        clip.publishing_status = "published"

        db.commit()
        logger.info(f"[Publish] Clip {clip.id} published: {post_ids}")
        return {"clip_id": clip.id, "platforms": post_ids}

    except Exception as e:
        logger.error(f"[Publish] Error publishing queue item {queue_item_id}: {e}")
        db.rollback()
        try:
            item = db.query(PublishingQueueItem).filter(PublishingQueueItem.id == queue_item_id).first()
            if item:
                item.status = "failed"
                item.error_message = str(e)
                db.commit()
        except Exception:
            pass
        raise self.retry(exc=e, countdown=300)  # Retry in 5 minutes

    finally:
        db.close()
