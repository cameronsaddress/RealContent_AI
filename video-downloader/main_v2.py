"""
Video Processing Service for n8n - Hybird Version (Legacy + CapCut Upgrade)
- Downloads videos from TikTok, Instagram, YouTube using yt-dlp
- Processes videos with MoviePy (TikTok looks) + FFmpeg (NVENC on DGX)
- Combines avatar videos with background footage (Legacy/RealEstate)
- Viral Clips: TradWest-style Grid, Pulse, Grading, Captions/Karaoke (MoviePy TikTok Style)
"""

import os
# Force MoviePy to use system FFmpeg (which has NVENC support)
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

import subprocess
import uuid
import json
from pathlib import Path
from typing import Optional, List, Tuple
import random

# Monkey patch for Pillow 10+ (removes ANTIALIAS) used by moviepy 1.0.3
import PIL.Image
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# MoviePy imports (works with MoviePy 1.x + 2.x patterns used below)
from moviepy import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
    concatenate_audioclips,
    CompositeAudioClip,
)
# fx classes in your original file (kept)
from moviepy.video.fx import Resize, MultiplyColor, LumContrast, FadeOut
from moviepy.audio.fx import MultiplyVolume
import librosa

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from datetime import datetime
import shutil

# ============ CONFIG & DIRS ============
app = FastAPI(
    title="Video Processing Service",
    description="GPU-accelerated video download and processing (Hybrid V2)",
    version="2.2.0"
)

DOWNLOAD_DIR = Path("/downloads")
OUTPUT_DIR = Path("/outputs")
MUSIC_DIR = Path("/music")
DOWNLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MUSIC_DIR.mkdir(exist_ok=True)

# ============ MODELS ============

class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None
    format: Optional[str] = "mp4"

class DownloadResponse(BaseModel):
    success: bool
    filename: str
    path: str
    size: int
    duration: Optional[float] = None
    title: Optional[str] = None
    platform: Optional[str] = None

class ComposeRequest(BaseModel):
    script_id: str
    avatar_path: str
    background_path: str
    audio_path: Optional[str] = None
    music_path: Optional[str] = None
    output_filename: Optional[str] = None
    use_gpu: bool = True
    avatar_scale: float = 0.75
    avatar_offset_x: int = -250
    avatar_offset_y: int = 600
    greenscreen_color: str = "0x00FF00"
    original_volume: float = 0.7
    avatar_volume: float = 1.0
    music_volume: float = 0.3

class TranscribeRequest(BaseModel):
    audio_path: str
    output_format: str = "srt"

class FetchChannelRequest(BaseModel):
    url: str
    limit: int = 10
    platform: Optional[str] = None

class RenderClipRequest(BaseModel):
    video_path: str
    clip_id: str
    start_time: float
    end_time: float
    style: str = "split_screen" 
    caption_style: str = "karaoke"
    font: str = "Arial"

# ============ LAZY LOADERS ============
whisper_model = None
def get_whisper_model():
    global whisper_model
    import whisper
    import torch
    if whisper_model is None:
        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model("medium", device=device)
    return whisper_model

# ============ HELPER FUNCTIONS ============

def get_gpu_encoder() -> tuple[str, list]:
    """
    Encode: NVENC on DGX when available.
    MoviePy will call ffmpeg; we pass these as ffmpeg_params in write_videofile.
    """
    # FIRST: Check if GPU is actually alive (prevent NVML errors)
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        print("⚠️ GPU not detected by nvidia-smi. Falling back to CPU.")
        return "libx264", ["-preset", "fast", "-crf", "20", "-b:v", "8M"]

    try:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10)
        if "h264_nvenc" in result.stdout:
            # DGX-friendly: quality-speed balance
            return "h264_nvenc", ["-preset", "p4", "-tune", "hq", "-b:v", "8M", "-maxrate", "10M", "-bufsize", "20M"]
    except:
        pass
    return "libx264", ["-preset", "fast", "-crf", "20", "-b:v", "8M"]

def get_video_info(path: str) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"duration": 0, "size": 0, "width": 0, "height": 0}
    data = json.loads(result.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fps = 30
    try:
        if video_stream.get("r_frame_rate"):
            fps = eval(video_stream["r_frame_rate"])
    except:
        fps = 30
    return {
        "duration": float(data.get("format", {}).get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps or 30,
        "codec": video_stream.get("codec_name", "unknown"),
        "size": int(data.get("format", {}).get("size", 0))
    }

def create_cross_image():
    size = (200, 200)
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, 150)
    except:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), "†", font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except:
        w, h = 100, 100

    draw.text(((size[0] - w) / 2, (size[1] - h) / 2), "†", font=font, fill=(128, 0, 128))
    cross_path = OUTPUT_DIR / f"cross_{uuid.uuid4().hex[:8]}.png"
    img.save(cross_path)
    return str(cross_path)

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def build_trigger_intervals(trigger_words: Optional[List[dict]]) -> List[Tuple[float, float]]:
    """
    Returns intervals in *clip-relative* time (seconds).
    Each trigger word gets:
      - quick pulse window: (start, end) used in pulse function
      - VHS fuzz window: 0.5s starting at trigger start (or within provided end)
    We'll return the VHS windows; pulse uses the same list but can weight it differently.
    """
    if not trigger_words:
        return []
    intervals = []
    for trig in trigger_words[:50]:
        t0 = _safe_float(trig.get("start", 0.0), 0.0)
        t1 = _safe_float(trig.get("end", t0 + 0.20), t0 + 0.20)
        # fuzzy VHS effect for half a second
        v0 = t0
        v1 = min(t0 + 0.50, max(t1, t0 + 0.50))
        intervals.append((max(0.0, v0), max(0.0, v1)))
    return intervals

def detect_beats(audio_path: str) -> List[float]:
    """
    Uses librosa to find beat timestamps in seconds.
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return list(beat_times)
    except Exception as e:
        print(f"Beat detection failed: {e}")
        return []

def tiktok_pulse_scale_fn(base_sin_hz: float, trigger_intervals: List[Tuple[float, float]], beat_times: List[float] = None):
    """
    MoviePy resize(scale=callable(t)->float).
    - subtle breathing pulse always on
    - Big Pulse: Trigger Keywords (0.08 kick)
    - Small Pulse: Music Beats (0.02 kick)
    """
    # Pre-pack intervals for quick checks
    intervals = trigger_intervals or []
    beats = sorted(beat_times) if beat_times else []

    def scale(t: float) -> float:
        # baseline: light breathing
        s = 1.0 + 0.012 * np.sin(2 * np.pi * base_sin_hz * t)
        
        # 1. Big Pulse (Keywords) - Strong
        for a, b in intervals:
            if a <= t <= b:
                prog = (t - a) / max(1e-6, (b - a))
                bump = 0.08 * (1.0 - prog)
                s += bump
                break
        
        # 2. Small Pulse (Beats) - Subtle
        # Check if t is within 0.15s after a beat
        # Naive optimization: just check beats around t
        if beats:
            # Simple linear scan or just check if t is close to any beat? 
            # Given standard BPM, checking nearby is fast enough. 
            # We assume beats are sorted. 
            for b in beats:
                if b > t: 
                    break # passed possible target
                if b <= t <= b + 0.15:
                    prog = (t - b) / 0.15
                    bump = 0.025 * (1.0 - prog)
                    s += bump
                    break
                    
        return float(min(max(s, 0.90), 1.35))
    return scale

def apply_vhs_fuzz_effect(clip: VideoFileClip, vhs_intervals: List[Tuple[float, float]]):
    """
    Fuzzy VHS effect for 0.5s on triggers:
    - horizontal jitter
    - scanlines
    - chroma wobble (cheap)
    - noise + slight blur proxy (noise + down-up sample)
    Done per-frame in MoviePy so it composes with all other TikTok looks.
    """
    if not vhs_intervals:
        return clip

    intervals = vhs_intervals

    def in_interval(t: float) -> bool:
        for a, b in intervals:
            if a <= t <= b:
                return True
        return False

    def vhs_frame(get_frame, t: float):
        frame = get_frame(t)
        if not in_interval(t):
            return frame

        # Work in float for effects
        f = frame.astype(np.float32)

        h, w = f.shape[:2]

        # 1) Horizontal jitter (line-based roll)
        jitter = int(6 * np.sin(2 * np.pi * 18.0 * t) + np.random.randint(-3, 4))
        if jitter != 0:
            f = np.roll(f, shift=jitter, axis=1)

        # 2) Scanlines
        # darken every other row slightly
        f[::2, :, :] *= 0.90

        # 3) Chroma wobble (simple channel shifts)
        cshift = int(2 * np.sin(2 * np.pi * 7.0 * t))
        if cshift != 0:
            f[:, :, 0] = np.roll(f[:, :, 0], shift=cshift, axis=1)  # R
            f[:, :, 2] = np.roll(f[:, :, 2], shift=-cshift, axis=1) # B

        # 4) Noise
        noise = np.random.normal(loc=0.0, scale=18.0, size=f.shape).astype(np.float32)
        f = f + noise

        # 5) “Fuzzy” blur proxy (downscale then upscale quickly)
        # (keeps CPU cost modest vs true blur kernels)
        if w > 32 and h > 32:
            # downsample by 2
            f2 = f[::2, ::2, :]
            # upsample nearest
            f = np.repeat(np.repeat(f2, 2, axis=0), 2, axis=1)
            f = f[:h, :w, :]

        return np.clip(f, 0, 255).astype(np.uint8)

    return clip.fl(lambda gf, t: vhs_frame(gf, t), apply_to=["mask"])

def make_tiktok_captions(
    transcript_segments: list[dict],
    clip_start_offset: float,
    clip_duration: float,
    font: str = "Arial",
    preset: str = "trad_west"
) -> List[TextClip]:
    """
    TikTok-ish caption look:
    - large, center-bottom
    - thick black stroke
    - slight shadow pop
    Uses transcript_segments with word timestamps when present.
    """
    captions: List[TextClip] = []
    if not transcript_segments:
        return captions

    # Style knobs per preset
    if preset == "trad_west":
        fontsize = 86
        stroke_w = 4
        y = 0.78
        # "Dirty" font approximation: Nimbus Sans Narrow Bold (Tall/Impactful)
        font_path = "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf"
        if not os.path.exists(font_path):
            font_path = "Arial" # Fallback
    else:
        fontsize = 78
        stroke_w = 4
        y = 0.80
        font_path = "Arial"

    # We’ll create short phrases per segment (or per word if provided).
    for seg in transcript_segments:
        s0 = _safe_float(seg.get("start", 0.0), 0.0) - clip_start_offset
        s1 = _safe_float(seg.get("end", s0 + 0.2), s0 + 0.2) - clip_start_offset
        if s1 <= 0 or s0 >= clip_duration:
            continue
        s0 = max(0.0, s0)
        s1 = min(clip_duration, s1)

        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # --- Layer 1: The "Glow" (Thick Dark Orange Stroke, semi-transparent) ---
        # "glowing dark orange (glows a bit outside of the letters)"
        try:
            glow = (
                TextClip(
                    text=text,
                    font=font_path,
                    font_size=fontsize,
                    color="#D35400",  # Dark Orange
                    stroke_color="#D35400",
                    stroke_width=20, # Wide stroke for glow
                    method="caption",
                    size=(980, None),
                )
                .with_start(s0)
                .with_end(s1)
                .with_opacity(0.6) # Soften it
                .with_position(("center", y), relative=True)
            )
        except:
             # Fallback simple
            glow = None

        # --- Layer 2: Black Outline (Contrast) ---
        try:
            outline = (
                TextClip(
                    text=text,
                    font=font_path,
                    font_size=fontsize,
                    color="black",
                    stroke_color="black",
                    stroke_width=stroke_w + 4, # Slightly wider than fill
                    method="caption",
                    size=(980, None),
                )
                .with_start(s0)
                .with_end(s1)
                .with_position(("center", y), relative=True)
            )
        except:
            outline = None

        # --- Layer 3: Main Text (Bright Orange Fill) ---
        try:
            main_text = (
                TextClip(
                    text=text,
                    font=font_path,
                    font_size=fontsize,
                    color="#FF8C00", # Dark Orange / Bright Orange
                    stroke_color="black", # Thin crisp stroke
                    stroke_width=0, # Let outline layer handle stroke or keep 0
                    method="caption",
                    size=(980, None),
                )
                .with_start(s0)
                .with_end(s1)
                .with_position(("center", y), relative=True)
            )
        except Exception:
            # Fallback if font/method differs in env
            main_text = (
                TextClip(
                    text=text,
                    font="Arial",
                    font_size=fontsize,
                    color="white",
                    stroke_color="black",
                    stroke_width=stroke_w,
                )
                .with_start(s0)
                .with_end(s1)
                .with_position(("center", y), relative=True)
            )

        # Stack order: Glow -> Outline -> Main
        if glow:
            captions.append(glow)
        if outline:
            captions.append(outline)
        captions.append(main_text)

    return captions

def apply_tiktok_looks(clip: VideoFileClip, preset: str) -> VideoFileClip:
    """
    “All the TikTok looks” (practical set that runs well):
    - contrast/luma pop
    - slight saturation
    - slight warm tint
    - vignette-ish darkening approximation (cheap)
    Implemented with MoviePy effects + lightweight per-frame.
    """
    # Effects from your imports
    if preset == "trad_west":
        clip = clip.with_effects([
            LumContrast(lum=0, contrast=0.20, contrast_thr=127),
            MultiplyColor(1.05),
        ])
    else:
        clip = clip.with_effects([
            LumContrast(lum=0, contrast=0.15, contrast_thr=127),
            MultiplyColor(1.03),
        ])

    # Cheap vignette approximation via per-frame radial mask
    def vignette(get_frame, t):
        fr = get_frame(t).astype(np.float32)
        h, w = fr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r /= (np.sqrt(cx ** 2 + cy ** 2) + 1e-6)
        # vignette strength
        strength = 0.22 if preset == "trad_west" else 0.16
        mask = 1.0 - strength * (r ** 1.6)
        mask = np.clip(mask, 0.0, 1.0)
        fr *= mask[..., None]
        return np.clip(fr, 0, 255).astype(np.uint8)

    return clip.fl(lambda gf, t: vignette(gf, t), apply_to=["mask"])

def select_background_music(transcript_text: str, mood_hint: Optional[str] = None) -> Optional[str]:
    """
    Selects the best music track based on mood hint or transcript keywords.
    Keywords in filenames are matched against mood keywords.
    """
    if not MUSIC_DIR.exists():
        return None
        
    music_files = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))
    if not music_files:
        return None

    # 1. Define Mood Keywords mapping
    # Maps internal mood keys to filename keywords
    mood_map = {
        "epic": ["epic", "battle", "war", "hero", "orchestra", "cinematic"],
        "industrial": ["industrial", "metal", "cyber", "dark", "heavy", "factory", "machine"],
        "phonk": ["phonk", "aggressive", "drift", "cowbell", "gym", "workout"],
        "sad": ["sad", "emotional", "piano", "slow", "melancholy", "cry"],
        "happy": ["happy", "upbeat", "summer", "pop", "fun", "joy"],
        "tech": ["tech", "glitch", "future", "matrix", "cyberpunk"]
    }

    # 2. Determine Mood
    detected_mood = "epic" # Default
    
    if mood_hint and mood_hint.lower() in mood_map:
        detected_mood = mood_hint.lower()
    else:
        # Heuristic from transcript
        text = (transcript_text or "").lower()
        
        # Simple scoring
        scores = {k: 0 for k in mood_map}
        
        for k, keywords in mood_map.items():
            for kw in keywords:
                if kw in text:
                    scores[k] += 1
        
        # Add specific trigger words handling
        if "money" in text or "rich" in text or "hustle" in text:
            scores["phonk"] += 2
        if "war" in text or "fight" in text or "win" in text:
            scores["epic"] += 2
        if "robot" in text or "ai" in text or "code" in text:
            scores["industrial"] += 2
            
        # Get max
        best_mood = max(scores, key=scores.get)
        if scores[best_mood] > 0:
            detected_mood = best_mood
            
    print(f"Smart Music Selection: Detected Mood '{detected_mood}' from text/hint.")

    # 3. Filter Files
    target_keywords = mood_map.get(detected_mood, [])
    scored_files = []
    
    for f in music_files:
        fname = f.name.lower()
        score = 0
        for kw in target_keywords:
            if kw in fname:
                score += 10
        # Soft match for generic terms
        if detected_mood in fname:
            score += 20
            
        # Random jitter to rotate tracks with same score
        score += random.random()
        scored_files.append((score, f))
        
    # Sort desc
    scored_files.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top 1 (or random from top 3 if close?)
    # For now, just top
    selected = scored_files[0][1]
    print(f"Selected Track: {selected.name} (Score: {scored_files[0][0]:.2f})")
    return str(selected)

# ============ MUSIC ENDPOINTS ============

class MusicFile(BaseModel):
    filename: str
    size: int
    created_at: str
    is_active: bool = False

@app.get("/music", response_model=dict)
async def list_music():
    files = []
    if MUSIC_DIR.exists():
        # sort by time
        for f in sorted(MUSIC_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.suffix.lower() in [".mp3", ".wav"]:
                files.append({
                    "filename": f.name,
                    "size": f.stat().st_size,
                    "created_at": str(datetime.fromtimestamp(f.stat().st_mtime)),
                    "is_active": True # All available tracks are candidates for random choice
                })
    return {"files": files}

@app.post("/music")
async def upload_music_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".mp3", ".wav")):
             raise HTTPException(status_code=400, detail="Only .mp3 and .wav allowed")
        
        file_path = MUSIC_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"success": True, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/music/{filename}")
async def delete_music_file(filename: str):
    file_path = MUSIC_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        file_path.unlink()
        return {"success": True, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ ENDPOINTS ============

@app.get("/health")
async def health_check():
    return {"status": "healthy_v2_2", "gpu_encoder": get_gpu_encoder()[0]}

@app.post("/download", response_model=DownloadResponse)
async def download_video(request: DownloadRequest):
    base_filename = request.filename or str(uuid.uuid4())[:8]
    output_path = DOWNLOAD_DIR / f"{base_filename}.{request.format}"

    if output_path.exists():
        output_path.unlink()

    platform_opts = []
    if "tiktok" in request.url:
        platform_opts = ["--impersonate", "chrome"]
    if "rumble" in request.url:
        platform_opts = ["--impersonate", "safari"]
    if "instagram" in request.url:
        platform_opts = ["--impersonate", "chrome"]

    cmd = [
        "yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", str(output_path), "--no-playlist", "--merge-output-format", "mp4",
        "--no-warnings", "--print-json", *platform_opts, request.url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
        if result.returncode == 0 and output_path.exists():
            return DownloadResponse(success=True, filename=output_path.name, path=str(output_path), size=output_path.stat().st_size)
    except Exception as e:
        print(f"Download Error: {e}")

    raise HTTPException(status_code=500, detail="Download failed")

@app.post("/transcribe-whisper")
async def transcribe_whisper(request: TranscribeRequest):
    try:
        # 1. Explicitly extract audio to 16kHz WAV first (Much faster/safer than Whisper's internal ffmpeg)
        # This isolates the heavy I/O and ffmpeg work from the GPU inference
        import uuid
        temp_audio_filename = f"temp_whisper_{uuid.uuid4().hex[:8]}.wav"
        temp_audio_path = OUTPUT_DIR / temp_audio_filename
        
        # Optimized FFmpeg command for audio extraction
        # -vn: Drop video (no decode needed!) -> Fastest way
        # -ar 16000: NATIVE resolution for Whisper AI (Lossless for this model, prevents aliasing)
        # -ac 1: NATIVE channel count for Whisper AI
        # -threads 32: Utilize DGX CPU power for rapid processing
        extract_cmd = [
            "ffmpeg", "-y",
            "-threads", "32", 
            "-i", request.audio_path,
            "-vn", 
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(temp_audio_path)
        ]
        
        print(f"Extracting 16k audio for Whisper: {' '.join(extract_cmd)}")
        # Run extraction in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: subprocess.run(extract_cmd, check=True))
        
        if not temp_audio_path.exists():
             raise Exception("Audio extraction failed")

        model = get_whisper_model()
        
        # 2. Run Inference in ThreadPool to prevent blocking FastAPI heartbeat/other requests
        # This prevents the "ReadTimeout" or "Server disconnected" errors on large files
        def _run_inference():
            return model.transcribe(str(temp_audio_path), word_timestamps=True)
            
        print(f"Starting GPU Inference on: {temp_audio_path}")
        result = await loop.run_in_executor(None, _run_inference)
        
        # Cleanup
        try:
            os.unlink(temp_audio_path)
        except:
            pass

        if request.output_format == "json":
            return result
        return {"text": result["text"], "segments": result["segments"]}
    except Exception as e:
        print(f"Transcription Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch-channel")
async def fetch_channel_videos(request: FetchChannelRequest):
    cmd = ["yt-dlp", "--dump-json", "--playlist-end", str(request.limit), "--no-warnings", request.url]
    if request.platform == "rumble":
        cmd.extend(["--impersonate", "safari"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    videos.append({"id": data.get("id"), "title": data.get("title"), "url": data.get("webpage_url")})
                except:
                    pass
        return {"success": True, "videos": videos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compose")
async def compose_video(request: ComposeRequest):
    return {"success": True, "note": "Legacy composition endpoint preserved but simplified."}

@app.post("/render-viral-clip")
async def render_viral_clip(request: RenderClipRequest):
    """
    MoviePy pipeline:
      - GPU decode hint (best-effort via ffmpeg_params)
      - TikTok looks (grade + vignette)
      - Pulse zoom (Resize with callable)
      - Keyword hits:
          * quick pulse (already in zoom scale)
                # simple loop by concatenation - USE AUDIO CONCAT
                loops = int(np.ceil(duration / max(1e-6, bgm.duration)))
                bgm = concatenate_audioclips([bgm] * loops)
            bgm = bgm.subclipped(0, duration)

            # volume ramp last 5 seconds (Outro Swell)
            ramp_start = max(0.0, duration - 5.0)
            def bgm_gain(t):
                # Standard volume: 0.25 (ducked slightly for voice)
                base = 0.25
                if t < ramp_start:
                    return base
                # Ramp to 0.80 at end
                prog = min(1.0, (t - ramp_start) / 5.0)
                return base + 0.55 * prog 

            # Apply gain to BGM
            bgm = bgm.fl(lambda gf, t: gf(t) * bgm_gain(t))
            
            # Mix: Source (Original Raw Volume) + BGM (Ducked/Ramped)
            # User request: Ensure source is NOT turned down.
            mixed = CompositeAudioClip([audio, bgm])
        else:
            mixed = audio
            
        # ---- Beat-Sync Scaling (Moved here to use beat_times)
        # Re-apply pulse with beats if we have them
        scale_fn = tiktok_pulse_scale_fn(base_sin_hz=1.5, trigger_intervals=trigger_intervals, beat_times=beat_times)
        pulsed = styled.with_effects([Resize(lambda t: scale_fn(t))]).cropped(
            x_center=target_w / 2,
            y_center=target_h / 2,
            width=target_w,
            height=target_h
        )

        # ---- NEW: VHS fuzz effect for 0.5s on triggers (in addition to pulse)
        vhsd = apply_vhs_fuzz_effect(pulsed, trigger_intervals)
        
        # Update vhsd audio with the mixed one for final composite
        vhsd = vhsd.with_audio(mixed)

        # ---- Final composite
        final = CompositeVideoClip(
            [vhsd, *caption_layers, cross, handle_txt],
            size=(target_w, target_h)
        ).with_audio(mixed)

        # ---- DGX / GPU encode (NVENC) via ffmpeg_params
        # Notes:
        # - -movflags +faststart helps TikTok/IG upload
        # - -pix_fmt yuv420p for compatibility
        # - We also hint CUDA again; harmless if ignored during encode.
        ffmpeg_params = [
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-bf", "2",
            "-rc-lookahead", "20",
            "-spatial_aq", "1",
            "-temporal_aq", "1",
            "-aq-strength", "8",
            # best-effort hw hints (some builds ignore):
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            *gpu_opts,
        ]

        # Write
        final.write_videofile(
            str(output_path),
            codec=encoder,
            audio_codec="aac",
            fps=get_video_info(request.video_path).get("fps", 30) or 30,
            threads=0,
            preset=None,          # codec opts already in ffmpeg_params
            ffmpeg_params=ffmpeg_params,
            logger=None
        )

        info = get_video_info(str(output_path))
        return {"success": True, "path": str(output_path), "filename": output_filename, "duration": info["duration"]}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Render failed: {str(e)}")
    finally:
        # cleanup cross image if we generated it
        if cross_path and os.path.exists(cross_path):
            try:
                os.unlink(cross_path)
            except:
                pass
        
        # cleanup precut temp file
        try:
             # Look for local variable 'precut_path' if it exists in scope
             local_vars = locals()
             if 'precut_path' in local_vars and local_vars['precut_path'].exists():
                  local_vars['precut_path'].unlink()
        except:
             pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
