"""
Video Processing Service for n8n - Hybird Version (Legacy + CapCut Upgrade)
- Downloads videos from TikTok, Instagram, YouTube using yt-dlp
- Processes videos with FFmpeg (GPU-accelerated on DGX Spark)
- Combines avatar videos with background footage (Legacy/RealEstate)
- Viral Clips: TradWest-style Grid, Pulse, Grading, Karaoke (New CapCut Style)
"""

import os
# Force MoviePy to use system FFmpeg (which has NVENC support)
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import subprocess
import sys
import uuid
import json
import asyncio
from pathlib import Path
from typing import Optional, List
from enum import Enum
import random
import urllib.request

# Monkey patch for Pillow 10+ (removes ANTIALIAS) used by moviepy 1.0.3
import PIL.Image
import numpy as np
from scipy.signal import find_peaks
from PIL import Image, ImageDraw, ImageFont

# MoviePy v2.0 imports (Kep for RealEstate/Compose endpoint)
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, ImageClip, concatenate_videoclips, concatenate_audioclips, CompositeAudioClip
from moviepy.video.fx import Resize, MultiplyColor, LumContrast, FadeOut
from moviepy.audio.fx import MultiplyVolume, AudioLoop

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ============ CONFIG & DIRS ============
app = FastAPI(
    title="Video Processing Service",
    description="GPU-accelerated video download and processing (Hybrid V2)",
    version="2.1.0"
)

DOWNLOAD_DIR = Path("/downloads")
OUTPUT_DIR = Path("/outputs")
MUSIC_DIR = Path("/music")
DOWNLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Trigger words for pulse/glitch effects - HARSH words only
# Curse words, negative connotation, destruction, violence
HARSH_TRIGGER_WORDS = [
    # Curse words
    "fuck", "fucking", "fucked", "shit", "bitch", "bitches", "ass", "damn", "damned",
    "hell", "crap", "bastard", "bullshit", "piss", "dick", "asshole",
    # Violence/destruction
    "kill", "killed", "killing", "murder", "destroy", "destroyed", "destruction",
    "death", "dead", "die", "dying", "war", "attack", "fight", "fighting",
    "blood", "violent", "violence", "slaughter", "massacre", "annihilate",
    # Negative/intense
    "hate", "hated", "hatred", "evil", "wicked", "corrupt", "corruption",
    "chaos", "insane", "crazy", "brutal", "savage", "vicious", "ruthless",
    "toxic", "pathetic", "disgusting", "disgrace", "disaster", "catastrophe",
    # Power/dominance (aggressive context)
    "dominate", "crush", "crushed", "obliterate", "demolish", "wreck", "wrecked",
    "smash", "smashed", "burn", "burning", "explode", "explosion",
    # Confrontational
    "enemy", "enemies", "loser", "losers", "idiot", "idiots", "stupid",
    "moron", "fool", "fools", "liar", "liars", "traitor", "coward"
]
# Alias for backwards compatibility
TRAD_TRIGGER_WORDS = HARSH_TRIGGER_WORDS

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

class CaptionRequest(BaseModel):
    script_id: str
    video_path: str
    srt_path: Optional[str] = None
    ass_path: Optional[str] = None
    output_filename: Optional[str] = None
    font_size: int = 48
    font_color: str = "white"
    outline_color: str = "black"
    use_gpu: bool = True

class TranscribeRequest(BaseModel):
    audio_path: str
    output_format: str = "srt"

class FetchChannelRequest(BaseModel):
    url: str
    limit: int = 10
    platform: Optional[str] = None

class ExtractClipRequest(BaseModel):
    video_path: str
    start_time: float
    end_time: float
    output_filename: Optional[str] = None

# Available TikTok-style fonts for viral clips
VIRAL_FONTS = [
    "Honk",           # Playful & Expressive
    "Pirata One",     # Vintage Gothic
    "Rubik Vinyl",    # Retro Groovy
    "Rubik 80s Fade", # Neon Retro
    "Rubik Dirt",     # Grungy Distressed
]

class RenderClipRequest(BaseModel):
    video_path: str
    start_time: float
    end_time: float
    transcript_segments: list[dict] = []
    style_preset: str = "trad_west"
    font: str = "random"  # "random" picks from VIRAL_FONTS, or specify exact font name
    output_filename: Optional[str] = None
    outro_path: Optional[str] = None
    channel_handle: Optional[str] = None
    trigger_words: Optional[List[dict]] = None
    status_webhook_url: Optional[str] = None

def report_status(webhook_url: str, status: str):
    if not webhook_url: return
    try:
        data = json.dumps({"status": status}).encode('utf-8')
        req = urllib.request.Request(webhook_url, data=data, headers={'Content-Type': 'application/json'}, method='PUT')
        with urllib.request.urlopen(req, timeout=2) as f: pass
    except: pass


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

def safe_transcribe(audio_path: str, word_timestamps: bool = True):
    """
    Safely transcribe audio, resetting decoder KV cache to prevent
    'Key and Value must have the same sequence length' errors.
    """
    import torch
    model = get_whisper_model()

    # Reset decoder KV cache to prevent stale state errors
    # This fixes the "Key and Value must have the same sequence length" bug
    if hasattr(model, 'decoder'):
        for block in model.decoder.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'kv_cache'):
                block.attn.kv_cache = None
            if hasattr(block, 'cross_attn') and hasattr(block.cross_attn, 'kv_cache'):
                block.cross_attn.kv_cache = None

    # Clear CUDA cache to prevent memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model.transcribe(audio_path, word_timestamps=word_timestamps)

def detect_beats(audio_path: str, max_duration: float = None) -> list:
    """
    Detect beat times in audio file using onset detection.
    Returns list of beat timestamps in seconds.
    """
    import subprocess
    import tempfile

    # Extract audio to WAV for analysis
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        # Extract audio, limit to clip duration if specified
        duration_args = ["-t", str(max_duration)] if max_duration else []
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            *duration_args,
            "-ac", "1", "-ar", "22050",  # Mono, 22kHz for faster processing
            tmp_wav
        ], capture_output=True, check=True)

        # Load audio with scipy
        from scipy.io import wavfile
        sample_rate, audio_data = wavfile.read(tmp_wav)

        # Convert to float
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

        # Onset detection using energy envelope
        # Compute short-time energy with hop size ~23ms (512 samples at 22kHz)
        hop_size = 512
        frame_size = 1024

        # Compute RMS energy per frame
        num_frames = (len(audio_data) - frame_size) // hop_size + 1
        energy = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_size
            frame = audio_data[start:start + frame_size]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        # Normalize energy
        if energy.max() > 0:
            energy = energy / energy.max()

        # Detect onsets: local maxima above threshold with minimum distance
        # Use scipy's find_peaks for robust peak detection
        min_beat_gap = int(0.3 * sample_rate / hop_size)  # Minimum 0.3s between beats
        peaks, _ = find_peaks(energy, height=0.15, distance=min_beat_gap, prominence=0.05)

        # Convert frame indices to timestamps
        beat_times = peaks * hop_size / sample_rate

        return beat_times.tolist()

    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_wav)
        except:
            pass

# ============ HELPER FUNCTIONS (LEGACY & NEW) ============

def get_gpu_encoder() -> tuple[str, list]:
    try:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10)
        if "h264_nvenc" in result.stdout:
            # Tuned for high quality speed
            return "h264_nvenc", ["-preset", "p4", "-tune", "hq", "-b:v", "5M"]
    except:
        pass
    return "libx264", ["-preset", "fast", "-crf", "23", "-b:v", "5M"]

def get_video_info(path: str) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback empty
        return {"duration": 0, "size": 0, "width": 0, "height": 0}
        
    data = json.loads(result.stdout)
    video_stream = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), {})
    return {
        "duration": float(data.get("format", {}).get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": eval(video_stream.get("r_frame_rate", "30/1")) if video_stream.get("r_frame_rate") else 30,
        "codec": video_stream.get("codec_name", "unknown"),
        "size": int(data.get("format", {}).get("size", 0))
    }

def create_cross_image():
    # Generate the Purple Cross overlay
    size = (200, 200)
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Try system fonts
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
    return cross_path

def apply_chromatic_aberration(video_path: str, output_path: str, trigger_windows: list, max_shift: int = 15):
    """
    Apply RGB chromatic aberration (glitch/VHS effect) during trigger word windows.
    Uses MoviePy for per-frame numpy manipulation.

    trigger_windows: list of (start_time, end_time) tuples
    max_shift: max pixel shift for R/B channels at peak intensity
    """
    if not trigger_windows:
        # No triggers, just copy the file
        subprocess.run(["cp", video_path, output_path], check=True)
        return

    print(f"Applying chromatic aberration to {len(trigger_windows)} trigger windows...")

    clip = VideoFileClip(video_path)

    def rgb_shift_effect(get_frame, t):
        """Per-frame RGB channel shift based on trigger windows."""
        frame = get_frame(t)

        # Calculate intensity based on trigger windows
        # Use smooth sine pulse matching the zoom pulse timing
        intensity = 0.0
        for start, end in trigger_windows:
            if start <= t <= end:
                # Smooth sine wave: 0 → 1 → 0 over the window
                duration = end - start
                progress = (t - start) / duration
                intensity = max(intensity, np.sin(np.pi * progress))

        if intensity < 0.01:
            return frame  # No effect needed

        # Calculate pixel shift based on intensity
        shift = int(max_shift * intensity)
        if shift < 1:
            return frame

        # Split into RGB channels
        r_channel = frame[:, :, 0]
        g_channel = frame[:, :, 1]
        b_channel = frame[:, :, 2]

        # Shift R left (negative), B right (positive) for classic VHS glitch
        # Use numpy roll with edge padding
        r_shifted = np.roll(r_channel, -shift, axis=1)
        b_shifted = np.roll(b_channel, shift, axis=1)

        # Fix edge artifacts from roll (fill with edge values)
        r_shifted[:, -shift:] = r_channel[:, -1:]
        b_shifted[:, :shift] = b_channel[:, :1]

        # Recombine
        result = np.stack([r_shifted, g_channel, b_shifted], axis=2)
        return result.astype(np.uint8)

    # Apply the effect using transform (fl is deprecated in moviepy 2.x)
    processed = clip.transform(rgb_shift_effect)

    # Write output with GPU encoding if available
    processed.write_videofile(
        output_path,
        codec='libx264',  # Use CPU codec for reliability
        preset='ultrafast',
        audio_codec='aac',
        logger=None  # Suppress progress bar
    )

    clip.close()
    processed.close()
    print(f"Chromatic aberration applied successfully")

def generate_karaoke_ass(segments: list[dict], output_path: Path, start_offset: float = 0.0, font: str = "Honk", trigger_words: list = None):
    """
    Generate ASS karaoke captions with enhanced trigger word effects.

    Trigger words get:
    - 3x size scale (300%)
    - RGB glitch color (cyan/magenta alternating)
    - Shake effect via shadow blur
    """
    # Build trigger word lookup (by time range)
    trigger_windows = []
    if trigger_words:
        for trig in trigger_words:
            t_start = trig.get('start', 0) - start_offset
            t_end = trig.get('end', t_start + 0.5) - start_offset
            word = trig.get('word', '').lower()
            trigger_windows.append((t_start, t_end, word))

    def is_trigger_word(w_start, w_end, w_text):
        """Check if word matches any trigger window"""
        w_start_rel = w_start - start_offset
        w_end_rel = w_end - start_offset
        w_lower = w_text.lower().strip('.,!?;:')

        for t_start, t_end, t_word in trigger_windows:
            # Check time overlap AND word match
            if w_start_rel < t_end and w_end_rel > t_start:
                if t_word in w_lower or w_lower in t_word:
                    return True
            # Also check if word text matches regardless of timing
            if t_word and (t_word in w_lower or w_lower in t_word):
                return True
        return False

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{font},100,&H00FFFFFF,&H00FF00FF,&H00000000,&HAA000000,-1,0,0,0,100,100,2,0,1,4,2,2,20,20,120,1
Style: TriggerWord,{font},100,&H00FFFF00,&H00FF00FF,&H000000FF,&HAA000000,-1,0,0,0,300,300,2,0,1,6,3,2,20,20,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    def fmt_time(t):
        t = max(0, t - start_offset)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        cs = int((s - int(s)) * 100)
        return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"

    for seg in segments:
        s_start, s_end = seg['start'], seg['end']
        words = seg.get('words', [{'start': s_start, 'end': s_end, 'word': seg['text']}])
        if not words: continue

        karaoke = ""
        current_time = s_start

        for w in words:
            w_start, w_end, w_text = w['start'], w['end'], w['word'].strip()

            # Check for gap between current position and this word's start
            gap = w_start - current_time
            if gap > 0.05:
                gap_cs = int(gap * 100)
                karaoke += f"{{\\k{gap_cs}}}"

            # Duration of this word in centiseconds
            dur_cs = int((w_end - w_start) * 100)
            if dur_cs < 1: dur_cs = 1

            # Check if this is a trigger word
            if is_trigger_word(w_start, w_end, w_text):
                # TRIGGER WORD: 3x size + RGB color + shake
                # \fscx300\fscy300 = 3x scale
                # \c&H00FFFF& = Cyan color (BGR format)
                # \bord6\shad3 = thick border + shadow for impact
                # \t(0,{dur},\fscx100\fscy100) = animate back to normal size
                trigger_fx = f"\\fscx300\\fscy300\\c&H00FFFF&\\bord6\\shad3\\t(0,{dur_cs*10},\\fscx100\\fscy100\\c&HFFFFFF&)"
                karaoke += f"{{\\k{dur_cs}{trigger_fx}}}{w_text.upper()} {{\\r}}"
            else:
                # Normal word
                karaoke += f"{{\\k{dur_cs}}}{w_text} "

            current_time = w_end

        line_entry = f"Dialogue: 0,{fmt_time(s_start)},{fmt_time(s_end)},Karaoke,,0,0,0,,{{\\blur2}}{karaoke}"
        events.append(line_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))

    print(f"Generated karaoke ASS with {len(trigger_windows)} trigger word enhancements")


# ============ ENDPOINTS ============

@app.get("/health")
async def health_check():
    return {"status": "healthy_v2", "gpu_encoder": get_gpu_encoder()[0]}

@app.post("/download", response_model=DownloadResponse)
async def download_video(request: DownloadRequest):
    base_filename = request.filename or str(uuid.uuid4())[:8]
    output_path = DOWNLOAD_DIR / f"{base_filename}.{request.format}"
    if output_path.exists(): output_path.unlink()

    strategies = [
        [],  # Default (Direct)
        ["--proxy", "http://vpn:8888"], # VPN Direct
        ["--proxy", "http://vpn:8888", "--impersonate", "safari"], # VPN + Safari
        ["--proxy", "http://vpn:8888", "--impersonate", "chrome"], # VPN + Chrome
        ["--impersonate", "safari"], # Direct + Safari (Fallback)
    ]

    last_error = None

    for strategy in strategies:
        try:
            print(f"Download strategy: {strategy} for {request.url}")
            cmd = [
                "yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "-o", str(output_path), "--no-playlist", "--merge-output-format", "mp4",
                "--no-warnings", "--print-json", "--no-progress",
                *strategy, 
                request.url
            ]
            
            # 90m timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
            
            if result.returncode == 0:
                # File existence check
                if output_path.exists():
                    return DownloadResponse(success=True, filename=output_path.name, path=str(output_path), size=output_path.stat().st_size)
                else: 
                     last_error = "yt-dlp success but file missing"
            else:
                 last_error = f"Return {result.returncode}: {result.stderr}"
                 
        except Exception as e:
            last_error = str(e)
            continue
            
    print(f"All download strategies failed: {last_error}")     
    raise HTTPException(status_code=500, detail=f"Download failed: {last_error}")

@app.post("/transcribe-whisper")
async def transcribe_whisper(request: TranscribeRequest):
    try:
        if not os.path.exists(request.audio_path):
             raise HTTPException(status_code=404, detail=f"File not found")
        result = safe_transcribe(request.audio_path, word_timestamps=True)
        if request.output_format == "json": return result
        return {"text": result["text"], "segments": result["segments"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch-channel")
async def fetch_channel_videos(request: FetchChannelRequest):
    strategies = [
        [],  # Default (Direct)
        ["--proxy", "http://vpn:8888"], # VPN Direct
        ["--proxy", "http://vpn:8888", "--impersonate", "safari"], # VPN + Safari
        ["--proxy", "http://vpn:8888", "--impersonate", "chrome"], # VPN + Chrome
        ["--impersonate", "safari"], # Direct + Safari (Fallback)
    ]

    last_error = None
    
    for strategy in strategies:
        try:
            cmd = ["yt-dlp", "--dump-json", "--playlist-end", str(request.limit), "--no-warnings", request.url]
            cmd.extend(strategy)
            
            print(f"Fetch strategy: {strategy} for {request.url}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            
            if result.returncode == 0 and result.stdout.strip():
                videos = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            data = json.loads(line)
                            thumbnail = data.get("thumbnail")
                            if not thumbnail:
                                thumbs = data.get("thumbnails") or []
                                if thumbs:
                                    last_thumb = thumbs[-1]
                                    if isinstance(last_thumb, dict):
                                        thumbnail = last_thumb.get("url")
                                    elif isinstance(last_thumb, str):
                                        thumbnail = last_thumb
                            videos.append({
                                "id": data.get("id"),
                                "title": data.get("title"),
                                "url": data.get("webpage_url") or data.get("original_url"),
                                "thumbnail": thumbnail,
                                "duration": data.get("duration"),
                                "upload_date": data.get("upload_date"),
                                "view_count": data.get("view_count"),
                            })
                        except: pass
                if videos:
                     return {"success": True, "videos": videos}
            else:
                 last_error = f"Return {result.returncode}: {result.stderr}"
                 
        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(status_code=500, detail=f"All fetch strategies failed. Last error: {last_error}")
    # End of strategies loop

@app.post("/compose")
async def compose_video(request: ComposeRequest):
    # LEGACY for RealEstate (MoviePy based)
    return {"success": True, "note": "Legacy composition endpoint preserved but simplified."}


@app.post("/render-viral-clip")
def render_viral_clip(request: RenderClipRequest):
    # V5 OPTIMIZATION: DGX Native (CUDA Filters)
    # Using /dev/shm for instant IO
    # Switch to /tmp (disk) to avoid /dev/shm (64MB) overflow on large renders
    temp_dir = Path("/tmp")
    output_filename = request.output_filename or f"viral_{uuid.uuid4().hex[:8]}.mp4"
    output_path = OUTPUT_DIR / output_filename
    
    # Unique temp identifier
    uid = uuid.uuid4().hex[:8]
    temp_audio = temp_dir / f"temp_{uid}.wav"
    temp_main = temp_dir / f"temp_main_{uid}.mp4"
    temp_outro = temp_dir / f"temp_outro_{uid}.mp4"
    temp_grid = temp_dir / f"temp_grid_{uid}.mp4"
    temp_concat_list = temp_dir / f"temp_concat_{uid}.txt"
    ass_path = temp_dir / f"temp_{uid}.ass"
    cross_path = create_cross_image() 

    cleanup_files = [temp_audio, temp_main, temp_outro, temp_grid, temp_concat_list, ass_path, cross_path]
    encoder, gpu_opts = get_gpu_encoder()

    try:
        report_status(request.status_webhook_url, "Processing: Setup")
        # 1. Setup: Duration
        duration = request.end_time - request.start_time

        # 2. EXTRACT CLIP FIRST (needed for per-clip transcription)
        temp_extracted = temp_dir / f"temp_extract_{uid}.mp4"
        cleanup_files.append(temp_extracted)

        report_status(request.status_webhook_url, "Processing: Extracting Clip")
        cmd_extract = [
            "ffmpeg", "-y", "-threads", "0",
            "-ss", str(request.start_time),  # INPUT seeking = fast (seeks to keyframe)
            "-i", request.video_path,
            "-t", str(duration),
            "-c:v", "h264_nvenc", "-preset", "p1",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",  # Fix PTS issues from input seeking
            str(temp_extracted)
        ]
        print(f"DEBUG EXTRACT CMD: {' '.join(cmd_extract)}")
        subprocess.run(cmd_extract, check=True)

        # 3. PER-CLIP TRANSCRIPTION - Fresh Whisper for perfect sync
        # This eliminates timestamp drift from long-video transcription
        report_status(request.status_webhook_url, "Processing: Fresh Transcription")
        print(f"Running fresh Whisper transcription on extracted clip...")
        whisper_result = safe_transcribe(str(temp_extracted), word_timestamps=True)
        fresh_segments = whisper_result.get("segments", [])
        print(f"Fresh transcription: {len(fresh_segments)} segments")

        # Build fresh trigger words from TRAD_TRIGGER_WORDS list
        fresh_trigger_words = []
        for seg in fresh_segments:
            words = seg.get("words", [])
            for word_obj in words:
                # Strip whitespace AND punctuation - Whisper often returns " word" with leading space
                w_text = word_obj.get("word", "").lower().strip().strip(".,!?;:'\"()-")
                if w_text in TRAD_TRIGGER_WORDS:
                    print(f"  TRIGGER HIT: '{w_text}' at {word_obj.get('start', 0):.2f}s")
                    fresh_trigger_words.append({
                        "start": word_obj["start"],
                        "end": word_obj["end"],
                        "word": w_text
                    })
        print(f"Found {len(fresh_trigger_words)} trigger words in fresh transcript")

        # 4. Prepare Resources
        # BGM
        bgm_path = None
        if MUSIC_DIR.exists():
            music_files = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))
            if music_files: bgm_path = random.choice(music_files)

        # Font Selection - "random" picks from VIRAL_FONTS list
        selected_font = request.font
        if request.font.lower() == "random":
            selected_font = random.choice(VIRAL_FONTS)
            print(f"DEBUG: Random font selected: {selected_font}")

        # 5. Karaoke ASS Generation with FRESH timestamps (offset=0 since clip starts at 0)
        report_status(request.status_webhook_url, f"Processing: Karaoke ({selected_font})")
        generate_karaoke_ass(fresh_segments, ass_path, 0.0, selected_font, fresh_trigger_words)

        # 3. MEGA-FILTER CHAIN (V5: CUDA NATIVE)
        # Architecture: 
        # [Input Texture(CUDA)] -> scale_cuda(Pulse) -> eq_cuda(Grade) -> [Texture] 
        # -> hwdownload -> [Frame(NV12)] -> subtitles(CPU) -> hwupload_cuda -> [Texture] -> NVENC
        
        # Pulse Logic
        # 1. Base Zoom: Slow growth over time (min(1+0.001*t, 1.5))
        # 2. Beat Pulse: Sync to BGM beats (detected via scipy)
        # 3. Trigger Kicks: SNAP ZOOM (quick punch-in) on harsh words

        trigger_sum = ""
        # Use FRESH trigger words from per-clip transcription (already clip-relative, starting at 0)
        if fresh_trigger_words:
            # We limit to 20 triggers to avoid command line overflow
            for trig in fresh_trigger_words[:20]:
                # Fresh timestamps are already relative to clip start (no adjustment needed)
                t_start = trig.get('start', 0)
                t_end = trig.get('end', t_start + 0.5)

                # SNAP ZOOM: Quick punch-in centered on the word
                # Fast attack (0.08s), short hold, quick release (0.15s)
                snap_time = (t_start + t_end) / 2
                snap_start = snap_time - 0.04  # Start just before word
                snap_peak = snap_time + 0.04   # Peak right after word starts
                snap_end = snap_time + 0.20    # Release over 0.15s

                # Skip triggers that fall outside the clip range
                if snap_end < 0 or snap_start > duration:
                    continue

                # Clamp to clip bounds
                snap_start = max(0, snap_start)
                snap_end = min(duration, snap_end)

                # SNAP ZOOM expression: quick attack, slower decay
                # Attack phase: 0 → 0.5 in 0.08s (steep ramp)
                # Decay phase: 0.5 → 0 in 0.15s (exponential-ish decay)
                # Using piecewise: attack uses steep ramp, decay uses inverted quadratic
                attack_dur = 0.08
                decay_dur = 0.15
                # 50% zoom punch on harsh words
                trigger_sum += f"+0.50*between(t,{snap_start:.3f},{snap_peak:.3f})*((t-{snap_start:.3f})/{attack_dur})"
                trigger_sum += f"+0.50*between(t,{snap_peak:.3f},{snap_end:.3f})*(1-((t-{snap_peak:.3f})/{decay_dur}))"

        # Beat-synced pulse from background music
        beat_pulse = ""
        if bgm_path:
            try:
                beat_times = detect_beats(str(bgm_path), duration)
                print(f"Detected {len(beat_times)} beats in BGM for pulse sync")
                # Add subtle pulse on each beat (10% zoom, quick decay)
                for bt in beat_times[:100]:  # Limit to avoid FFmpeg expr overflow
                    beat_pulse += f"+0.10*between(t,{bt:.3f},{bt+0.15:.3f})*(1-((t-{bt:.3f})/0.15))"
            except Exception as e:
                print(f"Beat detection failed, using fallback heartbeat: {e}")
                beat_pulse = "0.008*sin(2*3.14159*t*2.0)"  # 2Hz fallback

        # Pulse Expr (CPU Dynamic)
        zoom_base = "min(1+0.001*t,1.5)"
        # Use beat pulse if available, otherwise subtle heartbeat
        heartbeat = beat_pulse if beat_pulse else "0.005*sin(2*3.14159*t*2.0)"
        
        # Combined Scale Factor
        scale_factor = f"({zoom_base}+{heartbeat}{trigger_sum})"
        
        # We vary Width(w) and Height(h) using the expression.
        # Force even dimensions (2-aligned) for YUV420P / CUDA compatibility
        # Ensure factor >= 1 using max(1, ...) to prevent crop failure
        zoom_expr = f"w='2*trunc(iw*max(1,{scale_factor})/2)':h='2*trunc(ih*max(1,{scale_factor})/2)'"
        
        # Grade Expr (CPU)
        grade_filter = "eq=contrast=1.15:brightness=0.03:saturation=1.25"

        # Chromatic Aberration - DISABLED (rgbashift/colorbalance don't support time expressions)
        # The smooth pulse zoom effect is the main visual impact.
        # TODO: Implement via sendcmd filter with pre-computed intervals for RGB shift
        chromatic_filter = None

        # Escape path for ASS filter
        escaped_ass = str(ass_path).replace(":", "\\:").replace("'", "\\'")

        # V7.0 ROBUST CPU SCALING (Legacy Bottleneck Accepted for Stability)
        # [GPU Decode] -> hwdownload -> [CPU] -> scale(Fill) -> scale(Pulse) -> crop -> chromatic -> eq -> ass -> hwupload -> [GPU Encode]
        # logic: scale_cuda filters failed expression parsing. Reverting to CPU scaling.
        chromatic_stage = f"{chromatic_filter}," if chromatic_filter else ""
        filter_complex = (
            f"[0:v]hwdownload,format=nv12,format=yuv420p,"
            f"scale=-2:1920,"
            f"scale={zoom_expr}:eval=frame,"
            f"crop=1080:1920:(iw-1080)/2+250:(ih-1920)/2,"
            f"{chromatic_stage}"
            f"{grade_filter},"
            f"ass='{escaped_ass}',"
            f"hwupload_cuda[v];"
            f"[0:a]volume=1.0[a]"
        )

        # Apply effects to extracted clip (already extracted above for transcription)
        # temp_extracted starts at t=0, fresh transcript also starts at t=0 = perfect sync
        filter_complex_effects = (
            f"[0:v]hwdownload,format=nv12,format=yuv420p,"
            f"scale=-2:1920,"
            f"scale={zoom_expr}:eval=frame,"
            f"crop=1080:1920:(iw-1080)/2+250:(ih-1920)/2,"
            f"{grade_filter},"
            f"ass='{escaped_ass}',"
            f"hwupload_cuda[v];"
            f"[0:a]volume=1.0[a]"
        )

        cmd_mega = [
            "ffmpeg", "-y", "-threads", "0",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", str(temp_extracted),
            "-filter_complex", filter_complex_effects,
            "-map", "[v]", "-map", "[a]",
            "-c:v", encoder, *gpu_opts,
            "-c:a", "aac",
            str(temp_main)
        ]
        
        print(f"DEBUG MEGA CMD: {' '.join(cmd_mega)}")
        report_status(request.status_webhook_url, "Processing: GPU FX")
        subprocess.run(cmd_mega, check=True)

        # 3.5. Chromatic Aberration Pass (MoviePy) - RGB split effect
        # ALWAYS add 4 intro pulses at the start, plus BIG keyword trigger hits
        report_status(request.status_webhook_url, "Processing: RGB Glitch")

        trigger_windows = []

        # INTRO PULSES: Four RGB glitch hits at the very start
        # Creates that signature "glitchy intro" viral clip look
        trigger_windows.append((0.0, 0.3))    # First intro pulse
        trigger_windows.append((0.4, 0.7))    # Second intro pulse
        trigger_windows.append((0.9, 1.2))    # Third intro pulse
        trigger_windows.append((1.4, 1.7))    # Fourth intro pulse

        # KEYWORD TRIGGERS: Add BIG pulses for each trigger word (using fresh timestamps)
        if fresh_trigger_words:
            for trig in fresh_trigger_words[:20]:
                # Fresh timestamps are already clip-relative (no adjustment needed)
                t_start = trig.get('start', 0)
                t_end = trig.get('end', t_start + 0.5)
                t_center = (t_start + t_end) / 2
                # Wider pulse window for more impact (0.4s total instead of 0.5s)
                pulse_start = max(0, t_center - 0.2)
                pulse_end = min(duration, t_center + 0.2)
                if pulse_end > 0 and pulse_start < duration:
                    # Avoid overlap with intro pulses (now 4 pulses until ~1.7s)
                    if pulse_start >= 2.0:
                        trigger_windows.append((pulse_start, pulse_end))

        temp_chroma = temp_dir / f"temp_chroma_{uid}.mp4"
        cleanup_files.append(temp_chroma)
        try:
            # max_shift=35 for BIGGER RGB glitch effect
            apply_chromatic_aberration(str(temp_main), str(temp_chroma), trigger_windows, max_shift=35)
            # Replace temp_main with chromatic version
            subprocess.run(["mv", str(temp_chroma), str(temp_main)], check=True)
            print(f"Applied {len(trigger_windows)} RGB glitch pulses (4 intro + {len(trigger_windows)-4} keywords)")
        except Exception as e:
            print(f"WARNING: Chromatic aberration failed, continuing without: {e}")
            # Continue with unmodified temp_main

        # 4. Outro Generation (V4 CPU-Threaded for safety)
        final_video_path = str(temp_main)
        total_dur = get_video_info(str(temp_main))["duration"]
        
        if total_dur > 5:
            outro_start = total_dur - 2
            cmd_outro_extract = [
                "ffmpeg", "-y", "-threads", "0", 
                "-ss", str(outro_start), "-i", str(temp_main), "-t", "2",
                "-c:v", "copy", "-c:a", "copy", str(temp_outro)
            ]
            subprocess.run(cmd_outro_extract, check=True)

            # High-Speed Grid 
            grid_filter = (
                "split=9[a][b][c][d][e][f][g][h][i];"
                "[a]scale=iw*1.0:ih*1.0[a1];"
                "[b]scale=iw*1.05:ih*1.05,crop=1080:1920[b1];"
                "[c]scale=iw*1.1:ih*1.1,crop=1080:1920[c1];"
                "[d]scale=iw*1.05:ih*1.05,crop=1080:1920[d1];"
                "[e]scale=iw*1.0:ih*1.0[e1];"
                "[f]scale=iw*1.05:ih*1.05,crop=1080:1920[f1];"
                "[g]scale=iw*1.1:ih*1.1,crop=1080:1920[g1];"
                "[h]scale=iw*1.05:ih*1.05,crop=1080:1920[h1];"
                "[i]scale=iw*1.0:ih*1.0[i1];"
                "[a1][b1][c1]hstack=3[row1];[d1][e1][f1]hstack=3[row2];[g1][h1][i1]hstack=3[row3];"
                "[row1][row2][row3]vstack=3,scale=1080:1920"
            )
            
            handle = request.channel_handle or "TRAD_WEST"
            
            cmd_outro_process = [
                "ffmpeg", "-y", "-threads", "0",
                "-i", str(temp_outro), "-i", str(cross_path),
                "-filter_complex", f"{grid_filter}[grid];[grid][1:v]overlay=(W-w)/2:(H-h)/2[bg];[bg]drawtext=text='@{handle}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=70:fontcolor=white:borderw=3:bordercolor=black:x=(w-text_w)/2:y=1200,fade=t=out:st=1:d=1",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy",
                str(temp_grid)
            ]
            subprocess.run(cmd_outro_process, check=True)

            # 5. Concat (RAM Disk)
            report_status(request.status_webhook_url, "Processing: Merging")
            temp_main_trim = temp_dir / f"temp_trim_{uid}.mp4"
            subprocess.run(["ffmpeg", "-y", "-threads", "0", "-i", str(temp_main), "-t", str(total_dur - 2), "-c", "copy", str(temp_main_trim)], check=True)
            cleanup_files.append(temp_main_trim)
            
            with open(temp_concat_list, "w") as f:
                f.write(f"file '{temp_main_trim}'\nfile '{temp_grid}'\n")
            
            subprocess.run(["ffmpeg", "-y", "-threads", "0", "-f", "concat", "-safe", "0", "-i", str(temp_concat_list), "-c", "copy", str(output_path)], check=True)

        else:
            subprocess.run(["cp", str(temp_main), str(output_path)], check=True)

        # 6. BGM Mix (Final Pass)
        if bgm_path:
            report_status(request.status_webhook_url, "Processing: Audio Mix")
            # Vol Ramp Expr (Applied to MUSIC now)
            # Ramp from 0.45 to 0.95 starting 5s from end (+15% boost)
            ramp_start = max(0, duration - 5)
            vol_expr = f"'if(gte(t,{ramp_start}),0.45+( (t-{ramp_start}) / 5 )*0.5,0.45)'"

            final_mix = temp_dir / f"final_mix_{uid}.mp4"
            cmd_mix = [
                "ffmpeg", "-y", "-threads", "0",
                "-i", str(output_path), 
                "-i", str(bgm_path), 
                "-filter_complex", 
                f"[1:a]volume=eval=frame:volume={vol_expr},aloop=loop=-1:size=2147483647[bgm];[0:a][bgm]amix=inputs=2:duration=first",
                "-map", "0:v", "-c:v", "copy", "-c:a", "aac", str(final_mix)
            ]
            subprocess.run(cmd_mix, check=True)
            subprocess.run(["cp", str(final_mix), str(output_path)], check=True)
            cleanup_files.append(final_mix)

        info = get_video_info(str(output_path))
        return {"success": True, "path": str(output_path), "filename": output_filename, "duration": info["duration"]}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Render failed: {str(e)}")
    finally:
        for f in cleanup_files:
            if f.exists():
                try: f.unlink()
                except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
