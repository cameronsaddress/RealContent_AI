"""
Video Processing Service for n8n
- Downloads videos from TikTok, Instagram, YouTube using yt-dlp
- Processes videos with FFmpeg (GPU-accelerated on DGX Spark)
- Combines avatar videos with background footage
- Burns captions/subtitles
- Combines avatar videos with background footage
"""

import os
# Force MoviePy to use system FFmpeg (which has NVENC support)
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import subprocess
import sys
import uuid

# Monkey patch for Pillow 10+ (removes ANTIALIAS) used by moviepy 1.0.3
import PIL.Image
# MoviePy v2.0 imports
# MoviePy v2.0 imports
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, ImageClip, concatenate_videoclips, concatenate_audioclips, CompositeAudioClip
# Import effects individually for v2.0 compatibility
# Note: Colorx -> MultiplyColor, Vignette missing (implemented manually)
from moviepy.video.fx import Resize, MultiplyColor, LumContrast, FadeOut
from moviepy.audio.fx import MultiplyVolume, AudioLoop, AudioFadeOut

# Manual Vignette Implementation for MoviePy v2
def vignette_effect(clip, intensity=0.6):
    import numpy as np
    def filter(get_frame, t):
        img = get_frame(t)
        h, w = img.shape[:2]
        
        # Create radial gradient mask
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        # Normalize distance: 0 at center, 1 at corners
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        norm_dist = dist_from_center / max_dist
        
        # Vignette mask: 1 at center, darkening towards edges
        # intensity controls how dark the edges get
        mask = 1 - (intensity * norm_dist)
        mask = np.clip(mask, 0, 1)
        
        # Apply mask to all channels
        return (img * mask[..., np.newaxis]).astype(np.uint8)
        
    return clip.transform(filter)

# Fallback map for code using vfx.func
class VFX:
    resize = Resize
    colorx = MultiplyColor
    lum_contrast = LumContrast
    vignette = vignette_effect
    fadeout = FadeOut
vfx = VFX()

class AFX:
    volumex = MultiplyVolume
    audio_loop = AudioLoop
    # audio_fadeout = AudioFadeOut
afx = AFX()

import numpy as np
from scipy.signal import find_peaks
from PIL import Image, ImageDraw, ImageFont
import os
import subprocess
import uuid
import json
import asyncio
from pathlib import Path
from typing import Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Lazy import helpers
whisper_model = None

def get_whisper_model():
    global whisper_model
    import whisper
    import torch
    if whisper_model is None:
        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model("small", device=device)
    return whisper_model

app = FastAPI(
    title="Video Processing Service",
    description="GPU-accelerated video download and processing for n8n pipeline",
    version="2.0.0"
)

# Directories (shared with n8n via volumes)
DOWNLOAD_DIR = Path("/downloads")
OUTPUT_DIR = Path("/outputs")
DOWNLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


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
    """Request to compose final video with avatar overlay"""
    script_id: str
    avatar_path: str  # Path to avatar video (green screen)
    background_path: str  # Path to background/source video
    audio_path: Optional[str] = None  # Optional separate audio track
    music_path: Optional[str] = None  # Optional background music
    output_filename: Optional[str] = None
    use_gpu: bool = True
    # Avatar positioning settings
    avatar_scale: float = 0.75  # Scale factor for avatar
    avatar_offset_x: int = -250  # Horizontal offset (negative = left)
    avatar_offset_y: int = 600  # Vertical offset from bottom
    greenscreen_color: str = "0x00FF00"  # Chroma key color
    # Audio mixing settings
    original_volume: float = 0.7
    avatar_volume: float = 1.0
    music_volume: float = 0.3


class CaptionRequest(BaseModel):
    """Request to burn captions onto video"""
    script_id: str
    video_path: str
    srt_path: Optional[str] = None  # Path to SRT subtitle file
    ass_path: Optional[str] = None  # Path to ASS subtitle file (for karaoke)
    output_filename: Optional[str] = None
    font_size: int = 48
    font_color: str = "white"
    outline_color: str = "black"
    use_gpu: bool = True


class TranscribeRequest(BaseModel):
    """Request to transcribe audio"""
    audio_path: str
    output_format: str = "srt"  # srt, vtt, json


class VideoInfoResponse(BaseModel):
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    size: int


class FetchChannelRequest(BaseModel):
    url: str
    limit: int = 10
    platform: Optional[str] = None


class ExtractClipRequest(BaseModel):
    video_path: str
    start_time: float
    end_time: float
    output_filename: Optional[str] = None
    



class RenderClipRequest(BaseModel):
    video_path: str
    start_time: float
    end_time: float
    transcript_segments: list[dict] = []
    style_preset: str = "trad_west"
    font: str = "Arial"
    output_filename: Optional[str] = None
    outro_path: Optional[str] = None
    channel_handle: Optional[str] = None
    trigger_words: Optional[List[dict]] = None


# ============ UTILITIES ============

def get_gpu_encoder() -> tuple[str, list]:
    """Check if NVIDIA GPU encoding is available"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        if "h264_nvenc" in result.stdout:
            return "h264_nvenc", ["-preset", "p4", "-tune", "hq"]
    except:
        pass
    return "libx264", ["-preset", "fast", "-crf", "23"]


def get_video_info(path: str) -> dict:
    """Get video metadata using ffprobe"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    video_stream = next(
        (s for s in data.get("streams", []) if s["codec_type"] == "video"),
        {}
    )

    return {
        "duration": float(data.get("format", {}).get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": eval(video_stream.get("r_frame_rate", "0/1")) if video_stream.get("r_frame_rate") else 0,
        "codec": video_stream.get("codec_name", "unknown"),
        "size": int(data.get("format", {}).get("size", 0))
    }


# ============ HEALTH & INFO ============

@app.get("/health")
async def health_check():
    """Health check with GPU and tool status"""
    # Check yt-dlp
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=10)
        ytdlp_version = result.stdout.strip()
    except Exception as e:
        ytdlp_version = f"error: {e}"

    # Check FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        ffmpeg_version = result.stdout.split('\n')[0]
    except Exception as e:
        ffmpeg_version = f"error: {e}"

    # Check GPU encoder
    encoder, _ = get_gpu_encoder()
    gpu_available = encoder == "h264_nvenc"

    # Check NVIDIA GPU
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10)
        gpu_info = result.stdout.strip()
    except:
        gpu_info = "not available"

    return {
        "status": "healthy",
        "yt_dlp_version": ytdlp_version,
        "ffmpeg_version": ffmpeg_version,
        "gpu_encoder_available": gpu_available,
        "gpu_info": gpu_info,
        "download_dir": str(DOWNLOAD_DIR),
        "output_dir": str(OUTPUT_DIR)
    }


# ============ DOWNLOAD ENDPOINTS ============

def detect_platform(url: str) -> str:
    """Detect platform from URL"""
    url_lower = url.lower()
    if "tiktok.com" in url_lower:
        return "tiktok"
    elif "instagram.com" in url_lower:
        return "instagram"
    elif "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    elif "twitter.com" in url_lower or "x.com" in url_lower:
        return "twitter"
    elif "reddit.com" in url_lower:
        return "reddit"
    elif "facebook.com" in url_lower or "fb.watch" in url_lower:
        return "facebook"
    elif "rumble.com" in url_lower:
        return "rumble"
    return "unknown"


def download_with_ytdlp(url: str, output_path: Path, platform: str) -> tuple[bool, str, dict]:
    """
    Try downloading with yt-dlp using platform-specific settings.
    Returns (success, error_message, metadata)
    """
    # Platform-specific options
    platform_opts = []

    if platform == "tiktok":
        # TikTok needs impersonation to avoid IP blocks
        platform_opts = [
            "--impersonate", "chrome",
            "--extractor-args", "tiktok:api_hostname=api22-normal-c-alisg.tiktokv.com"
        ]
    elif platform == "instagram":
        # Instagram also benefits from impersonation
        platform_opts = [
            "--impersonate", "chrome"
        ]
    elif platform == "youtube":
        # YouTube works but may need specific format selection
        platform_opts = [
            "--extractor-args", "youtube:player_client=web"
        ]
    elif platform == "twitter":
        # Twitter/X - try guest token first
        platform_opts = []
    elif platform == "rumble":
        # Rumble needs impersonation to bypass Cloudflare/bot protection
        platform_opts = ["--impersonate", "safari"]

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", str(output_path),
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--no-warnings",
        "--print-json",
        *platform_opts,
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and output_path.exists():
            # Parse metadata from stdout
            metadata = {}
            try:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('{'):
                        metadata = json.loads(line)
                        break
            except:
                pass
            return True, "", metadata

        return False, result.stderr, {}
    except subprocess.TimeoutExpired:
        return False, "Download timed out", {}
    except Exception as e:
        return False, str(e), {}


def download_with_ytdlp_fallback(url: str, output_path: Path, platform: str) -> tuple[bool, str, dict]:
    """
    Try alternative yt-dlp approaches if main method fails
    """
    # Try without impersonation but with cookies simulation
    fallback_opts = [
        ["--cookies-from-browser", "chrome"],  # Try browser cookies
        ["--user-agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15"],  # Mobile UA
        [],  # Plain attempt
    ]

    for opts in fallback_opts:
        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",
            "-o", str(output_path),
            "--no-playlist",
            "--merge-output-format", "mp4",
            *opts,
            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                return True, "", {}
        except:
            continue

    return False, "All fallback methods failed", {}


@app.post("/download", response_model=DownloadResponse)
async def download_video(request: DownloadRequest):
    """
    Download a video from TikTok, Instagram, YouTube, Twitter, etc.
    Uses platform-specific strategies and fallbacks.
    """
    base_filename = request.filename or str(uuid.uuid4())[:8]
    output_path = DOWNLOAD_DIR / f"{base_filename}.{request.format}"

    # Remove existing file if present
    if output_path.exists():
        output_path.unlink()

    platform = detect_platform(request.url)

    # Try primary download method
    success, error, metadata = download_with_ytdlp(request.url, output_path, platform)

    # Try fallback if primary failed
    if not success or not output_path.exists():
        success, error, metadata = download_with_ytdlp_fallback(request.url, output_path, platform)

    # Check for file
    if not output_path.exists():
        # Check for files with different extensions
        possible_files = list(DOWNLOAD_DIR.glob(f"{base_filename}.*"))
        if possible_files:
            output_path = possible_files[0]
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Download failed for {platform}: {error}. The platform may be blocking server IPs."
            )

    file_size = output_path.stat().st_size
    if file_size < 1000:
        raise HTTPException(status_code=500, detail="Downloaded file is too small, likely failed")

    return DownloadResponse(
        success=True,
        filename=output_path.name,
        path=str(output_path),
        size=file_size,
        duration=metadata.get("duration"),
        title=metadata.get("title"),
        platform=platform
    )


# ============ VIDEO PROCESSING ENDPOINTS ============

@app.post("/compose")
async def compose_video(request: ComposeRequest):
    """
    Compose final video: background + avatar overlay + optional music

    This creates TikTok-style video with:
    - Source video as background (scaled to 9:16, looped if shorter than avatar)
    - Avatar with chroma key removal overlaid with configurable position
    - Optional background music mixed with avatar audio
    """
    output_filename = request.output_filename or f"{request.script_id}_combined.mp4"
    output_path = OUTPUT_DIR / output_filename

    # Get avatar duration to set output duration limit
    avatar_info = get_video_info(request.avatar_path)
    avatar_duration = avatar_info["duration"]

    # Get encoder settings
    encoder, encoder_opts = get_gpu_encoder() if request.use_gpu else ("libx264", ["-preset", "fast", "-crf", "23"])

    # Build FFmpeg filter complex
    filter_parts = []
    # Loop background video to match avatar duration (avatar is usually longer than source clip)
    inputs = ["-stream_loop", "-1", "-i", request.background_path, "-i", request.avatar_path]
    input_count = 2
    audio_index = 1  # Avatar audio by default

    # Add separate audio track if provided
    if request.audio_path and Path(request.audio_path).exists():
        inputs.extend(["-i", request.audio_path])
        audio_index = input_count
        input_count += 1

    # Add music if provided
    music_index = -1
    if request.music_path and Path(request.music_path).exists():
        inputs.extend(["-stream_loop", "-1", "-i", request.music_path])
        music_index = input_count
        input_count += 1

    # Scale and crop background to 9:16 (1080x1920)
    filter_parts.append("[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[bg]")

    # Chroma key and scale the avatar
    filter_parts.append(
        f"[1:v]chromakey={request.greenscreen_color}:0.1:0.2,"
        f"scale=iw*{request.avatar_scale}:ih*{request.avatar_scale}[avatar_keyed]"
    )

    # Calculate overlay position (bottom-left with offsets)
    # X: 10 + offset_x (negative offset moves left)
    # Y: (H-h-10) + offset_y (positive offset moves down from bottom)
    x_pos = f"(10+{request.avatar_offset_x})"
    y_pos = f"((H-h-10)+{request.avatar_offset_y})"
    # Use eof_action=repeat to keep showing last frame if bg runs out, but with loop it won't
    # Remove shortest=1 so video runs for full avatar duration
    filter_parts.append(f"[bg][avatar_keyed]overlay={x_pos}:{y_pos}[outv]")

    # Audio mixing
    if music_index >= 0:
        # Mix avatar/voice audio with background music
        filter_parts.append(f"[{audio_index}:a]volume={request.avatar_volume}[voice]")
        filter_parts.append(f"[{music_index}:a]volume={request.music_volume}[music_low]")
        filter_parts.append("[voice][music_low]amix=inputs=2:duration=first[outa]")
    else:
        filter_parts.append(f"[{audio_index}:a]volume={request.avatar_volume},aformat=channel_layouts=stereo[outa]")

    filter_complex = ";".join(filter_parts)

    # Build full command with duration limit to match avatar length
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-t", str(avatar_duration),  # Limit output to avatar duration
        "-c:v", encoder,
        *encoder_opts,
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg compose failed: {result.stderr[-1000:]}")

        info = get_video_info(str(output_path))

        return {
            "success": True,
            "output_path": str(output_path),
            "filename": output_filename,
            "size": info["size"],
            "duration": info["duration"],
            "encoder_used": encoder
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Video composition timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Composition failed: {str(e)}")


@app.post("/caption")
async def burn_captions(request: CaptionRequest):
    """
    Burn SRT or ASS captions onto video (TikTok-style text overlay)
    ASS files are used for karaoke-style word-by-word highlighting.
    """
    output_filename = request.output_filename or f"{request.script_id}_final.mp4"
    output_path = OUTPUT_DIR / output_filename

    encoder, encoder_opts = get_gpu_encoder() if request.use_gpu else ("libx264", ["-preset", "fast", "-crf", "23"])

    # Use ASS if provided (for karaoke), otherwise SRT
    if request.ass_path and Path(request.ass_path).exists():
        # ASS has all styling built-in
        subtitle_filter = f"ass={request.ass_path}"
    elif request.srt_path:
        # SRT with inline styling
        subtitle_filter = (
            f"subtitles={request.srt_path}:force_style='"
            f"FontSize={request.font_size},"
            f"PrimaryColour=&H00FFFFFF,"  # White
            f"OutlineColour=&H00000000,"  # Black outline
            f"Outline=2,"
            f"Shadow=1,"
            f"Alignment=2,"  # Bottom center
            f"MarginV=80'"  # Margin from bottom
        )
    else:
        raise HTTPException(status_code=400, detail="Either srt_path or ass_path must be provided")

    cmd = [
        "ffmpeg", "-y",
        "-i", request.video_path,
        "-vf", subtitle_filter,
        "-c:v", encoder,
        *encoder_opts,
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg caption failed: {result.stderr[-1000:]}")

        info = get_video_info(str(output_path))

        return {
            "success": True,
            "output_path": str(output_path),
            "filename": output_filename,
            "size": info["size"],
            "duration": info["duration"]
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Caption burning timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Caption burning failed: {str(e)}")


@app.get("/info/{filename}")
async def get_file_info(filename: str):
    """Get video/audio file information"""
    # Check both directories
    for directory in [DOWNLOAD_DIR, OUTPUT_DIR]:
        file_path = directory / filename
        if file_path.exists():
            info = get_video_info(str(file_path))
            return {
                "filename": filename,
                "path": str(file_path),
                **info
            }

    raise HTTPException(status_code=404, detail="File not found")


@app.get("/file/{filename}")
async def get_file(filename: str):
    """Retrieve a file by filename"""
    for directory in [DOWNLOAD_DIR, OUTPUT_DIR]:
        file_path = directory / filename
        if file_path.exists():
            return FileResponse(path=file_path, filename=filename)

    raise HTTPException(status_code=404, detail="File not found")


@app.delete("/file/{filename}")
async def delete_file(filename: str):
    """Delete a file"""
    for directory in [DOWNLOAD_DIR, OUTPUT_DIR]:
        file_path = directory / filename
        if file_path.exists():
            file_path.unlink()
            return {"success": True, "deleted": filename}

    raise HTTPException(status_code=404, detail="File not found")


@app.get("/list")
async def list_files():
    """List all files in download and output directories"""
    files = {"downloads": [], "outputs": []}

    for f in DOWNLOAD_DIR.iterdir():
        if f.is_file():
            files["downloads"].append({
                "filename": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            })

    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            files["outputs"].append({
                "filename": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            })

    return files


@app.post("/fetch-channel")
async def fetch_channel_videos(request: FetchChannelRequest):
    """
    Fetch recent videos from a channel (Youtube/Rumble) using yt-dlp flat playlist
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--playlist-end", str(request.limit),
        "--no-warnings",
        request.url
    ]
    
    try:
        # Check platform settings
        if request.platform == "tiktok": # Use browser impersonation for scraping if needed
             cmd.extend(["--impersonate", "chrome"])
        elif request.platform == "rumble":
             cmd.extend(["--impersonate", "safari"])

        print(f"Executing CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(f"CMD Return Code: {result.returncode}")
        print(f"Stderr: {result.stderr}")
        print(f"Stdout (first 500 chars): {result.stdout[:500]}")
        
        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    # Handle case where it returns a single playlist object with entries
                    if data.get("_type") == "playlist" and "entries" in data:
                        print("Detected playlist object, iterating entries...")
                        for entry in data["entries"]:
                            if entry:
                                videos.append({
                                    "id": entry.get("id"),
                                    "title": entry.get("title", "Untitled"),
                                    "url": entry.get("url") or entry.get("webpage_url"),
                                    "thumbnail": entry.get("thumbnail"),
                                    "duration": entry.get("duration"),
                                    "upload_date": entry.get("upload_date"),
                                    "view_count": entry.get("view_count")
                                })
                    else:
                        # Standard per-line video object
                        videos.append({
                            "id": data.get("id"),
                            "title": data.get("title", "Untitled"),
                            "url": data.get("url") or data.get("webpage_url"),
                            "thumbnail": data.get("thumbnail"),
                            "duration": data.get("duration"),
                            "upload_date": data.get("upload_date"),
                            "view_count": data.get("view_count")
                        })
                except json.JSONDecodeError:
                    continue
                    
        return {"success": True, "videos": videos}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")


@app.post("/transcribe-whisper")
async def transcribe_whisper(request: TranscribeRequest):
    """
    Transcribe audio/video using OpenAI Whisper (GPU) with word-level timestamps
    """
    try:
        model = get_whisper_model()
        
        if not os.path.exists(request.audio_path):
             raise HTTPException(status_code=404, detail=f"File not found: {request.audio_path}")

        # Run transcription
        result = model.transcribe(request.audio_path, word_timestamps=True)
        
        if request.output_format == "json":
            return result # Returns full detail including segments and words
        
        return {"text": result["text"], "segments": result["segments"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/extract-clip")
async def extract_clip(request: ExtractClipRequest):
    """
    Extract a sub-clip from a video without re-encoding (copy) if possible, 
    or precise cut with re-encoding.
    """
    output_filename = request.output_filename or f"clip_{uuid.uuid4().hex[:8]}.mp4"
    output_path = OUTPUT_DIR / output_filename
    
    # Use re-encoding for frame-accurate cuts (copy codec is usually keyframe inaccurate)
    # Using specific viral editing settings (fast but decent quality)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(request.start_time),
        "-i", request.video_path,
        "-t", str(request.end_time - request.start_time),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise Exception(result.stderr[-1000:])
            
        return {
            "success": True,
            "output_path": str(output_path),
            "filename": output_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clip extraction failed: {str(e)}")


def generate_karaoke_ass(segments: list[dict], output_path: Path, start_offset: float = 0.0, font: str = "Arial"):
    """
    Generate ASS subtitle file with Karaoke highlighting.
    Highlights words as they are spoken.
    """
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke, {font}, 80, &H00FFFF00, &H00FFFFFF, &H00000000, &H80000000, -1, 0, 0, 0, 100, 100, 0, 0, 1, 3, 0, 2, 50, 50, 500, 1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    events = []
    
    def fmt_time(t):
        t = max(0, t - start_offset)
        m, s = divmod(t, 60)
        h, m = divmod(m, 60)
        cs = int((s - int(s)) * 100)
        return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"

    for seg in segments:
        s_start = seg['start']
        s_end = seg['end']
        
        # If no word timestamps, treat whole segment as one block
        words = seg.get('words', [])
        if not words:
            # Fallback: Just show text
            events.append(f"Dialogue: 0,{fmt_time(s_start)},{fmt_time(s_end)},Karaoke,,0,0,0,,{seg['text'].strip()}")
            continue
            
        # Build karaoke string: {\k10}Word {\k20}
        karaoke_line = ""
        last_end = s_start
        
        for w in words:
            w_start = w['start']
            w_end = w['end']
            w_text = w['word'].strip()
            
            # Duration in centiseconds
            duration = int((w_end - w_start) * 100)
            
            # Gap handling (if gap > 0, append space or wait?)
            # Usually just append duration.
            # \K matches 'secondary color' logic (fill from left) or \kf 
            # Standard \k highlights immediately? 
            # Let's use \k which is standard karaoke (Fill). 
            # To have "future" words white and "current/past" yellow:
            # We set SecondaryColour (Unplayed) = White, PrimaryColour (Played) = Yellow.
            # \k fills the text.
            
            # Add space if needed
            pre_gap = w_start - last_end
            if pre_gap > 0.5: # If long silence, maybe split line? For now just keep.
                pass
                
            karaoke_line += f"{{\\k{duration}}}{w_text} "
            last_end = w_end
            
        events.append(f"Dialogue: 0,{fmt_time(s_start)},{fmt_time(s_end)},Karaoke,,0,0,0,,{karaoke_line}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))


def create_cross_image(size=(100, 100), color=(128, 0, 128)):  # Purple
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
         # Try to find a font, or use default
         font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
    except:
         try:
            font = ImageFont.truetype("arial.ttf", 80)
         except:
            font = ImageFont.load_default()

    # Draw centered cross
    # Need text size to center
    try:
        w, h = draw.textsize("†", font=font)
    except:
        w, h = 40, 60 # approx
        
    draw.text(((size[0]-w)/2, (size[1]-h)/2), "†", font=font, fill=color)
    
    img_path = OUTPUT_DIR / "temp_cross.png"
    img.save(img_path)
    return str(img_path)

def apply_warmth(clip, factor=1.2):
    def warm(get_frame, t):
        frame = get_frame(t)
        frame = frame.astype(float)
        frame[:, :, 0] = np.clip(frame[:, :, 0] * factor, 0, 255)  # Boost red
        frame[:, :, 2] = np.clip(frame[:, :, 2] * 0.8, 0, 255)   # Reduce blue
        return frame.astype(np.uint8)
    return clip.transform(warm)

def create_grid_clip(base_clip, grid_size=(3, 3)):
    w, h = base_clip.size
    small_w, small_h = w // grid_size[0], h // grid_size[1]
    grid_clips = []
    
    # Pre-resize to small to avoid doing it per frame in loop (slow)
    # But we need time variance for pulse. 
    # Logic: 
    # offset_clip = base_clip.resize((small_w, small_h))
    # offset_clip = offset_clip.resize(lambda t: zoom ... )
    # This might be heavy.
    
    for i in range(grid_size[0] * grid_size[1]):
        # Static resize first
        cell = base_clip.resized((small_w, small_h)) 
        
        # Position in grid
        pos = ((i % 3) * small_w, (i // 3) * small_h)
        cell = cell.with_position(pos)
        
        # Pulse/Zoom effect
        zoom_factor = 1 + (i * 0.05)
        # Note: resize(lambda) is very slow in MoviePy as it renders every frame.
        # Minimal pulse:
        cell = cell.resized(lambda t: 1.0 + 0.05 * np.sin(t * 3 + i)) 
        
        grid_clips.append(cell)
        
    return CompositeVideoClip(grid_clips, size=(w, h))

def ensure_stereo(clip):
    if clip.nchannels == 2:
        return clip
    import numpy as np
    def make_stereo(get_frame, t):
        frame = get_frame(t)
        if frame.ndim == 1:
            frame = frame[:, np.newaxis]
        if frame.shape[1] == 1:
            return np.hstack([frame, frame])
        return frame
    new_clip = clip.transform(make_stereo)
    new_clip.nchannels = 2
    return new_clip

@app.post("/render-viral-clip")
async def render_viral_clip(request: RenderClipRequest):
    output_filename = request.output_filename or f"viral_{uuid.uuid4().hex[:8]}.mp4"
    output_path = OUTPUT_DIR / output_filename
    temp_path = output_path.with_name(f"temp_{output_filename}")
    ass_path = output_path.with_suffix(".ass")
    cross_path = None

    try:
        temp_audio_clean = output_path.with_name(f"temp_clean_{uuid.uuid4().hex[:8]}.wav")

        # 1. Load and Crop
        # Force 44100Hz Audio extraction via FFmpeg CLI to guarantee sample rate match
        # MoviePy's fps argument isn't robust enough for some containers
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", str(request.video_path), 
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", 
            str(temp_audio_clean)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        full_clip_video = VideoFileClip(request.video_path).subclipped(request.start_time, request.end_time)
        full_clip_audio = AudioFileClip(str(temp_audio_clean)).subclipped(request.start_time, request.end_time)
        full_clip = full_clip_video.with_audio(full_clip_audio)
        
        # Vertical 9:16 Crop
        if full_clip.h != 1920: full_clip = full_clip.resized(height=1920)
        if full_clip.w > 1080: full_clip = full_clip.cropped(x_center=full_clip.w/2, width=1080)
        full_clip = full_clip.resized(height=1920).cropped(x_center=full_clip.w/2, width=1080, height=1920)

        # Trad West Color Grade (Global)
        full_clip = full_clip.with_effects([vfx.colorx(1.2), vfx.lum_contrast(0, 0.2)])
        
        
        # Split into Main and Outro segments
        # If clip is short (< 5s), maybe don't do full outro?
        total_dur = full_clip.duration
        outro_dur = 2.0
        
        if total_dur > 5.0:
            main_part = full_clip.subclipped(0, total_dur - outro_dur)
            outro_base = full_clip.subclipped(total_dur - outro_dur, total_dur)
            
            # --- Main Part Effects ---
            # === APPLY PULSE EFFECT (TradWest) ===
            # Select random Backround Music if any exist
            bg_music = None
            MUSIC_DIR = Path("/music")
            if MUSIC_DIR.exists():
                music_files = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))
                if music_files:
                     import random
                     chosen_track = random.choice(music_files)
                     print(f"Rendering with music: {chosen_track.name}")
                     
                     # Clean Room Audio: Convert BGM to 44100Hz WAV
                     temp_bg_wav = output_path.with_name(f"temp_bg_{uuid.uuid4().hex[:8]}.wav")
                     subprocess.run([
                        "ffmpeg", "-y", "-i", str(chosen_track),
                        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                        str(temp_bg_wav)
                     ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                     
                     bg_music = AudioFileClip(str(temp_bg_wav))
                     
                     # Lower BGM volume (30%)
                     bg_music = bg_music.with_effects([afx.volumex(0.3)])
            
            # Use BGM for pulse ref? Or original audio? 
            # TradWest usually pulses to the MUSIC beat.
            # So pass BGM to the pulse function!
            audio_for_pulse = bg_music if bg_music else main_part.audio
            
            # Apply Pulse + Volume Ramp (returns video with audio_for_pulse set as AUDIO)
            main_part = add_pulse_and_volume_ramp(main_part, audio_for_pulse, request.trigger_words, output_path)
            
            # If we used BGM, we need to Mix it with Speech
            if bg_music:
                 # main_part now has BGM as its audio (from pulse function)
                 # We need to overlay original speech audio
                 # Get original audio for the main_duration
                 # Get original audio for the main_duration
                 # Get original audio for the main_duration
                 original_speech = full_clip.subclipped(0, main_part.duration).audio
                 # Composite: [BGM (already rammed/pulsed in main_part.audio), Speech]
                 # Composite: [BGM (already rammed/pulsed in main_part.audio), Speech]
                 # Composite: [BGM (already rammed/pulsed in main_part.audio), Speech]
                 # Wait, main_part.audio IS the rammed/pulsed music.
                 mixed_audio = CompositeAudioClip([main_part.audio, original_speech])
                 main_part = main_part.with_audio(mixed_audio)
            
            # --- Outro Effects (TradWest) ---
            # 1. Grid Pulse
            grid_outro = create_grid_clip(outro_base)
            
            # 2. Warmth & Vignette
            enhanced_outro = apply_warmth(grid_outro)
            # Use manual vignette wrapper via vfx.vignette
            enhanced_outro = vignette_effect(enhanced_outro, intensity=0.6)
            
            # 3. Overlays
            # Cross
            cross_file = create_cross_image()
            cross_path = Path(cross_file)
            cross_clip = ImageClip(cross_file).with_duration(outro_dur).with_position(("center", "bottom")).with_opacity(0.8)
            
            # Handle Text
            handle_text = f"@{request.channel_handle}" if request.channel_handle else "@TRAD_WEST_"
            txt_clip = TextClip(text=handle_text, font_size=70, color='white', font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', stroke_color='black', stroke_width=3)
            txt_clip = txt_clip.with_duration(outro_dur).with_position(("center", "center"))
            
            # Fadeout
            final_outro = CompositeVideoClip([enhanced_outro, txt_clip, cross_clip]).with_duration(outro_dur).with_effects([vfx.fadeout(1.0)])
            
            # Concatenate
            final_clip = concatenate_videoclips([main_part, final_outro])
        else:
            # Too short for fancy outro, just run main effects
            final_clip = full_clip.resized(lambda t: 1 + 0.03 * (t / max(0.1, full_clip.duration)))
            
        # Write Temp
        encoder, encoder_opts = get_gpu_encoder()
        print(f"Rendering {output_filename} using encoder: {encoder}")
        final_clip.write_videofile(
            str(temp_path), 
            codec=encoder, 
            audio_codec="aac", 
            fps=30, 
            ffmpeg_params=encoder_opts,
            threads=4
            # preset defaults to 'medium', which gets overridden by appropriate flags in ffmpeg_params if needed
        )
        
        # 2. Burn Karaoke Captions (FFmpeg ASS)
        try:
             generate_karaoke_ass(request.transcript_segments, ass_path, start_offset=request.start_time, font=request.font)
        except Exception as e:
             print(f"Error creating karaoke subs: {e}")

        # Burn ASS
        # Escape path
        ass_path_esc = str(ass_path).replace("\\", "/").replace(":", "\\:")
        
        encoder, encoder_opts = get_gpu_encoder()
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_path),
            "-vf", f"ass='{ass_path_esc}'",
            "-c:v", encoder, *encoder_opts,
            "-c:a", "copy",
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True)
        
        # Cleanup
        if temp_path.exists(): temp_path.unlink()
        if ass_path.exists(): ass_path.unlink()
        if temp_path.exists(): temp_path.unlink()
        if ass_path.exists(): ass_path.unlink()
        if 'temp_audio_clean' in locals() and temp_audio_clean.exists(): temp_audio_clean.unlink()
        if 'temp_bg_wav' in locals() and temp_bg_wav.exists(): temp_bg_wav.unlink()
        if cross_path and cross_path.exists(): cross_path.unlink()
        
        full_clip.close()
        final_clip.close()

        info = get_video_info(str(output_path))
        return {
            "success": True,
            "path": str(output_path),
            "filename": output_filename,
            "duration": info["duration"]
        }

    except Exception as e:
        if temp_path.exists(): temp_path.unlink()
        if ass_path.exists(): ass_path.unlink()
        if temp_path.exists(): temp_path.unlink()
        if ass_path.exists(): ass_path.unlink()
        if 'temp_audio_clean' in locals() and temp_audio_clean.exists(): temp_audio_clean.unlink()
        if 'temp_bg_wav' in locals() and temp_bg_wav.exists(): temp_bg_wav.unlink()
        if cross_path and Path(cross_path).exists(): Path(cross_path).unlink()
        
        # Print detailed error for debugging
        # clip_id is directly on request object if defined, else use request.video_path as identifier
        retry_id = getattr(request, 'clip_id', 'unknown')
        print(f"Error rendering clip {retry_id}: {str(e)}")
        # raise HTTPException(status_code=500, detail=str(e))
        # Return error as JSON to avoid 500 spam on client logic if preferred, or raise
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

def add_pulse_and_volume_ramp(video_clip, audio_clip, trigger_words_timestamps, output_path):
    """
    Adds subtle pulse effect to video synced to audio beats.
    - trigger_words_timestamps: List of tuples [(start_time, end_time, word)] for exciting words.
    - Pulses start subtle on triggers, build globally (increase amplitude over time).
    - Volume ramps up in last 3s before outro.
    """
    total_dur = video_clip.duration
    outro_start = max(0, total_dur - 2)
    ramp_start = max(0, outro_start - 3)

    # Safety: ensure audio matches video duration
    if audio_clip.duration < total_dur:
        audio_clip = audio_clip.with_effects([afx.audio_loop(duration=total_dur)])
    else:
        # If longer, take random slice or just start? Just start for now.
        audio_clip = audio_clip.subclipped(0, total_dur)

    # Step 1: Beat Detection (using SciPy envelope peaks)
    try:
        # Get audio array (mono, sample rate)
        # fps=44100 as per user request
        audio_array = audio_clip.to_soundarray(fps=44100, nbytes=2, quantize=False)
        if len(audio_array.shape) > 1:
            audio_array = audio_array[:, 0]
            
        # Envelope: Absolute value + low-pass filter (simple moving average)
        envelope = np.abs(audio_array)
        window_size = 2205  # ~0.05s at 44.1kHz for beat sensitivity
        envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Find peaks (beats) - adjust prominence for sensitivity (lower for more pulses)
        peaks, _ = find_peaks(envelope, prominence=0.1 * np.max(envelope), distance=44100/4)  # Min 0.25s between beats
        beat_times = peaks / 44100  # Convert indices to seconds
    except Exception as e:
        print(f"Beat detection failed: {e}")
        beat_times = np.arange(0, total_dur, 0.5)

    # Filter beats to start only on trigger words (subtle at first)
    # trigger_words_timestamps structure: [{'start': s, 'end': e, 'word': w}] (dict from Pydantic)
    # User sample used structure of tuples, but request sends dicts. Access accordingly.
    
    trigger_ranges = []
    if trigger_words_timestamps:
        for t in trigger_words_timestamps:
            if isinstance(t, dict):
                 trigger_ranges.append((t.get('start',0), t.get('end',0)))
            elif isinstance(t, (list, tuple)) and len(t) >= 2:
                 trigger_ranges.append((t[0], t[1]))

    trigger_beats = [t for t in beat_times if any(start <= t <= end for start, end in trigger_ranges)]

    # Step 2: Use FFmpeg for Pulse Effect (GPU Acceleration)
    # Instead of Python loop, we use FFmpeg's zoompan filter.
    # zoompan is CPU-based but highly optimized C, and we wrap it with GPU decoding/encoding.
    
    # 1. Write the input video segment to a temp file (fast copy)
    temp_pulse_input = output_path.with_name(f"temp_pulse_in_{uuid.uuid4().hex[:8]}.mp4")
    temp_pulse_output = output_path.with_name(f"temp_pulse_out_{uuid.uuid4().hex[:8]}.mp4")
    
    # Check if video_clip is file-based or memory-based
    # If it's a subclip, we might need to write it first.
    # To cover all bases, we write `video_clip` to `temp_pulse_input`.
    # Using NVENC if available to speed this up.
    encoder, encoder_opts = get_gpu_encoder()
    video_clip.write_videofile(
        str(temp_pulse_input), 
        codec=encoder, 
        audio=False, # We handle audio separately
        ffmpeg_params=encoder_opts,
        threads=4,
        logger=None # Silence logs
    )
    
    # 2. Run FFmpeg Zoompan + Scale command
    # Zoom expression: 1 + 0.05 * sin(time * 3). 
    # Oscillation matching the previous Python lambda: 1.0 + 0.05 * np.sin(t * 3)
    # zoompan works on frames. d=1 means update every frame.
    # s=1080x1920 ensures output resolution.
    
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda", # GPU Decode
        "-hwaccel_output_format", "cuda", # Output CUDA frames
        "-i", str(temp_pulse_input),
        "-vf", "hwdownload,format=nv12,zoompan=z='1.05+0.05*sin(time*3)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920:fps=30,hwupload_cuda,scale_cuda=1080:1920",
        "-c:v", "h264_nvenc", # GPU Encode
        "-preset", "p4",
        "-tune", "hq",
        "-b:v", "5M",
        str(temp_pulse_output)
    ]
    
    # Note: zoompan requires software frames, so we hwdownload before it and hwupload after.
    # This pipeline: GPU Decode -> VRAM -> System RAM -> ZoomPan -> System RAM -> VRAM -> Scale -> GPU Encode
    # Much faster than Python Loop.
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Pulse failed: {e.stderr.decode()}")
        # Fallback to original input if pulse fails
        temp_pulse_output = temp_pulse_input

    # 3. Load the result back as a clip
    pulsed_video = VideoFileClip(str(temp_pulse_output))
    
    # Cleanup input temp immediately
    # On Linux we can unlink even if open, but safest to try/except
    try:
        if temp_pulse_input.exists():
            temp_pulse_input.unlink()
    except Exception as e:
        print(f"Warning: could not delete temp pulse input: {e}")
    # Note: temp_pulse_output needs to survive until final render? 
    # VideoFileClip locks the file? No, ffmpeg reader does.
    # We should add it to a cleanup list if possible, but for now we leave it.

    # Step 3: Volume Ramp-Up (linear increase in last 3s before outro)
    # Apply global volume transform to avoid concatenation artifacts
    def global_volume_ramp(get_frame, t):
        # Base audio frame
        frame = get_frame(t)
        
        # Calculate volume factor based on time t
        # If t is an array (which it is for audio), we need vectorized logic
        # But t might be absolute time or relative to clip start? 
        # t passed to transform is relative to clip start usually.
        
        # We need a volume array 'vol' corresponding to 't'
        # ramp_start and outro_start are defined relative to video start.
        # audio_clip might have different duration? 
        # Assuming audio_clip is synced or looped to video duration.
        
        if isinstance(t, np.ndarray):
            vol = np.ones_like(t, dtype=float)
            # Find indices where t > ramp_start and t < outro_start
            mask = (t > ramp_start) & (t < outro_start)
            if np.any(mask):
                # ramp_dur = outro_start - ramp_start
                # factor = 1 + ((t - ramp_start) / ramp_dur) * 0.5
                ramp_dur_val = outro_start - ramp_start
                if ramp_dur_val > 0:
                    vol[mask] = 1.0 + ((t[mask] - ramp_start) / ramp_dur_val) * 0.5
            
            # Reshape vol for broadcasting against stereo/mono channels
            vol = vol[:, np.newaxis]
        else:
            # Singleton float logic
            if ramp_start < t < outro_start:
                vol = 1.0 + ((t - ramp_start) / (outro_start - ramp_start)) * 0.5
            else:
                vol = 1.0
        
        return frame * vol

    # Apply transform to the whole audio clip (no splitting/concat)
    final_audio = audio_clip.transform(global_volume_ramp)
    
    # Return video with new audio
    return pulsed_video.with_audio(final_audio)
