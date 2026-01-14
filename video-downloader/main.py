"""
Video Processing Service for n8n
- Downloads videos from TikTok, Instagram, YouTube using yt-dlp
- Processes videos with FFmpeg (GPU-accelerated on DGX Spark)
- Combines avatar videos with background footage
- Burns captions/subtitles
"""

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
