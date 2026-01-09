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
    music_path: Optional[str] = None  # Optional background music
    output_filename: Optional[str] = None
    use_gpu: bool = True


class CaptionRequest(BaseModel):
    """Request to burn captions onto video"""
    script_id: str
    video_path: str
    srt_path: str  # Path to SRT subtitle file
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

@app.post("/download", response_model=DownloadResponse)
async def download_video(request: DownloadRequest):
    """
    Download a video from TikTok, Instagram, or YouTube using yt-dlp
    """
    base_filename = request.filename or str(uuid.uuid4())[:8]
    output_path = DOWNLOAD_DIR / f"{base_filename}.{request.format}"

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", str(output_path),
        "--no-playlist",
        "--merge-output-format", request.format,
        "--no-warnings",
        "--print-json",
        request.url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"yt-dlp failed: {result.stderr}")

        # Find the downloaded file
        if not output_path.exists():
            possible_files = list(DOWNLOAD_DIR.glob(f"{base_filename}.*"))
            if possible_files:
                output_path = possible_files[0]
            else:
                raise HTTPException(status_code=500, detail="Download completed but file not found")

        file_size = output_path.stat().st_size

        # Parse metadata
        metadata = {}
        try:
            for line in result.stdout.strip().split('\n'):
                if line.startswith('{'):
                    metadata = json.loads(line)
                    break
        except:
            pass

        return DownloadResponse(
            success=True,
            filename=output_path.name,
            path=str(output_path),
            size=file_size,
            duration=metadata.get("duration"),
            title=metadata.get("title"),
            platform=metadata.get("extractor_key")
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Download timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


# ============ VIDEO PROCESSING ENDPOINTS ============

@app.post("/compose")
async def compose_video(request: ComposeRequest):
    """
    Compose final video: background + avatar overlay + optional music

    This creates TikTok-style video with:
    - Source video as background (scaled to 9:16)
    - Avatar with chroma key removal overlaid at bottom
    - Optional background music mixed with avatar audio
    """
    output_filename = request.output_filename or f"{request.script_id}_combined.mp4"
    output_path = OUTPUT_DIR / output_filename

    # Get encoder settings
    encoder, encoder_opts = get_gpu_encoder() if request.use_gpu else ("libx264", ["-preset", "fast", "-crf", "23"])

    # Build FFmpeg filter complex
    filter_parts = []
    inputs = ["-i", request.avatar_path, "-i", request.background_path]
    input_count = 2

    # Add music if provided
    music_index = -1
    if request.music_path and Path(request.music_path).exists():
        inputs.extend(["-stream_loop", "-1", "-i", request.music_path])
        music_index = input_count
        input_count += 1

    # Scale and crop background to 9:16 (1080x1920)
    filter_parts.append("[1:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[bg]")

    # Chroma key the avatar (green screen removal)
    filter_parts.append("[0:v]chromakey=0x00FF00:0.1:0.2[avatar_keyed]")

    # Overlay avatar on background (centered at bottom with 100px margin)
    filter_parts.append("[bg][avatar_keyed]overlay=(W-w)/2:H-h-100:shortest=1[outv]")

    # Audio mixing
    if music_index >= 0:
        filter_parts.append(f"[{music_index}:a]volume=0.1[music_low]")
        filter_parts.append("[0:a][music_low]amix=inputs=2:duration=first[outa]")
    else:
        filter_parts.append("[0:a]aformat=channel_layouts=stereo[outa]")

    filter_complex = ";".join(filter_parts)

    # Build full command
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
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
    Burn SRT captions onto video (TikTok-style text overlay)
    """
    output_filename = request.output_filename or f"{request.script_id}_captioned.mp4"
    output_path = OUTPUT_DIR / output_filename

    encoder, encoder_opts = get_gpu_encoder() if request.use_gpu else ("libx264", ["-preset", "fast", "-crf", "23"])

    # Subtitle filter with styling
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
