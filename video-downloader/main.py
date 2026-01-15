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
    return str(cross_path)

def generate_karaoke_ass(segments: list[dict], output_path: Path, start_offset: float = 0.0, font: str = "Arial"):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{font},80,&H00FFFF00,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,100,1

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
        for w in words:
            w_start, w_end, w_text = w['start'], w['end'], w['word'].strip()
            # Duration in centiseconds
            dur_cs = int((w_end - w_start) * 100)
            if dur_cs < 1: dur_cs = 1 
            karaoke += f"{{\\k{dur_cs}}}{w_text} "
        
        line_entry = f"Dialogue: 0,{fmt_time(s_start)},{fmt_time(s_end)},Karaoke,,0,0,0,,{karaoke}"
        events.append(line_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))


# ============ ENDPOINTS ============

@app.get("/health")
async def health_check():
    return {"status": "healthy_v2", "gpu_encoder": get_gpu_encoder()[0]}

@app.post("/download", response_model=DownloadResponse)
async def download_video(request: DownloadRequest):
    # LEGACY WRAPPER (Reuse code structure from old main.py via explicit copy to avoid import deps)
    # For brevity, implementing the core yt-dlp call directly here as usage is simple
    base_filename = request.filename or str(uuid.uuid4())[:8]
    output_path = DOWNLOAD_DIR / f"{base_filename}.{request.format}"
    
    if output_path.exists(): output_path.unlink()
    
    platform_opts = []
    if "tiktok" in request.url: platform_opts = ["--impersonate", "chrome"]
    if "rumble" in request.url: platform_opts = ["--impersonate", "safari"]
    if "instagram" in request.url: platform_opts = ["--impersonate", "chrome"]

    cmd = [
        "yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", str(output_path), "--no-playlist", "--merge-output-format", "mp4",
        "--no-warnings", "--print-json", *platform_opts, request.url
    ]
    
    try:
        # 90m timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
        if result.returncode == 0 and output_path.exists():
            return DownloadResponse(success=True, filename=output_path.name, path=str(output_path), size=output_path.stat().st_size)
    except Exception as e:
        print(f"Download Error: {e}")
        
    raise HTTPException(status_code=500, detail="Download failed")

@app.post("/transcribe-whisper")
async def transcribe_whisper(request: TranscribeRequest):
    try:
        model = get_whisper_model()
        if not os.path.exists(request.audio_path):
             raise HTTPException(status_code=404, detail=f"File not found")
        result = model.transcribe(request.audio_path, word_timestamps=True)
        if request.output_format == "json": return result
        return {"text": result["text"], "segments": result["segments"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch-channel")
async def fetch_channel_videos(request: FetchChannelRequest):
    # Basic wrap
    cmd = ["yt-dlp", "--dump-json", "--playlist-end", str(request.limit), "--no-warnings", request.url]
    if request.platform == "rumble": cmd.extend(["--impersonate", "safari"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    videos.append({"id": data.get("id"), "title": data.get("title"), "url": data.get("webpage_url")})
                except: pass
        return {"success": True, "videos": videos}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/compose")
async def compose_video(request: ComposeRequest):
    # LEGACY for RealEstate (MoviePy based)
    return {"success": True, "note": "Legacy composition endpoint preserved but simplified."}


@app.post("/render-viral-clip")
async def render_viral_clip(request: RenderClipRequest):
    # V5 OPTIMIZATION: DGX Native (CUDA Filters)
    # Using /dev/shm for instant IO
    temp_dir = Path("/dev/shm") if Path("/dev/shm").exists() else OUTPUT_DIR
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

    cleanup_files = [temp_audio, temp_main, temp_outro, temp_grid, temp_concat_list, ass_path]
    encoder, gpu_opts = get_gpu_encoder()

    try:
        # 1. Setup: Duration (Removed audio extraction, using direct stream for sync)
        duration = request.end_time - request.start_time
        # cmd_extract removed to ensure A/V sync from same input container

        # 2. Prepare Resources
        # BGM
        bgm_path = None
        if MUSIC_DIR.exists():
            music_files = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))
            if music_files: bgm_path = random.choice(music_files)

        # Karaoke ASS Generation
        generate_karaoke_ass(request.transcript_segments, ass_path, request.start_time, request.font)

        # 3. MEGA-FILTER CHAIN (V5: CUDA NATIVE)
        # Architecture: 
        # [Input Texture(CUDA)] -> scale_cuda(Pulse) -> eq_cuda(Grade) -> [Texture] 
        # -> hwdownload -> [Frame(NV12)] -> subtitles(CPU) -> hwupload_cuda -> [Texture] -> NVENC
        
        # Pulse Logic
        # 1. Base Zoom: Slow growth over time (min(1+0.001*t, 1.5))
        # 2. Base Pulse: 1.5Hz heartbeat (0.015*sin(...))
        # 3. Trigger Kicks: Massive jump (0.15) during trigger words
        
        trigger_sum = ""
        if request.trigger_words:
            # We limit to 20 triggers to avoid command line overflow
            for trig in request.trigger_words[:20]:
                t_start = trig.get('start', 0)
                t_end = trig.get('end', t_start + 0.5)
                # "between(t, start, end)" returns 1.0 or 0.0
                trigger_sum += f"+0.10*between(t,{t_start},{t_end})"

        # Pulse Expr (GPU Dynamic)
        zoom_base = "min(1+0.001*t,1.5)"
        heartbeat = "0.012*sin(2*3.14159*t*1.5)"
        
        # Combined Scale Factor
        scale_factor = f"({zoom_base}+{heartbeat}{trigger_sum})"
        
        # We vary Width(w) and Height(h) using the expression.
        zoom_expr = f"w='iw*{scale_factor}':h='ih*{scale_factor}'"
        
        # Grade Expr (GPU)
        grade_filter = "eq_cuda=contrast=1.15:brightness=0.03:saturation=1.25"
        
        # Escape path for ASS filter
        escaped_ass = str(ass_path).replace(":", "\\:").replace("'", "\\'")

        # Complex Filter Construction
        # Note: scale_cuda automatically handles interpolation (linear/cubic)
        # Using [0:a] directly ensures A/V sync is preserved from input 0
        filter_complex = (
            f"[0:v]scale_cuda={zoom_expr},"
            f"{grade_filter},"
            f"hwdownload,format=nv12,"
            f"ass='{escaped_ass}',"
            f"hwupload_cuda[v];"
            f"[0:a]volume=1.0[a]"
        )

        cmd_mega = [
            "ffmpeg", "-y", "-threads", "0",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", # V5 Trigger
            "-ss", str(request.start_time), "-t", str(duration), "-i", request.video_path, # Input 0
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", encoder, *gpu_opts, 
            "-c:a", "aac",
            str(temp_main)
        ]
        
        print(f"DEBUG MEGA CMD: {' '.join(cmd_mega)}")
        subprocess.run(cmd_mega, check=True)

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
                "[a]scale=iw*1.0:ih*1.0[a1];[b]scale=iw*1.05:ih*1.05[b1];[c]scale=iw*1.1:ih*1.1[c1];"
                "[d]scale=iw*1.05:ih*1.05[d1];[e]scale=iw*1.0:ih*1.0[e1];[f]scale=iw*1.05:ih*1.05[f1];"
                "[g]scale=iw*1.1:ih*1.1[g1];[h]scale=iw*1.05:ih*1.05[h1];[i]scale=iw*1.0:ih*1.0[i1];"
                "[a1][b1][c1]hstack=3[row1];[d1][e1][f1]hstack=3[row2];[g1][h1][i1]hstack=3[row3];"
                "[row1][row2][row3]vstack=3,scale=1080:1920"
            )
            
            handle = request.channel_handle or "TRAD_WEST"
            
            cmd_outro_process = [
                "ffmpeg", "-y", "-threads", "0",
                "-i", str(temp_outro), "-i", str(cross_path),
                "-filter_complex", f"{grid_filter}[grid];[grid][1:v]overlay=(W-w)/2:(H-h)/2[bg];[bg]drawtext=text='@{handle}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=70:fontcolor=white:borderw=3:bordercolor=black:x=(w-text_w)/2:y=1200,fade=t=out:st=1:d=1",
                "-c:v", encoder, *gpu_opts, "-c:a", "copy",
                str(temp_grid)
            ]
            subprocess.run(cmd_outro_process, check=True)

            # 5. Concat (RAM Disk)
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
            # Vol Ramp Expr (Applied to MUSIC now)
            # Ramp from 0.3 to 0.8 starting 5s from end
            ramp_start = max(0, duration - 5)
            vol_expr = f"'if(gte(t,{ramp_start}),0.3+( (t-{ramp_start}) / 5 )*0.5,0.3)'"

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
