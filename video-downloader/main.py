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
import re
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

# CURSE WORDS - These get FULL intensity RGB glitch effect
# Other trigger words get reduced (1/4 to 1/2) intensity randomly
CURSE_WORDS = {
    "fuck", "fucking", "fucked", "shit", "bitch", "bitches", "ass", "damn", "damned",
    "hell", "crap", "bastard", "bullshit", "piss", "dick", "asshole", "whore", "slut"
}

# ============ COLOR PRESETS ============
# Each preset defines FFmpeg eq filter parameters + optional tint adjustments
COLOR_PRESETS = {
    "vibrant": {"contrast": 1.15, "brightness": 0.03, "saturation": 1.25},
    "high_contrast": {"contrast": 1.25, "brightness": 0.02, "saturation": 1.3},
    "dark_cinematic": {"contrast": 1.2, "brightness": -0.05, "saturation": 0.9},
    "cyberpunk": {"contrast": 1.15, "brightness": 0.0, "saturation": 1.4, "tint": "cyan"},
    "clean_modern": {"contrast": 1.1, "brightness": 0.02, "saturation": 1.1},
    "crusade_dark": {"contrast": 1.3, "brightness": -0.08, "saturation": 0.7, "tint": "gold"},
    "documentary": {"contrast": 1.1, "brightness": -0.03, "saturation": 0.85},
    "high_energy": {"contrast": 1.2, "brightness": 0.03, "saturation": 1.2},
    "bw": {"contrast": 1.15, "brightness": 0.03, "saturation": 0},
}

# ============ LUT PRESETS ============
# Maps creative names to .cube LUT files stored in /assets/luts/
LUT_DIR = "/assets/luts"
LUT_PRESETS = {
    "kodak_warm": {"file": "kodak_warm.cube", "description": "Warm golden cinema look (serious monologues, wisdom)"},
    "teal_orange": {"file": "teal_orange.cube", "description": "Hollywood blockbuster teal shadows + orange highlights (confrontation, drama)"},
    "film_noir": {"file": "film_noir.cube", "description": "Deep shadows, high contrast (dark topics, conspiracy)"},
    "bleach_bypass": {"file": "bleach_bypass.cube", "description": "Desaturated, silvery, gritty (war, violence, raw moments)"},
    "cross_process": {"file": "cross_process.cube", "description": "Surreal color shifts, greens and magentas (funny, absurd)"},
    "golden_hour": {"file": "golden_hour.cube", "description": "Warm amber glow (inspirational, hopeful)"},
    "cold_chrome": {"file": "cold_chrome.cube", "description": "Steel blue metallic feel (tech, power, authority)"},
    "vintage_tobacco": {"file": "vintage_tobacco.cube", "description": "Aged warm desaturated (retro, nostalgia)"},
}

# ============ EFFECT REGISTRY ============
# Complete catalog of available effects with descriptions for Grok director prompt
EFFECT_REGISTRY = {
    # --- Color Grading ---
    "lut_color_grade": {
        "type": "ffmpeg_filter",
        "category": "color",
        "description": "Cinema-quality color grading using film emulation LUT",
        "params": {"lut_file": "string (LUT preset name or filename)"},
        "options": list(LUT_PRESETS.keys()),
    },
    # --- Motion Effects ---
    "camera_shake": {
        "type": "ffmpeg_expression",
        "category": "motion",
        "description": "Handheld documentary camera shake feel",
        "params": {"intensity": "int 2-15 pixels", "frequency": "float 1.0-4.0 Hz"},
    },
    "speed_ramp": {
        "type": "ffmpeg_segment",
        "category": "motion",
        "description": "Slow-motion emphasis on key moments, speed up between",
        "params": {"ramps": "[{time, speed, duration}] max 3"},
    },
    "temporal_trail": {
        "type": "ffmpeg_filter",
        "category": "motion",
        "description": "Ghosting/motion blur streaks for trippy dreamlike moments",
        "params": {"decay": "float 0.8-0.99", "segments": "[[start, end]] time ranges"},
    },
    # --- Glitch Effects ---
    "retro_glow": {
        "type": "ffmpeg_filter",
        "category": "glitch",
        "description": "Soft neon bloom with RGB fringe glow effect",
        "params": {"intensity": "float 0.2-0.6"},
    },
    "wave_displacement": {
        "type": "ffmpeg_filter",
        "category": "glitch",
        "description": "Melting/warping wave distortion in brief bursts",
        "params": {"triggers": "[{start, end, amplitude}] max 3 bursts"},
    },
    "heavy_vhs": {
        "type": "config_override",
        "category": "glitch",
        "description": "Extra VHS tracking distortion and grain",
        "params": {"intensity": "float 1.0-2.0"},
    },
    # --- Caption Styles ---
    "caption_pop_scale": {
        "type": "ass_style",
        "category": "captions",
        "description": "Words pop in with bouncy overshoot scale animation",
    },
    "caption_shake": {
        "type": "ass_style",
        "category": "captions",
        "description": "Vibrating/shaking text on trigger words (aggressive energy)",
    },
    "caption_blur_reveal": {
        "type": "ass_style",
        "category": "captions",
        "description": "Words sharpen from blur as spoken (cinematic reveal)",
    },
    # --- Audio Reactive ---
    "beat_sync": {
        "type": "audio_analysis",
        "category": "audio",
        "description": "Zoom pulses sync to background music beats (amplified)",
    },
    "audio_saturation": {
        "type": "audio_analysis",
        "category": "audio",
        "description": "Colors intensify dynamically on loud/intense moments",
    },
    # --- Transitions (B-roll) ---
    "transition_pixelize": {
        "type": "xfade",
        "category": "transition",
        "description": "Pixel dissolve between B-roll cuts (glitch aesthetic)",
    },
    "transition_radial": {
        "type": "xfade",
        "category": "transition",
        "description": "Circular wipe between B-roll cuts (dramatic reveal)",
    },
    "transition_dissolve": {
        "type": "xfade",
        "category": "transition",
        "description": "Smooth cross-fade between B-roll cuts (cinematic)",
    },
    "transition_glitch": {
        "type": "xfade",
        "category": "transition",
        "description": "Noise burst transition between B-roll cuts (aggressive)",
    },
    # --- Rare Effects (MoviePy post-processing) ---
    "datamosh": {
        "type": "moviepy_segment",
        "category": "rare",
        "description": "Frame-melting I-frame removal glitch (use SPARINGLY, max 1-2 clips per video)",
        "params": {"segments": "[{start, end}] max 3, max 2s each"},
    },
    "pixel_sort": {
        "type": "moviepy_segment",
        "category": "rare",
        "description": "Glitch art pixel sorting by brightness (use SPARINGLY, max 1-2 clips per video)",
        "params": {"segments": "[{start, end}] max 3, max 2s each"},
    },
}

# ============ FACE DETECTION FOR AUTO-CENTERING ============
def detect_face_offset(video_path: str, target_width: int = 1080) -> int:
    """
    Detect face position in the first frame and calculate horizontal offset
    to center the speaker in the final vertical crop.

    Returns: horizontal pixel offset (positive = shift crop right, negative = shift left)
    """
    try:
        import cv2

        # Extract first frame using FFmpeg (faster than OpenCV VideoCapture for remote files)
        temp_frame = f"/tmp/face_detect_{uuid.uuid4().hex[:8]}.jpg"
        result = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vframes", "1", "-q:v", "2",
            temp_frame
        ], capture_output=True, timeout=10)

        if result.returncode != 0 or not os.path.exists(temp_frame):
            print(f"[FaceDetect] Failed to extract frame from {video_path}")
            return 0

        # Load the frame
        frame = cv2.imread(temp_frame)
        if frame is None:
            os.remove(temp_frame)
            return 0

        frame_height, frame_width = frame.shape[:2]
        print(f"[FaceDetect] Frame size: {frame_width}x{frame_height}")

        # Use OpenCV's DNN face detector (more accurate than Haar)
        # Fall back to Haar cascade if DNN model not available
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces with multiple scale factors for better detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Clean up temp file
        os.remove(temp_frame)

        if len(faces) == 0:
            print(f"[FaceDetect] No faces detected, using default center crop")
            return 0

        # Find the largest face (likely the main speaker)
        largest_face = max(faces, key=lambda f: f[2] * f[3])  # w * h
        x, y, w, h = largest_face

        # Calculate face center X position
        face_center_x = x + w // 2
        frame_center_x = frame_width // 2

        # Calculate offset needed to center the face
        # Positive offset = face is right of center, need to shift crop right
        # Negative offset = face is left of center, need to shift crop left
        offset = face_center_x - frame_center_x

        # Limit offset to prevent cropping outside frame bounds
        # After scaling to 1920 height, width is approximately frame_width * (1920/frame_height)
        scaled_width = int(frame_width * (1920 / frame_height))
        max_offset = (scaled_width - target_width) // 2

        # Clamp offset to safe range (leave some margin)
        safe_margin = 50
        offset = max(-max_offset + safe_margin, min(max_offset - safe_margin, offset))

        print(f"[FaceDetect] Face at ({face_center_x}, {y + h//2}), frame center at {frame_center_x}")
        print(f"[FaceDetect] Calculated offset: {offset}px (max: ±{max_offset})")

        return offset

    except ImportError:
        print("[FaceDetect] OpenCV not available, using default center crop")
        return 0
    except Exception as e:
        print(f"[FaceDetect] Error detecting face: {e}")
        return 0


# ============ EFFECT DISPATCHER ============
def get_effect_chain(effect_settings: dict) -> dict:
    """
    Parse effect_settings and return a configuration dict for rendering.
    This is the main dispatcher that interprets template effect_settings.
    """
    config = {
        # Color grading
        "contrast": 1.15,
        "brightness": 0.03,
        "saturation": 1.25,
        "tint": None,
        # Pulse/zoom effects
        "pulse_intensity": 0.25,
        "aggressive_pulse": False,
        # Glitch effects
        "rgb_max_shift": 35,
        "glitch_intensity": 1.0,
        "heavy_glitch": False,
        # VHS effects
        "vhs_enabled": True,
        "vhs_intensity": 1.0,
        "grain_intensity": 15,
        # Visual effects
        "letterbox": False,
        "letterbox_ratio": 2.35,
        "vignette": False,
        "vignette_intensity": 0.3,
        "scan_lines": False,
        "scan_line_opacity": 0.1,
        "film_grain": False,
        # Animation effects
        "velocity_enabled": False,
        "velocity_speed": 2.0,
        "flash_enabled": False,
        "flash_intensity": 0.8,
        "flash_color": "#FFFFFF",
        "whip_pan": False,
        "motion_blur": False,
        # Text effects
        "kinetic_text": False,
        "text_animation": "pop_scale",
        # Grid effects
        "grid_enabled": False,
        "grid_type": 4,
        # B-roll settings
        "broll_cut_interval": 0.5,
        # Cross overlay (for Crusade template)
        "cross_overlay": False,
        # === NEW EFFECTS (Grok Director) ===
        # LUT color grading (overrides eq grade when set)
        "lut_file": None,  # filename in /assets/luts/ or LUT_PRESETS key
        # Retro glow (neon bloom)
        "retro_glow": False,
        "retro_glow_intensity": 0.3,
        # Temporal trail (motion ghosting)
        "temporal_trail": False,
        "temporal_trail_segments": [],  # [[start, end], ...] in clip-relative time
        "temporal_trail_decay": 0.95,
        # Camera shake
        "camera_shake": False,
        "shake_intensity": 8,  # pixels
        "shake_frequency": 2.0,  # Hz
        # Wave displacement (brief distortion bursts)
        "wave_displacement": False,
        "wave_triggers": [],  # [{"start": t, "end": t, "amplitude": 15}]
        # Speed ramps
        "speed_ramps": [],  # [{"time": t, "speed": 0.3, "duration": 1.5}]
        # Audio reactive
        "beat_sync_intensity": 0.006,  # Default heartbeat intensity
        "audio_saturation": False,
        "audio_saturation_amount": 0.3,
        # B-roll transitions
        "broll_transition_type": None,  # "pixelize"|"radial"|"dissolve"|"slideleft"|"fadeblack"
        "broll_transition_duration": 0.3,
        # Caption style
        "caption_style": "standard",  # "standard"|"pop_scale"|"shake"|"blur_reveal"
        # Rare effects (MoviePy post-processing)
        "datamosh_segments": [],  # [{"start": 5.0, "end": 7.0}] max 3, 2s each
        "pixel_sort_segments": [],  # [{"start": 10.0, "end": 12.0}] max 3, 2s each
    }

    if not effect_settings:
        return config

    # Apply color preset if specified
    color_preset = effect_settings.get("color_preset")
    if color_preset and color_preset in COLOR_PRESETS:
        preset = COLOR_PRESETS[color_preset]
        config["contrast"] = preset.get("contrast", config["contrast"])
        config["brightness"] = preset.get("brightness", config["brightness"])
        config["saturation"] = preset.get("saturation", config["saturation"])
        config["tint"] = preset.get("tint", None)

    # Direct overrides from effect_settings
    for key in config.keys():
        if key in effect_settings:
            config[key] = effect_settings[key]

    # Parse effects list
    effects = effect_settings.get("effects", [])
    for effect in effects:
        if effect == "velocity":
            config["velocity_enabled"] = True
        elif effect == "flash":
            config["flash_enabled"] = True
        elif effect == "aggressive_pulse":
            config["aggressive_pulse"] = True
            config["pulse_intensity"] = 0.35
        elif effect == "letterbox":
            config["letterbox"] = True
        elif effect == "film_grain":
            config["film_grain"] = True
        elif effect == "slow_zoom":
            config["pulse_intensity"] = 0.1
        elif effect == "heavy_glitch":
            config["heavy_glitch"] = True
            config["glitch_intensity"] = 1.5
            config["rgb_max_shift"] = 50
        elif effect == "pixel_sort":
            pass  # Handled in MoviePy pass
        elif effect == "scan_lines":
            config["scan_lines"] = True
        elif effect == "kinetic_text":
            config["kinetic_text"] = True
        elif effect == "subtle_zoom":
            config["pulse_intensity"] = 0.15
        elif effect == "clean_grade":
            config["vhs_enabled"] = False
        elif effect == "heavy_vhs":
            config["vhs_intensity"] = 1.3
        elif effect == "cross_flash":
            config["cross_overlay"] = True
            config["flash_color"] = "#FFD700"  # Gold
        elif effect == "dark_grade":
            config["brightness"] = -0.08
        elif effect == "grid_4panel":
            config["grid_enabled"] = True
            config["grid_type"] = 4
        elif effect == "sync_reveal":
            pass  # Animation handled separately
        elif effect == "vibrant_grade":
            config["saturation"] = 1.3
        elif effect == "vignette":
            config["vignette"] = True
        elif effect == "whip_pan":
            config["whip_pan"] = True
        elif effect == "motion_blur":
            config["motion_blur"] = True

    # Handle saturation=0 from effect_settings (B&W override)
    if effect_settings.get("saturation") == 0 or effect_settings.get("color_grade") == "bw":
        config["saturation"] = 0

    # === LUT RESOLUTION ===
    # Resolve lut_file: can be a LUT_PRESETS key or direct filename
    lut_file = config.get("lut_file") or effect_settings.get("color_grade")
    if lut_file:
        if lut_file in LUT_PRESETS:
            lut_path = os.path.join(LUT_DIR, LUT_PRESETS[lut_file]["file"])
        else:
            lut_path = os.path.join(LUT_DIR, lut_file)
        if os.path.exists(lut_path):
            config["lut_file"] = lut_path
            print(f"[Effects] LUT resolved: {lut_file} -> {lut_path}")
        else:
            config["lut_file"] = None
            print(f"[Effects] LUT not found: {lut_path}, falling back to eq grade")

    # === New effect flags from effect_settings ===
    if effect_settings.get("retro_glow"):
        config["retro_glow"] = True
        if isinstance(effect_settings["retro_glow"], (int, float)):
            config["retro_glow_intensity"] = float(effect_settings["retro_glow"])
    if effect_settings.get("temporal_trail"):
        config["temporal_trail"] = True
        if isinstance(effect_settings["temporal_trail"], dict):
            config["temporal_trail_segments"] = effect_settings["temporal_trail"].get("segments", [])
            config["temporal_trail_decay"] = effect_settings["temporal_trail"].get("decay", 0.95)
    if effect_settings.get("camera_shake"):
        config["camera_shake"] = True
        if isinstance(effect_settings["camera_shake"], dict):
            config["shake_intensity"] = effect_settings["camera_shake"].get("intensity", 8)
            config["shake_frequency"] = effect_settings["camera_shake"].get("frequency", 2.0)
    if effect_settings.get("wave_displacement"):
        config["wave_displacement"] = True
        if isinstance(effect_settings["wave_displacement"], dict):
            config["wave_triggers"] = effect_settings["wave_displacement"].get("triggers", [])
    if effect_settings.get("speed_ramps"):
        config["speed_ramps"] = effect_settings["speed_ramps"][:3]  # Max 3
    if effect_settings.get("beat_sync"):
        config["beat_sync_intensity"] = 0.02  # Amplified when beat_sync enabled
    if effect_settings.get("audio_saturation"):
        config["audio_saturation"] = True
        if isinstance(effect_settings["audio_saturation"], (int, float)):
            config["audio_saturation_amount"] = float(effect_settings["audio_saturation"])
    if effect_settings.get("transition"):
        config["broll_transition_type"] = effect_settings["transition"]
    if effect_settings.get("caption_style"):
        config["caption_style"] = effect_settings["caption_style"]

    return config


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
FONTS_DIR = Path("/usr/share/fonts/truetype/custom")

def get_installed_fonts() -> list:
    """Scan custom fonts directory and return list of font family names."""
    if not FONTS_DIR.exists():
        return ["Arial"]  # Fallback
    fonts = []
    for f in FONTS_DIR.glob("*.ttf"):
        # Extract font name from filename (e.g., "Honk.ttf" -> "Honk")
        # For multi-word fonts like "PirataOne-Regular.ttf" -> "Pirata One"
        name = f.stem
        # Remove common suffixes
        for suffix in ["-Regular", "-Bold", "-Italic", "-Medium", "-Light"]:
            name = name.replace(suffix, "")
        # Convert CamelCase to spaces (PirataOne -> Pirata One)
        import re
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        if name and name not in fonts:
            fonts.append(name)
    return fonts if fonts else ["Arial"]

# Legacy compatibility - will be called dynamically
VIRAL_FONTS = get_installed_fonts()

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
    # B-roll montage parameters
    climax_time: Optional[float] = None  # Absolute timestamp in source video where climax occurs
    broll_paths: Optional[List[str]] = []  # List of B-roll video paths to use for montage
    broll_duration: Optional[float] = 0  # Duration of B-roll section in seconds
    # Template effect settings
    effect_settings: Optional[dict] = {}  # {"color_grade": "bw", "saturation": 0, etc.}
    # === NEW: Grok Director effect fields ===
    speed_ramps: Optional[List[dict]] = []  # [{"time": 15.2, "speed": 0.3, "duration": 1.5}]
    caption_style: str = "standard"  # "standard"|"pop_scale"|"shake"|"blur_reveal"
    broll_transition_type: Optional[str] = None  # "pixelize"|"radial"|"dissolve"|"slideleft"

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
    Safely transcribe audio/video file.
    Extracts audio to WAV first for reliable transcription.
    Returns empty result if audio is invalid/silent.
    """
    import torch
    import tempfile
    import os

    model = get_whisper_model()

    # Clear CUDA cache to prevent memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract audio to a clean WAV file (16kHz mono - Whisper's native format)
    # This ensures reliable transcription regardless of input format
    temp_wav = None
    transcribe_path = audio_path

    try:
        # Always extract to WAV for consistency
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()

        extract_cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",  # 16kHz (Whisper native)
            "-ac", "1",  # Mono
            temp_wav.name
        ]

        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(temp_wav.name):
            transcribe_path = temp_wav.name
            print(f"Extracted audio to WAV for transcription: {temp_wav.name}")

            # Validate WAV file has content (at least 1KB = ~0.03s of 16kHz mono)
            wav_size = os.path.getsize(temp_wav.name)
            if wav_size < 1024:
                print(f"[Whisper] WAV file too small ({wav_size} bytes), likely empty audio")
                return {"segments": [], "text": ""}

            # Check audio duration using ffprobe
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", temp_wav.name]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                try:
                    audio_duration = float(probe_result.stdout.strip())
                    if audio_duration < 0.1:  # Less than 100ms
                        print(f"[Whisper] Audio too short ({audio_duration:.3f}s), skipping transcription")
                        return {"segments": [], "text": ""}
                    print(f"[Whisper] Audio duration: {audio_duration:.2f}s")
                except ValueError:
                    print(f"[Whisper] Could not parse duration: {probe_result.stdout}")
        else:
            print(f"WAV extraction failed, using original: {result.stderr[:200] if result.stderr else 'unknown'}")

        # Transcribe the audio with error handling for problematic audio
        try:
            transcription = model.transcribe(transcribe_path, word_timestamps=word_timestamps)
            return transcription
        except RuntimeError as e:
            error_msg = str(e).lower()
            # Catch all tensor-related Whisper errors (empty audio, size mismatches, etc.)
            if any(phrase in error_msg for phrase in [
                "cannot reshape tensor",
                "size of tensor",
                "expected size for",
                "dimensions of batch"
            ]):
                print(f"[Whisper] Tensor error (problematic audio): {str(e)[:100]}")
                print(f"[Whisper] Returning empty transcription")
                return {"segments": [], "text": ""}
            raise
        except TypeError as e:
            # Happens when Whisper timing hooks get None (problematic audio)
            if "NoneType" in str(e) and "subscriptable" in str(e):
                print(f"[Whisper] Timing hook failed (NoneType) - returning empty transcription")
                return {"segments": [], "text": ""}
            raise
        except IndexError as e:
            # Happens when Whisper word alignment fails (e.g., "index X is out of bounds")
            print(f"[Whisper] Word alignment failed (IndexError): {str(e)[:100]}")
            print(f"[Whisper] Returning empty transcription")
            return {"segments": [], "text": ""}

    finally:
        # Clean up temp WAV file
        if temp_wav and os.path.exists(temp_wav.name):
            try:
                os.unlink(temp_wav.name)
            except:
                pass

def detect_beats(audio_path: str, max_duration: float = None) -> list:
    """
    Detect BASS DRUM hits in audio file using low-frequency onset detection.
    Focuses on kick drum frequencies (40-120Hz) for tight beat sync.
    Returns list of beat timestamps in seconds.
    """
    import subprocess
    import tempfile
    from scipy.signal import butter, filtfilt

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

        # LOW-PASS FILTER to isolate bass drum (kick drum is 40-120Hz)
        # Use 120Hz cutoff to capture kick drum fundamentals
        nyquist = sample_rate / 2
        low_cutoff = 120 / nyquist  # Normalize to Nyquist

        # Butterworth low-pass filter, 4th order for sharp rolloff
        b, a = butter(4, low_cutoff, btype='low')
        bass_audio = filtfilt(b, a, audio_data)

        # Onset detection on bass-filtered signal
        # Compute short-time energy with hop size ~23ms (512 samples at 22kHz)
        hop_size = 512
        frame_size = 1024

        # Compute RMS energy per frame on BASS signal
        num_frames = (len(bass_audio) - frame_size) // hop_size + 1
        energy = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_size
            frame = bass_audio[start:start + frame_size]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        # Normalize energy
        if energy.max() > 0:
            energy = energy / energy.max()

        # Detect bass drum hits: local maxima with tighter constraints
        # Kick drums typically hit every 0.4-0.6s in most music (100-150 BPM)
        min_beat_gap = int(0.35 * sample_rate / hop_size)  # Minimum 0.35s between kicks
        peaks, properties = find_peaks(
            energy,
            height=0.25,          # Higher threshold for cleaner kicks
            distance=min_beat_gap,
            prominence=0.10       # Must be prominent peaks
        )

        # Convert frame indices to timestamps
        beat_times = peaks * hop_size / sample_rate

        return beat_times.tolist()

    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_wav)
        except:
            pass

def detect_onset_peaks(audio_path: str, max_duration: float = None) -> list:
    """
    Detect high-energy onset peaks in audio (full spectrum, not just bass).
    Used for audio-reactive saturation pulses. Returns peak timestamps.
    """
    import tempfile
    from scipy.signal import butter, filtfilt

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        duration_args = ["-t", str(max_duration)] if max_duration else []
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            *duration_args,
            "-ac", "1", "-ar", "22050",
            tmp_wav
        ], capture_output=True, check=True)

        from scipy.io import wavfile
        sample_rate, audio_data = wavfile.read(tmp_wav)

        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

        # Full spectrum onset detection (speech emphasis, loud moments)
        hop_size = 512
        frame_size = 1024

        num_frames = (len(audio_data) - frame_size) // hop_size + 1
        energy = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_size
            frame = audio_data[start:start + frame_size]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        if energy.max() > 0:
            energy = energy / energy.max()

        # Detect onset peaks: high-energy moments with minimum gap
        min_gap = int(0.5 * sample_rate / hop_size)  # Minimum 0.5s between peaks
        peaks, _ = find_peaks(
            energy,
            height=0.4,       # Only loud moments
            distance=min_gap,
            prominence=0.15
        )

        onset_times = (peaks * hop_size / sample_rate).tolist()
        return onset_times[:20]  # Max 20 peaks

    except Exception as e:
        print(f"[AudioReactive] Onset detection failed: {e}")
        return []
    finally:
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


def detect_audio_volume(path: str, max_duration: float = 60) -> dict:
    """
    Detect audio volume levels using FFmpeg's volumedetect filter.
    Returns mean_volume and max_volume in dB.

    Args:
        path: Path to audio/video file
        max_duration: Max seconds to analyze (for long files)

    Returns:
        dict with 'mean_volume' and 'max_volume' in dB (negative values)
    """
    try:
        # Analyze first N seconds for speed
        cmd = [
            "ffmpeg", "-hide_banner", "-i", path,
            "-t", str(max_duration),
            "-af", "volumedetect",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Parse output for volume stats
        output = result.stderr
        mean_vol = -20.0  # Default fallback
        max_vol = -10.0

        for line in output.split('\n'):
            if 'mean_volume:' in line:
                try:
                    mean_vol = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                except:
                    pass
            if 'max_volume:' in line:
                try:
                    max_vol = float(line.split('max_volume:')[1].split('dB')[0].strip())
                except:
                    pass

        return {"mean_volume": mean_vol, "max_volume": max_vol}
    except Exception as e:
        print(f"Volume detection failed: {e}")
        return {"mean_volume": -20.0, "max_volume": -10.0}

def create_cross_image():
    # Use the Chi Rho symbol image (pre-processed with transparency)
    chi_rho_source = Path("/app/chi_rho.png")
    cross_path = OUTPUT_DIR / f"chi_rho_{uuid.uuid4().hex[:8]}.png"

    if chi_rho_source.exists():
        # Copy the Chi Rho image to output dir
        import shutil
        shutil.copy(chi_rho_source, cross_path)
    else:
        # Fallback: generate simple cross if Chi Rho not found
        size = (400, 400)
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        try:
            font = ImageFont.truetype(font_path, 300)
        except:
            font = ImageFont.load_default()
        draw.text((100, 50), "☧", font=font, fill=(128, 0, 128))
        img.save(cross_path)

    return cross_path

def apply_chromatic_aberration(video_path: str, output_path: str, trigger_windows: list, max_shift: int = 15):
    """
    Apply RGB chromatic aberration (glitch/VHS effect) during trigger word windows.
    Uses MoviePy for per-frame numpy manipulation.

    trigger_windows: list of tuples - either (start, end) or (start, end, intensity_multiplier)
                     intensity_multiplier: 1.0 = full intensity, 0.25-0.5 = reduced
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
        for window in trigger_windows:
            # Support both (start, end) and (start, end, multiplier) formats
            if len(window) >= 3:
                start, end, multiplier = window[0], window[1], window[2]
            else:
                start, end, multiplier = window[0], window[1], 1.0

            if start <= t <= end:
                # Smooth sine wave: 0 → 1 → 0 over the window
                duration = end - start
                progress = (t - start) / duration
                # Apply the intensity multiplier for this window
                window_intensity = np.sin(np.pi * progress) * multiplier
                intensity = max(intensity, window_intensity)

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
        fps=30,  # Force consistent 30fps to match B-roll segments
        audio_codec='aac',
        logger=None  # Suppress progress bar
    )

    clip.close()
    processed.close()
    print(f"Chromatic aberration applied successfully")


def apply_datamosh_effect(video_path: str, output_path: str, segments: list, duration: float) -> bool:
    """
    Apply datamosh-style effect to specified segments of a video.
    Uses frame blending with decay to simulate the I-frame removal look:
    motion vectors propagate without reference, creating smearing/melting artifacts.

    segments: [{"start": 5.0, "end": 7.0}] - max 3 segments, max 2s each
    Returns True on success, False on failure.
    """
    if not segments:
        return False

    # Validate and limit segments
    valid_segments = []
    for seg in segments[:3]:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start + 1.0))
        # Clamp to clip duration and max 2s
        end = min(end, start + 2.0, duration)
        if start < duration and end > start:
            valid_segments.append((start, end))

    if not valid_segments:
        return False

    print(f"[Datamosh] Applying to {len(valid_segments)} segments: {valid_segments}")

    try:
        clip = VideoFileClip(video_path)

        # Frame buffer for persistence effect
        prev_frames = [None]
        decay = 0.85  # How much of the previous frame persists (higher = more smear)

        def datamosh_frame(get_frame, t):
            frame = get_frame(t)

            # Check if current time is in any datamosh segment
            in_segment = False
            for start, end in valid_segments:
                if start <= t <= end:
                    in_segment = True
                    break

            if not in_segment:
                prev_frames[0] = frame.copy()
                return frame

            # Datamosh: blend current frame heavily with previous
            if prev_frames[0] is not None:
                # Mix: keep motion from current but colors/structure from previous
                # This simulates P-frames decoding against wrong reference
                blended = (decay * prev_frames[0].astype(np.float32) +
                          (1.0 - decay) * frame.astype(np.float32))

                # Add random block corruption (simulates macro-block errors)
                h, w = frame.shape[:2]
                block_size = 16
                n_corrupt = random.randint(2, 6)
                for _ in range(n_corrupt):
                    bx = random.randint(0, w // block_size - 1) * block_size
                    by = random.randint(0, h // block_size - 1) * block_size
                    # Shift this block from a random position
                    sx = random.randint(0, w // block_size - 1) * block_size
                    sy = random.randint(0, h // block_size - 1) * block_size
                    blended[by:by+block_size, bx:bx+block_size] = \
                        frame[sy:sy+block_size, sx:sx+block_size]

                result = np.clip(blended, 0, 255).astype(np.uint8)
                prev_frames[0] = result.copy()
                return result
            else:
                prev_frames[0] = frame.copy()
                return frame

        processed = clip.transform(datamosh_frame)

        processed.write_videofile(
            output_path,
            codec='libx264',
            preset='ultrafast',
            fps=30,
            audio_codec='aac',
            logger=None
        )

        clip.close()
        processed.close()
        print(f"[Datamosh] Applied successfully to {len(valid_segments)} segments")
        return True

    except Exception as e:
        print(f"[Datamosh] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_pixel_sort_effect(video_path: str, output_path: str, segments: list, duration: float) -> bool:
    """
    Apply pixel sorting effect to specified segments of a video.
    Sorts pixel rows by brightness within a threshold range, creating
    the signature "melting" glitch art look.

    segments: [{"start": 10.0, "end": 12.0}] - max 3 segments, max 2s each
    Returns True on success, False on failure.
    """
    if not segments:
        return False

    # Validate and limit segments
    valid_segments = []
    for seg in segments[:3]:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start + 1.0))
        end = min(end, start + 2.0, duration)
        if start < duration and end > start:
            valid_segments.append((start, end))

    if not valid_segments:
        return False

    print(f"[PixelSort] Applying to {len(valid_segments)} segments: {valid_segments}")

    try:
        clip = VideoFileClip(video_path)

        def pixel_sort_frame(get_frame, t):
            frame = get_frame(t)

            # Check if current time is in any pixel sort segment
            in_segment = False
            segment_progress = 0.0
            for start, end in valid_segments:
                if start <= t <= end:
                    in_segment = True
                    segment_progress = (t - start) / max(end - start, 0.01)
                    break

            if not in_segment:
                return frame

            # Convert to grayscale for brightness threshold
            gray = np.mean(frame, axis=2)
            h, w = frame.shape[:2]

            # Threshold range that gets sorted (mid-brightness pixels)
            # Animate threshold to create evolving effect
            low_thresh = 60 + 40 * np.sin(np.pi * segment_progress)
            high_thresh = 180 - 30 * np.sin(np.pi * segment_progress)

            result = frame.copy()

            # Sort every 3rd row for performance (1920 rows at 30fps is expensive)
            for row_idx in range(0, h, 3):
                row_brightness = gray[row_idx]
                # Find spans of pixels within threshold
                mask = (row_brightness > low_thresh) & (row_brightness < high_thresh)

                # Find contiguous spans to sort
                spans = []
                in_span = False
                span_start = 0
                for col in range(w):
                    if mask[col] and not in_span:
                        span_start = col
                        in_span = True
                    elif not mask[col] and in_span:
                        if col - span_start > 8:  # Min span width
                            spans.append((span_start, col))
                        in_span = False
                if in_span and w - span_start > 8:
                    spans.append((span_start, w))

                # Sort each span by brightness
                for s_start, s_end in spans:
                    span_pixels = result[row_idx, s_start:s_end].copy()
                    span_bright = gray[row_idx, s_start:s_end]
                    sort_indices = np.argsort(span_bright)
                    result[row_idx, s_start:s_end] = span_pixels[sort_indices]
                    # Also apply to the two rows below for visual consistency
                    if row_idx + 1 < h:
                        result[row_idx + 1, s_start:s_end] = span_pixels[sort_indices]
                    if row_idx + 2 < h:
                        result[row_idx + 2, s_start:s_end] = span_pixels[sort_indices]

            return result

        processed = clip.transform(pixel_sort_frame)

        processed.write_videofile(
            output_path,
            codec='libx264',
            preset='ultrafast',
            fps=30,
            audio_codec='aac',
            logger=None
        )

        clip.close()
        processed.close()
        print(f"[PixelSort] Applied successfully to {len(valid_segments)} segments")
        return True

    except Exception as e:
        print(f"[PixelSort] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_speed_ramps(input_path: Path, output_path: Path, speed_ramps: list, encoder: str, gpu_opts: list) -> bool:
    """
    Apply speed ramps to a rendered clip. Splits at ramp points, retimes segments,
    and re-concatenates with audio sync.

    speed_ramps: [{"time": 15.2, "speed": 0.3, "duration": 1.5}]
      - time: when the ramp starts (clip-relative seconds)
      - speed: playback speed (0.3=slow-mo, 2.0=fast)
      - duration: how long the source content spans (in original seconds)

    Returns True if speed ramps were applied, False if skipped/failed.
    """
    if not speed_ramps:
        return False

    # Sort ramps by time
    ramps = sorted(speed_ramps[:3], key=lambda r: r.get("time", 0))

    # Get clip duration
    try:
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)
        ], capture_output=True, text=True)
        clip_duration = float(probe.stdout.strip())
    except:
        print("[SpeedRamp] Failed to probe clip duration, skipping")
        return False

    # Build segment list: [(start, end, speed), ...]
    segments = []
    cursor = 0.0
    for ramp in ramps:
        ramp_start = ramp.get("time", 0)
        ramp_speed = max(0.25, min(4.0, ramp.get("speed", 1.0)))  # Clamp to [0.25, 4.0]
        ramp_dur = max(0.3, min(5.0, ramp.get("duration", 1.0)))  # Clamp to [0.3, 5.0]

        if ramp_start > cursor:
            # Normal speed segment before this ramp
            segments.append((cursor, ramp_start, 1.0))
        # Ramp segment
        ramp_end = min(ramp_start + ramp_dur, clip_duration)
        if ramp_end > ramp_start:
            segments.append((ramp_start, ramp_end, ramp_speed))
        cursor = ramp_end

    # Final normal segment after last ramp
    if cursor < clip_duration:
        segments.append((cursor, clip_duration, 1.0))

    if not segments:
        return False

    print(f"[SpeedRamp] Splitting into {len(segments)} segments: {[(f'{s:.1f}-{e:.1f}@{sp}x') for s, e, sp in segments]}")

    # Extract and retime each segment
    temp_dir = input_path.parent
    temp_segments = []

    def build_atempo_chain(speed: float) -> str:
        """Build chained atempo filters for speeds outside [0.5, 2.0] range."""
        filters = []
        remaining = speed
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        filters.append(f"atempo={remaining:.4f}")
        return ",".join(filters)

    try:
        for i, (start, end, speed) in enumerate(segments):
            seg_path = temp_dir / f"speedramp_seg_{i:02d}_{uuid.uuid4().hex[:4]}.mp4"
            seg_duration = end - start

            if speed == 1.0:
                # Normal speed - just extract
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start:.3f}", "-t", f"{seg_duration:.3f}",
                    "-i", str(input_path),
                    "-c", "copy",
                    str(seg_path)
                ]
            else:
                # Retimed segment
                atempo_chain = build_atempo_chain(speed)
                setpts = f"setpts=PTS/{speed}"
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start:.3f}", "-t", f"{seg_duration:.3f}",
                    "-i", str(input_path),
                    "-vf", setpts,
                    "-af", atempo_chain,
                    "-c:v", encoder, *gpu_opts,
                    "-c:a", "aac",
                    str(seg_path)
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not seg_path.exists():
                print(f"[SpeedRamp] Segment {i} failed: {result.stderr[-200:] if result.stderr else 'unknown'}")
                # Cleanup and abort
                for p in temp_segments:
                    if p.exists(): p.unlink()
                return False
            temp_segments.append(seg_path)

        # Concatenate all segments
        concat_file = temp_dir / f"speedramp_concat_{uuid.uuid4().hex[:6]}.txt"
        with open(concat_file, "w") as f:
            for seg in temp_segments:
                f.write(f"file '{seg}'\n")

        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy",
            str(output_path)
        ]
        result = subprocess.run(cmd_concat, capture_output=True, text=True)

        # Cleanup
        concat_file.unlink()
        for seg in temp_segments:
            if seg.exists(): seg.unlink()

        if result.returncode != 0 or not output_path.exists():
            print(f"[SpeedRamp] Concat failed: {result.stderr[-200:] if result.stderr else 'unknown'}")
            return False

        print(f"[SpeedRamp] Applied {len(ramps)} speed ramps successfully")
        return True

    except Exception as e:
        print(f"[SpeedRamp] Error: {e}")
        for seg in temp_segments:
            if seg.exists(): seg.unlink()
        return False


def generate_broll_montage(
    broll_paths: List[str],
    duration: float,
    beat_times: List[float],  # Ignored - kept for API compatibility
    output_path: Path,
    effect_settings: dict = None,
    caption_data: dict = None,
    cut_interval: float = 0.5,  # Fixed cut interval in seconds
    transition_type: Optional[str] = None,  # "pixelize"|"radial"|"dissolve"|"slideleft"|"fadeblack"|"wiperight"
    transition_duration: float = 0.3  # Transition duration in seconds
) -> Optional[str]:
    """
    Generate a B-roll montage with fixed-interval cuts (default 0.5s).
    Applies same visual effects (color grade, VHS grain, captions, RGB glitch) as main clip.

    Args:
        broll_paths: List of B-roll video file paths
        duration: Total duration of the montage in seconds
        beat_times: IGNORED - kept for API compatibility
        output_path: Output file path
        effect_settings: Template effects (e.g., {"saturation": 0} for B&W)
        caption_data: Dict with keys: segments, climax_clip_time, font, trigger_words
                     Used to burn ASS captions over B-roll
        cut_interval: Time between cuts in seconds (default 0.5s)

    Returns:
        Path to generated montage video, or None on failure
    """
    if not broll_paths:
        print("B-Roll Montage: No B-roll paths provided")
        return None

    print(f"B-Roll Montage: Generating {duration:.1f}s montage from {len(broll_paths)} clips")

    # Use fixed interval cuts (0.5s default) instead of beat-synced
    num_cuts = int(duration / cut_interval) + 1
    cut_times = [i * cut_interval for i in range(num_cuts)]

    # Ensure end boundary
    if cut_times[-1] < duration - 0.05:
        cut_times.append(duration)

    print(f"B-Roll Montage: Using {len(cut_times)} cut points at {cut_interval}s intervals")

    # Create FFmpeg concat demuxer input file
    concat_list_path = output_path.parent / f"broll_concat_{uuid.uuid4().hex[:8]}.txt"

    # Pre-trim each segment to temp files for reliable concatenation
    temp_segments = []
    temp_dir = output_path.parent

    try:
        # Get effect settings
        effect_settings = effect_settings or {}
        saturation = effect_settings.get("saturation", 1.25)  # Default vibrant, 0 = B&W
        if effect_settings.get("color_grade") == "bw":
            saturation = 0
            print(f"B-Roll Montage: Black & White mode enabled")
        grade_filter = f"eq=contrast=1.15:brightness=0.03:saturation={saturation}"
        vhs_grain = "noise=alls=15:allf=t+u"

        encoder, gpu_opts = get_gpu_encoder()
        clip_index = 0

        actual_duration = 0.0  # Track actual montage duration
        failed_segments = 0

        for i in range(len(cut_times) - 1):
            segment_start = cut_times[i]
            segment_end = cut_times[i + 1]
            segment_duration = segment_end - segment_start

            if segment_duration < 0.1:
                continue

            # Stop if we've used all unique B-roll clips (NO REPEATS)
            if clip_index >= len(broll_paths):
                print(f"B-Roll Montage: Used all {len(broll_paths)} unique clips, stopping at {segment_start:.1f}s (achieved {actual_duration:.1f}s)")
                break

            # Pick a B-roll clip (EACH CLIP USED ONLY ONCE)
            broll_clip = broll_paths[clip_index]

            # Pre-trim this segment with effects applied
            segment_path = temp_dir / f"broll_seg_{i:03d}_{uuid.uuid4().hex[:4]}.mp4"

            # Get clip duration to check if we need to loop
            try:
                probe = subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", broll_clip
                ], capture_output=True, text=True)
                clip_dur = float(probe.stdout.strip()) if probe.stdout.strip() else 10.0
            except:
                clip_dur = 10.0

            # If segment is longer than clip, we'll loop it
            loop_input = []
            if segment_duration > clip_dur:
                # Use stream_loop to loop the input
                loop_count = int(segment_duration / clip_dur) + 1
                loop_input = ["-stream_loop", str(loop_count)]

            cmd_segment = [
                "ffmpeg", "-y",
                *loop_input,
                "-i", broll_clip,
                "-t", f"{segment_duration:.3f}",
                "-vf",
                f"fps=30,"  # Force consistent 30fps for smooth concatenation
                f"scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920:(iw-1080)/2:(ih-1920)/2,"
                f"{grade_filter},"
                f"{vhs_grain}",
                "-c:v", encoder, *gpu_opts,
                "-r", "30",  # Output frame rate
                "-an",
                str(segment_path)
            ]

            result = subprocess.run(cmd_segment, capture_output=True, text=True)
            if result.returncode == 0 and segment_path.exists() and segment_path.stat().st_size > 1000:
                temp_segments.append(segment_path)
                actual_duration += segment_duration
                clip_index += 1  # Only increment on SUCCESS
            else:
                # Log actual error (skip FFmpeg version banner)
                stderr_lines = (result.stderr or "unknown").split('\n')
                error_line = next((l for l in stderr_lines if 'Error' in l or 'error' in l or 'Invalid' in l), stderr_lines[-1] if stderr_lines else "unknown")
                print(f"B-Roll Montage: Warning - segment {i} failed (clip: {Path(broll_clip).name}): {error_line[:150]}")
                failed_segments += 1
                clip_index += 1  # Move to next clip on failure too

        if failed_segments > 0:
            print(f"B-Roll Montage: {failed_segments} segments failed, {len(temp_segments)} succeeded")

        print(f"B-Roll Montage: Pre-trimmed {len(temp_segments)} segments")

        if not temp_segments:
            print("B-Roll Montage: No segments created!")
            return None

        # Concatenate segments (with or without transitions)
        valid_transitions = {"pixelize", "radial", "dissolve", "slideleft", "fadeblack", "wiperight", "smoothleft", "smoothright", "circleopen", "circleclose"}
        use_xfade = transition_type and transition_type in valid_transitions and len(temp_segments) >= 2

        if use_xfade:
            # === XFADE TRANSITIONS ===
            # Chain xfade filters between consecutive segments
            print(f"B-Roll Montage: Using xfade transition '{transition_type}' ({transition_duration:.1f}s)")

            # Get duration of each segment
            seg_durations = []
            for seg in temp_segments:
                try:
                    probe = subprocess.run([
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(seg)
                    ], capture_output=True, text=True)
                    seg_durations.append(float(probe.stdout.strip()))
                except:
                    seg_durations.append(cut_interval)

            # Build xfade filter chain
            inputs = []
            for seg in temp_segments:
                inputs.extend(["-i", str(seg)])

            # Chain xfade: [0][1]xfade=...[v01]; [v01][2]xfade=...[v012]; ...
            filter_parts = []
            cumulative_offset = 0.0
            td = min(transition_duration, cut_interval * 0.8)  # Don't exceed segment duration

            for i in range(len(temp_segments) - 1):
                if i == 0:
                    in_label = f"[{i}:v]"
                else:
                    in_label = f"[v{i}]"
                next_label = f"[{i+1}:v]"
                out_label = f"[v{i+1}]" if i < len(temp_segments) - 2 else "[vout]"

                offset = cumulative_offset + seg_durations[i] - td
                if offset < 0:
                    offset = cumulative_offset + seg_durations[i] * 0.5
                cumulative_offset = offset

                filter_parts.append(
                    f"{in_label}{next_label}xfade=transition={transition_type}:duration={td:.3f}:offset={offset:.3f}{out_label}"
                )

            filter_complex = ";".join(filter_parts)
            encoder_local, gpu_opts_local = get_gpu_encoder()

            cmd_xfade = [
                "ffmpeg", "-y",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-c:v", encoder_local, *gpu_opts_local,
                "-t", str(duration),
                "-an",
                str(output_path)
            ]

            result = subprocess.run(cmd_xfade, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"B-Roll Montage: xfade failed, falling back to concat: {result.stderr[-200:] if result.stderr else ''}")
                use_xfade = False  # Fall through to concat below

        if not use_xfade:
            # === CONCAT DEMUXER (default, no transitions) ===
            with open(concat_list_path, "w") as f:
                for seg in temp_segments:
                    f.write(f"file '{seg}'\n")

            cmd_concat = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_list_path),
                "-c", "copy",
                "-t", str(duration),
                str(output_path)
            ]

            print(f"B-Roll Montage: Concatenating {len(temp_segments)} segments (no transitions)")
            subprocess.run(cmd_concat, check=True)
            concat_list_path.unlink()

        # Clean up temp segments
        for seg in temp_segments:
            if seg.exists():
                seg.unlink()

        if not output_path.exists():
            print(f"B-Roll Montage: Output file not created")
            return None

        # === APPLY CAPTIONS AND EFFECTS TO B-ROLL ===
        if caption_data:
            segments = caption_data.get("segments", [])
            climax_clip_time = caption_data.get("climax_clip_time", 0)
            font = caption_data.get("font", "Honk")
            trigger_words = caption_data.get("trigger_words", [])
            broll_caption_style = caption_data.get("caption_style", "standard")

            # Filter segments that fall within the B-roll window
            broll_segments = []
            for seg in segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                # Check if segment overlaps with B-roll window
                if seg_end > climax_clip_time and seg_start < climax_clip_time + duration:
                    # Shift timestamps to be relative to B-roll start (0)
                    shifted_seg = seg.copy()
                    shifted_seg["start"] = max(0, seg_start - climax_clip_time)
                    shifted_seg["end"] = min(duration, seg_end - climax_clip_time)
                    # Also shift word timestamps if present
                    if "words" in shifted_seg:
                        shifted_words = []
                        for w in shifted_seg["words"]:
                            w_start = w.get("start", 0) - climax_clip_time
                            w_end = w.get("end", 0) - climax_clip_time
                            if w_end > 0 and w_start < duration:
                                shifted_w = w.copy()
                                shifted_w["start"] = max(0, w_start)
                                shifted_w["end"] = min(duration, w_end)
                                shifted_words.append(shifted_w)
                        shifted_seg["words"] = shifted_words
                    broll_segments.append(shifted_seg)

            # Filter trigger words that fall within B-roll window
            broll_triggers = []
            for tw in trigger_words:
                tw_start = tw.get("start", 0)
                tw_end = tw.get("end", 0)
                if tw_end > climax_clip_time and tw_start < climax_clip_time + duration:
                    shifted_tw = tw.copy()
                    shifted_tw["start"] = max(0, tw_start - climax_clip_time)
                    shifted_tw["end"] = min(duration, tw_end - climax_clip_time)
                    broll_triggers.append(shifted_tw)

            print(f"B-Roll Montage: {len(broll_segments)} segments, {len(broll_triggers)} triggers for captions")

            if broll_segments:
                # Generate ASS file for B-roll section
                broll_ass_path = temp_dir / f"broll_captions_{uuid.uuid4().hex[:8]}.ass"
                generate_karaoke_ass(broll_segments, broll_ass_path, 0.0, font, broll_triggers, broll_caption_style)

                # Burn ASS captions into B-roll
                temp_with_ass = temp_dir / f"broll_ass_{uuid.uuid4().hex[:8]}.mp4"
                escaped_ass = str(broll_ass_path).replace(":", "\\:").replace("'", "\\'")

                cmd_ass = [
                    "ffmpeg", "-y",
                    "-i", str(output_path),
                    "-vf", f"ass='{escaped_ass}'",
                    "-c:v", encoder, *gpu_opts,
                    "-an",  # B-roll has no audio at this point
                    str(temp_with_ass)
                ]
                print(f"B-Roll Montage: Burning ASS captions")
                result = subprocess.run(cmd_ass, capture_output=True, text=True)
                if result.returncode == 0 and temp_with_ass.exists():
                    # Replace output with captioned version
                    subprocess.run(["mv", str(temp_with_ass), str(output_path)], check=True)
                    print(f"B-Roll Montage: Captions burned successfully")
                else:
                    print(f"B-Roll Montage: ASS burn failed: {result.stderr[:200] if result.stderr else 'unknown'}")

                # Cleanup ASS file
                if broll_ass_path.exists():
                    broll_ass_path.unlink()

            # Apply RGB Glitch to B-roll (intro pulses + triggers)
            rgb_windows = []
            # Add 2 intro pulses for B-roll section
            rgb_windows.append((0.0, 0.3, 0.7))
            rgb_windows.append((0.5, 0.8, 0.7))

            # Add shifted trigger words as RGB pulses
            for tw in broll_triggers[:10]:
                tw_start = tw.get("start", 0)
                tw_end = tw.get("end", tw_start + 0.3)
                if tw_start >= 1.0:  # Skip overlap with intro pulses
                    rgb_windows.append((tw_start, tw_end, 0.5))

            if rgb_windows:
                temp_chroma = temp_dir / f"broll_chroma_{uuid.uuid4().hex[:8]}.mp4"
                try:
                    apply_chromatic_aberration(str(output_path), str(temp_chroma), rgb_windows, max_shift=25)
                    subprocess.run(["mv", str(temp_chroma), str(output_path)], check=True)
                    print(f"B-Roll Montage: RGB glitch applied ({len(rgb_windows)} pulses)")
                except Exception as e:
                    print(f"B-Roll Montage: RGB glitch failed, continuing: {e}")

            # Apply VHS Vintage Effects to B-roll
            temp_vintage = temp_dir / f"broll_vintage_{uuid.uuid4().hex[:8]}.mp4"
            try:
                apply_vintage_vhs_effects(str(output_path), str(temp_vintage), duration, rgb_trigger_windows=rgb_windows)
                subprocess.run(["mv", str(temp_vintage), str(output_path)], check=True)
                print(f"B-Roll Montage: VHS effects applied")
            except Exception as e:
                print(f"B-Roll Montage: VHS effects failed, continuing: {e}")

        print(f"B-Roll Montage: Successfully generated {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"B-Roll Montage: Failed - {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        if concat_list_path.exists():
            concat_list_path.unlink()
        return None


def apply_vintage_vhs_effects(video_path: str, output_path: str, duration: float, rgb_trigger_windows: list = None):
    """
    Apply vintage VHS effects after each RGB glitch trigger, plus a continuous VHS head error.

    Effects (triggered after RGB glitches):
    1. VHS Tracking Wrinkle - wavy horizontal distortion band rolling up screen
    2. Static Burst - TV snow/noise overlay
    3. Horizontal Tear - part of frame shifts horizontally
    4. Tape Dropout - random black horizontal lines
    5. Brightness Pump - brightness flash/surge

    Continuous effect:
    - VHS Head Error - small distortion in top right corner on 5 second loop

    Args:
        video_path: Input video file
        output_path: Output video file
        duration: Clip duration in seconds
        rgb_trigger_windows: List of (start, end, intensity) tuples from RGB glitch pass
    """
    # Schedule vintage effects to trigger 0.2-0.5s after each RGB glitch ends
    triggers = []
    if rgb_trigger_windows:
        for i, window in enumerate(rgb_trigger_windows):
            rgb_end = window[1]  # End time of RGB glitch

            # Skip intro glitches (first 4 are intro pulses at 0-1.7s)
            if rgb_end < 2.0:
                continue

            # Schedule vintage effect 0.2-0.5s after RGB glitch ends
            delay = random.uniform(0.2, 0.5)
            t_start = rgb_end + delay
            t_duration = random.uniform(0.5, 1.0)  # 0.5-1.0 second duration
            t_end = min(t_start + t_duration, duration - 2.0)

            # Make sure it fits
            if t_start < duration - 2.5 and t_end > t_start:
                # Randomly select effect type (0-4)
                effect_type = random.randint(0, 4)
                triggers.append((t_start, t_end, effect_type))

    effect_names = ["Tracking Wrinkle", "Static Burst", "Horizontal Tear", "Tape Dropout", "Brightness Pump"]
    for t_start, t_end, effect_type in triggers:
        print(f"  Vintage effect: {effect_names[effect_type]} at {t_start:.1f}s-{t_end:.1f}s")

    print(f"  VHS Head Error: continuous (random position each 1s cycle)")

    clip = VideoFileClip(video_path)

    def vintage_effect(get_frame, t):
        """Apply vintage VHS effects at trigger times + continuous head error."""
        frame = get_frame(t)

        # CONTINUOUS: VHS Head Error in top right corner (1 second loop)
        frame = _apply_vhs_head_error(frame, t, loop_seconds=1.0)

        # TRIGGERED: Vintage effects after RGB glitches
        for t_start, t_end, effect_type in triggers:
            if t_start <= t <= t_end:
                # Calculate intensity (ramp up then down)
                effect_duration = t_end - t_start
                progress = (t - t_start) / effect_duration
                # Smooth sine envelope
                intensity = np.sin(np.pi * progress)

                if effect_type == 0:
                    # VHS TRACKING WRINKLE - wavy horizontal band rolling up
                    frame = _apply_tracking_wrinkle(frame, t, t_start, intensity)
                elif effect_type == 1:
                    # STATIC BURST - TV snow overlay
                    frame = _apply_static_burst(frame, intensity)
                elif effect_type == 2:
                    # HORIZONTAL TEAR - part of frame shifts
                    frame = _apply_horizontal_tear(frame, intensity)
                elif effect_type == 3:
                    # TAPE DROPOUT - black horizontal lines
                    frame = _apply_tape_dropout(frame, intensity)
                elif effect_type == 4:
                    # BRIGHTNESS PUMP - flash/surge
                    frame = _apply_brightness_pump(frame, intensity)

                break  # Only apply one effect at a time

        return frame

    processed = clip.transform(vintage_effect)

    processed.write_videofile(
        output_path,
        codec='libx264',
        preset='ultrafast',
        audio_codec='aac',
        logger=None
    )

    clip.close()
    processed.close()
    print(f"Applied {len(triggers)} vintage VHS effects + continuous head error successfully")


def _apply_tracking_wrinkle(frame, t, t_start, intensity):
    """VHS tracking error - wavy horizontal distortion band that rolls up the screen."""
    h, w = frame.shape[:2]
    result = frame.copy()

    # Band position rolls up the screen over time
    time_offset = (t - t_start) * 0.5  # Speed of roll
    band_center = int(h * (1.0 - (time_offset % 1.0)))  # Roll from bottom to top
    band_height = int(80 * intensity)  # Band thickness

    if band_height < 5:
        return frame

    # Apply wavy distortion in the band region
    for y in range(max(0, band_center - band_height), min(h, band_center + band_height)):
        # Distance from band center (0 to 1)
        dist = abs(y - band_center) / band_height
        # Wave amplitude decreases toward edges
        wave_amp = int(15 * intensity * (1 - dist))
        # Horizontal wave offset
        wave_offset = int(wave_amp * np.sin(y * 0.1 + t * 10))

        if wave_offset != 0:
            # Shift this row horizontally
            if wave_offset > 0:
                result[y, wave_offset:, :] = frame[y, :-wave_offset, :]
                result[y, :wave_offset, :] = frame[y, 0:1, :]  # Fill edge
            else:
                result[y, :wave_offset, :] = frame[y, -wave_offset:, :]
                result[y, wave_offset:, :] = frame[y, -1:, :]  # Fill edge

    return result


def _apply_static_burst(frame, intensity):
    """TV static/snow noise overlay."""
    h, w = frame.shape[:2]

    # Generate noise
    noise_intensity = int(80 * intensity)
    noise = np.random.randint(-noise_intensity, noise_intensity + 1, (h, w, 3), dtype=np.int16)

    # Blend noise with frame
    result = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add some scanline-like horizontal variation
    for y in range(0, h, 2):
        if np.random.random() < 0.3 * intensity:
            brightness_shift = np.random.randint(-30, 30)
            result[y] = np.clip(result[y].astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)

    return result


def _apply_horizontal_tear(frame, intensity):
    """Horizontal tear/glitch - part of frame shifts horizontally."""
    h, w = frame.shape[:2]
    result = frame.copy()

    # Random tear position and size
    tear_y = int(h * (0.3 + 0.4 * np.random.random()))  # Middle 40% of frame
    tear_height = int(h * 0.1 * intensity)  # 10% of frame height max
    shift = int(50 * intensity * (1 if np.random.random() > 0.5 else -1))

    if abs(shift) < 3 or tear_height < 5:
        return frame

    # Shift the tear region
    y_start = max(0, tear_y - tear_height // 2)
    y_end = min(h, tear_y + tear_height // 2)

    if shift > 0:
        result[y_start:y_end, shift:, :] = frame[y_start:y_end, :-shift, :]
        result[y_start:y_end, :shift, :] = 0  # Black edge
    else:
        result[y_start:y_end, :shift, :] = frame[y_start:y_end, -shift:, :]
        result[y_start:y_end, shift:, :] = 0  # Black edge

    return result


def _apply_tape_dropout(frame, intensity):
    """Tape dropout - random thin black horizontal lines."""
    h, w = frame.shape[:2]
    result = frame.copy()

    # Number of dropout lines
    num_lines = int(5 + 10 * intensity)

    for _ in range(num_lines):
        if np.random.random() < intensity:
            y = np.random.randint(0, h)
            line_height = np.random.randint(1, 4)
            line_width = int(w * (0.3 + 0.7 * np.random.random()))  # 30-100% width
            x_start = np.random.randint(0, max(1, w - line_width))

            y_end = min(h, y + line_height)
            x_end = min(w, x_start + line_width)

            # Black or white dropout
            if np.random.random() > 0.7:
                result[y:y_end, x_start:x_end, :] = 255  # White speckle
            else:
                result[y:y_end, x_start:x_end, :] = 0  # Black dropout

    return result


def _apply_brightness_pump(frame, intensity):
    """Brightness pump/flash - sudden brightness surge."""
    # Increase brightness based on intensity
    brightness_boost = int(60 * intensity)

    result = np.clip(frame.astype(np.int16) + brightness_boost, 0, 255).astype(np.uint8)

    # Also slightly desaturate during flash (move toward white)
    if intensity > 0.5:
        gray = np.mean(result, axis=2, keepdims=True)
        blend = (intensity - 0.5) * 0.4  # 0-20% desaturation
        result = np.clip(result * (1 - blend) + gray * blend, 0, 255).astype(np.uint8)

    return result


def _apply_vhs_head_error(frame, t, loop_seconds: float = 5.0):
    """
    Apply a small VHS head switching error at random positions.
    Loops on a configurable cycle (default 5 seconds).
    Position changes each cycle but stays consistent within a cycle.

    The effect creates a small distortion area that mimics VHS head
    switching artifacts - horizontal noise bars and color distortion.
    """
    h, w = frame.shape[:2]
    result = frame.copy()

    # Calculate which loop cycle we're in and position within it
    loop_cycle = int(t // loop_seconds)
    loop_pos = t % loop_seconds  # 0 to 5

    # The head error appears for ~0.3s every 5 seconds
    # Error window: 2.5s to 2.8s in the loop (mid-cycle)
    error_start = loop_seconds / 2  # 2.5s
    error_duration = 0.3

    if error_start <= loop_pos <= error_start + error_duration:
        # Calculate intensity (fade in/out within the 0.3s window)
        progress = (loop_pos - error_start) / error_duration
        intensity = np.sin(np.pi * progress)

        # Region size (small: ~150x80 pixels)
        region_w = 150
        region_h = 80
        margin = 30  # Margin from edges

        # Use loop_cycle as seed for consistent position within each cycle
        rng = np.random.RandomState(seed=loop_cycle + 42)
        x_start = rng.randint(margin, w - region_w - margin)
        y_start = rng.randint(margin, h - region_h - margin)
        x_end = x_start + region_w
        y_end = y_start + region_h

        # Apply distortion effects to the region
        region = result[y_start:y_end, x_start:x_end, :].copy()

        # Effect 1: Horizontal noise bars
        num_bars = int(3 + 5 * intensity)
        for _ in range(num_bars):
            bar_y = np.random.randint(0, region_h)
            bar_height = np.random.randint(1, 3)
            bar_y_end = min(bar_y + bar_height, region_h)

            # Random horizontal shift for this bar
            shift = int(np.random.randint(-10, 10) * intensity)
            if shift != 0:
                if shift > 0:
                    region[bar_y:bar_y_end, shift:, :] = region[bar_y:bar_y_end, :-shift, :]
                    region[bar_y:bar_y_end, :shift, :] = 0
                else:
                    region[bar_y:bar_y_end, :shift, :] = region[bar_y:bar_y_end, -shift:, :]
                    region[bar_y:bar_y_end, shift:, :] = 0

        # Effect 2: Color channel bleeding (slight RGB shift)
        if intensity > 0.3:
            small_shift = int(2 * intensity)
            if small_shift > 0:
                # Shift red channel slightly
                r = region[:, :, 0].copy()
                region[:, small_shift:, 0] = r[:, :-small_shift]

        # Effect 3: Add noise speckles
        noise_mask = np.random.random((region_h, region_w)) < (0.1 * intensity)
        noise_values = np.random.randint(0, 255, (region_h, region_w, 3), dtype=np.uint8)
        for c in range(3):
            region[:, :, c] = np.where(noise_mask, noise_values[:, :, c], region[:, :, c])

        # Effect 4: Slight brightness flicker
        brightness_shift = int(20 * intensity * (np.random.random() - 0.5))
        region = np.clip(region.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)

        # Apply the distorted region back
        result[y_start:y_end, x_start:x_end, :] = region

    return result


# ============ NEW TEMPLATE EFFECT FUNCTIONS ============

def apply_letterbox(video_path: str, output_path: str, ratio: float = 2.35):
    """
    Add cinematic letterbox (black bars) to video.
    Ratio 2.35 = CinemaScope, 1.85 = Academy Flat
    """
    # Calculate bar heights for 1080x1920 (9:16) to simulate wider aspect
    # The visible area should have the specified ratio
    target_h = int(1080 / ratio)  # Width / ratio = height
    bar_h = (1920 - target_h) // 2

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"pad=1080:1920:0:{bar_h}:black",
        "-c:v", "h264_nvenc", "-preset", "p4",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Applied letterbox (ratio {ratio}, bars {bar_h}px)")


def apply_vignette_effect(video_path: str, output_path: str, intensity: float = 0.3):
    """
    Add a vignette (darkened edges) effect using FFmpeg vignette filter.
    Intensity 0.0 = no vignette, 1.0 = strong dark edges
    """
    # vignette filter: angle PI/2 is circular, a smaller angle makes it more elliptical
    # mode=forward makes it darker at edges
    angle = 0.5 + (0.3 * intensity)  # 0.5 to 0.8 based on intensity
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"vignette=angle={angle}:mode=forward",
        "-c:v", "h264_nvenc", "-preset", "p4",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Applied vignette (intensity {intensity})")


def apply_scan_lines(frame: np.ndarray, opacity: float = 0.1) -> np.ndarray:
    """
    Add CRT-style scan lines overlay to a frame.
    Called per-frame in MoviePy transform.
    """
    h, w = frame.shape[:2]
    result = frame.copy().astype(np.float32)

    # Create scan line pattern (every other row darker)
    for y in range(0, h, 2):
        result[y] = result[y] * (1 - opacity)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_film_grain_frame(frame: np.ndarray, intensity: float = 0.15) -> np.ndarray:
    """
    Add cinematic film grain to a frame (softer than VHS noise).
    Called per-frame in MoviePy transform.
    """
    h, w = frame.shape[:2]

    # Generate organic-looking grain (Gaussian noise)
    grain = np.random.normal(0, 25 * intensity, (h, w)).astype(np.float32)

    # Apply grain to all channels equally
    result = frame.astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] + grain

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_flash_frames(video_path: str, output_path: str, trigger_times: list,
                       flash_color: str = "#FFFFFF", flash_duration: float = 0.08):
    """
    Insert flash frames at trigger times using MoviePy.
    trigger_times: list of timestamps in seconds
    flash_color: hex color for flash (default white)
    """
    # Convert hex to RGB
    color = tuple(int(flash_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    clip = VideoFileClip(video_path)

    def add_flash(get_frame, t):
        frame = get_frame(t)

        # Check if we're within flash duration of any trigger
        for trigger_t in trigger_times:
            if trigger_t <= t < trigger_t + flash_duration:
                # Blend flash color with frame
                progress = (t - trigger_t) / flash_duration
                # Quick flash: 100% at start, fade to 0%
                blend = 1.0 - progress
                flash_frame = np.full_like(frame, color)
                result = (frame * (1 - blend) + flash_frame * blend).astype(np.uint8)
                return result

        return frame

    processed = clip.transform(add_flash)
    processed.write_videofile(
        output_path,
        codec='libx264',
        preset='ultrafast',
        audio_codec='aac',
        logger=None
    )
    clip.close()
    processed.close()
    print(f"Applied {len(trigger_times)} flash frames")


def apply_velocity_effect(video_path: str, output_path: str, trigger_times: list,
                         speed_multiplier: float = 2.0, ramp_duration: float = 0.15):
    """
    Apply speed ramping on trigger words.
    Speeds up to speed_multiplier then ramps back down.
    This is done via PTS manipulation in FFmpeg.

    Note: This is complex with variable speed, so we use a simplified approach:
    Just apply the visual flash/punch effect at trigger times instead.
    True velocity editing requires re-timing audio which is complex.
    """
    # For now, velocity effect is simulated via the aggressive pulse zoom
    # and flash frames which give the perception of speed change
    print(f"Velocity effect: {len(trigger_times)} trigger points (simulated via pulse/flash)")
    subprocess.run(["cp", video_path, output_path], check=True)


def apply_whip_pan_transition(frame: np.ndarray, t: float, transition_times: list,
                              whip_duration: float = 0.15, direction: str = "right") -> np.ndarray:
    """
    Apply horizontal whip pan motion blur effect during transitions.
    Called per-frame in MoviePy transform.
    """
    h, w = frame.shape[:2]

    for trans_t in transition_times:
        if trans_t <= t < trans_t + whip_duration:
            progress = (t - trans_t) / whip_duration
            # Intensity peaks in middle of whip
            intensity = np.sin(np.pi * progress)

            # Apply horizontal motion blur
            blur_amount = int(50 * intensity)
            if blur_amount > 0:
                result = frame.copy()
                # Simple horizontal smear
                kernel_size = blur_amount * 2 + 1
                # Create horizontal kernel
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size//2, :] = 1.0 / kernel_size

                from scipy.ndimage import convolve
                for c in range(3):
                    result[:, :, c] = convolve(frame[:, :, c], kernel, mode='reflect')

                return result

    return frame


def apply_combined_template_effects(video_path: str, output_path: str, effect_config: dict,
                                    trigger_words: list, duration: float):
    """
    Apply all template-specific visual effects in a single MoviePy pass.
    This consolidates letterbox, vignette, scan_lines, film_grain, and flash effects.
    """
    clip = VideoFileClip(video_path)

    # Pre-compute trigger times for flash effect
    flash_times = []
    if effect_config.get("flash_enabled") and trigger_words:
        for tw in trigger_words[:15]:  # Limit flashes
            flash_times.append(tw.get("start", 0))

    # Pre-compute whip pan times (on every other trigger)
    whip_times = []
    if effect_config.get("whip_pan") and trigger_words:
        for i, tw in enumerate(trigger_words[:10]):
            if i % 2 == 0:  # Every other trigger
                whip_times.append(tw.get("start", 0))

    letterbox_enabled = effect_config.get("letterbox", False)
    letterbox_ratio = effect_config.get("letterbox_ratio", 2.35)
    vignette_enabled = effect_config.get("vignette", False)
    vignette_intensity = effect_config.get("vignette_intensity", 0.3)
    scan_lines_enabled = effect_config.get("scan_lines", False)
    scan_line_opacity = effect_config.get("scan_line_opacity", 0.1)
    film_grain_enabled = effect_config.get("film_grain", False)
    grain_intensity = effect_config.get("grain_intensity", 15) / 100.0
    flash_enabled = effect_config.get("flash_enabled", False)
    flash_color = effect_config.get("flash_color", "#FFFFFF")
    flash_duration = 0.08
    whip_enabled = effect_config.get("whip_pan", False)
    cross_overlay = effect_config.get("cross_overlay", False)

    # Convert flash color hex to RGB tuple
    try:
        flash_rgb = tuple(int(flash_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    except:
        flash_rgb = (255, 255, 255)

    # Calculate letterbox bar height
    bar_h = 0
    if letterbox_enabled:
        target_h = int(1080 / letterbox_ratio)
        bar_h = (1920 - target_h) // 2

    def combined_effect(get_frame, t):
        frame = get_frame(t)
        h, w = frame.shape[:2]

        # 1. SCAN LINES (subtle CRT effect)
        if scan_lines_enabled:
            frame = apply_scan_lines(frame, scan_line_opacity)

        # 2. FILM GRAIN (cinematic noise)
        if film_grain_enabled:
            frame = apply_film_grain_frame(frame, grain_intensity)

        # 3. VIGNETTE (darkened edges)
        if vignette_enabled:
            # Simple radial vignette
            y, x = np.ogrid[:h, :w]
            cy, cx = h / 2, w / 2
            r = np.sqrt((x - cx)**2 / (cx**2) + (y - cy)**2 / (cy**2))
            vignette_mask = np.clip(1 - r * vignette_intensity, 0.3, 1.0)
            frame = (frame * vignette_mask[:, :, np.newaxis]).astype(np.uint8)

        # 4. FLASH FRAMES (white/colored flash on triggers)
        if flash_enabled and flash_times:
            for flash_t in flash_times:
                if flash_t <= t < flash_t + flash_duration:
                    progress = (t - flash_t) / flash_duration
                    blend = 1.0 - progress  # Quick fade
                    flash_frame = np.full_like(frame, flash_rgb)
                    frame = (frame * (1 - blend) + flash_frame * blend).astype(np.uint8)
                    break

        # 5. WHIP PAN (motion blur on transitions)
        if whip_enabled and whip_times:
            frame = apply_whip_pan_transition(frame, t, whip_times)

        # 6. LETTERBOX (black bars)
        if letterbox_enabled and bar_h > 0:
            frame[:bar_h, :, :] = 0  # Top bar
            frame[-bar_h:, :, :] = 0  # Bottom bar

        # 7. CROSS OVERLAY (for Crusade template)
        if cross_overlay:
            # Add subtle cross watermark in corner (gold color)
            cross_size = 60
            cx, cy = w - 80, 100  # Top right corner
            # Vertical bar
            frame[cy-cross_size//2:cy+cross_size//2, cx-5:cx+5, :] = [0, 215, 255]  # Gold BGR
            # Horizontal bar (Chi Rho style - shorter)
            frame[cy-5:cy+5, cx-cross_size//3:cx+cross_size//3, :] = [0, 215, 255]

        return frame

    processed = clip.transform(combined_effect)

    processed.write_videofile(
        output_path,
        codec='libx264',
        preset='ultrafast',
        audio_codec='aac',
        logger=None
    )

    clip.close()
    processed.close()

    effects_applied = []
    if letterbox_enabled: effects_applied.append("letterbox")
    if vignette_enabled: effects_applied.append("vignette")
    if scan_lines_enabled: effects_applied.append("scan_lines")
    if film_grain_enabled: effects_applied.append("film_grain")
    if flash_enabled: effects_applied.append(f"flash({len(flash_times)})")
    if whip_enabled: effects_applied.append(f"whip_pan({len(whip_times)})")
    if cross_overlay: effects_applied.append("cross_overlay")

    print(f"Applied template effects: {', '.join(effects_applied) if effects_applied else 'none'}")


def generate_karaoke_ass(segments: list[dict], output_path: Path, start_offset: float = 0.0, font: str = "Honk", trigger_words: list = None, caption_style: str = "standard"):
    """
    Generate ASS karaoke captions with enhanced trigger word effects.

    caption_style options:
    - "standard": trigger words get 3x scale + cyan color + border (default)
    - "pop_scale": ALL words bounce in with overshoot scale, triggers get bigger
    - "shake": trigger words vibrate/shake with position offsets
    - "blur_reveal": words sharpen from blur as spoken, triggers get extra blur + scale
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
            if w_start_rel < t_end and w_end_rel > t_start:
                if t_word in w_lower or w_lower in t_word:
                    return True
            if t_word and (t_word in w_lower or w_lower in t_word):
                return True
        return False

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{font},149,&H00FFFFFF,&H00FF00FF,&H00000000,&HAA000000,-1,0,0,0,100,100,2,0,1,5,2,2,20,20,120,1
Style: TriggerWord,{font},149,&H00FFFF00,&H00FF00FF,&H000000FF,&HAA000000,-1,0,0,0,300,300,2,0,1,8,3,2,20,20,120,1

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

    def build_word_fx(w_text, dur_cs, is_trigger, style):
        """Build ASS override tags for a word based on caption_style."""
        dur_ms = dur_cs * 10

        if style == "pop_scale":
            # Pop scale: words bounce in with overshoot
            if is_trigger:
                # Trigger: 4x overshoot → 3x settle + cyan color
                fx = f"\\fscx400\\fscy400\\c&H00FFFF&\\bord6\\t(0,{min(dur_ms, 100)},\\fscx350\\fscy350)\\t({min(dur_ms, 100)},{min(dur_ms, 250)},\\fscx300\\fscy300)"
                return f"{{\\k{dur_cs}{fx}}}{w_text.upper()} {{\\r}}"
            else:
                # Normal: 1.5x overshoot → 1x settle (bouncy appear)
                fx = f"\\fscx150\\fscy150\\t(0,{min(dur_ms, 100)},\\fscx110\\fscy110)\\t({min(dur_ms, 100)},{min(dur_ms, 200)},\\fscx100\\fscy100)"
                return f"{{\\k{dur_cs}{fx}}}{w_text} "

        elif style == "shake":
            # Shake: trigger words vibrate with position and rotation
            if is_trigger:
                # Rapid shake via alternating frz (rotation) + scale
                fx = f"\\fscx300\\fscy300\\c&H00FFFF&\\bord6\\shad3\\frz-3\\t(0,50,\\frz3)\\t(50,100,\\frz-2)\\t(100,150,\\frz2)\\t(150,{dur_ms},\\frz0\\fscx100\\fscy100\\c&HFFFFFF&)"
                return f"{{\\k{dur_cs}{fx}}}{w_text.upper()} {{\\r}}"
            else:
                return f"{{\\k{dur_cs}}}{w_text} "

        elif style == "blur_reveal":
            # Blur reveal: words sharpen from blur as they're spoken
            if is_trigger:
                # Heavy blur + scale → sharp + normal
                fx = f"\\blur12\\fscx250\\fscy250\\c&H00FFFF&\\bord6\\t(0,{min(dur_ms, 200)},\\blur0\\fscx100\\fscy100\\c&HFFFFFF&)"
                return f"{{\\k{dur_cs}{fx}}}{w_text.upper()} {{\\r}}"
            else:
                # Light blur → sharp
                fx = f"\\blur8\\t(0,{min(dur_ms, 200)},\\blur0)"
                return f"{{\\k{dur_cs}{fx}}}{w_text} "

        else:
            # Standard (original behavior)
            if is_trigger:
                trigger_fx = f"\\fscx300\\fscy300\\c&H00FFFF&\\bord6\\shad3\\t(0,{dur_ms},\\fscx100\\fscy100\\c&HFFFFFF&)"
                return f"{{\\k{dur_cs}{trigger_fx}}}{w_text.upper()} {{\\r}}"
            else:
                return f"{{\\k{dur_cs}}}{w_text} "

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

            is_trigger = is_trigger_word(w_start, w_end, w_text)
            karaoke += build_word_fx(w_text, dur_cs, is_trigger, caption_style)

            current_time = w_end

        # Line-level blur for blur_reveal style
        line_blur = "\\blur2" if caption_style != "blur_reveal" else ""
        line_entry = f"Dialogue: 0,{fmt_time(s_start)},{fmt_time(s_end)},Karaoke,,0,0,0,,{{{line_blur}}}{karaoke}"
        events.append(line_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))

    print(f"Generated karaoke ASS ({caption_style}) with {len(trigger_windows)} trigger word enhancements")


# ============ ENDPOINTS ============

@app.get("/health")
async def health_check():
    return {"status": "healthy_v2", "gpu_encoder": get_gpu_encoder()[0]}

@app.get("/effects")
async def get_effects_catalog():
    """Return available effects catalog with descriptions for Grok director prompt."""
    # Check which LUTs are actually present on disk
    available_luts = {}
    for name, info in LUT_PRESETS.items():
        lut_path = os.path.join(LUT_DIR, info["file"])
        available_luts[name] = {
            "file": info["file"],
            "description": info["description"],
            "available": os.path.exists(lut_path)
        }
    return {
        "effects": EFFECT_REGISTRY,
        "lut_presets": available_luts,
        "color_presets": COLOR_PRESETS,
        "caption_styles": ["standard", "pop_scale", "shake", "blur_reveal"],
        "transition_types": ["pixelize", "radial", "dissolve", "slideleft", "fadeblack", "wiperight"],
    }

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
        print(f"DEBUG DURATION: request.start_time={request.start_time}, request.end_time={request.end_time}, calculated duration={duration}")

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

        # 2.5. FACE DETECTION FOR AUTO-CENTERING
        # Detect face position in first frame to calculate crop offset
        report_status(request.status_webhook_url, "Processing: Face Detection")
        face_offset = detect_face_offset(str(temp_extracted))
        print(f"[FaceDetect] Using crop offset: {face_offset}px (0 = center)")

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
                raw_word = word_obj.get("word", "")
                # Aggressive cleaning: lowercase, strip whitespace, remove ALL punctuation
                w_text = raw_word.lower().strip()
                # Remove all punctuation including unicode variants
                w_text = re.sub(r"[^\w\s]", "", w_text)  # Keep only letters, numbers, spaces
                w_text = re.sub(r"'s$|'s$", "", w_text)  # Remove possessive (both quote types)
                w_text = w_text.strip()

                if w_text in TRAD_TRIGGER_WORDS:
                    print(f"  TRIGGER HIT: '{w_text}' (raw: '{raw_word.strip()}') at {word_obj.get('start', 0):.2f}s")
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

        # Font Selection - "random" picks from installed custom fonts
        selected_font = request.font
        if request.font.lower() == "random":
            available_fonts = get_installed_fonts()
            selected_font = random.choice(available_fonts)
            print(f"DEBUG: Random font selected: {selected_font} (from {len(available_fonts)} available)")

        # 5. Karaoke ASS Generation with FRESH timestamps (offset=0 since clip starts at 0)
        # Determine caption style from request or effect_settings
        caption_style = request.caption_style or "standard"
        if request.effect_settings and request.effect_settings.get("caption_style"):
            caption_style = request.effect_settings["caption_style"]
        report_status(request.status_webhook_url, f"Processing: Karaoke ({selected_font}, {caption_style})")
        generate_karaoke_ass(fresh_segments, ass_path, 0.0, selected_font, fresh_trigger_words, caption_style)

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

        # Beat-synced pulse from background music (bass drum only)
        # Read beat_sync intensity from effect_settings early (before get_effect_chain)
        beat_intensity = 0.006  # Default barely perceptible
        if request.effect_settings and request.effect_settings.get("beat_sync"):
            beat_intensity = 0.02  # Amplified when explicitly enabled by Grok director
        beat_pulse = ""
        if bgm_path:
            try:
                beat_times = detect_beats(str(bgm_path), duration)
                print(f"Detected {len(beat_times)} bass drum hits in BGM for pulse sync (intensity={beat_intensity})")
                for bt in beat_times[:60]:  # Limit to avoid FFmpeg expr overflow
                    attack = 0.02   # Instant snap in
                    decay = 0.20    # Smooth ease out
                    peak = bt + attack
                    end = peak + decay
                    beat_pulse += f"+{beat_intensity}*between(t,{bt:.3f},{peak:.3f})*((t-{bt:.3f})/{attack})"
                    beat_pulse += f"+{beat_intensity}*between(t,{peak:.3f},{end:.3f})*(1-((t-{peak:.3f})/{decay}))"
            except Exception as e:
                print(f"Beat detection failed, using fallback heartbeat: {e}")
                beat_pulse = "+0.003*sin(2*3.14159*t*2.0)"  # Very subtle 2Hz fallback

        # Pulse Expr (CPU Dynamic)
        zoom_base = "min(1+0.001*t,1.5)"
        # Use beat pulse if available, otherwise very subtle sine fallback
        heartbeat = beat_pulse if beat_pulse else "+0.003*sin(2*3.14159*t*2.0)"
        
        # Combined Scale Factor
        scale_factor = f"({zoom_base}+{heartbeat}{trigger_sum})"
        
        # We vary Width(w) and Height(h) using the expression.
        # Force even dimensions (2-aligned) for YUV420P / CUDA compatibility
        # Ensure factor >= 1 using max(1, ...) to prevent crop failure
        zoom_expr = f"w='2*trunc(iw*max(1,{scale_factor})/2)':h='2*trunc(ih*max(1,{scale_factor})/2)'"
        
        # ============ TEMPLATE EFFECT CONFIGURATION ============
        # Use effect dispatcher to parse template settings into rendering config
        effect_settings = request.effect_settings or {}
        effect_config = get_effect_chain(effect_settings)

        # Extract color grading parameters from config
        saturation = effect_config["saturation"]
        contrast = effect_config["contrast"]
        brightness = effect_config["brightness"]
        pulse_intensity_base = effect_config["pulse_intensity"]

        # Print template configuration for debugging
        effects_summary = []
        if effect_config.get("lut_file"): effects_summary.append(f"LUT({os.path.basename(effect_config['lut_file'])})")
        if effect_config.get("retro_glow"): effects_summary.append(f"retro_glow({effect_config.get('retro_glow_intensity', 0.3)})")
        if effect_config.get("temporal_trail"): effects_summary.append("temporal_trail")
        if effect_config.get("camera_shake"): effects_summary.append(f"shake({effect_config['shake_intensity']}px)")
        if effect_config.get("wave_displacement"): effects_summary.append("wave_disp")
        if effect_config.get("speed_ramps"): effects_summary.append(f"speed_ramps({len(effect_config['speed_ramps'])})")
        if effect_config.get("audio_saturation"): effects_summary.append("audio_sat")
        if effect_config.get("broll_transition_type"): effects_summary.append(f"transition({effect_config['broll_transition_type']})")
        if effect_config["letterbox"]: effects_summary.append(f"letterbox({effect_config['letterbox_ratio']})")
        if effect_config["vignette"]: effects_summary.append("vignette")
        if effect_config["film_grain"]: effects_summary.append("film_grain")
        if effect_config["scan_lines"]: effects_summary.append("scan_lines")
        if effect_config["flash_enabled"]: effects_summary.append("flash")
        if effect_config["heavy_glitch"]: effects_summary.append("heavy_glitch")
        if effect_config["whip_pan"]: effects_summary.append("whip_pan")
        if effect_config["cross_overlay"]: effects_summary.append("cross_overlay")
        if not effect_config["vhs_enabled"]: effects_summary.append("no_vhs")
        if saturation == 0: effects_summary.append("B&W")

        print(f"Template Effects: {', '.join(effects_summary) if effects_summary else 'default'}")
        print(f"  Color: contrast={contrast}, brightness={brightness}, saturation={saturation}")
        print(f"  Pulse intensity: {pulse_intensity_base}")

        # === COLOR GRADING: LUT or EQ ===
        lut_file = effect_config.get("lut_file")
        if lut_file and os.path.exists(lut_file):
            escaped_lut = lut_file.replace(":", "\\:").replace("'", "\\'")
            grade_filter = f"lut3d=file='{escaped_lut}':interp=tetrahedral"
            print(f"[Effects] Using LUT grade: {lut_file}")
        else:
            grade_filter = f"eq=contrast={contrast}:brightness={brightness}:saturation={saturation}"

        # VHS Grain Filter - conditionally disabled for "clean" templates
        grain_intensity = effect_config["grain_intensity"]
        vhs_grain_enabled = effect_config["vhs_enabled"]
        vhs_grain = f"noise=alls={grain_intensity}:allf=t+u" if vhs_grain_enabled else ""

        # === CAMERA SHAKE: offset crop position with sine waves ===
        crop_x_offset = face_offset if face_offset != 0 else 0
        shake_x_expr = ""
        shake_y_expr = ""
        if effect_config.get("camera_shake"):
            si = effect_config["shake_intensity"]
            sf = effect_config["shake_frequency"]
            # Multi-frequency sine for natural shake (not simple oscillation)
            shake_x_expr = f"+{si}*(0.5*sin(2*PI*{sf}*t+0.7)+0.3*sin(2*PI*{sf}*2.7*t+1.3)+0.2*sin(2*PI*{sf}*4.1*t+2.1))"
            shake_y_expr = f"+{si}*0.5*(0.5*sin(2*PI*{sf}*1.1*t+3.2)+0.3*sin(2*PI*{sf}*2.3*t+0.5))"
            print(f"[Effects] Camera shake: intensity={si}px, freq={sf}Hz")

        # === RETRO GLOW: neon bloom effect ===
        retro_glow_stage = ""
        if effect_config.get("retro_glow"):
            glow_int = effect_config.get("retro_glow_intensity", 0.3)
            # Split → blur → blend screen (creates soft glow around bright areas)
            retro_glow_stage = f"split[rgmain][rgglow];[rgglow]gblur=sigma=20,curves=all='0/0 0.5/0.6 1/1'[rgglowed];[rgmain][rgglowed]blend=all_mode=screen:all_opacity={glow_int},"
            print(f"[Effects] Retro glow: intensity={glow_int}")

        # === TEMPORAL TRAIL: motion ghosting ===
        temporal_trail_stage = ""
        if effect_config.get("temporal_trail"):
            trail_segments = effect_config.get("temporal_trail_segments", [])
            if trail_segments:
                # Build enable expression for specified time ranges
                enables = [f"between(t,{s[0]},{s[1]})" for s in trail_segments[:5]]
                enable_expr = "+".join(enables)
                temporal_trail_stage = f"tmix=frames=5:weights='1 0.8 0.6 0.4 0.2':enable='{enable_expr}',"
                print(f"[Effects] Temporal trail: {len(trail_segments)} segments")
            else:
                # Apply to entire clip if no segments specified
                temporal_trail_stage = "tmix=frames=5:weights='1 0.8 0.6 0.4 0.2',"
                print(f"[Effects] Temporal trail: full clip")

        # === WAVE DISPLACEMENT: time-gated row-based sine distortion ===
        wave_stage = ""
        if effect_config.get("wave_displacement") and effect_config.get("wave_triggers"):
            wave_triggers = effect_config["wave_triggers"][:3]  # Max 3 bursts
            # Build enable expression (geq only processes during trigger windows)
            enables = []
            max_amp = 15
            for wt in wave_triggers:
                s = wt.get("start", 0)
                e = wt.get("end", s + 1.0)
                amp = wt.get("amplitude", 15)
                max_amp = max(max_amp, amp)
                enables.append(f"between(t,{s:.2f},{e:.2f})")
            enable_expr = "+".join(enables)
            # Y-dependent horizontal sine displacement (rows shift left/right)
            # freq=120 pixels per wave cycle, speed=6 radians/sec for animation
            wave_lum = f"lum(clip(X+{max_amp}*sin(2*3.14159*Y/120+T*6),0,W-1),Y)"
            wave_cb = f"cb(clip(X/2+{max_amp}/2*sin(2*3.14159*Y/240+T*6),0,W/2-1),Y/2)"
            wave_cr = f"cr(clip(X/2+{max_amp}/2*sin(2*3.14159*Y/240+T*6),0,W/2-1),Y/2)"
            wave_stage = f"geq=lum='{wave_lum}':cb='{wave_cb}':cr='{wave_cr}':enable='{enable_expr}',"
            print(f"[Effects] Wave displacement: {len(wave_triggers)} bursts, amplitude={max_amp}px")

        # === AUDIO-REACTIVE SATURATION: boost color on loud moments ===
        audio_sat_stage = ""
        if effect_config.get("audio_saturation"):
            try:
                onset_peaks = detect_onset_peaks(str(temp_extracted), duration)
                if onset_peaks:
                    sat_boost = 1.0 + effect_config.get("audio_saturation_amount", 0.3)
                    # Build time-gated enable expression for onset windows (0.2s each)
                    enables = [f"between(t,{t:.2f},{t+0.2:.2f})" for t in onset_peaks]
                    enable_expr = "+".join(enables)
                    audio_sat_stage = f"eq=saturation={sat_boost:.2f}:enable='{enable_expr}',"
                    print(f"[Effects] Audio-reactive saturation: {len(onset_peaks)} peaks, boost={sat_boost:.1f}x")
            except Exception as e:
                print(f"[Effects] Audio saturation failed: {e}")

        # Escape path for ASS filter
        escaped_ass = str(ass_path).replace(":", "\\:").replace("'", "\\'")

        # === BUILD MEGA-FILTER CHAIN ===
        # Pipeline: hwdownload → scale(fill) → scale(pulse) → crop(face+shake) →
        #           [LUT/eq grade] → [audio_sat] → [retro_glow] → [temporal_trail] → [wave] → [vhs_grain] → ass → hwupload
        vhs_grain_stage = f"{vhs_grain}," if vhs_grain else ""

        # Crop expression with optional shake
        crop_x = f"(iw-1080)/2+{crop_x_offset}{shake_x_expr}"
        crop_y = f"(ih-1920)/2{shake_y_expr}"

        filter_complex_effects = (
            f"[0:v]hwdownload,format=nv12,format=yuv420p,"
            f"scale=-2:1920,"
            f"scale={zoom_expr}:eval=frame,"
            f"crop=1080:1920:{crop_x}:{crop_y},"
            f"{grade_filter},"
            f"{audio_sat_stage}"
            f"{retro_glow_stage}"
            f"{temporal_trail_stage}"
            f"{wave_stage}"
            f"{vhs_grain_stage}"
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

        # 3.25. SPEED RAMPS - retime segments after all effects are baked
        speed_ramps = effect_config.get("speed_ramps", []) or request.speed_ramps
        if speed_ramps:
            report_status(request.status_webhook_url, "Processing: Speed Ramps")
            temp_ramped = temp_dir / f"ramped_{uuid.uuid4().hex[:8]}.mp4"
            if apply_speed_ramps(temp_main, temp_ramped, speed_ramps, encoder, gpu_opts):
                # Replace temp_main with ramped version
                temp_main.unlink()
                subprocess.run(["mv", str(temp_ramped), str(temp_main)], check=True)
            elif temp_ramped.exists():
                temp_ramped.unlink()

        # 3.5. Chromatic Aberration Pass (MoviePy) - RGB split effect
        # ALWAYS add 4 intro pulses at the start, plus BIG keyword trigger hits
        report_status(request.status_webhook_url, "Processing: RGB Glitch")

        trigger_windows = []

        # INTRO PULSES: Four RGB glitch hits at the very start (full intensity)
        # Creates that signature "glitchy intro" viral clip look
        trigger_windows.append((0.0, 0.3, 1.0))    # First intro pulse
        trigger_windows.append((0.4, 0.7, 1.0))    # Second intro pulse
        trigger_windows.append((0.9, 1.2, 1.0))    # Third intro pulse
        trigger_windows.append((1.4, 1.7, 1.0))    # Fourth intro pulse

        # KEYWORD TRIGGERS: Add pulses for each trigger word
        # CURSE WORDS get full intensity, others get random 1/4 to 1/2
        curse_count = 0
        reduced_count = 0
        skipped_early = 0
        if fresh_trigger_words:
            for trig in fresh_trigger_words[:20]:
                # Fresh timestamps are already clip-relative (no adjustment needed)
                t_start = trig.get('start', 0)
                t_end = trig.get('end', t_start + 0.5)
                word = trig.get('word', '').lower().strip()
                t_center = (t_start + t_end) / 2
                # Wider pulse window for more impact (0.4s total instead of 0.5s)
                pulse_start = max(0, t_center - 0.2)
                pulse_end = min(duration, t_center + 0.2)
                if pulse_end > 0 and pulse_start < duration:
                    # Avoid overlap with intro pulses (now 4 pulses until ~1.7s)
                    if pulse_start >= 2.0:
                        # CURSE WORDS = full intensity (1.0), others = random 0.25-0.5
                        is_curse = word in CURSE_WORDS
                        if is_curse:
                            intensity = 1.0
                            curse_count += 1
                            print(f"  RGB CURSE: '{word}' at {t_start:.1f}s -> FULL intensity")
                        else:
                            intensity = random.uniform(0.25, 0.5)
                            reduced_count += 1
                            print(f"  RGB OTHER: '{word}' at {t_start:.1f}s -> {intensity:.2f} intensity")
                        trigger_windows.append((pulse_start, pulse_end, intensity))
                    else:
                        skipped_early += 1
                        print(f"  RGB SKIP (early): '{word}' at {t_start:.1f}s (< 2.0s)")
        print(f"RGB glitch: {curse_count} curse (full), {reduced_count} other (reduced), {skipped_early} skipped (early)")

        temp_chroma = temp_dir / f"temp_chroma_{uid}.mp4"
        cleanup_files.append(temp_chroma)
        try:
            # Use template-specific RGB shift amount
            rgb_shift = effect_config["rgb_max_shift"]
            print(f"RGB glitch shift: {rgb_shift}px (template config)")
            apply_chromatic_aberration(str(temp_main), str(temp_chroma), trigger_windows, max_shift=rgb_shift)
            # Replace temp_main with chromatic version
            subprocess.run(["mv", str(temp_chroma), str(temp_main)], check=True)
            print(f"Applied {len(trigger_windows)} RGB glitch pulses (4 intro + {len(trigger_windows)-4} keywords)")
        except Exception as e:
            print(f"WARNING: Chromatic aberration failed, continuing without: {e}")
            # Continue with unmodified temp_main

        # 3.6. Vintage VHS Effects Pass - conditionally applied based on template
        if effect_config["vhs_enabled"]:
            report_status(request.status_webhook_url, "Processing: Vintage FX")
            temp_vintage = temp_dir / f"temp_vintage_{uid}.mp4"
            cleanup_files.append(temp_vintage)
            try:
                apply_vintage_vhs_effects(str(temp_main), str(temp_vintage), duration, rgb_trigger_windows=trigger_windows)
                # Replace temp_main with vintage effects version
                subprocess.run(["mv", str(temp_vintage), str(temp_main)], check=True)
            except Exception as e:
                print(f"WARNING: Vintage VHS effects failed, continuing without: {e}")
                # Continue with unmodified temp_main
        else:
            print("VHS effects: DISABLED (clean template)")

        # 3.6.2 TEMPLATE-SPECIFIC EFFECTS PASS (letterbox, vignette, film_grain, flash, whip_pan, etc.)
        # Only run if any of these effects are enabled
        has_template_effects = (
            effect_config["letterbox"] or effect_config["vignette"] or
            effect_config["film_grain"] or effect_config["scan_lines"] or
            effect_config["flash_enabled"] or effect_config["whip_pan"] or
            effect_config["cross_overlay"]
        )
        if has_template_effects:
            report_status(request.status_webhook_url, "Processing: Template FX")
            temp_template = temp_dir / f"temp_template_{uid}.mp4"
            cleanup_files.append(temp_template)
            try:
                apply_combined_template_effects(
                    str(temp_main), str(temp_template), effect_config,
                    fresh_trigger_words, duration
                )
                # Replace temp_main with template effects version
                subprocess.run(["mv", str(temp_template), str(temp_main)], check=True)
            except Exception as e:
                print(f"WARNING: Template effects failed, continuing without: {e}")
                import traceback
                traceback.print_exc()

        # 3.6.25 RARE EFFECTS: Datamosh + Pixel Sort (MoviePy post-processing)
        # These are expensive per-frame operations, used sparingly by Grok director
        datamosh_segs = effect_config.get("datamosh_segments", [])
        if datamosh_segs:
            report_status(request.status_webhook_url, "Processing: Datamosh")
            temp_datamosh = temp_dir / f"temp_datamosh_{uid}.mp4"
            cleanup_files.append(temp_datamosh)
            try:
                if apply_datamosh_effect(str(temp_main), str(temp_datamosh), datamosh_segs, duration):
                    subprocess.run(["mv", str(temp_datamosh), str(temp_main)], check=True)
                elif temp_datamosh.exists():
                    temp_datamosh.unlink()
            except Exception as e:
                print(f"WARNING: Datamosh effect failed, continuing without: {e}")

        pixel_sort_segs = effect_config.get("pixel_sort_segments", [])
        if pixel_sort_segs:
            report_status(request.status_webhook_url, "Processing: Pixel Sort")
            temp_psort = temp_dir / f"temp_psort_{uid}.mp4"
            cleanup_files.append(temp_psort)
            try:
                if apply_pixel_sort_effect(str(temp_main), str(temp_psort), pixel_sort_segs, duration):
                    subprocess.run(["mv", str(temp_psort), str(temp_main)], check=True)
                elif temp_psort.exists():
                    temp_psort.unlink()
            except Exception as e:
                print(f"WARNING: Pixel sort effect failed, continuing without: {e}")

        # 3.6.3 DURATION ENFORCEMENT - MoviePy can sometimes produce longer output
        # Re-mux with hard duration limit to ensure clip is exactly the right length
        actual_dur = get_video_info(str(temp_main))["duration"]
        print(f"DEBUG DURATION CHECK: temp_main actual={actual_dur:.1f}s, expected={duration:.1f}s, diff={actual_dur - duration:.1f}s")
        if actual_dur > duration + 1:  # More than 1 second over
            print(f"WARNING: MoviePy output is {actual_dur:.1f}s, expected {duration:.1f}s - truncating")
            temp_trimmed = temp_dir / f"temp_trimmed_{uid}.mp4"
            cleanup_files.append(temp_trimmed)
            subprocess.run([
                "ffmpeg", "-y", "-i", str(temp_main),
                "-t", str(duration),
                "-c:v", "copy", "-c:a", "copy",
                str(temp_trimmed)
            ], check=True)
            subprocess.run(["mv", str(temp_trimmed), str(temp_main)], check=True)
            print(f"Duration enforced: {duration:.1f}s")

        # 3.7. B-ROLL CLIMAX MONTAGE (if climax_time specified and B-roll available)
        if request.climax_time and request.broll_paths and request.broll_duration and request.broll_duration >= 3:
            report_status(request.status_webhook_url, "Processing: B-Roll Montage")

            # climax_time is absolute (in source video), convert to clip-relative
            climax_clip_time = request.climax_time - request.start_time

            # Validate climax is within the clip
            if 0 < climax_clip_time < duration - 3:
                print(f"B-Roll: Climax at {climax_clip_time:.1f}s (clip-relative), duration {request.broll_duration:.1f}s")
                print(f"B-Roll: {len(request.broll_paths)} clips available")

                try:
                    # Get beat times for B-roll cut sync
                    broll_beat_times = []
                    if bgm_path:
                        try:
                            # Detect beats for the B-roll section (offset by climax time)
                            all_beats = detect_beats(str(bgm_path), duration)
                            # Filter beats that fall within the B-roll window
                            broll_beat_times = [b - climax_clip_time for b in all_beats
                                               if climax_clip_time <= b < climax_clip_time + request.broll_duration]
                            print(f"B-Roll: {len(broll_beat_times)} beat-sync points in montage window")
                        except Exception as e:
                            print(f"B-Roll: Beat detection failed, using 1s intervals: {e}")

                    # Generate B-roll montage
                    temp_broll_montage = temp_dir / f"temp_broll_{uid}.mp4"
                    cleanup_files.append(temp_broll_montage)

                    # Pass caption data so B-roll gets same ASS captions and effects
                    broll_caption_data = {
                        "segments": fresh_segments,
                        "climax_clip_time": climax_clip_time,
                        "font": selected_font,
                        "trigger_words": fresh_trigger_words,
                        "caption_style": caption_style
                    }

                    # Use template-specific cut interval for B-roll pacing
                    broll_cut_interval = effect_config.get("broll_cut_interval", 0.5)
                    print(f"B-Roll: Using {broll_cut_interval}s cut interval (template config)")

                    broll_transition = effect_config.get("broll_transition_type") or request.broll_transition_type
                    broll_transition_dur = effect_config.get("broll_transition_duration", 0.3)

                    broll_montage_path = generate_broll_montage(
                        broll_paths=request.broll_paths,
                        duration=request.broll_duration,
                        beat_times=broll_beat_times,
                        output_path=temp_broll_montage,
                        effect_settings=request.effect_settings,
                        caption_data=broll_caption_data,
                        cut_interval=broll_cut_interval,
                        transition_type=broll_transition,
                        transition_duration=broll_transition_dur
                    )

                    if broll_montage_path and Path(broll_montage_path).exists():
                        # Splice: [pre-climax] + [B-roll with speaker audio] + [post-broll]
                        temp_pre_climax = temp_dir / f"temp_pre_climax_{uid}.mp4"
                        temp_post_broll = temp_dir / f"temp_post_broll_{uid}.mp4"
                        temp_climax_audio = temp_dir / f"temp_climax_audio_{uid}.aac"
                        temp_broll_with_audio = temp_dir / f"temp_broll_audio_{uid}.mp4"
                        temp_spliced = temp_dir / f"temp_spliced_{uid}.mp4"
                        cleanup_files.extend([temp_pre_climax, temp_post_broll, temp_climax_audio,
                                            temp_broll_with_audio, temp_spliced])

                        # 1. Extract pre-climax portion (video + audio)
                        subprocess.run([
                            "ffmpeg", "-y", "-i", str(temp_main),
                            "-t", str(climax_clip_time),
                            "-c:v", "copy", "-c:a", "copy",
                            str(temp_pre_climax)
                        ], check=True)
                        print(f"B-Roll: Extracted pre-climax (0 to {climax_clip_time:.1f}s)")

                        # 2. Extract speaker audio from climax to end of B-roll section
                        subprocess.run([
                            "ffmpeg", "-y", "-i", str(temp_main),
                            "-ss", str(climax_clip_time),
                            "-t", str(request.broll_duration),
                            "-vn", "-c:a", "aac",
                            str(temp_climax_audio)
                        ], check=True)
                        print(f"B-Roll: Extracted speaker audio ({climax_clip_time:.1f}s to {climax_clip_time + request.broll_duration:.1f}s)")

                        # 3. Add speaker audio to B-roll montage (video from B-roll, audio from speaker)
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-i", str(broll_montage_path),
                            "-i", str(temp_climax_audio),
                            "-c:v", "copy", "-c:a", "aac",
                            "-shortest",
                            str(temp_broll_with_audio)
                        ], check=True)
                        print(f"B-Roll: Combined B-roll video with speaker audio")

                        # 4. Extract post-broll portion (if any remains before outro)
                        post_broll_start = climax_clip_time + request.broll_duration
                        post_broll_duration = duration - post_broll_start - 2  # Leave 2s for outro

                        if post_broll_duration > 0.5:
                            subprocess.run([
                                "ffmpeg", "-y", "-i", str(temp_main),
                                "-ss", str(post_broll_start),
                                "-t", str(post_broll_duration),
                                "-c:v", "copy", "-c:a", "copy",
                                str(temp_post_broll)
                            ], check=True)
                            print(f"B-Roll: Extracted post-broll ({post_broll_start:.1f}s to {post_broll_start + post_broll_duration:.1f}s)")

                        # 5. Concatenate: pre-climax + B-roll + post-broll (if exists)
                        # Debug: Check durations of files being concatenated
                        pre_climax_dur = get_video_info(str(temp_pre_climax))["duration"]
                        broll_audio_dur = get_video_info(str(temp_broll_with_audio))["duration"]
                        print(f"DEBUG SPLICE: pre_climax={pre_climax_dur:.1f}s, broll_with_audio={broll_audio_dur:.1f}s")

                        splice_list = temp_dir / f"splice_concat_{uid}.txt"
                        cleanup_files.append(splice_list)
                        with open(splice_list, "w") as f:
                            f.write(f"file '{temp_pre_climax}'\n")
                            f.write(f"file '{temp_broll_with_audio}'\n")
                            if post_broll_duration > 0.5 and temp_post_broll.exists():
                                f.write(f"file '{temp_post_broll}'\n")

                        subprocess.run([
                            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", str(splice_list),
                            "-c:v", "copy", "-c:a", "aac",
                            str(temp_spliced)
                        ], check=True)

                        # Replace temp_main with spliced version
                        subprocess.run(["mv", str(temp_spliced), str(temp_main)], check=True)
                        print(f"B-Roll: Montage spliced successfully at {climax_clip_time:.1f}s for {request.broll_duration:.1f}s")

                except Exception as e:
                    print(f"WARNING: B-roll montage failed, continuing without: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"B-Roll: Climax time {climax_clip_time:.1f}s invalid for clip duration {duration:.1f}s, skipping")
        elif request.broll_paths:
            print(f"B-Roll: Skipping - climax_time={request.climax_time}, broll_duration={request.broll_duration}")

        # 4. Outro Generation (V4 CPU-Threaded for safety)
        final_video_path = str(temp_main)
        total_dur = get_video_info(str(temp_main))["duration"]
        print(f"DEBUG OUTRO: temp_main duration before outro = {total_dur:.1f}s")

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
            
            # Re-encode audio to fix timestamp issues from concat (Non-monotonic DTS)
            subprocess.run(["ffmpeg", "-y", "-threads", "0", "-fflags", "+genpts", "-f", "concat", "-safe", "0", "-i", str(temp_concat_list), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-avoid_negative_ts", "make_zero", str(output_path)], check=True)

        else:
            subprocess.run(["cp", str(temp_main), str(output_path)], check=True)

        # 6. BGM Mix (Final Pass) with AUTO VOLUME DETECTION
        if bgm_path:
            report_status(request.status_webhook_url, "Processing: Audio Mix")

            # Detect volume levels of video (voice) and music
            print("Detecting audio levels...")
            video_vol = detect_audio_volume(str(output_path), max_duration=min(60, duration))
            music_vol = detect_audio_volume(str(bgm_path), max_duration=60)

            video_mean_db = video_vol["mean_volume"]
            music_mean_db = music_vol["mean_volume"]

            print(f"  Video voice: {video_mean_db:.1f} dB (mean)")
            print(f"  Music track: {music_mean_db:.1f} dB (mean)")

            # Calculate volume adjustment to make music 10% quieter than voice
            # Target: music should be (video_mean - 3dB) which is roughly 30% quieter
            # -3dB ≈ 70% volume, -6dB ≈ 50% volume
            target_music_db = video_mean_db - 3  # Music 3dB below voice (about 30% quieter)
            db_adjustment = target_music_db - music_mean_db

            # Convert dB adjustment to linear volume multiplier
            # volume = 10^(dB/20)
            import math
            base_volume = math.pow(10, db_adjustment / 20)

            # Clamp to reasonable range (0.1 to 1.5)
            base_volume = max(0.1, min(1.5, base_volume))

            print(f"  Target music: {target_music_db:.1f} dB -> adjustment: {db_adjustment:.1f} dB -> volume: {base_volume:.2f}")

            # Volume ramp: start at base_volume, ramp up to base_volume * 2 in final 5 seconds
            ramp_start = max(0, duration - 5)
            end_volume = min(1.0, base_volume * 2)  # Cap at 1.0 for end ramp

            vol_expr = f"'if(gte(t,{ramp_start}),{base_volume:.3f}+((t-{ramp_start})/5)*{end_volume - base_volume:.3f},{base_volume:.3f})'"

            print(f"  BGM volume: {base_volume:.2f} -> {end_volume:.2f} (ramp at {ramp_start:.1f}s)")

            # Mix BGM with voice and normalize to -14 LUFS (social media standard)
            # This ensures output volume matches other TikTok/Instagram videos
            final_mix = temp_dir / f"final_mix_{uid}.mp4"
            cmd_mix = [
                "ffmpeg", "-y", "-threads", "0",
                "-i", str(output_path),
                "-i", str(bgm_path),
                "-filter_complex",
                f"[1:a]volume=eval=frame:volume={vol_expr},aloop=loop=-1:size=2147483647[bgm];"
                f"[0:a][bgm]amix=inputs=2:duration=first,loudnorm=I=-14:TP=-1:LRA=11",
                "-map", "0:v", "-c:v", "copy", "-c:a", "aac", str(final_mix)
            ]
            subprocess.run(cmd_mix, check=True)
            print(f"Audio normalized to -14 LUFS (social media standard)")
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

# ============ FONT MANAGEMENT ENDPOINTS ============

class GoogleFontRequest(BaseModel):
    font_name: str  # e.g., "Bebas Neue", "Oswald", etc.

@app.get("/fonts")
def list_fonts():
    """List all installed custom fonts."""
    fonts = []
    if FONTS_DIR.exists():
        for f in FONTS_DIR.glob("*.ttf"):
            fonts.append({
                "filename": f.name,
                "name": f.stem,
                "size": f.stat().st_size
            })
    return {"fonts": fonts, "directory": str(FONTS_DIR)}

@app.get("/fonts/{filename}/file")
def serve_font_file(filename: str):
    """Serve a font file for browser loading."""
    from fastapi.responses import FileResponse

    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    font_path = FONTS_DIR / filename
    if not font_path.exists():
        raise HTTPException(status_code=404, detail="Font not found")

    # Determine MIME type
    mime_type = "font/ttf" if filename.endswith(".ttf") else "font/otf"

    return FileResponse(
        font_path,
        media_type=mime_type,
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.delete("/fonts/{filename}")
def delete_font(filename: str):
    """Delete a custom font."""
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    font_path = FONTS_DIR / filename
    if not font_path.exists():
        raise HTTPException(status_code=404, detail="Font not found")

    try:
        font_path.unlink()
        # Refresh font cache
        subprocess.run(["fc-cache", "-fv"], capture_output=True)
        return {"success": True, "deleted": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")

@app.post("/fonts/google")
async def download_google_font(req: GoogleFontRequest):
    """Download a font from Google Fonts via CSS API."""
    import urllib.parse

    font_name = req.font_name.strip()
    if not font_name:
        raise HTTPException(status_code=400, detail="Font name required")

    try:
        # Step 1: Fetch CSS to get TTF URL
        # Google Fonts uses + for spaces in family names (don't URL-encode the +)
        css_url = f"https://fonts.googleapis.com/css2?family={font_name.replace(' ', '+')}"

        # Use curl with browser user-agent to get TTF format (not woff2)
        result = subprocess.run(
            ["curl", "-sL", "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", css_url],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0 or "font-face" not in result.stdout:
            raise HTTPException(status_code=404, detail=f"Font '{font_name}' not found on Google Fonts")

        # Step 2: Extract TTF URL from CSS
        import re
        ttf_match = re.search(r'url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)', result.stdout)
        if not ttf_match:
            raise HTTPException(status_code=404, detail=f"Could not find TTF URL for '{font_name}'")

        ttf_url = ttf_match.group(1)

        # Step 3: Download TTF file
        # Create clean filename from font name
        clean_name = font_name.replace(' ', '')
        dest_path = FONTS_DIR / f"{clean_name}.ttf"

        dl_result = subprocess.run(
            ["curl", "-sL", "-o", str(dest_path), ttf_url],
            capture_output=True,
            timeout=30
        )

        if dl_result.returncode != 0 or not dest_path.exists():
            raise HTTPException(status_code=500, detail=f"Failed to download font file")

        # Verify it's a valid font file
        if dest_path.stat().st_size < 1000:
            dest_path.unlink()
            raise HTTPException(status_code=404, detail=f"Font '{font_name}' download failed")

        # Refresh font cache
        subprocess.run(["fc-cache", "-fv"], capture_output=True)

        return {"success": True, "font": font_name, "files": [f"{clean_name}.ttf"]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


# ============ B-ROLL TAGGING ENDPOINT ============

class TagBrollRequest(BaseModel):
    force: bool = False  # Re-tag all clips
    limit: int = 0  # Limit number of clips to process (0=all)

class TagBrollResponse(BaseModel):
    success: bool
    clips_tagged: int
    total_clips: int
    categories: dict
    message: str

@app.post("/tag-broll", response_model=TagBrollResponse)
async def tag_broll_clips(req: TagBrollRequest):
    """
    Tag B-roll clips with AI-generated metadata using BLIP (captioning) and CLIP (classification).

    This enables smart B-roll matching during viral clip rendering by categorizing clips into:
    war, wealth, faith, strength, nature, people, chaos, victory, power, history

    Run with force=True to re-tag all clips, or let it incrementally tag new ones.
    """
    import tempfile
    from datetime import datetime

    BROLL_DIR = Path("/broll")
    METADATA_FILE = BROLL_DIR / "metadata.json"

    # Semantic categories for CLIP classification
    BROLL_CATEGORIES = [
        "war and military combat", "soldiers and troops", "weapons and firearms", "explosions and destruction",
        "political leaders and speeches", "money and cash", "luxury and wealth", "business and corporate",
        "stock market and trading", "cars and vehicles", "church and religion", "prayer and worship",
        "crosses and religious symbols", "bible and scripture", "boxing and fighting", "gym and weightlifting",
        "sports and athletics", "muscles and bodybuilding", "mountains and landscapes", "ocean and water",
        "sunset and sunrise", "animals and wildlife", "crowds and masses", "family and children",
        "women and femininity", "men and masculinity", "chaos and disorder", "victory and celebration",
        "defeat and failure", "anger and rage", "fire and flames", "darkness and shadows",
        "light and brightness", "american flag and patriotism", "computers and technology", "city and urban",
        "historical footage", "news and media"
    ]

    # Simplified category mapping
    CATEGORY_KEYWORDS = {
        "war": ["war and military combat", "soldiers and troops", "weapons and firearms", "explosions and destruction"],
        "wealth": ["money and cash", "luxury and wealth", "business and corporate", "stock market and trading"],
        "faith": ["church and religion", "prayer and worship", "crosses and religious symbols", "bible and scripture"],
        "strength": ["boxing and fighting", "gym and weightlifting", "sports and athletics", "muscles and bodybuilding"],
        "nature": ["mountains and landscapes", "ocean and water", "sunset and sunrise", "animals and wildlife"],
        "people": ["crowds and masses", "family and children", "women and femininity", "men and masculinity"],
        "chaos": ["chaos and disorder", "fire and flames", "darkness and shadows", "anger and rage"],
        "victory": ["victory and celebration", "light and brightness", "american flag and patriotism"],
        "power": ["political leaders and speeches", "cars and vehicles", "city and urban"],
        "history": ["historical footage", "news and media"]
    }

    if not BROLL_DIR.exists():
        raise HTTPException(status_code=404, detail="B-roll directory not found")

    # Load existing metadata
    existing_metadata = {}
    if METADATA_FILE.exists() and not req.force:
        try:
            with open(METADATA_FILE, "r") as f:
                data = json.load(f)
                existing_metadata = {item["filename"]: item for item in data.get("clips", [])}
            print(f"Loaded {len(existing_metadata)} existing clip records")
        except Exception as e:
            print(f"Failed to load existing metadata: {e}")

    # Find clips to process
    clips = list(BROLL_DIR.glob("*.mp4"))
    total_clips = len(clips)

    if not req.force:
        clips = [c for c in clips if c.name not in existing_metadata]

    if not clips:
        return TagBrollResponse(
            success=True,
            clips_tagged=0,
            total_clips=total_clips,
            categories={},
            message="No new clips to tag"
        )

    if req.limit > 0:
        clips = clips[:req.limit]

    print(f"Tagging {len(clips)} B-roll clips...")

    # Load AI models
    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("Loading BLIP model...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

        print("Loading CLIP model...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load AI models: {str(e)}")

    # Helper functions
    def extract_frame(video_path: str, output_path: str, timestamp: float = 1.0) -> bool:
        try:
            cmd = ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path, "-vframes", "1", "-q:v", "2", output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0 and os.path.exists(output_path)
        except:
            return False

    def caption_image(image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = blip_processor(image, return_tensors="pt").to(device)
            with torch.no_grad():
                output = blip_model.generate(**inputs, max_length=50)
            return blip_processor.decode(output[0], skip_special_tokens=True)
        except:
            return ""

    def classify_image(image_path: str) -> list:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(text=BROLL_CATEGORIES, images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)

            top_indices = probs[0].topk(5).indices.tolist()
            top_probs = probs[0].topk(5).values.tolist()

            results = []
            for idx, prob in zip(top_indices, top_probs):
                if prob > 0.05:
                    results.append({"category": BROLL_CATEGORIES[idx], "confidence": round(prob, 3)})
            return results
        except:
            return []

    def get_simplified_categories(classifications: list) -> list:
        simplified = set()
        for item in classifications:
            cat = item["category"]
            for keyword, detailed_cats in CATEGORY_KEYWORDS.items():
                if cat in detailed_cats:
                    simplified.add(keyword)
        return list(simplified)

    # Process clips
    results = list(existing_metadata.values())
    clips_tagged = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, clip_path in enumerate(clips):
            print(f"[{i+1}/{len(clips)}] Processing {clip_path.name}")

            try:
                frame_path = os.path.join(temp_dir, f"{clip_path.name}.jpg")

                if not extract_frame(str(clip_path), frame_path, 1.0):
                    if not extract_frame(str(clip_path), frame_path, 0.5):
                        print(f"  SKIPPED - Could not extract frame")
                        continue

                caption = caption_image(frame_path)
                classifications = classify_image(frame_path)
                categories = get_simplified_categories(classifications)

                # Get duration
                try:
                    probe = subprocess.run([
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path)
                    ], capture_output=True, text=True, timeout=10)
                    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 2.0
                except:
                    duration = 2.0

                metadata = {
                    "filename": clip_path.name,
                    "path": str(clip_path),
                    "caption": caption,
                    "classifications": classifications,
                    "categories": categories,
                    "duration": duration,
                    "tagged_at": datetime.now().isoformat()
                }

                results.append(metadata)
                clips_tagged += 1
                print(f"  Caption: {caption[:60]}...")
                print(f"  Categories: {', '.join(categories)}")

                if os.path.exists(frame_path):
                    os.unlink(frame_path)

            except Exception as e:
                print(f"  ERROR: {e}")

    # Build category index
    category_index = {}
    for clip in results:
        for cat in clip.get("categories", []):
            if cat not in category_index:
                category_index[cat] = []
            category_index[cat].append(clip["filename"])

    # Save metadata
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "total_clips": len(results),
        "categories": list(category_index.keys()),
        "category_counts": {k: len(v) for k, v in category_index.items()},
        "category_index": category_index,
        "clips": results
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Tagged {clips_tagged} clips, saved metadata to {METADATA_FILE}")

    return TagBrollResponse(
        success=True,
        clips_tagged=clips_tagged,
        total_clips=len(results),
        categories=output_data["category_counts"],
        message=f"Tagged {clips_tagged} new clips"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
