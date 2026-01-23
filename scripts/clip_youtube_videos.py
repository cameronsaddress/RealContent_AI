#!/usr/bin/env python3
"""
Clip specific YouTube videos for B-roll library
Downloads via video-processor container, cuts into 2-second clips
Uses transcripts to categorize clips when available
"""

import subprocess
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

BROLL_DIR = Path("/home/canderson/n8n/assets/broll")
TEMP_DIR = Path("/tmp/broll_youtube")
BROLL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Container paths
CONTAINER_DOWNLOADS = "/downloads"
CONTAINER_BROLL = "/broll"

# YouTube videos to process
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=3ZJQ7ey4a80",
    "https://www.youtube.com/watch?v=hzZkdf6dORM",
    "https://www.youtube.com/watch?v=M1wtXQslQUk",
    "https://www.youtube.com/watch?v=PPvSFICiHGk",
    "https://www.youtube.com/watch?v=nM-wp4oJ2LY",
    "https://www.youtube.com/watch?v=LbNP0FUItjI",
    "https://www.youtube.com/watch?v=-U1w6imCYLY",
    "https://www.youtube.com/watch?v=tL5_bC2syfQ",
    "https://www.youtube.com/watch?v=sm8AhYyBMeI",
    "https://www.youtube.com/watch?v=sUijN63Z7SU",
    "https://www.youtube.com/watch?v=5Ud39x20RI4",
    "https://www.youtube.com/watch?v=8TfyXTIk9_o",
    "https://www.youtube.com/watch?v=6cDaHKP2Hao",
    "https://www.youtube.com/shorts/KA-05QdDAK8",
]

# Keywords to categorize clips
CATEGORY_KEYWORDS = {
    "warfare": ["war", "battle", "combat", "military", "troops", "soldier", "army", "attack"],
    "ww2": ["world war", "ww2", "nazi", "hitler", "normandy", "1940s", "pearl harbor"],
    "vietnam": ["vietnam", "saigon", "napalm", "jungle"],
    "fighter_jets": ["jet", "fighter", "aircraft", "f-16", "f-22", "takeoff", "afterburner"],
    "helicopters": ["helicopter", "chopper", "blackhawk", "apache", "huey"],
    "tanks": ["tank", "armored", "panzer", "abrams"],
    "navy": ["ship", "carrier", "battleship", "destroyer", "submarine", "naval"],
    "explosions": ["explosion", "blast", "bomb", "nuclear", "fireball", "demolition"],
    "fire": ["fire", "flames", "burning", "inferno"],
    "rockets": ["rocket", "launch", "liftoff", "space", "apollo", "nasa", "spacex"],
    "lions": ["lion", "pride", "hunt", "savanna", "roar"],
    "wolves": ["wolf", "pack", "howl"],
    "eagles": ["eagle", "soar", "talons"],
    "sharks": ["shark", "jaws", "teeth"],
    "storms": ["tornado", "hurricane", "cyclone", "twister"],
    "lightning": ["lightning", "thunder", "electrical"],
    "strength": ["lift", "deadlift", "squat", "powerlifting", "gym", "muscle"],
    "boxing": ["punch", "knockout", "boxing", "mma", "fight"],
    "racing": ["race", "crash", "speed", "f1", "nascar"],
    "patriotic": ["america", "flag", "freedom", "liberty", "usa"],
    "mountains": ["mountain", "summit", "peak", "climb", "everest"],
    "ocean": ["ocean", "wave", "sea", "surf"],
}


def get_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|/)([a-zA-Z0-9_-]{11})(?:\?|&|$)',
        r'shorts/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url[-11:]


def download_video_with_subs(url: str, video_id: str) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Download video and subtitles using video-processor container"""
    output_name = f"broll_yt_{video_id}"
    container_output = f"{CONTAINER_DOWNLOADS}/{output_name}"
    host_output = Path(f"/home/canderson/n8n/assets/videos/{output_name}.mp4")
    host_subs = Path(f"/home/canderson/n8n/assets/videos/{output_name}.en.vtt")

    print(f"  Downloading video...")

    # Download video
    if not host_output.exists():
        try:
            result = subprocess.run([
                "docker", "exec", "SocialGen_video_processor",
                "yt-dlp",
                "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
                "--merge-output-format", "mp4",
                "-o", f"{container_output}.%(ext)s",
                "--no-playlist",
                url
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"  Download failed: {result.stderr[:200]}")
                return None, None, None
        except Exception as e:
            print(f"  Download error: {e}")
            return None, None, None

    # Get video title
    title = None
    try:
        result = subprocess.run([
            "docker", "exec", "SocialGen_video_processor",
            "yt-dlp", "--get-title", url
        ], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            title = result.stdout.strip()
    except:
        pass

    # Download subtitles
    if not host_subs.exists():
        try:
            subprocess.run([
                "docker", "exec", "SocialGen_video_processor",
                "yt-dlp",
                "--write-auto-subs",
                "--sub-lang", "en",
                "--skip-download",
                "-o", container_output,
                url
            ], capture_output=True, timeout=120)
        except:
            pass

    return host_output if host_output.exists() else None, host_subs if host_subs.exists() else None, title


def parse_vtt_subs(subs_path: Path) -> List[Dict]:
    """Parse VTT subtitles"""
    segments = []
    try:
        with open(subs_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?)(?=\n\n|\n\d{2}:|$)'
        matches = re.findall(pattern, content, re.DOTALL)

        for start_str, end_str, text in matches:
            def ts_to_sec(ts):
                parts = ts.replace(',', '.').split(':')
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

            text = re.sub(r'<[^>]+>', '', text).strip().replace('\n', ' ')
            if text:
                segments.append({
                    'start': ts_to_sec(start_str),
                    'end': ts_to_sec(end_str),
                    'text': text.lower()
                })
    except:
        pass
    return segments


def categorize_from_text(text: str) -> str:
    """Determine category from text"""
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return category
    return "misc"


def cut_clips(video_path: Path, video_id: str, title: str, segments: List[Dict]) -> int:
    """Cut video into 2-second clips"""
    # Copy video to broll dir for container access
    temp_name = f"temp_yt_{video_id}.mp4"
    temp_path = BROLL_DIR / temp_name
    subprocess.run(["cp", str(video_path), str(temp_path)], timeout=120)

    # Get video duration
    try:
        result = subprocess.run([
            "docker", "exec", "SocialGen_video_processor",
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            f"{CONTAINER_BROLL}/{temp_name}"
        ], capture_output=True, text=True, timeout=30)
        duration = int(float(result.stdout.strip()))
    except:
        duration = 60

    # Determine base category from title
    base_category = categorize_from_text(title or "")
    if base_category == "misc":
        base_category = "warfare"  # Default for these military videos

    # Create safe name from title
    safe_title = re.sub(r'[^\w\s-]', '', title or video_id).strip().replace(' ', '_').lower()[:20]

    clips_created = 0

    # If we have segments, use them to find interesting moments
    if segments:
        for seg in segments[:50]:  # Max 50 clips per video
            timestamp = seg['start']
            text = seg['text']

            # Categorize this specific moment
            category = categorize_from_text(text)
            if category == "misc":
                category = base_category

            # Get next clip number
            existing = [f for f in BROLL_DIR.glob(f"{category}_*.mp4") if not f.name.startswith("temp_")]
            clip_num = len(existing) + 1
            output_name = f"{category}_{safe_title}_{clip_num:02d}.mp4"

            try:
                result = subprocess.run([
                    "docker", "exec", "SocialGen_video_processor",
                    "ffmpeg", "-y",
                    "-ss", str(timestamp),
                    "-i", f"{CONTAINER_BROLL}/{temp_name}",
                    "-t", "2",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-an",
                    "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                    f"{CONTAINER_BROLL}/{output_name}"
                ], capture_output=True, timeout=60)
                if result.returncode == 0:
                    clips_created += 1
            except:
                pass
    else:
        # No segments - cut uniformly every 3 seconds
        for t in range(5, min(duration - 2, 120), 3):
            existing = [f for f in BROLL_DIR.glob(f"{base_category}_*.mp4") if not f.name.startswith("temp_")]
            clip_num = len(existing) + 1
            output_name = f"{base_category}_{safe_title}_{clip_num:02d}.mp4"

            try:
                result = subprocess.run([
                    "docker", "exec", "SocialGen_video_processor",
                    "ffmpeg", "-y",
                    "-ss", str(t),
                    "-i", f"{CONTAINER_BROLL}/{temp_name}",
                    "-t", "2",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-an",
                    "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                    f"{CONTAINER_BROLL}/{output_name}"
                ], capture_output=True, timeout=60)
                if result.returncode == 0:
                    clips_created += 1
            except:
                pass

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()

    return clips_created


def main():
    print("=" * 60)
    print("Clipping YouTube Videos for B-Roll Library")
    print("=" * 60)

    # Cleanup temp files
    for f in BROLL_DIR.glob("temp_*.mp4"):
        f.unlink()

    total_clips = 0

    for url in YOUTUBE_URLS:
        video_id = get_video_id(url)
        print(f"\n>>> Processing: {video_id}")
        print(f"    URL: {url}")

        # Download
        video_path, subs_path, title = download_video_with_subs(url, video_id)

        if not video_path:
            print("    Failed to download video")
            continue

        print(f"    Title: {title}")

        # Parse subtitles
        segments = []
        if subs_path:
            segments = parse_vtt_subs(subs_path)
            print(f"    Found {len(segments)} subtitle segments")

        # Cut clips
        clips = cut_clips(video_path, video_id, title, segments)
        print(f"    Created {clips} clips")
        total_clips += clips

    print("\n" + "=" * 60)
    print(f"Complete! Created {total_clips} clips from {len(YOUTUBE_URLS)} videos")
    print("=" * 60)

    # Final summary
    print("\nLibrary by category:")
    categories = {}
    for f in BROLL_DIR.glob("*.mp4"):
        if not f.name.startswith("temp_"):
            cat = f.name.split("_")[0]
            categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
