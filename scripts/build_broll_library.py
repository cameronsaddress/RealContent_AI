#!/usr/bin/env python3
"""
Build Real B-Roll Library for Viral Clip Factory
Uses YouTube transcripts to find and catalog clips by keywords
Cuts precise 2-second clips for beat-sync montages
"""

import subprocess
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

BROLL_DIR = Path("/home/canderson/n8n/assets/broll")
TEMP_DIR = Path("/tmp/broll_downloads")
BROLL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Keywords to search for in transcripts, mapped to categories
# Format: category -> [keywords that indicate relevant footage]
CATEGORY_KEYWORDS = {
    # MILITARY / WARFARE
    "warfare": ["combat", "battle", "attack", "assault", "troops", "soldiers", "military", "war", "fight", "invasion"],
    "ww2": ["world war", "ww2", "nazi", "hitler", "normandy", "d-day", "pearl harbor", "allies", "axis", "1940s", "1944", "1945"],
    "vietnam": ["vietnam", "vietcong", "saigon", "napalm", "jungle warfare", "huey", "helicopter assault"],
    "fighter_jets": ["jet", "fighter", "aircraft", "f-16", "f-22", "raptor", "takeoff", "afterburner", "supersonic", "dogfight"],
    "helicopters": ["helicopter", "chopper", "blackhawk", "apache", "huey", "rotors", "hover"],
    "tanks": ["tank", "armored", "panzer", "abrams", "turret", "tracked vehicle", "artillery"],
    "navy": ["ship", "carrier", "battleship", "destroyer", "submarine", "naval", "fleet", "warship"],

    # EXPLOSIONS / DESTRUCTION
    "explosions": ["explosion", "blast", "detonate", "bomb", "nuclear", "fireball", "mushroom cloud", "demolition"],
    "fire": ["fire", "flames", "burning", "inferno", "wildfire"],
    "lightning": ["lightning", "thunder", "storm", "electric", "bolt"],

    # ROCKETS / SPACE
    "rockets": ["rocket", "launch", "liftoff", "thrust", "engine ignition", "countdown", "space launch"],
    "space": ["astronaut", "spacewalk", "orbit", "satellite", "iss", "moon", "mars", "nasa"],

    # PREDATORS / ANIMALS
    "lions": ["lion", "pride", "hunt", "prey", "kill", "savanna", "roar"],
    "wolves": ["wolf", "pack", "howl", "hunt", "prey"],
    "eagles": ["eagle", "soar", "dive", "talons", "prey", "hunting"],
    "sharks": ["shark", "attack", "jaws", "teeth", "feeding"],
    "bulls": ["bull", "charge", "rodeo", "matador"],
    "snakes": ["snake", "cobra", "viper", "strike", "venom"],

    # STORMS / NATURE
    "storms": ["tornado", "hurricane", "cyclone", "twister", "funnel"],
    "ocean": ["wave", "tsunami", "surf", "crash", "ocean storm"],
    "mountains": ["summit", "peak", "climb", "everest", "mountain"],

    # GYM / SPORTS
    "strength": ["lift", "deadlift", "squat", "bench press", "powerlifting", "strongman"],
    "boxing": ["punch", "knockout", "fight", "ring", "boxing", "mma"],
    "racing": ["race", "crash", "speed", "f1", "nascar", "drift"],

    # PATRIOTIC
    "patriotic": ["america", "flag", "freedom", "liberty", "anthem", "usa"],

    # WEALTH / SUCCESS
    "money": ["money", "cash", "dollar", "wealth", "gold", "rich"],
    "luxury": ["yacht", "mansion", "private jet", "supercar", "lamborghini", "ferrari"],
}

# YouTube sources with documentary/real footage
# Format: (url, primary_category, description)
YOUTUBE_SOURCES = [
    # WW2 Documentary Footage
    ("https://www.youtube.com/watch?v=VxLacN2Dp6A", "ww2", "WW2 Rare Color Combat Footage"),
    ("https://www.youtube.com/watch?v=SRX4b7hF0f8", "ww2", "D-Day Real Footage Documentary"),

    # Fighter Jets
    ("https://www.youtube.com/watch?v=bVR8FLF0eGk", "fighter_jets", "F-22 Raptor Demo"),
    ("https://www.youtube.com/watch?v=K6v3awH-BFY", "fighter_jets", "Fighter Jet Compilation"),

    # Rockets & Space
    ("https://www.youtube.com/watch?v=wbSwFU6tY1c", "rockets", "SpaceX Launches Compilation"),
    ("https://www.youtube.com/watch?v=T3_Voh7NgDE", "rockets", "Saturn V Apollo Launches"),

    # Explosions
    ("https://www.youtube.com/watch?v=T2I66dHbSRA", "explosions", "Nuclear Test Footage"),
    ("https://www.youtube.com/watch?v=Dkr7MzHvHgc", "explosions", "Building Demolitions"),

    # Tanks
    ("https://www.youtube.com/watch?v=_zCJk3Rxd88", "tanks", "Tank Battles Documentary"),

    # Helicopters
    ("https://www.youtube.com/watch?v=PYOjRCxG-Qk", "helicopters", "Apache Attack Helicopter"),
    ("https://www.youtube.com/watch?v=cPRCYU0KWHI", "helicopters", "Vietnam Helicopter Footage"),

    # Wildlife
    ("https://www.youtube.com/watch?v=TGwLJWPjgLg", "lions", "Lion Hunt Documentary"),
    ("https://www.youtube.com/watch?v=SvPnPaLpOao", "wolves", "Wolf Pack Hunt"),
    ("https://www.youtube.com/watch?v=VklTs-Tid_I", "eagles", "Eagle Hunting"),

    # Storms
    ("https://www.youtube.com/watch?v=DZkkG_bXpXk", "storms", "Tornado Footage"),
    ("https://www.youtube.com/watch?v=dukkO7c2eUE", "lightning", "Lightning Strikes"),

    # Sports
    ("https://www.youtube.com/watch?v=1cNOtEiNXac", "strength", "Powerlifting Competition"),
    ("https://www.youtube.com/watch?v=QWM--13VYxo", "boxing", "Boxing Knockouts"),

    # Racing
    ("https://www.youtube.com/watch?v=_CG9hCnOCNo", "racing", "F1 Crashes and Moments"),
]


def download_with_subtitles(url: str, output_base: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Download video and subtitles using yt-dlp"""
    video_path = output_base.with_suffix(".mp4")
    subs_path = output_base.with_suffix(".en.vtt")

    if video_path.exists():
        print(f"  Video already exists: {video_path.name}")
    else:
        print(f"  Downloading video...")
        try:
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
                "--merge-output-format", "mp4",
                "-o", str(video_path),
                "--no-playlist",
                "--socket-timeout", "60",
                "--retries", "3",
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0 or not video_path.exists():
                print(f"  Video download failed: {result.stderr[:200] if result.stderr else 'Unknown'}")
                return None, None
        except Exception as e:
            print(f"  Error: {e}")
            return None, None

    # Download subtitles
    if not subs_path.exists():
        print(f"  Downloading subtitles...")
        try:
            cmd = [
                "yt-dlp",
                "--write-auto-subs",
                "--sub-lang", "en",
                "--skip-download",
                "-o", str(output_base),
                url
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except:
            pass

    # Check for various subtitle formats
    for ext in [".en.vtt", ".en.srt", ".vtt", ".srt"]:
        potential_subs = output_base.with_suffix(ext)
        if potential_subs.exists():
            subs_path = potential_subs
            break

    return video_path, subs_path if subs_path.exists() else None


def parse_vtt_subtitles(subs_path: Path) -> List[Dict]:
    """Parse VTT subtitles into list of {start, end, text}"""
    segments = []

    with open(subs_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse VTT format
    # Pattern: 00:00:00.000 --> 00:00:00.000
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?)(?=\n\n|\n\d{2}:|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    for start_str, end_str, text in matches:
        # Convert timestamp to seconds
        def ts_to_sec(ts):
            parts = ts.replace(',', '.').split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

        # Clean text (remove VTT tags)
        text = re.sub(r'<[^>]+>', '', text)
        text = text.strip().replace('\n', ' ')

        if text:
            segments.append({
                'start': ts_to_sec(start_str),
                'end': ts_to_sec(end_str),
                'text': text.lower()
            })

    return segments


def find_keyword_moments(segments: List[Dict], keywords: List[str]) -> List[float]:
    """Find timestamps where keywords appear in subtitles"""
    moments = []

    for segment in segments:
        text = segment['text']
        for keyword in keywords:
            if keyword.lower() in text:
                # Use the start of the segment
                moments.append(segment['start'])
                break

    return sorted(set(moments))


def cut_clip(video_path: Path, timestamp: float, output_path: Path, duration: float = 2.0) -> bool:
    """Cut a clip from video at timestamp"""
    if output_path.exists():
        return True

    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(max(0, timestamp - 0.5)),  # Start slightly before
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-an",  # No audio for B-roll
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"    Error cutting: {e}")
        return False


def process_video_with_transcript(url: str, primary_category: str, description: str) -> Dict[str, int]:
    """Process a video using its transcript to find and catalog clips"""
    print(f"\n{'='*60}")
    print(f"Processing: {description}")
    print(f"URL: {url}")
    print(f"Primary Category: {primary_category}")
    print(f"{'='*60}")

    # Create safe filename from description
    safe_name = re.sub(r'[^\w\s-]', '', description).strip().replace(' ', '_').lower()[:30]
    output_base = TEMP_DIR / f"{primary_category}_{safe_name}"

    # Download video and subtitles
    video_path, subs_path = download_with_subtitles(url, output_base)

    if not video_path:
        print("  Failed to download video")
        return {}

    clips_by_category = {}

    if subs_path:
        # Parse subtitles and find moments
        print(f"  Parsing subtitles: {subs_path.name}")
        segments = parse_vtt_subtitles(subs_path)
        print(f"  Found {len(segments)} transcript segments")

        # Find moments for each category
        for category, keywords in CATEGORY_KEYWORDS.items():
            moments = find_keyword_moments(segments, keywords)
            if moments:
                print(f"  Found {len(moments)} moments for '{category}'")

                # Cut clips for each moment (max 15 per category from this video)
                clip_count = 0
                for i, timestamp in enumerate(moments[:15]):
                    # Check how many clips we already have for this category
                    existing = list(BROLL_DIR.glob(f"{category}_*.mp4"))
                    clip_num = len(existing) + 1

                    output_path = BROLL_DIR / f"{category}_{safe_name}_{clip_num:02d}.mp4"

                    if cut_clip(video_path, timestamp, output_path):
                        clip_count += 1

                if clip_count > 0:
                    clips_by_category[category] = clip_count
                    print(f"    Created {clip_count} clips for {category}")
    else:
        # No subtitles - fall back to uniform sampling for primary category
        print("  No subtitles available - using uniform sampling")

        # Get video duration
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ], capture_output=True, text=True, timeout=30)
            vid_duration = int(float(result.stdout.strip()))
        except:
            vid_duration = 120

        # Sample every 5 seconds for 60 seconds
        clip_count = 0
        for t in range(10, min(vid_duration - 2, 70), 5):
            existing = list(BROLL_DIR.glob(f"{primary_category}_*.mp4"))
            clip_num = len(existing) + 1

            output_path = BROLL_DIR / f"{primary_category}_{safe_name}_{clip_num:02d}.mp4"

            if cut_clip(video_path, t, output_path):
                clip_count += 1

        if clip_count > 0:
            clips_by_category[primary_category] = clip_count
            print(f"  Created {clip_count} clips for {primary_category}")

    return clips_by_category


def main():
    print("=" * 60)
    print("Building Real B-Roll Library from YouTube Transcripts")
    print("=" * 60)

    total_by_category = {}

    for url, category, description in YOUTUBE_SOURCES:
        try:
            clips = process_video_with_transcript(url, category, description)
            for cat, count in clips.items():
                total_by_category[cat] = total_by_category.get(cat, 0) + count
        except Exception as e:
            print(f"  Error processing: {e}")
            continue

    # Final summary
    print("\n" + "=" * 60)
    print("Library Build Complete!")
    print("=" * 60)

    total_clips = sum(total_by_category.values())
    print(f"\nTotal clips created: {total_clips}")
    print("\nClips by category:")
    for cat, count in sorted(total_by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Also count existing clips
    print("\nAll clips in library:")
    all_clips = {}
    for f in BROLL_DIR.glob("*.mp4"):
        if not f.name.startswith("pexels_"):
            cat = f.name.split("_")[0]
            all_clips[cat] = all_clips.get(cat, 0) + 1

    for cat, count in sorted(all_clips.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
