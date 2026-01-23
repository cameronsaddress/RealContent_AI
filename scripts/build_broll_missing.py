#!/usr/bin/env python3
"""
Build clips for missing B-Roll categories from Archive.org
"""

import subprocess
import os
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

BROLL_DIR = Path("/home/canderson/n8n/assets/broll")
TEMP_DIR = Path("/tmp/broll_downloads")
BROLL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CONTAINER_BROLL_DIR = "/broll"

# Missing categories with search queries
MISSING_CATEGORIES = [
    ("tank battle armored military combat", "tanks"),
    ("lion predator hunt africa", "lions"),
    ("wolf pack hunting wild", "wolves"),
    ("eagle bird prey hunting", "eagles"),
    ("shark attack underwater ocean", "sharks"),
    ("tornado storm twister", "storms"),
    ("lightning storm thunder electrical", "lightning"),
    ("weightlifting powerlifting gym", "strength"),
    ("boxing knockout fight ring", "boxing"),
    ("car racing crash motorsport", "racing"),
    ("american flag patriotic usa", "patriotic"),
    ("money cash wealth gold", "money"),
    ("cathedral church interior gothic", "cathedrals"),
    ("castle medieval fortress", "castles"),
    ("mountain summit climb alpine", "mountains"),
    ("fire wildfire burning flames", "fire"),
    ("bull rodeo charging", "bulls"),
]


def search_archive_org(query: str, max_results: int = 5) -> List[Dict]:
    """Search Archive.org for video content"""
    encoded_query = urllib.parse.quote(query)
    url = f"https://archive.org/advancedsearch.php?q={encoded_query}+AND+mediatype%3Amovies&fl=identifier,title&output=json&rows={max_results}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data.get('response', {}).get('docs', [])
    except Exception as e:
        print(f"  Search failed: {e}")
        return []


def get_video_url(identifier: str) -> Optional[str]:
    """Get direct video URL from Archive.org metadata"""
    url = f"https://archive.org/metadata/{identifier}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())

        files = data.get('files', [])

        # Find MP4 file
        mp4_files = [f for f in files if f.get('name', '').endswith('.mp4')]
        if mp4_files:
            mp4_files.sort(key=lambda x: int(x.get('size', 0)))
            idx = min(len(mp4_files) // 2, len(mp4_files) - 1)
            name = mp4_files[idx].get('name', '')
            encoded_name = urllib.parse.quote(name)
            return f"https://archive.org/download/{identifier}/{encoded_name}"

        # Fallback to OGV
        for f in files:
            name = f.get('name', '')
            if name.endswith('.ogv'):
                encoded_name = urllib.parse.quote(name)
                return f"https://archive.org/download/{identifier}/{encoded_name}"
    except:
        pass

    return None


def download_video(url: str, output_path: Path) -> bool:
    """Download video from URL"""
    if output_path.exists() and output_path.stat().st_size > 1000:
        return True

    try:
        cmd = ["wget", "-q", "-O", str(output_path), url]
        result = subprocess.run(cmd, timeout=180)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000
    except:
        if output_path.exists():
            output_path.unlink()
        return False


def process_item(identifier: str, category: str, name: str) -> int:
    """Download and cut clips from an Archive.org item"""
    video_url = get_video_url(identifier)
    if not video_url:
        return 0

    temp_file = TEMP_DIR / f"{category}_{name}_raw.mp4"
    if not download_video(video_url, temp_file):
        return 0

    # Copy to broll dir for container access
    temp_source_name = f"temp_source_{category}_{name}.mp4"
    temp_source_path = BROLL_DIR / temp_source_name
    subprocess.run(["cp", str(temp_file), str(temp_source_path)], timeout=120)

    # Get duration
    try:
        result = subprocess.run([
            "docker", "exec", "SocialGen_video_processor",
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            f"{CONTAINER_BROLL_DIR}/{temp_source_name}"
        ], capture_output=True, text=True, timeout=30)
        duration = int(float(result.stdout.strip()))
    except:
        duration = 60

    # Cut clips
    clips = 0
    for t in range(30, min(duration - 2, 120), 3):
        existing = [f for f in BROLL_DIR.glob(f"{category}_*.mp4") if not f.name.startswith("temp_")]
        clip_num = len(existing) + 1
        output_name = f"{category}_{name}_{clip_num:02d}.mp4"

        try:
            result = subprocess.run([
                "docker", "exec", "SocialGen_video_processor",
                "ffmpeg", "-y",
                "-ss", str(t),
                "-i", f"{CONTAINER_BROLL_DIR}/{temp_source_name}",
                "-t", "2",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-an",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                f"{CONTAINER_BROLL_DIR}/{output_name}"
            ], capture_output=True, timeout=60)
            if result.returncode == 0:
                clips += 1
        except:
            pass

    # Cleanup
    if temp_source_path.exists():
        temp_source_path.unlink()

    return clips


def main():
    print("=" * 60)
    print("Adding Missing B-Roll Categories")
    print("=" * 60)

    for f in BROLL_DIR.glob("temp_*.mp4"):
        f.unlink()

    total = 0

    for query, category in MISSING_CATEGORIES:
        print(f"\n>>> {category}: '{query}'")

        # Check existing count
        existing = [f for f in BROLL_DIR.glob(f"{category}_*.mp4") if not f.name.startswith("temp_")]
        if len(existing) >= 20:
            print(f"  Already have {len(existing)} clips, skipping")
            continue

        results = search_archive_org(query, 5)
        clips = 0

        for item in results[:3]:
            identifier = item.get('identifier', '')
            if identifier.startswith('youtube-'):
                continue
            if len(identifier) > 80:
                continue

            title = item.get('title', '')
            safe_name = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_').lower()[:15]
            if not safe_name:
                safe_name = identifier[:15]

            print(f"  Processing: {identifier[:40]}...")
            c = process_item(identifier, category, safe_name)
            clips += c

            if clips >= 20:
                break

        print(f"  Created {clips} clips for {category}")
        total += clips

    print("\n" + "=" * 60)
    print(f"Added {total} new clips")
    print("=" * 60)

    # Final count
    print("\nFinal library by category:")
    categories = {}
    for f in BROLL_DIR.glob("*.mp4"):
        if not f.name.startswith("temp_"):
            cat = f.name.split("_")[0]
            categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
