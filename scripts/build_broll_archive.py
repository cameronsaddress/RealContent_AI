#!/usr/bin/env python3
"""
Build Real B-Roll Library from Archive.org (Public Domain)
Downloads via direct links, cuts into 2-second clips using video-processor container
"""

import subprocess
import os
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Host paths
BROLL_DIR = Path("/home/canderson/n8n/assets/broll")
TEMP_DIR = Path("/tmp/broll_downloads")
BROLL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Container paths (video-processor mounts)
CONTAINER_BROLL_DIR = "/broll"  # Mounted from ./assets/broll

# Search queries to find good Archive.org content
SEARCH_QUERIES = [
    ("world war 2 color footage", "ww2"),
    ("d-day invasion normandy", "ww2"),
    ("battle of britain", "ww2"),
    ("fighter jet aircraft military", "fighter_jets"),
    ("rocket launch apollo nasa", "rockets"),
    ("space shuttle launch", "rockets"),
    ("nuclear test explosion atomic", "explosions"),
    ("building demolition explosion", "explosions"),
    ("tank battle military armor", "tanks"),
    ("helicopter combat vietnam military", "helicopters"),
    ("aircraft carrier navy ship", "navy"),
    ("lightning storm thunder", "lightning"),
    ("tornado storm destruction", "storms"),
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

        # Find MP4 file (prefer smaller ones for faster download)
        mp4_files = [f for f in files if f.get('name', '').endswith('.mp4')]
        if mp4_files:
            # Sort by size if available, prefer medium sized files
            mp4_files.sort(key=lambda x: int(x.get('size', 0)))
            # Pick a medium-sized one
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

    except Exception as e:
        print(f"  Metadata fetch failed: {e}")

    return None


def download_video(url: str, output_path: Path) -> bool:
    """Download video from URL"""
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"  Already downloaded: {output_path.name}")
        return True

    print(f"  Downloading: {url[:80]}...")
    try:
        cmd = ["wget", "-q", "-O", str(output_path), url]
        result = subprocess.run(cmd, timeout=300)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000
    except Exception as e:
        print(f"  Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def get_video_duration(video_path: Path) -> int:
    """Get video duration in seconds using docker container"""
    try:
        # Copy file to broll dir temporarily for container access
        temp_name = f"temp_probe_{video_path.name}"
        temp_path = BROLL_DIR / temp_name
        subprocess.run(["cp", str(video_path), str(temp_path)], timeout=60)

        result = subprocess.run([
            "docker", "exec", "SocialGen_video_processor",
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            f"{CONTAINER_BROLL_DIR}/{temp_name}"
        ], capture_output=True, text=True, timeout=30)

        # Clean up
        if temp_path.exists():
            temp_path.unlink()

        return int(float(result.stdout.strip()))
    except:
        return 60


def cut_clip_docker(source_path: Path, timestamp: float, output_name: str, duration: float = 2.0) -> bool:
    """Cut a 2-second clip using video-processor container"""
    output_path = BROLL_DIR / output_name

    if output_path.exists() and output_path.stat().st_size > 1000:
        return True

    # Copy source to broll dir for container access
    temp_source = f"temp_source_{source_path.name}"
    temp_source_path = BROLL_DIR / temp_source

    try:
        if not temp_source_path.exists():
            subprocess.run(["cp", str(source_path), str(temp_source_path)], timeout=60)

        cmd = [
            "docker", "exec", "SocialGen_video_processor",
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", f"{CONTAINER_BROLL_DIR}/{temp_source}",
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-an",  # No audio for B-roll
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            f"{CONTAINER_BROLL_DIR}/{output_name}"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000
    except Exception as e:
        print(f"    Error cutting: {e}")
        return False


def process_archive_item(identifier: str, category: str, name: str, start: int = 30, max_duration: int = 90) -> int:
    """Download and cut clips from an Archive.org item"""
    print(f"\n{'='*60}")
    print(f"Processing: {identifier[:50]}")
    print(f"Category: {category}, Name: {name}")
    print(f"{'='*60}")

    # Get video URL
    video_url = get_video_url(identifier)
    if not video_url:
        print("  No video URL found")
        return 0

    # Download to temp dir
    temp_file = TEMP_DIR / f"{category}_{name}_raw.mp4"
    if not download_video(video_url, temp_file):
        return 0

    # Get duration
    vid_duration = get_video_duration(temp_file)
    print(f"  Video duration: {vid_duration}s")

    # Determine range to cut
    end = min(start + max_duration, vid_duration - 2)
    if end <= start:
        print("  Video too short")
        return 0

    # Copy source file to broll dir for container access (once)
    temp_source_name = f"temp_source_{category}_{name}.mp4"
    temp_source_path = BROLL_DIR / temp_source_name
    subprocess.run(["cp", str(temp_file), str(temp_source_path)], timeout=120)

    # Cut clips every 3 seconds
    clips_created = 0
    for t in range(start, end, 3):
        # Get next available clip number for this category
        existing = list(BROLL_DIR.glob(f"{category}_*.mp4"))
        # Filter out temp files
        existing = [f for f in existing if not f.name.startswith("temp_")]
        clip_num = len(existing) + 1

        output_name = f"{category}_{name}_{clip_num:02d}.mp4"
        output_path = BROLL_DIR / output_name

        if output_path.exists() and output_path.stat().st_size > 1000:
            clips_created += 1
            continue

        try:
            cmd = [
                "docker", "exec", "SocialGen_video_processor",
                "ffmpeg", "-y",
                "-ss", str(t),
                "-i", f"{CONTAINER_BROLL_DIR}/{temp_source_name}",
                "-t", "2",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-an",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                f"{CONTAINER_BROLL_DIR}/{output_name}"
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                clips_created += 1
        except Exception as e:
            print(f"    Error at {t}s: {e}")

    # Clean up temp source
    if temp_source_path.exists():
        temp_source_path.unlink()

    print(f"  Created {clips_created} clips")
    return clips_created


def discover_and_process(query: str, category: str, max_items: int = 3) -> int:
    """Search Archive.org and process found items"""
    print(f"\n{'='*60}")
    print(f"Searching: '{query}' -> {category}")
    print(f"{'='*60}")

    results = search_archive_org(query, max_items)
    print(f"  Found {len(results)} items")

    total_clips = 0
    for item in results:
        identifier = item.get('identifier', '')
        title = item.get('title', '')

        # Skip YouTube mirrors
        if identifier.startswith('youtube-'):
            print(f"  Skipping YouTube mirror: {identifier}")
            continue

        # Skip very long identifiers (usually duplicates)
        if len(identifier) > 80:
            print(f"  Skipping: {identifier[:50]}...")
            continue

        # Create safe name from title
        safe_name = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_').lower()[:20]
        if not safe_name:
            safe_name = re.sub(r'[^\w]', '', identifier)[:20]

        clips = process_archive_item(identifier, category, safe_name, 30, 90)
        total_clips += clips

        # Limit clips per category to avoid too many from one source
        if total_clips >= 30:
            break

    return total_clips


def main():
    print("=" * 60)
    print("Building Real B-Roll Library from Archive.org")
    print("Using video-processor container for FFmpeg")
    print("=" * 60)

    # Clean up any temp files first
    for f in BROLL_DIR.glob("temp_*.mp4"):
        f.unlink()

    total_clips = 0

    # Discover and process via search
    for query, category in SEARCH_QUERIES:
        clips = discover_and_process(query, category, max_items=3)
        total_clips += clips

    # Summary
    print("\n" + "=" * 60)
    print("Library Build Complete!")
    print(f"Total clips created: {total_clips}")
    print("=" * 60)

    # Count by category
    print("\nClips by category:")
    clips_by_cat = {}
    for f in BROLL_DIR.glob("*.mp4"):
        if not f.name.startswith("pexels_") and not f.name.startswith("temp_"):
            cat = f.name.split("_")[0]
            clips_by_cat[cat] = clips_by_cat.get(cat, 0) + 1

    for cat, count in sorted(clips_by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
