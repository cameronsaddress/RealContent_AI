#!/usr/bin/env python3
"""
B-Roll Downloader & Splitter
Downloads authentic influencer footage from YouTube, splits into 3-5 second clips,
and names them with category prefixes for the viral clip factory.
"""

import subprocess
import json
import os
import sys
import random
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIGURATION ===
# When running inside video-processor container:
BROLL_DIR = Path("/broll")
TEMP_DIR = Path("/tmp/broll_download")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Clip duration range
MIN_CLIP_DURATION = 3.0
MAX_CLIP_DURATION = 5.0

# Max videos to download per search query
MAX_VIDEOS_PER_QUERY = 2

# Max video duration to download (seconds) - skip anything longer than 10 min
MAX_VIDEO_DURATION = 600

# === CATEGORY SEARCH QUERIES ===
# Each category has multiple YouTube search queries.
# We search for "edit" and "compilation" style videos - these are already
# highly visual, short-form content that the target audience watches.
CATEGORY_QUERIES = {
    "money": [
        "hustle motivation edit sigma",
        "wolf of wall street edit",
        "money motivation edit shorts",
        "stock market trading edit",
        "cash money counting compilation",
    ],
    "luxury": [
        "luxury lifestyle edit supercar mansion",
        "dubai lifestyle edit 2024",
        "billionaire lifestyle edit",
        "private jet yacht edit",
        "andrew tate bugatti edit",
    ],
    "gym": [
        "gym motivation edit 2024",
        "sam sulek workout edit",
        "chris bumstead cbum edit",
        "david goggins motivation edit",
        "bodybuilding motivation sigma edit",
    ],
    "relationships": [
        "fresh and fit podcast best moments",
        "whatever podcast heated moments",
        "dating podcast argument edit",
        "red pill moments edit",
        "relationship debate edit",
    ],
    "city": [
        "new york city night edit cinematic",
        "city nightlife edit aesthetic",
        "tokyo night edit cyberpunk",
        "dubai city night drone edit",
        "city timelapse night edit",
    ],
    "crowd": [
        "trump rally crowd edit",
        "stadium crowd celebration edit",
        "concert crowd edit epic",
        "massive crowd timelapse",
        "political rally crowd usa",
    ],
    "sports": [
        "ufc knockout compilation 2024",
        "ufc best knockouts edit",
        "football highlights edit motivation",
        "basketball dunks edit",
        "boxing knockout edit sigma",
    ],
    "fashion": [
        "fashion show runway compilation",
        "streetwear fashion edit",
        "mens fashion transformation edit",
        "designer clothing haul edit",
    ],
    "cars": [
        "supercar exhaust sound compilation",
        "car meet edit 2024",
        "lamborghini ferrari edit",
        "street racing edit",
        "widebody car edit cinematic",
    ],
    "nature_power": [
        "volcano eruption compilation",
        "massive wave compilation",
        "tornado close up footage",
        "thunderstorm lightning 4k",
        "avalanche footage compilation",
    ],
}


def search_youtube(query: str, max_results: int = 3) -> list:
    """Search YouTube and return video URLs (prefer short videos)."""
    print(f"  Searching: '{query}'")
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                f"ytsearch{max_results}:{query}",
                "--dump-json",
                "--flat-playlist",
                "--no-download",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                video_id = data.get("id", "")
                title = data.get("title", "Unknown")
                duration = data.get("duration", 0)

                # Skip videos longer than MAX_VIDEO_DURATION
                if duration and duration > MAX_VIDEO_DURATION:
                    print(f"    SKIP (too long: {duration}s): {title}")
                    continue

                # Skip very short videos (< 10s) - not enough content
                if duration and duration < 10:
                    print(f"    SKIP (too short: {duration}s): {title}")
                    continue

                url = f"https://www.youtube.com/watch?v={video_id}"
                videos.append({
                    "url": url,
                    "title": title,
                    "duration": duration,
                    "id": video_id,
                })
                print(f"    FOUND ({duration}s): {title}")
            except json.JSONDecodeError:
                continue
        return videos[:MAX_VIDEOS_PER_QUERY]
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT searching for: {query}")
        return []
    except Exception as e:
        print(f"    ERROR searching: {e}")
        return []


def download_video(url: str, output_path: Path) -> bool:
    """Download a YouTube video to the specified path."""
    print(f"  Downloading: {url}")
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
                "--merge-output-format", "mp4",
                "-o", str(output_path),
                "--no-playlist",
                "--socket-timeout", "30",
                "--retries", "3",
                "--no-warnings",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for download
        )
        if output_path.exists() and output_path.stat().st_size > 10000:
            print(f"    Downloaded: {output_path.name} ({output_path.stat().st_size // 1024}KB)")
            return True
        else:
            # Sometimes yt-dlp adds .mp4 extension
            alt_path = Path(str(output_path) + ".mp4")
            if alt_path.exists():
                alt_path.rename(output_path)
                return True
            print(f"    FAILED: File not created or too small")
            if result.stderr:
                err_lines = [l for l in result.stderr.split('\n') if 'ERROR' in l or 'error' in l.lower()]
                for l in err_lines[:3]:
                    print(f"      {l[:120]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT downloading")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def get_video_duration(path: Path) -> float:
    """Get video duration using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip()) if result.stdout.strip() else 0
    except:
        return 0


def split_video_into_clips(video_path: Path, category: str, source_name: str) -> list:
    """Split a video into 3-5 second clips, return list of output paths."""
    duration = get_video_duration(video_path)
    if duration < MIN_CLIP_DURATION:
        print(f"    Video too short ({duration:.1f}s), skipping split")
        return []

    clips = []
    clip_index = 1
    current_time = 0.0

    # Clean source name for filenames
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in source_name[:40]).strip("_").lower()

    while current_time + MIN_CLIP_DURATION <= duration:
        # Random clip duration between MIN and MAX
        clip_dur = random.uniform(MIN_CLIP_DURATION, MAX_CLIP_DURATION)
        clip_dur = min(clip_dur, duration - current_time)

        if clip_dur < MIN_CLIP_DURATION:
            break

        output_name = f"{category}_{safe_name}_{clip_index:02d}.mp4"
        output_path = BROLL_DIR / output_name

        # Skip if already exists
        if output_path.exists():
            print(f"    EXISTS: {output_name}")
            clips.append(output_path)
            current_time += clip_dur
            clip_index += 1
            continue

        # Extract clip with re-encode for consistent format
        # Scale to 1080x1920 (9:16 portrait) and crop center
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{current_time:.3f}",
                    "-i", str(video_path),
                    "-t", f"{clip_dur:.3f}",
                    "-vf", (
                        "scale=1080:1920:force_original_aspect_ratio=increase,"
                        "crop=1080:1920:(iw-1080)/2:(ih-1920)/2,"
                        "fps=30"
                    ),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-an",  # No audio for B-roll
                    "-movflags", "+faststart",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if output_path.exists() and output_path.stat().st_size > 5000:
                clips.append(output_path)
            else:
                # Try without the scale filter (in case of weird formats)
                pass
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT on clip {clip_index}")
        except Exception as e:
            print(f"    ERROR on clip {clip_index}: {e}")

        current_time += clip_dur
        clip_index += 1

    return clips


def process_category(category: str, queries: list) -> dict:
    """Process an entire category: search, download, split."""
    print(f"\n{'='*60}")
    print(f"CATEGORY: {category.upper()}")
    print(f"{'='*60}")

    category_clips = []

    for query in queries:
        videos = search_youtube(query)

        for video in videos:
            # Download to temp
            video_id = video["id"]
            temp_path = TEMP_DIR / f"{category}_{video_id}.mp4"

            if not temp_path.exists():
                success = download_video(video["url"], temp_path)
                if not success:
                    continue
            else:
                print(f"  Using cached: {temp_path.name}")

            # Split into clips
            safe_title = video.get("title", video_id)
            clips = split_video_into_clips(temp_path, category, safe_title)
            category_clips.extend(clips)
            print(f"    Split into {len(clips)} clips")

    print(f"\n  TOTAL {category}: {len(category_clips)} clips")
    return {"category": category, "clips": [str(c) for c in category_clips]}


def update_metadata(all_results: list):
    """Update the metadata.json with new categories."""
    metadata_path = BROLL_DIR / "metadata.json"

    # Load existing metadata
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except:
            metadata = {}

    # Ensure structure
    if "clips" not in metadata or not isinstance(metadata["clips"], dict):
        metadata["clips"] = {}
    if "category_index" not in metadata:
        metadata["category_index"] = {}

    # Add new clips
    for result in all_results:
        category = result["category"]
        if category not in metadata["category_index"]:
            metadata["category_index"][category] = []

        for clip_path in result["clips"]:
            filename = Path(clip_path).name
            # Add to clips dict
            metadata["clips"][filename] = {
                "categories": [category],
                "source": "youtube_influencer",
                "format": "9:16",
            }
            # Add to category index
            if filename not in metadata["category_index"][category]:
                metadata["category_index"][category].append(filename)

    # Update counts
    metadata["total_clips"] = len(metadata["clips"])
    metadata["category_counts"] = {
        cat: len(clips) for cat, clips in metadata["category_index"].items()
    }
    metadata["generated_at"] = str(Path("/proc/self/fd/0"))  # timestamp placeholder

    # Write
    import datetime
    metadata["generated_at"] = datetime.datetime.now().isoformat()
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nUpdated metadata.json: {metadata['total_clips']} total clips, {len(metadata['category_index'])} categories")


def main():
    """Main entry point."""
    print("=" * 60)
    print("B-ROLL DOWNLOADER & SPLITTER")
    print("Downloading authentic influencer footage for viral clips")
    print("=" * 60)

    # Parse args for specific categories
    categories_to_process = list(CATEGORY_QUERIES.keys())
    if len(sys.argv) > 1:
        categories_to_process = [c for c in sys.argv[1:] if c in CATEGORY_QUERIES]
        if not categories_to_process:
            print(f"Unknown categories. Available: {list(CATEGORY_QUERIES.keys())}")
            sys.exit(1)

    print(f"\nCategories to process: {categories_to_process}")
    print(f"Output directory: {BROLL_DIR}")
    print(f"Clip duration: {MIN_CLIP_DURATION}-{MAX_CLIP_DURATION}s")
    print(f"Max videos per query: {MAX_VIDEOS_PER_QUERY}")

    all_results = []
    for category in categories_to_process:
        queries = CATEGORY_QUERIES[category]
        result = process_category(category, queries)
        all_results.append(result)

    # Update metadata
    update_metadata(all_results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = 0
    for r in all_results:
        count = len(r["clips"])
        total += count
        print(f"  {r['category']:15s}: {count} clips")
    print(f"  {'TOTAL':15s}: {total} clips")
    print("\nDone! New clips are in:", BROLL_DIR)


if __name__ == "__main__":
    main()
