#!/bin/bash
# Build Real B-Roll Library from Archive.org and YouTube
# Clips are cut into 2-second segments for beat-sync montages

set -e

BROLL_DIR="/home/canderson/n8n/assets/broll"
TEMP_DIR="/tmp/broll_downloads"
mkdir -p "$BROLL_DIR" "$TEMP_DIR"

# Function to download from Archive.org and cut into 2-second clips
download_archive_and_cut() {
    local url="$1"
    local category="$2"
    local name="$3"
    local start="${4:-0}"
    local duration="${5:-60}"

    echo "=== Downloading $category/$name from Archive.org ==="
    local temp_file="$TEMP_DIR/${category}_${name}_raw.mp4"

    # Download
    if [ ! -f "$temp_file" ]; then
        wget -q --show-progress -O "$temp_file" "$url" || { echo "Failed to download $url"; return 1; }
    fi

    # Get video duration
    local vid_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$temp_file" 2>/dev/null | cut -d. -f1)
    vid_duration=${vid_duration:-$duration}

    # Cut into 2-second segments starting from $start
    local end=$((start + duration))
    [ $end -gt $vid_duration ] && end=$vid_duration

    local clip_num=1
    for ((t=start; t<end-2; t+=2)); do
        local outfile="$BROLL_DIR/${category}_${name}_$(printf '%02d' $clip_num).mp4"
        if [ ! -f "$outfile" ]; then
            echo "  Cutting clip $clip_num: ${t}s-$((t+2))s"
            ffmpeg -y -ss $t -i "$temp_file" -t 2 -c:v libx264 -preset fast -crf 23 -an -vf "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" "$outfile" 2>/dev/null
        fi
        ((clip_num++))
    done
    echo "  Created $((clip_num-1)) clips for $category/$name"
}

# Function to download from YouTube and cut into 2-second clips
download_youtube_and_cut() {
    local url="$1"
    local category="$2"
    local name="$3"
    local start="${4:-0}"
    local duration="${5:-60}"

    echo "=== Downloading $category/$name from YouTube ==="
    local temp_file="$TEMP_DIR/${category}_${name}_raw.mp4"

    # Download with yt-dlp
    if [ ! -f "$temp_file" ]; then
        yt-dlp -f "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best" \
            --merge-output-format mp4 \
            -o "$temp_file" \
            "$url" || { echo "Failed to download $url"; return 1; }
    fi

    # Get video duration
    local vid_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$temp_file" 2>/dev/null | cut -d. -f1)
    vid_duration=${vid_duration:-$duration}

    # Cut into 2-second segments
    local end=$((start + duration))
    [ $end -gt $vid_duration ] && end=$vid_duration

    local clip_num=1
    for ((t=start; t<end-2; t+=2)); do
        local outfile="$BROLL_DIR/${category}_${name}_$(printf '%02d' $clip_num).mp4"
        if [ ! -f "$outfile" ]; then
            echo "  Cutting clip $clip_num: ${t}s-$((t+2))s"
            ffmpeg -y -ss $t -i "$temp_file" -t 2 -c:v libx264 -preset fast -crf 23 -an -vf "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" "$outfile" 2>/dev/null
        fi
        ((clip_num++))
    done
    echo "  Created $((clip_num-1)) clips for $category/$name"
}

echo "============================================"
echo "Building Real B-Roll Library"
echo "============================================"

# =============================================
# ARCHIVE.ORG SOURCES (Public Domain)
# =============================================

# WW2 - D-Day Footage (Public Domain)
download_archive_and_cut \
    "https://archive.org/download/1944-06-06_D-Day/1944-06-06_D-Day.mp4" \
    "ww2" "dday" 30 120

# WW2 - Battle of Britain
download_archive_and_cut \
    "https://archive.org/download/BattleOfBritain1943/BattleOfBritain1943.mp4" \
    "ww2" "britain" 60 120

# Fighter Jets - Blue Angels (Public Domain)
download_archive_and_cut \
    "https://archive.org/download/BlueAngels1954/BlueAngels1954.mp4" \
    "fighter_jets" "blueangels" 30 90

# Rockets - Apollo 11 Launch (NASA Public Domain)
download_archive_and_cut \
    "https://archive.org/download/Apollo1116mmOnboardFilm1/Apollo1116mmOnboardFilm1.mp4" \
    "rockets" "apollo11" 0 120

# Explosions - Nuclear Tests (Public Domain)
download_archive_and_cut \
    "https://archive.org/download/OperationCrossroads1946/OperationCrossroads1946.mp4" \
    "explosions" "crossroads" 120 90

# Navy - Aircraft Carrier Operations
download_archive_and_cut \
    "https://archive.org/download/TheFleetThatCameToStay/TheFleetThatCameToStay.mp4" \
    "navy" "carrier" 60 90

# Tanks - WW2 Tank Battles
download_archive_and_cut \
    "https://archive.org/download/BattleOfSanPietro1945/BattleOfSanPietro1945.mp4" \
    "tanks" "sanpietro" 120 90

echo ""
echo "============================================"
echo "Library Build Complete!"
echo "============================================"
echo ""
ls -la "$BROLL_DIR"/*.mp4 2>/dev/null | wc -l
echo "total clips created"
