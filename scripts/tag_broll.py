#!/usr/bin/env python3
"""
B-Roll Tagging Script

Analyzes B-roll clips using BLIP-2 (captioning) and CLIP (classification) to generate
metadata for smart B-roll matching during viral clip rendering.

Usage:
    # Tag all untagged clips
    python scripts/tag_broll.py

    # Force re-tag all clips
    python scripts/tag_broll.py --force

    # Tag specific directory
    python scripts/tag_broll.py --input /path/to/clips

    # Run on video-processor container (recommended - has GPU)
    docker exec -it SocialGen_video_processor python /app/scripts/tag_broll.py

Requirements (installed in video-processor container):
    pip install transformers torch pillow

Output:
    assets/broll/metadata.json - Contains descriptions, categories, and embeddings for each clip
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile

# Default paths (container paths)
DEFAULT_BROLL_DIR = "/broll"
DEFAULT_METADATA_FILE = "/broll/metadata.json"

# Semantic categories for B-roll classification
BROLL_CATEGORIES = [
    # Conflict & Power
    "war and military combat",
    "soldiers and troops",
    "weapons and firearms",
    "explosions and destruction",
    "political leaders and speeches",

    # Wealth & Success
    "money and cash",
    "luxury and wealth",
    "business and corporate",
    "stock market and trading",
    "cars and vehicles",

    # Faith & Spirituality
    "church and religion",
    "prayer and worship",
    "crosses and religious symbols",
    "bible and scripture",

    # Strength & Masculinity
    "boxing and fighting",
    "gym and weightlifting",
    "sports and athletics",
    "muscles and bodybuilding",

    # Nature & Beauty
    "mountains and landscapes",
    "ocean and water",
    "sunset and sunrise",
    "animals and wildlife",

    # Society & People
    "crowds and masses",
    "family and children",
    "women and femininity",
    "men and masculinity",

    # Emotions & States
    "chaos and disorder",
    "victory and celebration",
    "defeat and failure",
    "anger and rage",

    # Abstract & Symbolic
    "fire and flames",
    "darkness and shadows",
    "light and brightness",
    "american flag and patriotism",

    # Technology & Modern
    "computers and technology",
    "city and urban",
    "historical footage",
    "news and media"
]

# Simplified category mapping for Grok
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


def extract_frame(video_path: str, output_path: str, timestamp: float = 1.0) -> bool:
    """Extract a single frame from video at given timestamp."""
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", str(timestamp),
            "-i", video_path, "-vframes", "1",
            "-q:v", "2", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"  Error extracting frame: {e}")
        return False


def load_models():
    """Load BLIP and CLIP models."""
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load BLIP for captioning
    print("Loading BLIP model for captioning...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    # Load CLIP for classification
    print("Loading CLIP model for classification...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    return {
        "blip_processor": blip_processor,
        "blip_model": blip_model,
        "clip_processor": clip_processor,
        "clip_model": clip_model,
        "device": device
    }


def caption_image(image_path: str, models: dict) -> str:
    """Generate a caption for an image using BLIP."""
    from PIL import Image
    import torch

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = models["blip_processor"](image, return_tensors="pt").to(models["device"])

        with torch.no_grad():
            output = models["blip_model"].generate(**inputs, max_length=50)

        caption = models["blip_processor"].decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"  Error captioning: {e}")
        return ""


def classify_image(image_path: str, models: dict, categories: list) -> list:
    """Classify an image into categories using CLIP zero-shot."""
    from PIL import Image
    import torch

    try:
        image = Image.open(image_path).convert("RGB")

        # Prepare inputs
        inputs = models["clip_processor"](
            text=categories,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(models["device"])

        with torch.no_grad():
            outputs = models["clip_model"](**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)

        # Get top 5 categories
        top_indices = probs[0].topk(5).indices.tolist()
        top_probs = probs[0].topk(5).values.tolist()

        results = []
        for idx, prob in zip(top_indices, top_probs):
            if prob > 0.05:  # Only include if >5% confidence
                results.append({
                    "category": categories[idx],
                    "confidence": round(prob, 3)
                })

        return results
    except Exception as e:
        print(f"  Error classifying: {e}")
        return []


def get_simplified_categories(classifications: list) -> list:
    """Convert detailed classifications to simplified category keywords."""
    simplified = set()
    for item in classifications:
        cat = item["category"]
        for keyword, detailed_cats in CATEGORY_KEYWORDS.items():
            if cat in detailed_cats:
                simplified.add(keyword)
    return list(simplified)


def process_clip(clip_path: str, models: dict, temp_dir: str) -> dict:
    """Process a single B-roll clip and return metadata."""
    clip_name = os.path.basename(clip_path)

    # Extract a frame from the middle of the clip
    frame_path = os.path.join(temp_dir, f"{clip_name}.jpg")

    # Try to extract from 1 second in, fallback to 0.5s
    if not extract_frame(clip_path, frame_path, 1.0):
        if not extract_frame(clip_path, frame_path, 0.5):
            return None

    # Generate caption with BLIP
    caption = caption_image(frame_path, models)

    # Classify with CLIP
    classifications = classify_image(frame_path, models, BROLL_CATEGORIES)

    # Get simplified categories
    categories = get_simplified_categories(classifications)

    # Get video duration
    try:
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", clip_path
        ], capture_output=True, text=True, timeout=10)
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 2.0
    except:
        duration = 2.0

    # Clean up temp frame
    if os.path.exists(frame_path):
        os.unlink(frame_path)

    return {
        "filename": clip_name,
        "path": clip_path,
        "caption": caption,
        "classifications": classifications,
        "categories": categories,
        "duration": duration,
        "tagged_at": datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description="Tag B-roll clips with AI-generated metadata")
    parser.add_argument("--input", "-i", default=DEFAULT_BROLL_DIR, help="Input directory containing B-roll clips")
    parser.add_argument("--output", "-o", default=DEFAULT_METADATA_FILE, help="Output metadata JSON file")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-tag all clips")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit number of clips to process (0=all)")
    args = parser.parse_args()

    broll_dir = Path(args.input)
    metadata_file = Path(args.output)

    if not broll_dir.exists():
        print(f"Error: B-roll directory not found: {broll_dir}")
        sys.exit(1)

    # Load existing metadata
    existing_metadata = {}
    if metadata_file.exists() and not args.force:
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
                existing_metadata = {item["filename"]: item for item in data.get("clips", [])}
            print(f"Loaded {len(existing_metadata)} existing clip records")
        except:
            pass

    # Find all clips
    clips = list(broll_dir.glob("*.mp4"))
    print(f"Found {len(clips)} B-roll clips in {broll_dir}")

    # Filter to only untagged clips
    if not args.force:
        clips = [c for c in clips if c.name not in existing_metadata]
        print(f"{len(clips)} clips need tagging")

    if not clips:
        print("No clips to process!")
        return

    if args.limit > 0:
        clips = clips[:args.limit]
        print(f"Limited to {len(clips)} clips")

    # Load AI models
    print("\nLoading AI models...")
    models = load_models()

    # Process clips
    print(f"\nProcessing {len(clips)} clips...")
    results = list(existing_metadata.values())  # Start with existing

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, clip_path in enumerate(clips):
            print(f"[{i+1}/{len(clips)}] {clip_path.name}")

            try:
                metadata = process_clip(str(clip_path), models, temp_dir)
                if metadata:
                    results.append(metadata)
                    print(f"  Caption: {metadata['caption'][:60]}...")
                    print(f"  Categories: {', '.join(metadata['categories'])}")
                else:
                    print(f"  SKIPPED - Could not extract frame")
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

    with open(metadata_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Tagged {len(clips)} new clips")
    print(f"Total clips in metadata: {len(results)}")
    print(f"Categories: {', '.join(category_index.keys())}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(category_index.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(count)} clips")


if __name__ == "__main__":
    main()
