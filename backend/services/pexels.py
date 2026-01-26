"""
Pexels API Service for fetching and caching B-roll footage.
Focuses on masculine/warfare themes for viral clip factory climax montages.
"""

import os
import random
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from config import settings
from models import BRollClip, SessionLocal

import logging
logger = logging.getLogger(__name__)


# Pre-defined search queries for trad/masculine B-roll themes
# Each category has multiple search terms for variety
# Focus on REAL footage - documentary, news, nature - NOT reenactments
BROLL_SEARCH_QUERIES = {
    # MILITARY / WARFARE - real footage, documentary style
    "warfare": ["war documentary", "combat footage", "military operation", "troops deployment", "battlefield", "war zone", "military patrol", "armed forces"],
    "ww2": ["world war 2 documentary", "ww2 archive", "1940s war footage", "ww2 bombing", "war archive"],
    "vietnam": ["vietnam documentary", "helicopter vietnam", "military chopper", "war helicopter"],
    "fighter_jets": ["fighter jet takeoff", "f16 flying", "jet engine", "military jet", "air force", "jet afterburner", "supersonic"],
    "helicopters": ["helicopter flying", "military chopper", "helicopter takeoff", "blackhawk", "apache", "chopper"],
    "tanks": ["tank driving", "military tank", "armored vehicle", "tank column"],
    "navy": ["warship", "aircraft carrier", "battleship ocean", "submarine", "naval", "destroyer"],

    # ARCHITECTURE - real buildings
    "crusades": ["jerusalem", "holy land", "ancient ruins", "medieval architecture", "stone walls", "fortress"],
    "swords": ["sword closeup", "blade steel", "knife sharp", "weapon metal"],
    "castles": ["castle ruins", "fortress walls", "medieval castle", "ancient castle", "stone fortress"],
    "cathedrals": ["cathedral interior", "gothic architecture", "stained glass", "church interior", "notre dame", "cologne cathedral", "chapel"],

    # GYM / STRENGTH / FITNESS - real athletes
    "strength": ["powerlifting competition", "weightlifting", "deadlift", "squat", "bench press", "strongman"],
    "gym": ["gym workout", "weight training", "muscle", "fitness", "bodybuilding"],
    "boxing": ["boxing fight", "boxing ring", "mma cage", "knockout", "punch", "fighter"],
    "running": ["sprinter", "marathon", "track race", "runner", "athletics"],

    # EXPLOSIONS / FIRE / DESTRUCTION - real
    "explosions": ["explosion real", "building demolition", "bomb blast", "fireball", "detonation", "blast"],
    "fire": ["wildfire", "fire burning", "flames", "inferno", "house fire", "forest fire"],
    "lightning": ["lightning strike", "thunder storm", "lightning bolt", "electrical storm"],

    # RACING / MOTORSPORT / SPEED - real races
    "racing": ["f1 race", "nascar crash", "drag race", "race car", "motorsport", "racing crash"],
    "cars": ["supercar", "lamborghini", "ferrari", "muscle car burnout", "car drift", "fast car"],
    "motorcycles": ["motorcycle race", "superbike", "harley", "motocross", "bike wheelie"],

    # SPACE / EXPLORATION - real NASA footage
    "space": ["rocket launch nasa", "spacex launch", "space shuttle", "astronaut spacewalk", "satellite", "iss"],
    "rockets": ["rocket liftoff", "missile launch", "spacex", "rocket engine", "launch pad"],
    "mars": ["mars surface", "mars rover", "red planet", "mars landscape"],
    "frontier": ["mountain summit", "everest climb", "expedition", "explorer", "wilderness"],

    # PATRIOTIC / AMERICA
    "patriotic": ["american flag", "usa flag waving", "eagle soaring", "fireworks", "statue liberty"],
    "flags": ["flag waving wind", "american flag", "flags", "banner"],

    # MONEY / SUCCESS / HUSTLE
    "money": ["cash money", "dollar bills", "gold bars", "counting money", "money stack", "wealth"],
    "success": ["stock market", "wall street", "trading", "business", "corporate"],
    "luxury": ["yacht", "private jet", "mansion", "luxury car", "wealthy lifestyle", "penthouse"],

    # NATURE / POWER - real nature footage
    "mountains": ["mountain peak", "summit", "snowy mountain", "alps", "rocky mountains", "everest"],
    "storms": ["tornado", "hurricane", "storm clouds", "severe weather", "cyclone", "supercell"],
    "ocean": ["ocean storm", "massive waves", "tsunami", "rough sea", "surfing big wave"],
    "wilderness": ["wilderness", "forest", "jungle", "backcountry", "wild nature"],

    # HISTORICAL / WARRIORS - sculptures, art, architecture (not cosplay)
    "spartans": ["greek statue", "roman sculpture", "colosseum", "ancient rome", "parthenon", "gladiator statue"],
    "vikings": ["viking ship", "norse", "longship", "scandinavian"],
    "samurai": ["katana sword", "japanese temple", "japan shrine", "samurai armor museum"],
}

# Default categories for B-roll if none specified
# Includes both new influencer footage and legacy warfare clips
DEFAULT_BROLL_CATEGORIES = [
    # New authentic influencer categories
    "money", "luxury", "gym", "sports", "cars", "city",
    "crowd", "relationships", "fashion", "nature_power",
    # Legacy warfare/religious categories
    "warfare", "ww2", "fighter", "explosions",
    "boxing", "patriotic", "cathedrals", "castles"
]

# Category weights for selection
# Higher weight = more likely to be selected
# All categories weighted equally now since Grok selects the right ones
CATEGORY_WEIGHTS = {
    # New authentic influencer categories - HIGH priority (weight 4)
    "money": 4,
    "luxury": 4,
    "gym": 4,
    "sports": 4,
    "cars": 4,
    "city": 4,
    "crowd": 4,
    "relationships": 4,
    "fashion": 3,
    "nature_power": 3,

    # Legacy warfare/military categories (weight 3)
    "war": 3,
    "warfare": 3,
    "ww2": 3,
    "fighter_jets": 3,
    "helicopters": 3,
    "navy": 3,
    "explosions": 3,

    # Chaos/destruction (weight 3)
    "chaos": 3,
    "fire": 3,
    "lightning": 3,
    "storms": 3,

    # Faith/History (weight 2)
    "faith": 2,
    "cathedrals": 2,
    "castles": 2,
    "history": 2,
    "patriotic": 2,

    # Legacy aliases (weight 2)
    "boxing": 3,
    "rockets": 2,
    "mountains": 2,

    # Racing - moderate now (weight 2)
    "racing": 2,
}

# Blacklisted B-roll prefixes - these never fit contextually
BLACKLISTED_PREFIXES = [
    "fire_firemen",  # Firefighters - never fits
    # Animals - too generic, don't match content themes
    "eagles", "wolves", "lions", "sharks", "bulls", "predators",
    "snakes", "animals", "wildlife",
]

# Per-category limits - max clips per montage
# Prevents over-representation of certain content types
CATEGORY_LIMITS = {
    "racing": 3,       # Max 3 racing clips per montage
    "motorcycles": 2,  # Max 2 motorcycle clips per montage
}


class PexelsService:
    """
    Service for fetching B-roll footage from Pexels API.
    Includes local caching to avoid redundant downloads.
    """

    PEXELS_API_URL = "https://api.pexels.com/videos"
    BROLL_DIR = Path("/app/assets/broll")

    def __init__(self):
        self.api_key = settings.PEXELS_API_KEY
        self.BROLL_DIR.mkdir(parents=True, exist_ok=True)
        self._used_clips_path = self.BROLL_DIR / "used_clips.json"

    def _load_used_clips(self) -> dict:
        """Load the used clips tracking state per category."""
        if self._used_clips_path.exists():
            try:
                with open(self._used_clips_path, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}  # category -> [list of used filenames]

    def _save_used_clips(self, used: dict):
        """Save the used clips tracking state."""
        try:
            with open(self._used_clips_path, "w") as f:
                json.dump(used, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save used clips state: {e}")

    def _mark_clip_used(self, clip_path: str, category: str):
        """Mark a clip as used for a category. Resets when category exhausted."""
        used = self._load_used_clips()
        filename = Path(clip_path).name
        if category not in used:
            used[category] = []
        if filename not in used[category]:
            used[category].append(filename)
        self._save_used_clips(used)

    def _get_unused_clips(self, available: list, category: str) -> list:
        """
        Filter to unused clips for this category.
        If all clips exhausted, reset tracking and return all.
        """
        used = self._load_used_clips()
        used_in_cat = set(used.get(category, []))

        # Filter to unused
        unused = [p for p in available if Path(p).name not in used_in_cat]

        if not unused and available:
            # Category exhausted - reset and return all
            logger.info(f"B-roll category '{category}' exhausted ({len(used_in_cat)} clips), resetting")
            used[category] = []
            self._save_used_clips(used)
            return available

        return unused if unused else available

    def search_videos(
        self,
        query: str,
        per_page: int = 15,
        orientation: str = "portrait",
        size: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        Search Pexels for videos matching query.
        Returns list of video metadata with download URLs.
        """
        if not self.api_key:
            logger.warning("PEXELS_API_KEY not configured")
            return []

        headers = {"Authorization": self.api_key}
        params = {
            "query": query,
            "per_page": per_page,
            "orientation": orientation,
            "size": size
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self.PEXELS_API_URL}/search",
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                return data.get("videos", [])
        except Exception as e:
            logger.error(f"Pexels search failed for '{query}': {e}")
            return []

    def download_video(
        self,
        pexels_video: Dict[str, Any],
        category: str,
        search_query: str
    ) -> Optional[str]:
        """
        Download a Pexels video to local B-roll library.
        Creates/updates BRollClip record.
        Returns local path or None on failure.
        """
        video_id = str(pexels_video["id"])
        db = SessionLocal()

        try:
            # Check if already cached
            existing = db.query(BRollClip).filter(
                BRollClip.pexels_video_id == video_id
            ).first()

            if existing and existing.local_path and os.path.exists(existing.local_path):
                # Update usage stats
                existing.use_count += 1
                existing.last_used_at = datetime.utcnow()
                db.commit()
                logger.info(f"Using cached B-roll: {existing.local_path}")
                return existing.local_path

            # Find best quality MP4 file (prefer portrait for 9:16)
            video_files = pexels_video.get("video_files", [])
            best_file = None

            # First pass: look for portrait HD
            for vf in video_files:
                if vf.get("file_type") == "video/mp4":
                    h = vf.get("height", 0)
                    w = vf.get("width", 0)
                    # Prefer portrait (height > width)
                    if h > w and vf.get("quality") in ["hd", "sd"]:
                        if not best_file or h > best_file.get("height", 0):
                            best_file = vf

            # Second pass: any HD MP4
            if not best_file:
                for vf in video_files:
                    if vf.get("file_type") == "video/mp4" and vf.get("quality") == "hd":
                        best_file = vf
                        break

            # Third pass: any MP4
            if not best_file:
                for vf in video_files:
                    if vf.get("file_type") == "video/mp4":
                        best_file = vf
                        break

            if not best_file:
                logger.warning(f"No suitable video file found for Pexels video {video_id}")
                return None

            # Download video
            download_url = best_file["link"]
            local_path = self.BROLL_DIR / f"pexels_{video_id}.mp4"

            logger.info(f"Downloading B-roll from Pexels: {video_id} -> {local_path}")

            with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                response = client.get(download_url)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(response.content)

            # Create/update DB record
            if existing:
                existing.local_path = str(local_path)
                existing.duration = pexels_video.get("duration", 0)
                existing.width = best_file.get("width", 0)
                existing.height = best_file.get("height", 0)
                existing.use_count += 1
                existing.last_used_at = datetime.utcnow()
            else:
                broll = BRollClip(
                    pexels_video_id=video_id,
                    search_query=search_query,
                    category=category,
                    local_path=str(local_path),
                    duration=pexels_video.get("duration", 0),
                    width=best_file.get("width", 0),
                    height=best_file.get("height", 0),
                    use_count=1,
                    last_used_at=datetime.utcnow()
                )
                db.add(broll)

            db.commit()
            logger.info(f"Downloaded B-roll: {local_path}")
            return str(local_path)

        except Exception as e:
            logger.error(f"Failed to download Pexels video {video_id}: {e}")
            db.rollback()
            return None
        finally:
            db.close()

    def _load_metadata(self) -> Optional[Dict]:
        """Load B-roll metadata.json if it exists."""
        metadata_path = self.BROLL_DIR / "metadata.json"
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata.json: {e}")
        return None

    def _get_clip_categories(self, clip_path: str, metadata: Dict) -> List[str]:
        """
        Get all categories a clip belongs to (from AI metadata + filename).
        Used for enforcing per-category limits.
        """
        categories = set()
        filename = Path(clip_path).name

        # Check AI metadata - clips can be a dict (filename -> info) or list (legacy)
        if metadata and "clips" in metadata and isinstance(metadata["clips"], dict):
            clip_info = metadata["clips"].get(filename, {})
            if isinstance(clip_info, dict):
                categories.update(clip_info.get("categories", []))

        # Check filename prefix (e.g., "racing_nascar_01.mp4" -> "racing")
        parts = filename.split("_")
        if parts:
            prefix = parts[0].lower()
            categories.add(prefix)

        # Map filename prefixes to limit categories
        racing_prefixes = ["racing", "cars", "nascar", "f1", "motorsport", "drag", "drift"]
        if any(prefix in filename.lower() for prefix in racing_prefixes):
            categories.add("racing")

        return list(categories)

    def _check_category_limits(self, clip_path: str, category_counts: Dict[str, int], metadata: Dict) -> bool:
        """
        Check if adding this clip would exceed any category limits.
        Returns True if clip is allowed, False if it would exceed a limit.
        """
        clip_categories = self._get_clip_categories(clip_path, metadata)

        for cat in clip_categories:
            limit = CATEGORY_LIMITS.get(cat)
            if limit is not None and category_counts.get(cat, 0) >= limit:
                logger.debug(f"Skipping clip {Path(clip_path).name} - category '{cat}' limit ({limit}) reached")
                return False

        return True

    def _update_category_counts(self, clip_path: str, category_counts: Dict[str, int], metadata: Dict):
        """Update category counts after adding a clip."""
        clip_categories = self._get_clip_categories(clip_path, metadata)
        for cat in clip_categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    def _build_weighted_category_list(self, categories: List[str]) -> List[str]:
        """
        Build a weighted list of categories for random selection.
        Categories with higher weights appear more frequently.
        """
        weighted_list = []
        for cat in categories:
            weight = CATEGORY_WEIGHTS.get(cat, 1)  # Default weight = 1
            # Multiply category by weight (rounded to int)
            occurrences = max(1, int(weight * 2))  # Scale up for better granularity
            weighted_list.extend([cat] * occurrences)

        random.shuffle(weighted_list)
        return weighted_list

    def get_broll_clips(
        self,
        categories: List[str] = None,
        count: int = 10
    ) -> List[str]:
        """
        Get B-roll clips for specified categories with weighted selection.

        FEATURES:
        - Weighted selection: war/combat clips prioritized (5x weight)
        - Category limits: racing/NASCAR limited to 1 clip per montage
        - AI-tagged metadata used when available
        - Filename-based fallback

        PRIORITY ORDER:
        1. AI-tagged clips from metadata.json (if available)
        2. Filename-based category matching (e.g., warfare_*.mp4)
        3. Fallback: any available local clips

        Categories should be from: war, wealth, faith, strength, nature, people, chaos, victory, power, history

        Returns list of local file paths - ONE UNIQUE CLIP PER BEAT SYNC.
        """
        if not categories:
            categories = DEFAULT_BROLL_CATEGORIES

        db = SessionLocal()
        paths = []
        category_counts = {}  # Track per-category usage for limits

        # Build weighted category list for selection
        weighted_categories = self._build_weighted_category_list(categories)
        logger.info(f"Built weighted category list: {len(weighted_categories)} entries from {len(categories)} categories")

        # Load AI-generated metadata if available
        metadata = self._load_metadata()
        category_index_from_metadata = {}  # category -> [list of paths]

        if metadata and "category_index" in metadata:
            # Use AI-tagged categories from metadata
            category_index_from_metadata = metadata.get("category_index", {})
            logger.info(f"Using AI-tagged metadata with {len(category_index_from_metadata)} categories")

        # STEP 1: Also scan for LOCAL CURATED clips by filename (fallback)
        # Use module-level blacklist for clips that never fit contextually
        local_curated = {}  # category -> [list of paths]
        if self.BROLL_DIR.exists():
            for f in self.BROLL_DIR.glob("*.mp4"):
                if f.name.startswith("pexels_"):
                    continue
                # Skip blacklisted clips
                if any(f.name.lower().startswith(bl) for bl in BLACKLISTED_PREFIXES):
                    continue
                parts = f.name.split("_")
                if parts:
                    cat = parts[0].lower()
                    if cat not in local_curated:
                        local_curated[cat] = []
                    local_curated[cat].append(str(f))

        if local_curated:
            total_curated = sum(len(v) for v in local_curated.values())
            logger.info(f"Found {total_curated} local curated B-roll clips in {len(local_curated)} filename categories")

        # We need AT LEAST 'count' unique clips
        max_iterations = count * 5  # More iterations since we may skip due to limits
        iterations = 0

        try:
            while len(paths) < count and iterations < max_iterations:
                iterations += 1

                # Weighted random category selection
                category = random.choice(weighted_categories)

                # PRIORITY 1: AI-tagged clips from metadata
                if category in category_index_from_metadata:
                    available = [
                        str(self.BROLL_DIR / fn)
                        for fn in category_index_from_metadata[category]
                        if str(self.BROLL_DIR / fn) not in paths
                    ]
                    # Filter by category limits
                    available = [p for p in available if self._check_category_limits(p, category_counts, metadata)]
                    # Prefer unused clips, reset when exhausted
                    available = self._get_unused_clips(available, category)

                    if available:
                        clip_path = random.choice(available)
                        if Path(clip_path).exists():
                            paths.append(clip_path)
                            self._mark_clip_used(clip_path, category)
                            self._update_category_counts(clip_path, category_counts, metadata)
                            clip_caption = metadata["clips"].get(Path(clip_path).name, {}).get("caption", "")
                            caption_preview = clip_caption[:60] if clip_caption else "no caption"
                            logger.info(f"Using AI-tagged B-roll for '{category}': {Path(clip_path).name} [{caption_preview}]")
                            continue

                # PRIORITY 2: Filename-based category matching
                # Map Grok category names to filename prefixes in /broll/
                category_to_prefix = {
                    # DESTRUCTION & CONFLICT
                    "war": ["warfare", "military", "ww2", "combat", "soldier", "battle", "tanks"],
                    "chaos": ["chaos", "destruction", "disorder"],
                    "explosions": ["explosions", "explosion", "blast", "detonate"],
                    "storms": ["storms", "storm", "lightning", "thunder", "weather"],
                    "fire": ["fire", "flames", "burning", "inferno"],

                    # MONEY & SUCCESS
                    "money": ["money", "hustle", "trading", "cash", "stock", "bills"],
                    "luxury": ["luxury", "supercar", "mansion", "yacht", "dubai", "rich"],
                    "wealth": ["wealth", "billionaire", "lavish", "opulent"],
                    "city": ["city", "urban", "nightlife", "skyline", "neon", "metropolitan"],

                    # FITNESS & STRENGTH
                    "gym": ["gym", "bodybuilding", "workout", "fitness", "lifting", "exercise", "weightlifting"],
                    "sports": ["sports", "ufc", "football", "basketball", "athletic", "competition"],
                    "boxing": ["boxing", "knockout", "punch", "fight"],
                    "strength": ["strength", "muscle", "powerful", "strong"],

                    # PATRIOTIC & FAITH
                    "patriotic": ["patriotic", "american", "usa", "flag", "america"],
                    "crowd": ["crowd", "rally", "stadium", "concert", "protest", "masses"],
                    "faith": ["faith", "church", "cross", "christian", "religion", "prayer"],
                    "cathedrals": ["cathedrals", "cathedral", "gothic", "castles", "castle"],

                    # ANIMALS & NATURE
                    "lions": ["lions", "lion", "predator", "apex"],
                    "eagles": ["eagles", "eagle", "hawk", "bird"],
                    "wolves": ["wolves", "wolf", "pack"],
                    "nature": ["nature", "mountains", "ocean", "landscape", "scenic"],

                    # MILITARY TECH
                    "jets": ["jets", "jet", "fighter", "aircraft", "aviation", "plane"],
                    "navy": ["navy", "warship", "carrier", "battleship", "destroyer", "ship"],
                    "helicopters": ["helicopters", "helicopter", "chopper", "heli"],

                    # VEHICLES
                    "cars": ["cars", "supercar", "automotive", "vehicle", "driving"],
                    "racing": ["racing", "drift", "motorsport", "race", "speed"],

                    # OTHER
                    "history": ["history", "historical", "ww2", "archive", "rockets", "vintage"],
                    "people": ["people", "men", "women", "person", "human", "relationships"],
                    "fashion": ["fashion", "runway", "streetwear", "designer"],

                    # Legacy mappings (for backwards compatibility)
                    "nature_power": ["nature_power", "volcano", "tsunami", "tornado", "avalanche"],
                    "power": ["power", "dominance", "control", "authority", "money"],
                    "victory": ["victory", "winning", "success", "celebrate", "champion"],
                    "warfare": ["warfare", "military", "combat", "soldier"],  # alias for war
                    "fighter_jets": ["fighter", "jets", "aircraft"],  # alias for jets
                }

                prefixes = category_to_prefix.get(category, [category])
                for prefix in prefixes:
                    if prefix in local_curated:
                        available = [p for p in local_curated[prefix] if p not in paths]
                        # Filter by category limits
                        available = [p for p in available if self._check_category_limits(p, category_counts, metadata)]
                        # Prefer unused clips, reset when exhausted
                        available = self._get_unused_clips(available, prefix)

                        if available:
                            clip_path = random.choice(available)
                            paths.append(clip_path)
                            self._mark_clip_used(clip_path, prefix)
                            self._update_category_counts(clip_path, category_counts, metadata)
                            logger.info(f"Using filename-matched B-roll for '{category}' (prefix: {prefix}): {Path(clip_path).name}")
                            break
                else:
                    # PRIORITY 3: Fallback - any available local clip (respecting limits)
                    if local_curated:
                        for fallback_cat, fallback_clips in local_curated.items():
                            available = [p for p in fallback_clips if p not in paths]
                            # Filter by category limits
                            available = [p for p in available if self._check_category_limits(p, category_counts, metadata)]
                            # Prefer unused clips, reset when exhausted
                            available = self._get_unused_clips(available, fallback_cat)

                            if available:
                                clip_path = random.choice(available)
                                paths.append(clip_path)
                                self._mark_clip_used(clip_path, fallback_cat)
                                self._update_category_counts(clip_path, category_counts, metadata)
                                logger.info(f"Using fallback B-roll (from '{fallback_cat}'): {Path(clip_path).name}")
                                break

        except Exception as e:
            logger.error(f"Error getting B-roll clips: {e}")
        finally:
            db.close()

        # Shuffle the final list to mix categories
        random.shuffle(paths)

        # Log category distribution
        logger.info(f"B-roll category counts: {category_counts}")
        logger.info(f"Returning {len(paths)} unique B-roll clips for {count} requested (categories: {categories})")
        return paths[:count]

    def get_cached_count(self) -> Dict[str, int]:
        """Get count of cached B-roll clips by category."""
        db = SessionLocal()
        try:
            from sqlalchemy import func
            results = db.query(
                BRollClip.category,
                func.count(BRollClip.id)
            ).group_by(BRollClip.category).all()
            return {cat: count for cat, count in results}
        finally:
            db.close()
