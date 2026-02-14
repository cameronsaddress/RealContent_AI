"""
Clip Analyzer Service
Uses LLM to analyze video transcripts and identify high-engagement moments based on persona.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import httpx
from pydantic import BaseModel

from config import settings
from models import ClipPersona, ViralClip, InfluencerVideo, LLMSettings, RenderTemplate
from services.base import BaseService

logger = logging.getLogger(__name__)

# Grok context window limits for chunking large transcripts
GROK_CONTEXT_LIMIT = 2_000_000  # 2M tokens
CHUNK_THRESHOLD = int(GROK_CONTEXT_LIMIT * 0.75)  # 1.5M tokens - trigger chunking

# ============ EFFECT CATALOG FOR GROK DIRECTOR ============
# Creative descriptions help Grok choose effects based on content energy and mood
EFFECT_CATALOG = """
=== COLOR GRADES (choose ONE per clip) ===
- "kodak_warm": Warm golden cinema - for wisdom, expert advice, market insights
- "teal_orange": Hollywood blockbuster teal/orange - for high-energy property reveals, dramatic transformations
- "film_noir": Deep shadows, high contrast - for luxury properties, evening showcases
- "bleach_bypass": Desaturated silvery grit - for renovation before/after, raw market analysis
- "cross_process": Surreal color shifts - for creative property showcases, unique listings
- "golden_hour": Warm amber glow - for aspirational lifestyle, dream home reveals
- "cold_chrome": Steel blue metallic - for data-driven analysis, market reports, tech-forward
- "vintage_tobacco": Aged warm desaturated - for historic properties, classic architecture
- "vibrant": Enhanced saturation (default) - for general property tours, high energy content
- "bw": Full black & white - for dramatic statements, key market predictions

=== MOTION EFFECTS (can combine, but limit to 2 max) ===
- camera_shake: Handheld documentary feel (intensity 2-15px, frequency 1-4Hz)
  USE FOR: exciting property reveals, market breaking news, high energy moments
  MAX 3 ramps per clip, each at critical word timestamps
- temporal_trail: Ghosting/afterimage streaks in short bursts
  USE FOR: dreamy property transitions, aspirational lifestyle content

=== GLITCH EFFECTS (use sparingly - max 1 per clip) ===
- retro_glow: Soft neon bloom glow (intensity 0.2-0.5)
  USE FOR: highlight moments, key statistics, impressive numbers
- wave_displacement: Melting wave distortion in brief bursts (2-3 bursts max, 1s each)
  USE FOR: surprising market data, unexpected price reveals
- heavy_vhs: Extra VHS tracking noise (intensity 1.0-1.5)
  USE FOR: throwback comparisons, historical market references

=== CAPTION STYLES (choose ONE per clip) ===
- "standard": Default karaoke with trigger word scale (general use)
- "pop_scale": Bouncy overshoot scale on every word (high energy, exciting reveals)
- "shake": Vibrating text on triggers (urgent market updates, breaking news)
- "blur_reveal": Words sharpen from blur (cinematic property tours, luxury reveals)

=== AUDIO REACTIVE (boolean flags) ===
- beat_sync: Amplified zoom pulses on music beats (true/false)
  USE FOR: high energy clips with rhythmic delivery
- audio_saturation: Color intensifies on loud moments (true/false)
  USE FOR: passionate market commentary, enthusiastic property reveals

=== B-ROLL TRANSITIONS (choose ONE for clips with B-roll) ===
- "pixelize": Pixel dissolve between cuts (modern aesthetic)
- "radial": Circular wipe (dramatic reveal)
- "dissolve": Smooth cross-fade (cinematic, elegant)
- "slideleft": Slide transition (fast-paced, news style)
- "fadeblack": Fade through black (premium, luxury feel)

=== PULSE INTENSITY ===
- pulse_intensity: 0.1 (subtle) to 0.4 (strong zoom punches on key terms)
  Default 0.25, increase for exciting reveals and breaking data

=== VHS GRAIN ===
- vhs_intensity: 0.5 (light film grain) to 1.5 (heavy VHS noise)
  Default 1.0, reduce for clean/modern content, increase for retro

=== RARE EFFECTS (use VERY SPARINGLY - max 1-2 clips per entire video) ===
- datamosh_segments: Frame-melting I-frame removal glitch [{start, end}] max 3 segments, max 2s each
  USE FOR: most dramatic price reveals, shocking market shifts
  WARNING: Expensive effect. Only use on the MOST impactful clip in the video.
- pixel_sort_segments: Glitch art pixel sorting by brightness [{start, end}] max 3 segments, max 2s each
  USE FOR: data visualization moments, dramatic stat reveals
  WARNING: Expensive effect. Only use on the MOST visually impactful clip.

=== BGM MOOD (choose ONE per clip) ===
Background music sets the EMOTIONAL TONE. Match the mood to the clip's energy and content.
- "dark": Serious, urgent - for market warnings, rate hikes, cautionary advice
- "triumphant": Epic, victorious, ascending - for dream home reveals, success stories, market wins
- "aggressive": High-energy, intense - for competitive markets, bidding wars, urgent opportunities
- "melancholic": Somber, reflective, bittersweet - for market downturns, lessons learned, buyer challenges
- "hype": High energy, pump-up, exciting - for new listings, open houses, exciting neighborhoods
"""

# Known effect keys for validation
VALID_EFFECT_KEYS = {
    "color_grade", "camera_shake", "temporal_trail",
    "retro_glow", "wave_displacement", "heavy_vhs", "caption_style",
    "beat_sync", "audio_saturation", "transition", "pulse_intensity",
    "vhs_intensity", "datamosh_segments", "pixel_sort_segments", "bgm_mood"
}

class ViralSegment(BaseModel):
    start_time: float
    end_time: float
    clip_type: str # educational, market_update, property_tour, testimonial
    virality_explanation: str
    suggested_caption: str
    hashtags: List[str]

class ClipAnalyzerService(BaseService):
    
    async def analyze_video(self, video_id: int, db_session) -> List[ViralClip]:
        """
        Full analysis pipeline for a video:
        1. Fetch video & transcript from DB
        2. Fetch Persona
        3. Send to Grok
        4. Parse results
        5. Create ViralClip records
        """
        video = db_session.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if not video or not video.transcript_json:
            raise ValueError("Video not found or no transcript available")
            
        influencer = video.influencer
        persona = influencer.persona
        if not persona:
            # Fallback to default if no persona attached
            persona = db_session.query(ClipPersona).first() 
            
        if not persona:
            raise ValueError("No persona available for analysis")

        # Prepare Transcript Text (Reader friendly)
        transcript_data = video.transcript_json
        full_text = transcript_data.get("text", "")
        segments = transcript_data.get("segments", [])
        
        # We need to send segments to LLM so it can reference timestamps
        # Optimization: Don't send word-level precision to LLM context window if too large, 
        # but Grok has large context window so sending ~60 mins is usually fine.
        concise_segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"]} 
            for s in segments
        ]

        # Check for VIRAL_SYSTEM_PROMPT setting override
        sys_prompt_setting = db_session.query(LLMSettings).filter(LLMSettings.key == "VIRAL_SYSTEM_PROMPT").first()
        prompt_override = sys_prompt_setting.value if sys_prompt_setting and sys_prompt_setting.value else None

        # Fetch all templates for Grok to choose from
        templates = db_session.query(RenderTemplate).order_by(RenderTemplate.sort_order).all()

        # Estimate token count to check if chunking is needed
        segments_json = json.dumps(concise_segments, indent=None)
        estimated_tokens = self._estimate_tokens(segments_json)
        logger.info(f"Estimated tokens for video {video_id}: {estimated_tokens:,} (threshold: {CHUNK_THRESHOLD:,})")

        # Route to chunked analysis if transcript is too large
        if estimated_tokens > CHUNK_THRESHOLD:
            logger.info(f"Large transcript detected, splitting into chunks for video {video_id}")
            analysis_result = await self._analyze_video_chunked(
                video, concise_segments, segments, persona, prompt_override, templates, db_session
            )
        else:
            video.status = "analyzing (grok)"
            db_session.commit()

            prompt = self._build_analysis_prompt(video, concise_segments, persona, prompt_override, templates)

            analysis_result = await self._call_grok(prompt)
        
        # Save analysis to video record
        video.status = "analyzing (processing clips)"
        db_session.commit()
        
        # Save analysis to video record
        video.analysis_json = analysis_result
        video.status = "analyzed"

        db_session.query(ViralClip).filter(
            ViralClip.source_video_id == video.id
        ).delete(synchronize_session=False)

        # Create Viral Clips
        created_clips = []
        target_max_duration = 60  # Target max - try to stay under this
        absolute_max_duration = 70  # Absolute max - can extend slightly to finish sentence
        min_clip_duration = 30  # Minimum duration

        def find_sentence_end(target_time: float, max_extension: float = 10.0) -> float:
            """Find the end of the current sentence near target_time.
            Returns a time up to max_extension seconds later if it helps complete a sentence."""
            # Look for transcript segments that end near or after target_time
            for seg in segments:
                seg_end = seg.get("end", 0)
                seg_text = seg.get("text", "")
                # If this segment ends within the extension window
                if target_time <= seg_end <= target_time + max_extension:
                    # Check if it ends with sentence-ending punctuation
                    if seg_text.strip().endswith(('.', '!', '?', '"', "'")):
                        return seg_end
            # No good sentence boundary found, return original
            return target_time

        for clip_data in analysis_result.get("clips", []):
            try:
                # Validate timestamps
                start = float(clip_data.get("start", 0))
                end = float(clip_data.get("end", 0))
                if end <= start: continue

                # Enforce duration limits with sentence extension
                duration = end - start
                if duration > target_max_duration:
                    # Clip exceeds target - truncate to target_max but try to find sentence end
                    tentative_end = start + target_max_duration
                    extended_end = find_sentence_end(tentative_end, absolute_max_duration - target_max_duration)
                    if extended_end > tentative_end:
                        logger.info(f"Clip extended from {target_max_duration:.1f}s to {extended_end - start:.1f}s to finish sentence")
                        end = extended_end
                    else:
                        logger.warning(f"Clip duration {duration:.1f}s exceeds target {target_max_duration}s, truncating")
                        end = tentative_end
                    duration = end - start
                if duration < min_clip_duration:
                    logger.warning(f"Clip duration {duration:.1f}s below min {min_clip_duration}s, skipping")
                    continue
                
                # Extract climax_time (Grok provides absolute timestamp)
                climax_time = clip_data.get("climax_time")
                if climax_time is not None:
                    climax_time = float(climax_time)
                    # Validate climax is within clip bounds
                    if climax_time < start or climax_time > end:
                        logger.warning(f"Climax time {climax_time} outside clip bounds [{start}, {end}], defaulting to middle of clip")
                        climax_time = start + (duration * 0.5)  # Default to 50% through clip

                # Extract recommended template from Grok
                recommended_template_id = clip_data.get("template_id")
                if recommended_template_id is not None:
                    recommended_template_id = int(recommended_template_id)

                # Extract hybrid B-roll insertions (new system)
                broll_insertions = clip_data.get("broll_insertions", [])
                # MUST match the 30 categories in assets/broll/metadata.json
                valid_local_categories = [
                    # Property & Architecture
                    "property", "kitchen", "interior", "exterior", "renovation",
                    # Lifestyle & Neighborhood
                    "neighborhood", "city", "nature", "luxury", "community",
                    # Business & Success
                    "money", "wealth", "office", "handshake", "celebration",
                    # People & Social
                    "family", "crowd", "people", "testimonial",
                    # Data & Market
                    "charts", "data", "growth", "construction",
                    # Scenic
                    "aerial", "sunset", "waterfront", "downtown",
                    # Vehicles & Transport
                    "cars", "moving",
                    # Other
                    "history", "victory", "power"
                ]
                # Legacy aliases - map old category names to new ones
                category_aliases = {
                    "house": "property",
                    "home": "property",
                    "building": "construction",
                    "skyline": "downtown",
                    "ocean": "waterfront",
                    "yard": "exterior",
                    "remodel": "renovation"
                }
                validated_insertions = []
                local_categories_used = set()
                for ins in broll_insertions:
                    if not isinstance(ins, dict):
                        continue
                    source = ins.get("source", "")
                    if source == "local":
                        cat = ins.get("category", "")
                        cat = category_aliases.get(cat, cat)
                        if cat in valid_local_categories:
                            validated_insertions.append({
                                "time": float(ins.get("time", 0)),
                                "source": "local",
                                "category": cat,
                                "visual": ins.get("visual", "")
                            })
                            local_categories_used.add(cat)
                    elif source == "youtube":
                        query = ins.get("query", "")
                        if query and len(query) > 5:
                            validated_insertions.append({
                                "time": float(ins.get("time", 0)),
                                "source": "youtube",
                                "query": query,
                                "visual": ins.get("visual", "")
                            })
                if validated_insertions:
                    logger.info(f"B-roll insertions: {len(validated_insertions)} points ({len([i for i in validated_insertions if i['source']=='local'])} local, {len([i for i in validated_insertions if i['source']=='youtube'])} youtube)")

                # Extract local categories for backwards compatibility
                broll_categories = list(local_categories_used) if local_categories_used else []

                # Extract and validate director effects from Grok
                raw_effects = clip_data.get("effects", {})
                director_effects = {}
                if isinstance(raw_effects, dict):
                    for key, value in raw_effects.items():
                        if key in VALID_EFFECT_KEYS:
                            director_effects[key] = value
                        else:
                            logger.warning(f"Unknown effect '{key}' from Grok, dropping")
                    if director_effects:
                        logger.info(f"Director effects for clip: {list(director_effects.keys())}")

                # Extract topic B-roll queries (for dynamic current-event footage)
                topic_broll = clip_data.get("topic_broll", [])
                if isinstance(topic_broll, list):
                    # Filter to valid strings only
                    topic_broll = [q for q in topic_broll if isinstance(q, str) and len(q) > 3][:3]
                    if topic_broll:
                        logger.info(f"Topic B-roll queries: {topic_broll}")

                # Extract topic keywords for YouTube transcript matching
                topic_broll_keywords = clip_data.get("topic_broll_keywords", [])
                if isinstance(topic_broll_keywords, list):
                    topic_broll_keywords = [k for k in topic_broll_keywords if isinstance(k, str) and len(k) > 1][:8]
                    if topic_broll_keywords:
                        logger.info(f"Topic B-roll keywords: {topic_broll_keywords}")

                # Extract cold open hook data for retention optimization
                hook_data = clip_data.get("hook", {})
                if isinstance(hook_data, dict):
                    hook_phrase = hook_data.get("hook_phrase", "")
                    hook_timestamp = hook_data.get("hook_timestamp")
                    visual_hook_time = hook_data.get("visual_hook_time")
                    # Validate timestamps are within clip bounds
                    if hook_timestamp is not None:
                        hook_timestamp = float(hook_timestamp)
                        if hook_timestamp < start or hook_timestamp > end:
                            hook_timestamp = None  # Invalid, will use fallback
                    if visual_hook_time is not None:
                        visual_hook_time = float(visual_hook_time)
                        if visual_hook_time < start or visual_hook_time > end:
                            visual_hook_time = climax_time  # Fallback to climax
                    if hook_phrase:
                        logger.info(f"Cold open hook: '{hook_phrase}' at {hook_timestamp}s, visual at {visual_hook_time}s")
                else:
                    hook_phrase = ""
                    hook_timestamp = None
                    visual_hook_time = None

                # Extract emotional arc data for dynamic intensity ramp
                emotional_arc = clip_data.get("emotional_arc", {})
                if isinstance(emotional_arc, dict):
                    setup_end = emotional_arc.get("setup_end")
                    escalation_peak = emotional_arc.get("escalation_peak")
                    quotable_line = emotional_arc.get("quotable_line", {})
                    # Validate setup_end and escalation_peak are within clip duration
                    if setup_end is not None:
                        setup_end = float(setup_end)
                        if setup_end < 0 or setup_end > duration:
                            setup_end = duration * 0.2  # Fallback to 20% mark
                    if escalation_peak is not None:
                        escalation_peak = float(escalation_peak)
                        if escalation_peak < 0 or escalation_peak > duration:
                            escalation_peak = duration * 0.7  # Fallback to 70% mark
                    if quotable_line and isinstance(quotable_line, dict):
                        quotable_text = quotable_line.get("text", "")
                        quotable_start = quotable_line.get("start")
                        quotable_end = quotable_line.get("end")
                        if quotable_start is not None:
                            quotable_start = float(quotable_start)
                        if quotable_end is not None:
                            quotable_end = float(quotable_end)
                        if quotable_text:
                            logger.info(f"Quotable line: '{quotable_text[:50]}...' at {quotable_start}-{quotable_end}s")
                    else:
                        quotable_text = ""
                        quotable_start = None
                        quotable_end = None
                    if setup_end or escalation_peak:
                        logger.info(f"Emotional arc: setup_end={setup_end}s, escalation_peak={escalation_peak}s")
                else:
                    setup_end = None
                    escalation_peak = None
                    quotable_text = ""
                    quotable_start = None
                    quotable_end = None

                # Extract dramatic pauses for effect hold
                key_pauses = clip_data.get("key_pauses", [])
                validated_pauses = []
                if isinstance(key_pauses, list):
                    for pause in key_pauses[:3]:  # Max 3 pauses per clip
                        if isinstance(pause, dict):
                            p_start = pause.get("start")
                            p_end = pause.get("end")
                            p_type = pause.get("type", "sink_in")
                            if p_start is not None and p_end is not None:
                                p_start = float(p_start)
                                p_end = float(p_end)
                                # Validate within clip duration
                                if 0 <= p_start < duration and 0 < p_end <= duration and p_end > p_start:
                                    validated_pauses.append({
                                        "start": p_start,
                                        "end": p_end,
                                        "type": p_type
                                    })
                if validated_pauses:
                    logger.info(f"Key pauses: {len(validated_pauses)} dramatic pauses detected")

                # Extract caption pacing data
                rapid_fire_sections = clip_data.get("rapid_fire_sections", [])
                validated_rapid_fire = []
                if isinstance(rapid_fire_sections, list):
                    for section in rapid_fire_sections[:3]:  # Max 3 sections
                        if isinstance(section, dict):
                            rf_start = section.get("start")
                            rf_end = section.get("end")
                            if rf_start is not None and rf_end is not None:
                                rf_start = float(rf_start)
                                rf_end = float(rf_end)
                                if 0 <= rf_start < duration and 0 < rf_end <= duration and rf_end > rf_start:
                                    validated_rapid_fire.append({"start": rf_start, "end": rf_end})

                question_moments = clip_data.get("question_moments", [])
                validated_questions = []
                if isinstance(question_moments, list):
                    for q_time in question_moments[:4]:  # Max 4 questions
                        if isinstance(q_time, (int, float)):
                            q_time = float(q_time)
                            if 0 <= q_time <= duration:
                                validated_questions.append(q_time)

                if validated_rapid_fire or validated_questions:
                    logger.info(f"Caption pacing: {len(validated_rapid_fire)} rapid-fire, {len(validated_questions)} questions")

                vc = ViralClip(
                    source_video_id=video.id,
                    start_time=start,
                    end_time=end,
                    duration=duration,
                    climax_time=climax_time,
                    clip_type=clip_data.get("type", "highlight"),
                    virality_explanation=clip_data.get("reason", ""),
                    title=clip_data.get("title", "Viral Clip"),
                    description=clip_data.get("caption", ""),
                    hashtags=clip_data.get("hashtags", []),
                    status="pending",
                    recommended_template_id=recommended_template_id,
                    template_id=recommended_template_id,
                    render_metadata={
                        "trigger_words": clip_data.get("trigger_words", []),
                        "climax_time": climax_time,
                        "broll_enabled": True,
                        "broll_categories": broll_categories,
                        "broll_insertions": validated_insertions,
                        "director_effects": director_effects,
                        "topic_broll": topic_broll if topic_broll else [],
                        "topic_broll_keywords": topic_broll_keywords if topic_broll_keywords else [],
                        # Platform recommendation from Grok (tiktok or reels - NOT youtube_shorts)
                        "recommended_platform": clip_data.get("recommended_platform", "tiktok"),
                        # Cold open hook for retention
                        "hook_phrase": hook_phrase if hook_phrase else None,
                        "hook_timestamp": hook_timestamp,
                        "visual_hook_time": visual_hook_time,
                        # Emotional arc for dynamic intensity
                        "setup_end": setup_end,
                        "escalation_peak": escalation_peak,
                        "quotable_line": {
                            "text": quotable_text,
                            "start": quotable_start,
                            "end": quotable_end
                        } if quotable_text else None,
                        # Dramatic pauses for effect hold
                        "key_pauses": validated_pauses if validated_pauses else [],
                        # Caption pacing
                        "rapid_fire_sections": validated_rapid_fire if validated_rapid_fire else [],
                        "question_moments": validated_questions if validated_questions else [],
                    }
                )
                db_session.add(vc)
                created_clips.append(vc)
            except Exception as e:
                logger.error(f"Error creating clip record: {e}")
                
        db_session.commit()
        return created_clips

    def _build_analysis_prompt(self, video: InfluencerVideo, segments: List[Dict], persona: ClipPersona, prompt_override: Optional[str] = None, templates: List[RenderTemplate] = None, chunk_info: Optional[str] = None) -> str:
        """Construct the prompt for Grok - includes climax detection for B-roll montages and template selection"""

        system_instructions = prompt_override if prompt_override else f"""
You are an expert viral content editor acting as '{persona.name}'.
{persona.description}
"""

        # Add chunk context if analyzing a portion of a large video
        chunk_context = ""
        if chunk_info:
            chunk_context = f"""
IMPORTANT: This is {chunk_info} of the video. Analyze ONLY this portion.
All timestamps are ABSOLUTE (relative to original video start, not chunk start).
"""
        # Target 45-second clips (30-60s range) for optimal virality
        target_duration = 45
        min_duration = 30
        max_duration = 60

        # Build template selection section with detailed descriptions for smart selection
        template_section = ""
        if templates:
            template_descriptions = []
            for t in templates:
                # Get keywords for matching hints
                keywords = t.keywords if isinstance(t.keywords, list) else []
                keyword_hint = f" [Keywords: {', '.join(keywords[:5])}]" if keywords else ""
                broll_status = "WITH B-roll montage" if t.broll_enabled else "NO B-roll (speaker visible throughout)"
                template_descriptions.append(
                    f"  ID {t.id}: {t.name} ({broll_status})\n"
                    f"    {t.description}\n"
                    f"    {keyword_hint}"
                )
            template_section = f"""

3. TEMPLATE SELECTION: Choose the BEST visual template for each clip based on CONTENT and ENERGY.

AVAILABLE TEMPLATES (read descriptions carefully - each creates a VERY different video):
{chr(10).join(template_descriptions)}

TEMPLATE SELECTION RULES:
- MATCH the template to the clip's EMOTIONAL ENERGY and CONTENT TYPE
- Market analysis/data → Glitch Storm (ID 5) or Documentary Dark (ID 9)
- Expert quotes/advice → Kinetic Quote (ID 6)
- Multi-point tips/lists → Grid Recap (ID 8)
- Serious market commentary → Documentary Dark (ID 9) or Black & White (ID 2)
- High energy property reveals → Velocity Beast (ID 3) or Whip Flash (ID 10)
- Major announcements/reveals → Trailer Drop (ID 4)
- General market updates → Maximum Impact (ID 1)

DO NOT default to Maximum Impact for everything - use the variety of templates based on content."""

        # Local thematic B-Roll categories
        local_broll_descriptions = """
=== LOCAL THEMATIC B-ROLL (for thematic/mood content) ===
Use these when the speaker discusses THEMES, IDEAS, or CONCEPTS — NOT specific events.

⚠️ CRITICAL: You MUST ONLY use these category names. Any other category name will FAIL:
  property, kitchen, interior, exterior, renovation,
  neighborhood, city, nature, luxury, community,
  money, wealth, office, handshake, celebration,
  family, crowd, people, testimonial,
  charts, data, growth, construction,
  aerial, sunset, waterfront, downtown,
  cars, moving,
  history, victory, power

CATEGORY DESCRIPTIONS:

=== PROPERTY & ARCHITECTURE ===
"property" - House exteriors, curb appeal, property walk-ups
   USE FOR: property references, "this home", real estate listings

"kitchen" - Modern kitchens, countertops, appliances, cooking spaces
   USE FOR: kitchen references, renovations, upgrades

"interior" - Living rooms, bedrooms, interior design, staging
   USE FOR: interior tours, design tips, staging advice

"exterior" - Backyards, patios, landscaping, outdoor spaces
   USE FOR: outdoor living, curb appeal, yard references

"renovation" - Before/after, construction work, remodeling
   USE FOR: flips, renovations, fixer-uppers, DIY

=== LIFESTYLE & NEIGHBORHOOD ===
"neighborhood" - Suburban streets, walkable areas, community scenes
   USE FOR: neighborhood reviews, location scouting, community

"city" - Urban skylines, downtown, metropolitan scenes
   USE FOR: urban living, city life, downtown areas

"nature" - Mountains, landscapes, scenic outdoor footage
   USE FOR: scenic areas, nature views, outdoor amenities

"luxury" - High-end properties, premium finishes, luxury lifestyle
   USE FOR: luxury listings, premium market, upscale living

"community" - Parks, schools, local businesses, community events
   USE FOR: community features, family-friendly areas

=== BUSINESS & SUCCESS ===
"money" - Financial imagery, investments, currency
   USE FOR: investment topics, ROI, financial advice

"wealth" - Success symbols, prosperity, achievement
   USE FOR: wealth building, financial freedom through real estate

"office" - Professional settings, meetings, workspaces
   USE FOR: agent life, office tours, professional settings

"handshake" - Deals closing, agreements, partnerships
   USE FOR: closing deals, partnerships, agreements

"celebration" - Celebrations, keys handover, moving day joy
   USE FOR: closing day, new homeowners, success stories

=== PEOPLE & SOCIAL ===
"family" - Families, couples, children in home settings
   USE FOR: family homes, first-time buyers, growing families

"crowd" - Open houses, events, groups of people
   USE FOR: open houses, community events, busy markets

"people" - General people footage, social interactions
   USE FOR: buyer/seller stories, community, demographics

"testimonial" - Happy clients, reviews, success stories
   USE FOR: client testimonials, reviews, social proof

=== DATA & MARKET ===
"charts" - Graphs, charts, data visualizations
   USE FOR: market data, statistics, trends

"data" - Numbers, analytics, market reports
   USE FOR: market analysis, pricing data, inventory stats

"growth" - Growth imagery, upward trends, progress
   USE FOR: market growth, appreciation, rising values

"construction" - New construction, building sites, development
   USE FOR: new builds, development, construction updates

=== SCENIC ===
"aerial" - Drone footage, bird's eye views, aerial shots
   USE FOR: neighborhood overviews, property aerials

"sunset" - Golden hour, sunsets, beautiful lighting
   USE FOR: aspirational content, lifestyle, beauty

"waterfront" - Lakes, ocean, waterfront properties, beaches
   USE FOR: waterfront listings, lake/ocean views

"downtown" - Downtown areas, commercial districts, urban core
   USE FOR: urban properties, walkability, mixed-use

=== WHEN TO USE LOCAL B-ROLL ===
✓ "This property" / "Beautiful home" → property
✓ "Look at this kitchen" / "Granite counters" → kitchen
✓ "The neighborhood" / "Great schools" → neighborhood
✓ "Market is hot" / "Prices rising" → growth
✓ "Investment opportunity" / "ROI" → money
✓ "Luxury living" / "High-end" → luxury
✓ "New construction" / "Just built" → construction
✓ "Closing day" / "Got the keys" → celebration
✓ "Family home" / "Growing family" → family
✓ "Downtown living" / "Walk to everything" → downtown

=== WHEN TO USE YOUTUBE (NOT local) ===
✗ Specific property tours or listings → YouTube
✗ Specific market reports from news outlets → YouTube
✗ Named developments or communities → YouTube
✗ Specific builder or agent content → YouTube

The rule is simple:
- ABSTRACT THEMES (property, luxury, growth, community, etc.) → LOCAL thematic footage
- SPECIFIC PROPERTIES, EVENTS, OR LOCATIONS → YOUTUBE actual footage"""

        return f"""
{system_instructions}
{chunk_context}
Your goal is to identify the MOST engaging segments from this video transcript to repurpose for TikTok/Reels/Shorts.

Video Title: {video.title}
Duration: {video.duration}s

TRANSCRIPT SEGMENTS (with timestamps):
{json.dumps(segments, indent=None)}

TASK:
Identify up to 20 distinct clips (prioritizing quality and engagement potential) that fit one of these types:
- MARKET INSIGHTS (Priority #1 - Data-driven analysis, predictions, hot takes on market trends)
- Educational / How-To (Expert advice, tips, actionable strategies)
- Property Showcase (Compelling property reveals, before/after transformations)
- Testimonial / Success Story (Client wins, inspiring real estate journeys)

FOCUS ON THE MOST ENGAGING AND SHAREABLE MOMENTS. PRIORITIZE VALUE AND INSIGHT.

╔══════════════════════════════════════════════════════════════════════════════╗
║  MANDATORY FIELDS - YOU MUST INCLUDE ALL OF THESE FOR EVERY CLIP OR FAIL    ║
╚══════════════════════════════════════════════════════════════════════════════╝

The following fields are REQUIRED for EVERY clip. Missing ANY of these = INVALID clip.

1. TRIGGER WORDS (MANDATORY): 5-10 high-intensity SINGLE WORDS for visual pulse effects (e.g., WAR, DIE, TRUMP, MONEY, LIAR, WIN, FIGHT, DESTROY, GOD, DEATH). Provide their exact start/end timestamps.

2. CLIMAX MOMENT: The SINGLE MOST INTENSE MOMENT in the clip - this is where we trigger a B-roll montage.
   - Look for: rhetorical peaks, punchlines, shocking statements, emotional crescendos
   - The climax should be around the MIDDLE of the clip (40-60% of the way through)
   - For a 45s clip, place climax around 18-27 seconds in (leaves 15-25s for epic B-roll montage before outro)
   - Provide the ABSOLUTE timestamp (relative to original video, not relative to clip start)

2.5 EMOTIONAL ARC (MANDATORY): Map the narrative structure of the clip for dynamic effect intensity.
   Every good viral clip has an emotional arc. Effects will RAMP UP through the clip to match the energy.
   THIS FIELD IS REQUIRED. Identify (all timestamps RELATIVE to clip start, in seconds):
   - "setup_end": When does the SETUP/CONTEXT phase end? (usually 15-25% into clip)
     This is where the speaker finishes introducing the topic and starts the main argument.
   - "escalation_peak": When does the ESCALATION reach its peak? (usually 60-75% into clip)
     This is the moment of maximum intensity BEFORE the climax/resolution.
   - "quotable_line": The ONE sentence that viewers will screenshot/share.
     Include "text" (the actual line), "start" and "end" timestamps (relative to clip).
     This line gets SPECIAL visual treatment - gold glow, larger scale, held on screen.

   Example emotional_arc for a 45s clip:
   {{
     "setup_end": 10.5,
     "escalation_peak": 32.0,
     "quotable_line": {{
       "text": "Truth doesn't care about your feelings",
       "start": 28.5,
       "end": 31.0
     }}
   }}

   ⚠️ REQUIRED OUTPUT: "emotional_arc": {{"setup_end": X.X, "escalation_peak": Y.Y, "quotable_line": {{...}}}}

2.6 DRAMATIC PAUSES (MANDATORY): Identify moments where the speaker pauses for emphasis (0.5s+ silence).
   These are POWERFUL moments - the silence BEFORE or AFTER a major statement creates tension.
   During these pauses, we will FREEZE all visual effects for dramatic impact.

   Look for:
   - "Pre-bomb pauses": Silence BEFORE delivering a shocking statement (tension building)
   - "Let it sink in": Silence AFTER a powerful line (letting impact resonate)
   - Rhetorical pauses: Mid-sentence pauses for emphasis

   Format: Array of pause objects with timestamps (RELATIVE to clip start):
   "key_pauses": [
     {{"start": 15.2, "end": 16.0, "type": "pre_bomb"}},
     {{"start": 32.5, "end": 33.2, "type": "sink_in"}}
   ]

   Rules:
   - Maximum 3 pauses per clip (only the most impactful ones)
   - Each pause must be at least 0.4 seconds
   - "type" is "pre_bomb" (before statement) or "sink_in" (after statement)

   ⚠️ REQUIRED OUTPUT: "key_pauses": [{{"start": X.X, "end": Y.Y, "type": "sink_in"}}]

2.7 CAPTION PACING (MANDATORY): Identify sections with different speech cadences for caption animation.
   Captions reveal word-by-word to match speech rhythm. Different sections need different pacing:

   "rapid_fire_sections": Sections where speaker fires off points quickly (lists, repeated phrases, rapid arguments)
   Format: Array of time ranges (RELATIVE to clip start)
   [
     {{"start": 20.0, "end": 25.0}},
     {{"start": 35.0, "end": 38.0}}
   ]
   During these sections, captions reveal FASTER to match the energy.

   "question_moments": Timestamps where speaker asks rhetorical questions
   Format: Array of question timestamps (RELATIVE to clip start)
   [12.5, 28.0, 35.5]
   Questions get special treatment: slight pause before, emphasized styling.

   Rules:
   - Maximum 3 rapid_fire_sections per clip
   - Maximum 4 question_moments per clip
   - Only mark sections with NOTICEABLY different pacing

   ⚠️ REQUIRED OUTPUT: "rapid_fire_sections": [...], "question_moments": [...]

3. B-ROLL INSERTIONS (HYBRID - LOCAL + YOUTUBE): Plan 5-10 B-Roll insertion points throughout the clip.
   For EACH insertion, decide whether to use LOCAL thematic footage or YOUTUBE event footage.

{local_broll_descriptions}

   ╔══════════════════════════════════════════════════════════════════════════════╗
   ║  CRITICAL: WHEN TO USE YOUTUBE vs LOCAL                                      ║
   ╚══════════════════════════════════════════════════════════════════════════════╝

   USE YOUTUBE (source: "youtube") FOR:
   ✓ ANY SPECIFIC PROPERTY or LISTING mentioned
   ✓ ANY SPECIFIC MARKET or LOCATION discussed
   ✓ ANY CURRENT NEWS or RECENT EVENTS (2024-2026)
   ✓ ANY CATEGORY NOT IN OUR LOCAL CATEGORIES
   ✓ When you need ACTUAL FOOTAGE of something they're discussing
   ✓ Property tours, market reports, neighborhood reviews, etc.

   USE LOCAL (source: "local") FOR:
   ✓ ABSTRACT THEMES: war, chaos, power, victory, strength, etc.
   ✓ EMOTIONAL MOMENTS: faith, crowd, nature
   ✓ METAPHORS: "be a lion" → lions, "soaring" → eagles
   ✓ Generic imagery to match mood/energy

   ⚠️ IF YOU'RE UNSURE OR THE CATEGORY DOESN'T EXIST → USE YOUTUBE
   Our system will fetch YouTube videos, transcribe them, and extract the best clips.

   For each insertion, specify:
   - "time": seconds into the clip (relative to clip START, not video start)
   - "source": "local" or "youtube"
   - For "local": include "category" - MUST be one of the 30 valid categories above
   - For "youtube": include "query" (specific search query - be DETAILED, include year if recent)
   - "visual": brief description of what should be shown

   Example broll_insertions for a clip about a hot housing market:
   [
     {{"time": 5, "source": "youtube", "query": "housing market update 2025 analysis", "visual": "Market data presentation"}},
     {{"time": 12, "source": "local", "category": "property", "visual": "Beautiful home exterior"}},
     {{"time": 18, "source": "local", "category": "growth", "visual": "Upward trend imagery"}},
     {{"time": 24, "source": "youtube", "query": "open house walkthrough tour 2025", "visual": "Busy open house"}},
     {{"time": 30, "source": "local", "category": "celebration", "visual": "Keys handover moment"}}
   ]

   Example for a clip discussing luxury waterfront properties:
   [
     {{"time": 3, "source": "youtube", "query": "luxury waterfront home tour 2025", "visual": "Waterfront property aerial"}},
     {{"time": 10, "source": "local", "category": "waterfront", "visual": "Ocean/lake views"}},
     {{"time": 18, "source": "local", "category": "interior", "visual": "Luxury interior design"}},
     {{"time": 25, "source": "local", "category": "luxury", "visual": "Premium finishes and amenities"}},
     {{"time": 32, "source": "local", "category": "sunset", "visual": "Golden hour property view"}}
   ]

   IMPORTANT: Mix sources naturally based on what the speaker is saying at each moment.
   - Specific events/people/news → YOUTUBE (with detailed search query + year)
   - Abstract themes/emotions → LOCAL (from our 30 categories)

4. TOPIC B-ROLL QUERIES (for YouTube video discovery): Provide 1-3 search queries to find relevant YouTube videos.
   IMPORTANT: Read the transcript BEFORE the clip start time to understand the FULL CONTEXT.
   The speaker often introduces a topic 30-60 seconds before the clip's most viral moment.
   Use that preceding context to understand WHAT SPECIFIC EVENT, PERSON, or SITUATION
   is being discussed, then find YouTube coverage of THAT EXACT thing.

   THIS IS THE PRIMARY B-ROLL. We will download YouTube videos, fetch their full TRANSCRIPTS,
   and then use a SECOND AI pass to pick the EXACT TIMESTAMPS showing the relevant visual content.
   So your search queries must find videos that actually SHOW the event being discussed.

   You must provide TWO things:
   a) "topic_broll": 1-3 YouTube search queries to find relevant videos
   b) "topic_broll_keywords": 3-8 keywords/phrases that would appear in the YouTube video's transcript
      at the exact moment the topic is being visually shown or discussed

   Rules for search queries (TARGET RAW FOOTAGE, NOT commentary):
   - NEVER include channel names (Fox News, CNN, Crowder, etc.) - those return desk commentary
   - USE footage-targeting terms: "footage", "raw video", "caught on camera", "bodycam",
     "live stream", "full video", "original clip", "compilation", "CCTV", "dashcam"
   - Be SPECIFIC about the event: "church protesters attack footage 2025" NOT "religious controversy"
   - Include the YEAR for current events to get recent uploads
   - For people: use their NAME + the specific event + "footage" or "video"
   - For political events: include "rally footage" or "live stream" or "crowd video"
   - For confrontations: include "caught on camera" or "raw footage" or "full incident"
   - For court/legal: include "hearing footage" or "courtroom video" or "press conference"
   - Goal: find videos that SHOW the event, not pundits TALKING ABOUT it

   Rules for keywords:
   - These are words that would be SPOKEN in the YouTube video's transcript at the relevant moment
   - Use the NAMES of people/places/events the speaker discussed BEFORE and DURING the clip
   - Include proper nouns: names of people, places, organizations, bills, events
   - Include both formal and informal references ("Biden" AND "Joe")
   - Single words or short 2-3 word phrases work best for matching
   - Order by importance (most specific/unique first)

   Examples:
   - Speaker discussed a housing market crash (context: mentioned inventory surge, rate hikes):
     queries: ["housing market crash analysis 2025", "real estate inventory surge footage"]
     keywords: ["housing", "market", "inventory", "rates", "crash", "correction", "buyers"]
   - Speaker discussed a luxury property (context: mentioned waterfront, custom build, $2M):
     queries: ["luxury waterfront home tour 2025", "custom built mansion walkthrough"]
     keywords: ["waterfront", "luxury", "custom", "million", "property", "tour"]
   - Speaker discussed mortgage rates (context: mentioned Fed, 30-year fixed, refinancing):
     queries: ["mortgage rate update 2025 analysis", "fed interest rate decision housing"]
     keywords: ["mortgage", "rates", "Fed", "refinance", "fixed", "housing", "market"]
   - Speaker discussed first-time buyers (context: mentioned down payment, FHA, market entry):
     queries: ["first time home buyer tips 2025", "FHA loan guide new buyers"]
     keywords: ["first-time", "buyer", "FHA", "down payment", "pre-approval", "closing"]

{template_section}

4. EFFECTS DIRECTION: You are the MOVIE DIRECTOR. For each clip, choose visual effects that match
   the ENERGY, MOOD, and CONTENT. Think like a music video director or film editor.

{EFFECT_CATALOG}

EFFECTS SELECTION PHILOSOPHY:
- MARKET URGENCY clips (rate changes, hot markets): camera_shake + shake captions + high pulse + bgm_mood:"aggressive"
- INSPIRATIONAL clips (success stories, dream homes): golden_hour/kodak_warm + retro_glow + pop_scale captions + bgm_mood:"triumphant"
- DATA/ANALYSIS clips (market reports, predictions): cold_chrome + blur_reveal + low vhs + bgm_mood:"dark"
- FUN/ENGAGING clips (open houses, neighborhood tours): cross_process + pop_scale captions + high pulse + bgm_mood:"hype"
- PROPERTY REVEALS (luxury listings, transformations): teal_orange + retro_glow + beat_sync + high pulse + bgm_mood:"triumphant"
- CAUTIONARY clips (market warnings, common mistakes): kodak_warm/bleach_bypass + blur_reveal + bgm_mood:"melancholic"
- Use temporal_trail sparingly for aspirational lifestyle content
- wave_displacement should be RARE - only for truly surprising market data
- DO NOT over-stack effects. 2-3 effects per clip is ideal. Max 4.

⚠️ bgm_mood IS REQUIRED IN EFFECTS - MUST be one of: "dark", "aggressive", "triumphant", "melancholic", "hype"

CRITICAL DURATION CONSTRAINTS (MUST FOLLOW):
- TARGET: {target_duration} seconds per clip
- MINIMUM: {min_duration} seconds (clips shorter than this are rejected)
- MAXIMUM: {max_duration} seconds (NEVER exceed this - clips over {max_duration}s are INVALID)
- If a great moment runs longer than {max_duration}s, SPLIT IT into multiple clips
- Better to have 2 punchy {target_duration}s clips than 1 bloated 90s clip
- Ensure the start and end times cut cleanly (complete sentences)
- Specific instructions: {persona.prompt_template}

5. PLATFORM SELECTION: For each clip, recommend the BEST platform between TikTok and Reels.
   ⚠️ DO NOT recommend YouTube Shorts - only "tiktok" or "reels".

   Platform guidelines:
   - "tiktok": Best for clips 24-38 seconds, high energy content, rapid market updates,
     trending real estate topics, data-driven hot takes, relatable buyer/seller content
   - "reels": Best for clips 7-30 seconds, polished property showcases, lifestyle/aspirational,
     expert advice clips, clean visuals, broader audience appeal, educational content

   Consider:
   - TikTok favors raw authenticity - quick tips and market takes perform well
   - Reels favors polished, aspirational content - luxury showcases and expert advice win
   - Market urgency clips -> TikTok
   - Expert advice / educational clips -> Reels (or TikTok if high energy delivery)
   - Property reveals / before-after -> Reels
   - Data-driven market commentary -> TikTok (algorithm rewards engagement)

╔══════════════════════════════════════════════════════════════════════════════╗
║  FINAL CHECKLIST - EVERY CLIP MUST HAVE ALL OF THESE FIELDS                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ✓ emotional_arc (setup_end, escalation_peak, quotable_line)                 ║
║  ✓ key_pauses (array of pause objects with start, end, type)                 ║
║  ✓ rapid_fire_sections (array of time ranges)                                ║
║  ✓ question_moments (array of timestamps)                                    ║
║  ✓ effects.bgm_mood (one of: dark, aggressive, triumphant, melancholic, hype)║
║  ✓ trigger_words, broll_insertions, climax_time, template_id                 ║
║  ✓ recommended_platform ("tiktok" or "reels" - NOT youtube_shorts)           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Return ONLY valid JSON in this format:
{{
  "clips": [
    {{
      "start": 10.5,
      "end": 55.2,
      "climax_time": 30.0,
      "template_id": 2,
      "type": "market_update",
      "title": "WHY THIS MARKET IS ABOUT TO SHIFT",
      "reason": "Compelling market analysis with actionable data",
      "caption": "Every buyer needs to hear this...",
      "hashtags": ["#realestate", "#housingmarket", "#fyp", "#homebuying"],
      "recommended_platform": "tiktok",
      "emotional_arc": {{
          "setup_end": 12.0,
          "escalation_peak": 35.0,
          "quotable_line": {{
              "text": "Truth doesn't care about your feelings",
              "start": 28.5,
              "end": 31.0
          }}
      }},
      "key_pauses": [
          {{"start": 15.2, "end": 15.8, "type": "pre_bomb"}},
          {{"start": 32.5, "end": 33.0, "type": "sink_in"}}
      ],
      "rapid_fire_sections": [
          {{"start": 22.0, "end": 26.5}}
      ],
      "question_moments": [18.5, 38.0],
      "trigger_words": [
          {{"word": "MARKET", "start": 20.5, "end": 21.0}},
          {{"word": "INVESTMENT", "start": 25.2, "end": 25.8}},
          {{"word": "OPPORTUNITY", "start": 30.0, "end": 30.3}}
      ],
      "broll_insertions": [
          {{"time": 5, "source": "youtube", "query": "luxury home tour walkthrough 2025", "visual": "Property exterior"}},
          {{"time": 12, "source": "local", "category": "interior", "visual": "Stunning interior design"}},
          {{"time": 20, "source": "local", "category": "kitchen", "visual": "Modern kitchen showcase"}},
          {{"time": 28, "source": "local", "category": "growth", "visual": "Market growth imagery"}},
          {{"time": 35, "source": "local", "category": "celebration", "visual": "Closing day celebration"}}
      ],
      "topic_broll": ["luxury home tour 2025 walkthrough", "real estate market update 2025"],
      "topic_broll_keywords": ["property", "market", "home", "listing", "price", "tour"],
      "effects": {{
          "color_grade": "teal_orange",
          "camera_shake": {{"intensity": 8, "frequency": 2.0}},
          "retro_glow": 0.3,
          "caption_style": "shake",
          "beat_sync": true,
          "audio_saturation": false,
          "transition": "pixelize",
          "pulse_intensity": 0.3,
          "vhs_intensity": 1.0,
          "bgm_mood": "aggressive"
      }}
    }},
    {{
      "start": 120.0,
      "end": 165.5,
      "climax_time": 145.0,
      "template_id": 7,
      "type": "educational",
      "title": "THE TRUTH ABOUT BUYING IN THIS MARKET",
      "reason": "Expert breakdown of current market conditions with actionable advice",
      "caption": "Every first-time buyer needs to hear this",
      "hashtags": ["#realestate", "#firsttimebuyer", "#housingmarket", "#homebuying"],
      "recommended_platform": "reels",
      "emotional_arc": {{
          "setup_end": 8.0,
          "escalation_peak": 28.0,
          "quotable_line": {{
              "text": "We don't inherit this world from our ancestors, we borrow it from our children",
              "start": 25.0,
              "end": 29.5
          }}
      }},
      "key_pauses": [
          {{"start": 22.0, "end": 23.0, "type": "pre_bomb"}},
          {{"start": 38.0, "end": 38.8, "type": "sink_in"}}
      ],
      "rapid_fire_sections": [],
      "question_moments": [15.0, 32.0],
      "trigger_words": [
          {{"word": "MARKET", "start": 125.0, "end": 125.5}},
          {{"word": "EQUITY", "start": 130.0, "end": 130.8}},
          {{"word": "INVESTMENT", "start": 142.5, "end": 143.0}},
          {{"word": "OPPORTUNITY", "start": 143.0, "end": 143.4}}
      ],
      "broll_insertions": [
          {{"time": 5, "source": "youtube", "query": "housing market analysis 2025", "visual": "Market data presentation"}},
          {{"time": 12, "source": "local", "category": "property", "visual": "Beautiful home exterior"}},
          {{"time": 20, "source": "local", "category": "neighborhood", "visual": "Tree-lined neighborhood street"}},
          {{"time": 28, "source": "local", "category": "family", "visual": "Happy family in new home"}}
      ],
      "topic_broll": ["real estate investing strategy 2025", "home buying tips first time buyer"],
      "topic_broll_keywords": ["market", "equity", "investment", "buyer", "mortgage", "rates"],
      "effects": {{
          "color_grade": "golden_hour",
          "retro_glow": 0.4,
          "caption_style": "blur_reveal",
          "beat_sync": false,
          "audio_saturation": false,
          "transition": "dissolve",
          "pulse_intensity": 0.2,
          "vhs_intensity": 0.6,
          "bgm_mood": "triumphant"
      }}
    }}
  ]
}}
"""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars = 1 token for English text)"""
        return len(text) // 4

    async def _analyze_video_chunked(
        self,
        video: InfluencerVideo,
        concise_segments: List[Dict],
        full_segments: List[Dict],
        persona: ClipPersona,
        prompt_override: Optional[str],
        templates: List,
        db_session
    ) -> Dict[str, Any]:
        """Split large transcripts into 2 chunks with 2-minute overlap for context continuity"""

        # Find midpoint by timestamp (not segment count)
        video_duration = video.duration
        midpoint_time = video_duration / 2

        # Find segment index closest to midpoint
        mid_idx = 0
        for i, seg in enumerate(concise_segments):
            if seg["start"] >= midpoint_time:
                mid_idx = i
                break

        # Add 2-minute overlap (120 seconds) for context continuity
        overlap_time = 120
        overlap_start_time = midpoint_time - overlap_time

        # Find overlap start index
        overlap_idx = mid_idx
        for i, seg in enumerate(concise_segments):
            if seg["start"] >= overlap_start_time:
                overlap_idx = i
                break

        # Chunk 1: Start to midpoint
        # Chunk 2: overlap_idx to end (includes 2-min overlap)
        chunk1_segments = concise_segments[:mid_idx]
        chunk2_segments = concise_segments[overlap_idx:]

        logger.info(f"Chunk 1: {len(chunk1_segments)} segments (0s - {midpoint_time:.0f}s)")
        logger.info(f"Chunk 2: {len(chunk2_segments)} segments ({overlap_start_time:.0f}s - {video_duration:.0f}s)")

        # Build prompts with chunk context
        chunk1_prompt = self._build_analysis_prompt(
            video, chunk1_segments, persona, prompt_override, templates,
            chunk_info=f"PART 1 of 2 (0:00 - {midpoint_time/60:.0f}:00)"
        )
        chunk2_prompt = self._build_analysis_prompt(
            video, chunk2_segments, persona, prompt_override, templates,
            chunk_info=f"PART 2 of 2 ({overlap_start_time/60:.0f}:00 - {video_duration/60:.0f}:00)"
        )

        # Sequential calls (not parallel to avoid double capacity errors)
        video.status = "analyzing (grok chunk 1/2)"
        db_session.commit()
        logger.info(f"Calling Grok for chunk 1 ({len(chunk1_segments)} segments)...")
        result1 = await self._call_grok(chunk1_prompt)

        video.status = "analyzing (grok chunk 2/2)"
        db_session.commit()
        logger.info(f"Calling Grok for chunk 2 ({len(chunk2_segments)} segments)...")
        result2 = await self._call_grok(chunk2_prompt)

        # Merge results with deduplication
        merged_clips = self._merge_chunk_results(result1, result2, overlap_start_time, midpoint_time)

        return {"clips": merged_clips}

    def _merge_chunk_results(
        self,
        result1: Dict,
        result2: Dict,
        overlap_start: float,
        overlap_end: float
    ) -> List[Dict]:
        """Merge clips from 2 chunks, deduplicating overlapping region"""

        clips1 = result1.get("clips", [])
        clips2 = result2.get("clips", [])

        merged = []

        # Add all clips from chunk 1
        for clip in clips1:
            merged.append(clip)

        # Add clips from chunk 2, but skip duplicates in overlap region
        for clip in clips2:
            clip_start = clip.get("start", 0)
            clip_end = clip.get("end", 0)

            # If clip starts in overlap region, check for duplicates
            if clip_start >= overlap_start and clip_start <= overlap_end:
                # Check if similar clip exists in merged list
                is_duplicate = False
                for existing in merged:
                    existing_start = existing.get("start", 0)
                    existing_end = existing.get("end", 0)
                    # Consider duplicate if >50% overlap
                    overlap = min(clip_end, existing_end) - max(clip_start, existing_start)
                    clip_duration = clip_end - clip_start
                    if clip_duration > 0 and overlap > 0 and overlap / clip_duration > 0.5:
                        is_duplicate = True
                        logger.info(f"Skipping duplicate clip in overlap region: {clip_start:.1f}-{clip_end:.1f}")
                        break
                if not is_duplicate:
                    merged.append(clip)
            else:
                # Clip is outside overlap region, add it
                merged.append(clip)

        # Sort by start time
        merged.sort(key=lambda c: c.get("start", 0))

        logger.info(f"Merged {len(clips1)} + {len(clips2)} clips -> {len(merged)} (after deduplication)")
        return merged

    async def _call_grok(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Call Grok (via OpenRouter or handling direct) with retry logic for capacity errors"""
        async with httpx.AsyncClient() as client:
            last_error = None
            for attempt in range(max_retries):
                try:
                    logger.info(f"Grok API attempt {attempt + 1}/{max_retries}")
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "x-ai/grok-4.1-fast",
                            "messages": [{"role": "user", "content": prompt}],
                            "response_format": {"type": "json_object"},
                            "temperature": 0.7,
                            "max_tokens": 16384  # Ensure full JSON response for 20 clips
                        },
                        timeout=300.0  # Increased timeout for large transcripts
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Check for capacity/service errors in response body
                    if "error" in data:
                        error_msg = data["error"].get("message", str(data["error"]))
                        error_code = data["error"].get("code", 0)
                        if error_code in [502, 503] or "capacity" in error_msg.lower():
                            logger.warning(f"Grok at capacity (attempt {attempt + 1}), waiting before retry...")
                            last_error = ValueError(f"Service at capacity: {error_msg}")
                            await asyncio.sleep(30 * (attempt + 1))  # Exponential backoff: 30s, 60s, 90s
                            continue
                        raise ValueError(f"API error: {error_msg}")

                    if "choices" not in data:
                        logger.error(f"Grok API response missing 'choices': {json.dumps(data)[:2000]}")
                        raise ValueError(f"Invalid API response: {data}")

                    content = data["choices"][0]["message"]["content"]
                    return json.loads(content)

                except httpx.HTTPStatusError as e:
                    logger.error(f"Grok API HTTP error {e.response.status_code}: {e.response.text[:1000]}")
                    last_error = e
                    if e.response.status_code in [502, 503, 429]:
                        await asyncio.sleep(30 * (attempt + 1))
                        continue
                    raise e
                except json.JSONDecodeError as e:
                    logger.warning(f"Grok returned malformed JSON (attempt {attempt + 1}): {e}")
                    last_error = e
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                except Exception as e:
                    if "capacity" in str(e).lower():
                        logger.warning(f"Grok capacity error (attempt {attempt + 1}): {e}")
                        last_error = e
                        await asyncio.sleep(30 * (attempt + 1))
                        continue
                    logger.error(f"Grok API failed: {e}")
                    raise e

            # All retries exhausted
            logger.error(f"Grok API failed after {max_retries} attempts")
            raise last_error or Exception("Grok API failed after all retries")
