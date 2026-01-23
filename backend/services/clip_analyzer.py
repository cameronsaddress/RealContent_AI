"""
Clip Analyzer Service
Uses Grok to analyze video transcripts and identify viral moments based on persona.
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
- "kodak_warm": Warm golden cinema - for wisdom, monologues, profound moments
- "teal_orange": Hollywood blockbuster teal/orange - for confrontation, drama, intensity
- "film_noir": Deep shadows, high contrast - for dark topics, conspiracy, threats
- "bleach_bypass": Desaturated silvery grit - for war, violence, raw conflict
- "cross_process": Surreal color shifts - for absurd humor, weird takes
- "golden_hour": Warm amber glow - for inspirational, hopeful, uplifting
- "cold_chrome": Steel blue metallic - for tech, power plays, authority
- "vintage_tobacco": Aged warm desaturated - for nostalgia, retro references
- "vibrant": Enhanced saturation (default) - for high energy, general use
- "bw": Full black & white - for profound statements, dramatic weight

=== MOTION EFFECTS (can combine, but limit to 2 max) ===
- camera_shake: Handheld documentary feel (intensity 2-15px, frequency 1-4Hz)
  USE FOR: raw emotion, angry rants, chaotic energy
- speed_ramps: Slow-mo emphasis on key words (speed 0.3-0.5, duration 1-2s)
  USE FOR: dramatic pauses, punchlines, "let that sink in" moments
  MAX 3 ramps per clip, each at critical word timestamps
- temporal_trail: Ghosting/afterimage streaks in short bursts
  USE FOR: dreamy sequences, philosophical tangents, altered states

=== GLITCH EFFECTS (use sparingly - max 1 per clip) ===
- retro_glow: Soft neon bloom glow (intensity 0.2-0.5)
  USE FOR: epic revelations, "based" moments, glory
- wave_displacement: Melting wave distortion in brief bursts (2-3 bursts max, 1s each)
  USE FOR: mind-bending takes, reality-breaking statements
- heavy_vhs: Extra VHS tracking noise (intensity 1.0-1.5)
  USE FOR: nostalgic references, "back in my day" content

=== CAPTION STYLES (choose ONE per clip) ===
- "standard": Default karaoke with trigger word scale (general use)
- "pop_scale": Bouncy overshoot scale on every word (high energy, hype)
- "shake": Vibrating text on triggers (aggressive, confrontational)
- "blur_reveal": Words sharpen from blur (cinematic, mysterious)

=== AUDIO REACTIVE (boolean flags) ===
- beat_sync: Amplified zoom pulses on music beats (true/false)
  USE FOR: high energy clips with rhythmic delivery
- audio_saturation: Color intensifies on loud moments (true/false)
  USE FOR: passionate speeches, yelling, emphasis

=== B-ROLL TRANSITIONS (choose ONE for clips with B-roll) ===
- "pixelize": Pixel dissolve between cuts (glitch aesthetic)
- "radial": Circular wipe (dramatic reveal)
- "dissolve": Smooth cross-fade (cinematic, calm)
- "slideleft": Slide transition (fast-paced, news style)
- "fadeblack": Fade through black (serious, weighty)

=== PULSE INTENSITY ===
- pulse_intensity: 0.1 (subtle) to 0.4 (aggressive zoom punches on triggers)
  Default 0.25, increase for confrontational/aggressive clips

=== VHS GRAIN ===
- vhs_intensity: 0.5 (light film grain) to 1.5 (heavy VHS noise)
  Default 1.0, reduce for clean/modern content, increase for retro

=== RARE EFFECTS (use VERY SPARINGLY - max 1-2 clips per entire video) ===
- datamosh_segments: Frame-melting I-frame removal glitch [{start, end}] max 3 segments, max 2s each
  USE FOR: most shocking/unhinged moments, reality-breaking statements, "everything is fake"
  WARNING: Expensive effect. Only use on the MOST viral-worthy clip in the video.
- pixel_sort_segments: Glitch art pixel sorting by brightness [{start, end}] max 3 segments, max 2s each
  USE FOR: surreal/psychedelic moments, existential dread, "simulation theory" takes
  WARNING: Expensive effect. Only use on the MOST visually impactful clip.
"""

# Known effect keys for validation
VALID_EFFECT_KEYS = {
    "color_grade", "camera_shake", "speed_ramps", "temporal_trail",
    "retro_glow", "wave_displacement", "heavy_vhs", "caption_style",
    "beat_sync", "audio_saturation", "transition", "pulse_intensity",
    "vhs_intensity", "datamosh_segments", "pixel_sort_segments"
}

class ViralSegment(BaseModel):
    start_time: float
    end_time: float
    clip_type: str # antagonistic, funny, controversial, inspirational
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

                # Extract B-roll categories from Grok response
                broll_categories = clip_data.get("broll_categories", ["war", "chaos", "power"])
                # Validate categories
                valid_categories = ["war", "wealth", "faith", "strength", "nature", "people", "chaos", "victory", "power", "history"]
                broll_categories = [c for c in broll_categories if c in valid_categories]
                if not broll_categories:
                    broll_categories = ["war", "chaos", "power"]  # Default fallback

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

                vc = ViralClip(
                    source_video_id=video.id,
                    start_time=start,
                    end_time=end,
                    duration=duration,  # Use pre-calculated (possibly truncated) duration
                    climax_time=climax_time,
                    clip_type=clip_data.get("type", "highlight"),
                    virality_explanation=clip_data.get("reason", ""),
                    title=clip_data.get("title", "Viral Clip"),
                    description=clip_data.get("caption", ""),
                    hashtags=clip_data.get("hashtags", []),
                    status="pending",
                    recommended_template_id=recommended_template_id,
                    template_id=recommended_template_id,  # Auto-apply recommendation (user can override)
                    render_metadata={
                        "trigger_words": clip_data.get("trigger_words", []),
                        "climax_time": climax_time,
                        "broll_enabled": True,
                        "broll_categories": broll_categories,
                        "director_effects": director_effects
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
- Religious/faith content → Crusade Core Phonk (ID 7)
- Conspiracy/system/tech topics → Glitch Storm (ID 5)
- Profound quotes/wisdom → Kinetic Quote (ID 6)
- Lists/multiple points → Grid Recap (ID 8)
- Serious/thoughtful monologues → Documentary Dark (ID 9) or Black & White (ID 2)
- High energy/hype moments → Velocity Beast (ID 3) or Whip Flash (ID 10)
- Epic announcements/revelations → Trailer Drop (ID 4)
- General controversy/rants → Maximum Impact (ID 1)

DO NOT default to Maximum Impact for everything - use the variety of templates based on content."""

        # B-roll categories available for visual montage
        broll_categories = [
            "war", "wealth", "faith", "strength", "nature",
            "people", "chaos", "victory", "power", "history"
        ]
        broll_category_descriptions = """
AVAILABLE B-ROLL CATEGORIES (choose 2-4 that match the climax content):
- war: military combat, soldiers, weapons, explosions, destruction
- wealth: money, luxury, business, stock market, expensive cars
- faith: church, prayer, religious symbols, bible imagery
- strength: boxing, fighting, gym, weightlifting, athletics
- nature: mountains, ocean, sunset, wildlife, landscapes
- people: crowds, families, men/women imagery
- chaos: fire, destruction, disorder, darkness
- victory: celebration, triumph, flags, glory
- power: political leaders, speeches, authority figures
- history: historical footage, archival clips, news media"""

        return f"""
{system_instructions}
{chunk_context}
Your goal is to identify the MOST viral segments from this video transcript to repurpose for TikTok/Reels/Shorts.

Video Title: {video.title}
Duration: {video.duration}s

TRANSCRIPT SEGMENTS (with timestamps):
{json.dumps(segments, indent=None)}

TASK:
Identify up to 20 distinct clips (prioritizing quality and viral potential) that fit one of these types:
- OUTRAGEOUS / CONTROVERSIAL (Priority #1 - Hot Takes, "Cancelled" moments)
- Antagonistic / Conflict (High emotion battles)
- Funny / Memeable (Out of context or hilarious)
- Inspirational / "Based" (Strong monologues)

FOCUS ON THE MOST CONTROVERSIAL AND SHOCKING MOMENTS. DO NOT HOLD BACK.

For EACH clip you MUST identify:

1. TRIGGER WORDS: 5-10 high-intensity SINGLE WORDS for visual pulse effects (e.g., WAR, DIE, TRUMP, MONEY, LIAR, WIN, FIGHT, DESTROY, GOD, DEATH). Provide their exact start/end timestamps.

2. CLIMAX MOMENT: The SINGLE MOST INTENSE MOMENT in the clip - this is where we trigger a B-roll montage.
   - Look for: rhetorical peaks, punchlines, shocking statements, emotional crescendos
   - The climax should be around the MIDDLE of the clip (40-60% of the way through)
   - For a 45s clip, place climax around 18-27 seconds in (leaves 15-25s for epic B-roll montage before outro)
   - Provide the ABSOLUTE timestamp (relative to original video, not relative to clip start)

3. B-ROLL CATEGORIES: Choose 2-4 visual categories that best match the CLIMAX CONTENT.
   The B-roll montage will use clips from these categories during the climax moment.
{broll_category_descriptions}
{template_section}

4. EFFECTS DIRECTION: You are the MOVIE DIRECTOR. For each clip, choose visual effects that match
   the ENERGY, MOOD, and CONTENT. Think like a music video director or film editor.

{EFFECT_CATALOG}

EFFECTS SELECTION PHILOSOPHY:
- AGGRESSIVE clips (rants, confrontation): camera_shake + heavy_vhs + shake captions + high pulse
- INSPIRATIONAL clips (wisdom, hope): golden_hour/kodak_warm + retro_glow + pop_scale captions
- CONSPIRACY/DARK clips: film_noir + wave_displacement + blur_reveal + low vhs
- FUNNY/ABSURD clips: cross_process + speed_ramps on punchlines + pop_scale
- EPIC/REVELATION clips: teal_orange + retro_glow + beat_sync + high pulse
- Use speed_ramps to emphasize the EXACT moment of a punchline or shocking statement
- Use temporal_trail sparingly for dreamlike/philosophical tangents
- wave_displacement should be RARE - only for truly mind-bending moments
- DO NOT over-stack effects. 2-3 effects per clip is ideal. Max 4.

CRITICAL DURATION CONSTRAINTS (MUST FOLLOW):
- TARGET: {target_duration} seconds per clip
- MINIMUM: {min_duration} seconds (clips shorter than this are rejected)
- MAXIMUM: {max_duration} seconds (NEVER exceed this - clips over {max_duration}s are INVALID)
- If a great moment runs longer than {max_duration}s, SPLIT IT into multiple clips
- Better to have 2 punchy {target_duration}s clips than 1 bloated 90s clip
- Ensure the start and end times cut cleanly (complete sentences)
- Specific instructions: {persona.prompt_template}

Return ONLY valid JSON in this format:
{{
  "clips": [
    {{
      "start": 10.5,
      "end": 55.2,
      "climax_time": 30.0,
      "template_id": 2,
      "broll_categories": ["war", "chaos", "power"],
      "type": "antagonistic",
      "title": "TOP G DESTROYS DEBATE OPPONENT",
      "reason": "High conflict moment, very engaging",
      "caption": "Bro didn't stand a chance...",
      "hashtags": ["#viral", "#shorts", "#fyp", "#sigma"],
      "trigger_words": [
          {{"word": "DESTROYED", "start": 20.5, "end": 21.0}},
          {{"word": "LIAR", "start": 25.2, "end": 25.8}},
          {{"word": "WAR", "start": 30.0, "end": 30.3}}
      ],
      "effects": {{
          "color_grade": "teal_orange",
          "camera_shake": {{"intensity": 8, "frequency": 2.0}},
          "speed_ramps": [{{"time": 30.0, "speed": 0.3, "duration": 1.5}}],
          "retro_glow": 0.3,
          "caption_style": "shake",
          "beat_sync": true,
          "audio_saturation": false,
          "transition": "pixelize",
          "pulse_intensity": 0.3,
          "vhs_intensity": 1.0
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
                            "temperature": 0.7
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
