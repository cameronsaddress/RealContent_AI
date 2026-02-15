"""
Script Generator service for creating viral video scripts using LLM.
Now uses brand persona and source video transcription for smarter scripts.
"""

import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from .base import BaseService
from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)


class ScriptRequest(BaseModel):
    """Request parameters for script generation."""
    content_idea_id: int
    pillar: str = "educational_tips"
    source_url: str = ""
    original_text: str = ""  # Caption/description
    source_transcription: str = ""  # What was actually SAID in the video
    suggested_hook: Optional[str] = None
    why_viral: Optional[str] = None
    target_duration: int = 60  # seconds


class BrandPersonaContext(BaseModel):
    """Brand persona for script context."""
    name: str = "Sarah Mitchell"
    title: str = "Real Estate Expert"
    location: str = "North Idaho & Spokane area"
    bio: str = ""
    tone: str = "professional"
    energy_level: str = "warm"
    humor_style: str = "light"
    core_values: List[str] = []
    content_boundaries: List[str] = []
    response_style: str = ""
    signature_intro: str = "Hey neighbors!"
    signature_cta: str = "DM me to chat about your home journey"
    hashtags: List[str] = []


class ScriptScene(BaseModel):
    """A single scene in the script with B-roll suggestion."""
    scene_number: int
    text: str
    duration_estimate: float  # seconds
    broll_keywords: List[str] = []
    broll_description: str = ""


class GeneratedScript(BaseModel):
    """Generated script with all components."""
    content_idea_id: int
    hook: str
    body: str
    cta: str
    full_text: str
    duration_estimate: float
    scenes: List[ScriptScene] = []
    pillar: str
    # Social captions
    tiktok_caption: str = ""
    ig_caption: str = ""
    yt_title: str = ""
    yt_description: str = ""
    linkedin_text: str = ""
    x_text: str = ""
    facebook_text: str = ""
    threads_text: str = ""
    metadata: Dict[str, Any] = {}


class ScriptGenerator(BaseService):
    """
    Service for generating video scripts using LLM.
    Uses brand persona and source transcription for contextual, authentic scripts.
    """

    # Content pillar guidance
    PILLAR_PROMPTS = {
        "market_intelligence": """
Content Angle: Data-driven insights, market trends, analysis
Style: Expert analyst, use statistics and concrete numbers
Phrases to use: "The data shows...", "Here's what the market is telling us...", "X% of buyers..."
""",
        "educational_tips": """
Content Angle: Actionable advice, how-to content, tips
Style: Helpful teacher, break down complex topics simply
Phrases to use: "Here's what you need to know...", "Pro tip...", "The secret is..."
""",
        "lifestyle_local": """
Content Angle: Community highlights, local life, area features
Style: Friendly neighbor, personal connection to the area
Phrases to use: "Here in [location]...", "Our community...", "What I love about..."
""",
        "brand_humanization": """
Content Angle: Personal stories, behind-the-scenes, relatability
Style: Authentic friend, vulnerable and genuine
Phrases to use: "I'll be honest...", "What people don't see...", "My experience..."
"""
    }

    async def get_brand_persona(self, db_session=None) -> BrandPersonaContext:
        """
        Get brand persona from database or use defaults.

        Args:
            db_session: Optional database session

        Returns:
            BrandPersonaContext with persona settings
        """
        if db_session:
            try:
                from models import BrandPersona
                persona = db_session.query(BrandPersona).filter(BrandPersona.id == 1).first()
                if persona:
                    return BrandPersonaContext(
                        name=persona.name,
                        title=persona.title,
                        location=persona.location,
                        bio=persona.bio,
                        tone=persona.tone,
                        energy_level=persona.energy_level,
                        humor_style=persona.humor_style,
                        core_values=persona.core_values or [],
                        content_boundaries=persona.content_boundaries or [],
                        response_style=persona.response_style,
                        signature_intro=persona.signature_intro,
                        signature_cta=persona.signature_cta,
                        hashtags=persona.hashtags or []
                    )
            except Exception as e:
                logger.warning(f"Could not load persona from DB: {e}")

        return BrandPersonaContext()

    def build_prompt(
        self,
        request: ScriptRequest,
        persona: BrandPersonaContext
    ) -> str:
        """
        Build the LLM prompt for script generation.
        Uses brand persona and source transcription for context.
        """
        pillar_guidance = self.PILLAR_PROMPTS.get(
            request.pillar,
            self.PILLAR_PROMPTS["educational_tips"]
        )

        # Build persona section
        values_text = "\n".join(f"  - {v}" for v in persona.core_values) if persona.core_values else "  - Professionalism and client focus"
        boundaries_text = "\n".join(f"  - {b}" for b in persona.content_boundaries) if persona.content_boundaries else "  - Maintain professional standards"

        prompt = f"""You are writing a script for {persona.name}, {persona.title} in {persona.location}.

=== WHO I AM ===
{persona.bio}

Tone: {persona.tone}
Energy: {persona.energy_level}
Humor: {persona.humor_style}

My Core Values:
{values_text}

Content Boundaries (what I will NEVER do):
{boundaries_text}

{persona.response_style}

=== THE VIRAL VIDEO I'M RESPONDING TO ===
Source URL: {request.source_url or "Direct submission"}
Post Caption: {request.original_text or "No caption"}
"""

        # Add transcription if available - this is the KEY context
        if request.source_transcription:
            prompt += f"""
WHAT THEY ACTUALLY SAY IN THE VIDEO:
\"\"\"{request.source_transcription}\"\"\"
"""
        else:
            prompt += """
(No transcription available - work with the caption only)
"""

        prompt += f"""
Why this might be viral: {request.why_viral or "Unknown"}
Suggested hook angle: {request.suggested_hook or "None provided"}

=== CONTENT DIRECTION ===
Content Pillar: {request.pillar}
{pillar_guidance}

=== YOUR TASK ===
Create a {request.target_duration}-second script where {persona.name} REACTS to this viral video.

The video will play BEHIND {persona.name} on screen. Write the script as if {persona.name} is:
1. Commenting on what's happening in the video
2. Adding professional expertise and local insights
3. Providing genuine value to viewers

STRUCTURE:
1. HOOK (3-5 seconds): Grab attention, reference the viral video
2. BODY (40-50 seconds): React to the content, add value, share expertise
3. CTA (5-7 seconds): Clear call-to-action using: "{persona.signature_cta}"

IMPORTANT:
- Start with "{persona.signature_intro}" or similar greeting
- Reference specific things from the video (if transcription provided)
- Stay in character - maintain the tone and respect the boundaries
- NO emojis in the spoken script
- Make it conversational, not scripted-sounding
- NEVER use "Coeur d'Alene" - the TTS mispronounces it. Use "North Idaho" or "the Spokane area" instead

Return as JSON:
{{
  "hook": "Opening hook (reference the video)...",
  "body": "Main content with reactions and value...",
  "cta": "Clear call-to-action...",
  "duration_estimate": 55.0,
  "tiktok_caption": "TikTok caption with hashtags (can use emojis)...",
  "ig_caption": "Instagram caption...",
  "yt_title": "YouTube Shorts title...",
  "yt_description": "YouTube description...",
  "linkedin_text": "LinkedIn post text (professional)...",
  "x_text": "X/Twitter text (under 280 chars)...",
  "facebook_text": "Facebook post...",
  "threads_text": "Threads post..."
}}"""

        return prompt

    async def generate_script(
        self,
        request: ScriptRequest,
        db_session=None
    ) -> GeneratedScript:
        """
        Generate a video script using Grok via OpenRouter.

        Args:
            request: Script generation request parameters
            db_session: Optional database session for loading persona

        Returns:
            Generated script with hook, body, CTA, and social captions
        """
        # Get brand persona
        persona = await self.get_brand_persona(db_session)

        # Build prompt with persona and transcription context
        prompt = self.build_prompt(request, persona)

        try:
            response = await self._post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "x-ai/grok-4.1-fast",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.8
                },
                timeout=60.0
            )

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            return self.parse_response(parsed, request, persona)

        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            raise

    def parse_response(
        self,
        raw_response: Dict[str, Any],
        request: ScriptRequest,
        persona: BrandPersonaContext
    ) -> GeneratedScript:
        """Parse LLM response into structured script."""
        hook = raw_response.get("hook", "")
        body = raw_response.get("body", "")
        cta = raw_response.get("cta", "")

        # Build full text
        full_text = f"{hook} {body} {cta}".strip()

        return GeneratedScript(
            content_idea_id=request.content_idea_id,
            hook=hook,
            body=body,
            cta=cta,
            full_text=full_text,
            duration_estimate=raw_response.get("duration_estimate", 55.0),
            scenes=[],  # Could be added later if needed
            pillar=request.pillar,
            # Social captions
            tiktok_caption=raw_response.get("tiktok_caption", ""),
            ig_caption=raw_response.get("ig_caption", ""),
            yt_title=raw_response.get("yt_title", ""),
            yt_description=raw_response.get("yt_description", ""),
            linkedin_text=raw_response.get("linkedin_text", ""),
            x_text=raw_response.get("x_text", ""),
            facebook_text=raw_response.get("facebook_text", ""),
            threads_text=raw_response.get("threads_text", ""),
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": "x-ai/grok-4.1-fast",
                "source_url": request.source_url,
                "had_transcription": bool(request.source_transcription),
                "persona_name": persona.name
            }
        )

    async def save_script(
        self,
        script: GeneratedScript,
        db_session
    ) -> int:
        """
        Save generated script to database.

        Args:
            script: Generated script to save
            db_session: Database session

        Returns:
            Script ID
        """
        from models import Script

        try:
            db_script = Script(
                content_idea_id=script.content_idea_id,
                hook=script.hook,
                body=script.body,
                cta=script.cta,
                full_script=script.full_text,
                duration_estimate=int(script.duration_estimate),
                # Social captions
                tiktok_caption=script.tiktok_caption,
                ig_caption=script.ig_caption,
                yt_title=script.yt_title,
                yt_description=script.yt_description,
                linkedin_text=script.linkedin_text,
                x_text=script.x_text,
                facebook_text=script.facebook_text,
                threads_text=script.threads_text,
                status="script_ready"
            )
            db_session.add(db_script)
            await db_session.commit()
            await db_session.refresh(db_script)

            logger.info(f"Saved script {db_script.id} for content idea {script.content_idea_id}")
            return db_script.id

        except Exception as e:
            logger.error(f"Failed to save script: {e}")
            await db_session.rollback()
            raise

    def estimate_duration(self, text: str, words_per_minute: int = 150) -> float:
        """
        Estimate speech duration from text.

        Args:
            text: Script text
            words_per_minute: Speaking rate (default 150 wpm)

        Returns:
            Duration in seconds
        """
        words = len(text.split())
        return (words / words_per_minute) * 60

    def format_for_tts(self, script: GeneratedScript) -> str:
        """
        Format script for text-to-speech.

        Adds appropriate pauses and formatting for natural speech.

        Args:
            script: Generated script

        Returns:
            Formatted text optimized for TTS
        """
        # Add short pauses between sections
        parts = []

        if script.hook:
            parts.append(script.hook.strip())
            parts.append("...")  # Brief pause after hook

        if script.body:
            parts.append(script.body.strip())
            parts.append("...")  # Brief pause before CTA

        if script.cta:
            parts.append(script.cta.strip())

        return " ".join(parts)
