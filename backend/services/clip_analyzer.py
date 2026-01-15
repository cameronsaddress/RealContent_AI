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
from models import ClipPersona, ViralClip, InfluencerVideo, LLMSettings
from services.base import BaseService

logger = logging.getLogger(__name__)

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

        video.status = "analyzing (grok)"
        db_session.commit()
        
        prompt = self._build_analysis_prompt(video, concise_segments, persona, prompt_override)
        
        analysis_result = await self._call_grok(prompt)
        
        # Save analysis to video record
        video.status = "analyzing (processing clips)"
        db_session.commit()
        
        # Save analysis to video record
        video.analysis_json = analysis_result
        video.status = "analyzed"
        
        # Create Viral Clips
        created_clips = []
        for clip_data in analysis_result.get("clips", []):
            try:
                # Validate timestamps
                start = float(clip_data.get("start", 0))
                end = float(clip_data.get("end", 0))
                if end <= start: continue
                
                vc = ViralClip(
                    source_video_id=video.id,
                    start_time=start,
                    end_time=end,
                    duration=end-start,
                    clip_type=clip_data.get("type", "highlight"),
                    virality_explanation=clip_data.get("reason", ""),
                    title=clip_data.get("title", "Viral Clip"),
                    description=clip_data.get("caption", ""),
                    hashtags=clip_data.get("hashtags", []),
                    status="pending",
                    render_metadata={"trigger_words": clip_data.get("trigger_words", [])}
                )
                db_session.add(vc)
                created_clips.append(vc)
            except Exception as e:
                logger.error(f"Error creating clip record: {e}")
                
        db_session.commit()
        return created_clips

    def _build_analysis_prompt(self, video: InfluencerVideo, segments: List[Dict], persona: ClipPersona, prompt_override: Optional[str] = None) -> str:
        """Construct the prompt for Grok"""
        
        system_instructions = prompt_override if prompt_override else f"""
You are an expert viral content editor acting as '{persona.name}'.
{persona.description}
"""

        return f"""
{system_instructions}

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

- Inspirational / "Based" (Strong monologues)

FOCUS ON THE MOST CONTROVERSIAL AND SHOCKING MOMENTS. DO NOT HOLD BACK.

Also identify 3-5 high-intensity SINGLE WORDS (triggers) within the clip for visual impact (e.g., WAR, DIE, TRUMP, MONEY, LIAR). Provide their exact start/end timestamps.

Constraints:
- Clips should be between {persona.min_clip_duration} and {persona.max_clip_duration} seconds.
- Ensure the start and end times cut cleanly (complete sentences).
- specific instructions: {persona.prompt_template}

Return ONLY valid JSON in this format:
{{
  "clips": [
    {{
      "start": 10.5,
      "end": 45.2,
      "type": "antagonistic",
      "title": "TOP G DESTROYS DEBATE OPPONENT",
      "reason": "High conflict moment, very engaging",
      "caption": "Bro didn't stand a chance... \ud83d\udc80 #owned #debate",
      "hashtags": ["#viral", "#shorts", "#fyp", "#sigma"],
      "trigger_words": [
          {{"word": "DESTROYED", "start": 30.5, "end": 31.0}},
          {{"word": "LIAR", "start": 40.2, "end": 40.8}}
      ]
    }}
  ]
}}
"""

    async def _call_grok(self, prompt: str) -> Dict[str, Any]:
        """Call Grok (via OpenRouter or handling direct)"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "x-ai/grok-4.1-fast", # Or beta version
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.7
                    },
                    timeout=120.0
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            except Exception as e:
                logger.error(f"Grok API failed: {e}")
                # Return empty or mock for robustness ? No, raise to retry
                raise e
