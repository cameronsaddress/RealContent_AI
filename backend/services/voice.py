"""
Voice service for text-to-speech generation using ElevenLabs.
Replaces n8n nodes: Generate Voice (ElevenLabs), Save Voice, Get Duration.
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel

from .base import BaseService
from config import settings
from utils.paths import asset_paths
from utils.logging import get_logger

logger = get_logger(__name__)


class VoiceRequest(BaseModel):
    """Request parameters for voice generation."""
    script_id: int
    text: str
    voice_id: Optional[str] = None
    model_id: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True


class VoiceResult(BaseModel):
    """Result of voice generation."""
    script_id: int
    audio_path: str
    duration_seconds: float
    file_size_bytes: int
    voice_id: str


class VoiceService(BaseService):
    """
    Service for generating voice audio using ElevenLabs.

    Replaces n8n nodes:
    - Prepare TTS Text
    - Check Voice Exists
    - Generate Voice (ElevenLabs)
    - Save Voice
    - Get Duration
    - Update Asset Voice
    """

    ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

    async def generate_voice(self, request: VoiceRequest) -> VoiceResult:
        """
        Generate voice audio from text using ElevenLabs.

        Args:
            request: Voice generation parameters

        Returns:
            VoiceResult with path and duration
        """
        voice_id = request.voice_id or settings.ELEVENLABS_VOICE_ID

        url = f"{self.ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}"

        try:
            response = await self._post(
                url,
                headers={
                    "xi-api-key": settings.ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg"
                },
                json={
                    "text": request.text,
                    "model_id": request.model_id,
                    "voice_settings": {
                        "stability": request.stability,
                        "similarity_boost": request.similarity_boost,
                        "style": request.style,
                        "use_speaker_boost": request.use_speaker_boost
                    }
                },
                timeout=120.0
            )

            # Save audio file
            audio_path = asset_paths.voice_path(request.script_id)
            audio_path.parent.mkdir(parents=True, exist_ok=True)

            with open(audio_path, "wb") as f:
                f.write(response.content)

            # Get duration using ffprobe
            duration = self.get_audio_duration(audio_path)

            result = VoiceResult(
                script_id=request.script_id,
                audio_path=str(audio_path),
                duration_seconds=duration,
                file_size_bytes=audio_path.stat().st_size,
                voice_id=voice_id
            )

            logger.info(f"Generated voice for script {request.script_id}: {duration:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
            raise

    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get audio duration using ffprobe.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e.stderr}")
            raise
        except ValueError as e:
            logger.error(f"Failed to parse duration: {e}")
            raise

    def voice_exists(self, script_id: int) -> bool:
        """
        Check if voice file already exists.

        Args:
            script_id: Script ID to check

        Returns:
            True if voice file exists
        """
        voice_path = asset_paths.voice_path(script_id)
        return voice_path.exists() and voice_path.stat().st_size > 0

    def get_voice_path(self, script_id: int) -> Optional[Path]:
        """
        Get path to existing voice file if it exists.

        Args:
            script_id: Script ID

        Returns:
            Path to voice file or None
        """
        voice_path = asset_paths.voice_path(script_id)
        if voice_path.exists():
            return voice_path
        return None

    async def list_voices(self) -> Dict[str, Any]:
        """
        List available voices from ElevenLabs.

        Returns:
            Dict with voice information
        """
        url = f"{self.ELEVENLABS_BASE_URL}/voices"

        try:
            response = await self._get(
                url,
                headers={"xi-api-key": settings.ELEVENLABS_API_KEY}
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            raise

    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """
        Get information about a specific voice.

        Args:
            voice_id: ElevenLabs voice ID

        Returns:
            Voice information dict
        """
        url = f"{self.ELEVENLABS_BASE_URL}/voices/{voice_id}"

        try:
            response = await self._get(
                url,
                headers={"xi-api-key": settings.ELEVENLABS_API_KEY}
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get voice info: {e}")
            raise

    def prepare_tts_text(self, text: str) -> str:
        """
        Prepare text for text-to-speech.

        Cleans up text and adds appropriate pauses.

        Args:
            text: Raw script text

        Returns:
            Cleaned text optimized for TTS
        """
        # Remove multiple spaces
        text = " ".join(text.split())

        # Add slight pauses after sentences if not already present
        text = text.replace(". ", "... ")
        text = text.replace("! ", "... ")
        text = text.replace("? ", "... ")

        # Clean up any triple dots that became too long
        while "...." in text:
            text = text.replace("....", "...")

        return text.strip()

    async def update_asset_status(
        self,
        script_id: int,
        db_session,
        status: str = "voice_ready",
        duration: Optional[float] = None
    ) -> None:
        """
        Update asset record with voice information.

        Args:
            script_id: Script ID
            db_session: Database session
            status: New status
            duration: Voice duration in seconds
        """
        from models import Asset

        try:
            voice_path = asset_paths.voice_path(script_id)

            # Find or create asset record
            result = await db_session.execute(
                Asset.__table__.select().where(Asset.script_id == script_id)
            )
            asset = result.fetchone()

            if asset:
                await db_session.execute(
                    Asset.__table__.update().where(Asset.script_id == script_id).values(
                        voiceover_path=str(voice_path),
                        voiceover_duration=duration,
                        status=status
                    )
                )
            else:
                new_asset = Asset(
                    script_id=script_id,
                    voiceover_path=str(voice_path),
                    voiceover_duration=duration,
                    status=status
                )
                db_session.add(new_asset)

            await db_session.commit()
            logger.info(f"Updated asset for script {script_id} with status {status}")

        except Exception as e:
            logger.error(f"Failed to update asset status: {e}")
            await db_session.rollback()
            raise
