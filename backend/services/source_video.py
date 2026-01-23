"""
Source Video service for downloading and transcribing viral source videos.
This extracts what creators actually SAY in their videos for smarter script generation.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel
import httpx

from .base import BaseService
from config import settings
from utils.paths import asset_paths
from utils.logging import get_logger

logger = get_logger(__name__)


class SourceVideoResult(BaseModel):
    """Result of source video processing."""
    content_idea_id: int
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    transcription: Optional[str] = None
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None


class SourceVideoService(BaseService):
    """
    Service for processing source videos from viral content.

    Workflow:
    1. Download video from TikTok/Instagram/etc via video-downloader service
    2. Extract audio from video
    3. Transcribe audio using Whisper
    4. Return transcription for script generation
    """

    VIDEO_DOWNLOADER_URL = "http://video-processor:8080"
    OPENAI_WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"

    async def download_source_video(
        self,
        content_idea_id: int,
        source_url: str
    ) -> Path:
        """
        Download source video using video-downloader service.

        Args:
            content_idea_id: Content idea ID for naming
            source_url: URL to the viral video

        Returns:
            Path to downloaded video
        """
        output_path = asset_paths.source_video_path(content_idea_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = await self._post(
                f"{self.VIDEO_DOWNLOADER_URL}/download",
                json={
                    "url": source_url,
                    "filename": f"{content_idea_id}_source",
                    "format": "mp4"
                },
                timeout=300.0  # 5 min timeout for downloads
            )

            result = response.json()

            if not result.get("success", False):
                raise Exception(result.get("error", "Download failed"))

            # The video-downloader saves to its own path, we need to copy/move it
            downloaded_path_raw = result.get("path") or result.get("file_path") or ""
            downloaded_path = Path(downloaded_path_raw) if downloaded_path_raw else None
            if not downloaded_path:
                filename = result.get("filename")
                if filename:
                    downloaded_path = asset_paths.videos_dir / filename

            if not downloaded_path:
                raise FileNotFoundError("Downloaded file not found: missing path")

            if not downloaded_path.exists():
                mapped_path = asset_paths.videos_dir / downloaded_path.name
                if mapped_path.exists():
                    downloaded_path = mapped_path

            if downloaded_path.exists():
                # Copy to our assets directory
                import shutil
                shutil.copy2(downloaded_path, output_path)
                logger.info(f"Downloaded source video: {output_path}")
                return output_path
            else:
                raise FileNotFoundError(f"Downloaded file not found: {downloaded_path}")

        except Exception as e:
            logger.error(f"Failed to download source video: {e}")
            raise

    def extract_audio(
        self,
        video_path: Path,
        content_idea_id: int
    ) -> Path:
        """
        Extract audio from video using FFmpeg.

        Args:
            video_path: Path to source video
            content_idea_id: For naming the output file

        Returns:
            Path to extracted audio file
        """
        audio_path = asset_paths.base_path / "audio" / f"{content_idea_id}_source_audio.mp3"
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-ar", "16000",  # 16kHz for Whisper
            "-ac", "1",  # Mono
            "-b:a", "64k",
            str(audio_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
                raise Exception(f"Audio extraction failed: {result.stderr[:200]}")

            logger.info(f"Extracted audio: {audio_path}")
            return audio_path

        except subprocess.TimeoutExpired:
            raise Exception("Audio extraction timed out")

    async def transcribe_audio(
        self,
        audio_path: Path
    ) -> str:
        """
        Transcribe audio using OpenAI Whisper.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Check file size (Whisper has 25MB limit)
        file_size = audio_path.stat().st_size
        if file_size > 25 * 1024 * 1024:
            logger.warning(f"Audio file too large ({file_size} bytes), may need to chunk")

        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            response = await self._post(
                self.OPENAI_WHISPER_URL,
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                },
                files={
                    "file": ("audio.mp3", audio_data, "audio/mpeg"),
                },
                data={
                    "model": "whisper-1",
                    "response_format": "text"  # Just get the text, not verbose JSON
                },
                timeout=120.0
            )

            # Response is plain text when response_format is "text"
            transcription = response.text.strip()
            logger.info(f"Transcribed {len(transcription)} characters from audio")
            return transcription

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    async def process_source_video(
        self,
        content_idea_id: int,
        source_url: str,
        skip_download: bool = False,
        existing_video_path: Optional[Path] = None
    ) -> SourceVideoResult:
        """
        Full pipeline: Download, extract audio, transcribe.

        Args:
            content_idea_id: Content idea ID
            source_url: URL to viral video
            skip_download: If True, use existing_video_path instead of downloading
            existing_video_path: Path to already-downloaded video

        Returns:
            SourceVideoResult with transcription
        """
        result = SourceVideoResult(content_idea_id=content_idea_id)

        try:
            # Step 1: Get video
            if skip_download and existing_video_path:
                video_path = existing_video_path
            else:
                video_path = await self.download_source_video(content_idea_id, source_url)

            result.video_path = str(video_path)

            # Step 2: Extract audio
            audio_path = self.extract_audio(video_path, content_idea_id)
            result.audio_path = str(audio_path)

            # Step 3: Transcribe
            transcription = await self.transcribe_audio(audio_path)
            result.transcription = transcription

            # Get duration
            result.duration_seconds = self._get_duration(video_path)

            logger.info(f"Processed source video for idea {content_idea_id}: {len(transcription)} chars")
            return result

        except Exception as e:
            logger.error(f"Source video processing failed: {e}")
            result.success = False
            result.error = str(e)
            return result

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    async def transcribe_from_url(
        self,
        content_idea_id: int,
        source_url: str
    ) -> Optional[str]:
        """
        Convenience method to just get transcription from a URL.

        Args:
            content_idea_id: Content idea ID
            source_url: URL to viral video

        Returns:
            Transcription text or None if failed
        """
        try:
            result = await self.process_source_video(content_idea_id, source_url)
            if result.success and result.transcription:
                return result.transcription
            return None
        except Exception as e:
            logger.error(f"Failed to transcribe from URL: {e}")
            return None
