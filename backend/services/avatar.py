"""
Avatar service for AI video generation using HeyGen.
Replaces n8n nodes: Upload HeyGen Audio, Create HeyGen Video, Poll Status, Save Avatar.
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel
from enum import Enum

from .base import BaseService
from config import settings
from utils.paths import asset_paths
from utils.logging import get_logger

logger = get_logger(__name__)


class AvatarType(str, Enum):
    """Types of HeyGen avatars."""
    TALKING_PHOTO = "talking_photo"
    VIDEO_AVATAR = "video_avatar"
    STUDIO_AVATAR = "studio_avatar"


class AvatarRequest(BaseModel):
    """Request parameters for avatar video generation."""
    script_id: int
    audio_url: str
    avatar_id: Optional[str] = None
    avatar_type: AvatarType = AvatarType.VIDEO_AVATAR  # Default to video avatar (not talking photo)
    avatar_style: str = "normal"
    greenscreen_enabled: bool = True
    greenscreen_color: str = "#00FF00"
    aspect_ratio: str = "9:16"
    test_mode: bool = False


class AvatarStatus(BaseModel):
    """Status of avatar generation job."""
    video_id: str
    status: str  # pending, processing, completed, failed
    video_url: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[float] = None


class AvatarResult(BaseModel):
    """Result of avatar video generation."""
    script_id: int
    video_id: str
    video_path: str
    duration_seconds: float
    file_size_bytes: int


class AvatarService(BaseService):
    """
    Service for generating AI avatar videos using HeyGen.

    Replaces n8n nodes:
    - Read Audio File
    - Avatar Already Exists?
    - Skip HeyGen?
    - Upload HeyGen Audio
    - Is Talking Photo?
    - Prepare HeyGen Data
    - Create HeyGen Video
    - Poll HeyGen Status (Wait nodes)
    - Download Avatar
    - Save Avatar
    """

    HEYGEN_BASE_URL = "https://api.heygen.com/v2"
    HEYGEN_STATUS_URL = "https://api.heygen.com/v1"  # Status endpoint uses v1
    HEYGEN_UPLOAD_URL = "https://upload.heygen.com/v1"

    async def upload_audio(self, audio_path: Path) -> str:
        """
        Upload audio file to HeyGen.

        Args:
            audio_path: Path to local audio file

        Returns:
            HeyGen audio URL
        """
        # HeyGen expects raw binary data with filename in query param
        filename = audio_path.name
        url = f"{self.HEYGEN_UPLOAD_URL}/asset?name={filename}"

        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            # Send raw binary data (not multipart) - matches n8n workflow
            response = await self._post(
                url,
                headers={
                    "X-Api-Key": settings.HEYGEN_API_KEY,
                    "Content-Type": "audio/mpeg",
                },
                content=audio_data,
                timeout=120.0
            )

            result = response.json()
            # HeyGen returns both url and id - we need the id for video generation
            audio_asset_id = result.get("data", {}).get("id")
            audio_url = result.get("data", {}).get("url")

            if not audio_asset_id:
                raise ValueError(f"Failed to get audio asset ID from HeyGen: {result}")

            logger.info(f"Uploaded audio to HeyGen: id={audio_asset_id}, url={audio_url}")
            return audio_asset_id  # Return asset ID, not URL

        except Exception as e:
            logger.error(f"Audio upload failed: {e}")
            raise

    async def create_video(self, request: AvatarRequest) -> str:
        """
        Create avatar video using HeyGen API.

        Args:
            request: Avatar generation request

        Returns:
            HeyGen video job ID
        """
        avatar_id = request.avatar_id or settings.HEYGEN_AVATAR_ID
        url = f"{self.HEYGEN_BASE_URL}/video/generate"

        # Build request based on avatar type
        # HeyGen v2 API uses audio_asset_id (returned from upload), not audio_url
        if request.avatar_type == AvatarType.TALKING_PHOTO:
            video_inputs = [{
                "character": {
                    "type": "talking_photo",
                    "talking_photo_id": avatar_id,
                },
                "voice": {
                    "type": "audio",
                    "audio_asset_id": request.audio_url  # This is actually the asset ID now
                }
            }]
        else:
            video_inputs = [{
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id,
                    "avatar_style": request.avatar_style
                },
                "voice": {
                    "type": "audio",
                    "audio_asset_id": request.audio_url  # This is actually the asset ID now
                }
            }]

        # Add background settings
        background = None
        if request.greenscreen_enabled:
            background = {
                "type": "color",
                "value": request.greenscreen_color
            }

        payload = {
            "video_inputs": video_inputs,
            "dimension": self._get_dimensions(request.aspect_ratio),
            "test": request.test_mode
        }

        if background:
            payload["video_inputs"][0]["background"] = background

        # Log payload for debugging
        import json
        logger.info(f"HeyGen video payload: {json.dumps(payload, indent=2)}")

        try:
            response = await self._post(
                url,
                headers={
                    "X-Api-Key": settings.HEYGEN_API_KEY,
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60.0
            )

            result = response.json()
            video_id = result.get("data", {}).get("video_id")

            if not video_id:
                raise ValueError(f"Failed to get video ID from HeyGen: {result}")

            logger.info(f"Created HeyGen video job: {video_id}")
            return video_id

        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            raise

    def _get_dimensions(self, aspect_ratio: str) -> Dict[str, int]:
        """Get dimensions for aspect ratio."""
        dimensions = {
            "9:16": {"width": 1080, "height": 1920},
            "16:9": {"width": 1920, "height": 1080},
            "1:1": {"width": 1080, "height": 1080}
        }
        return dimensions.get(aspect_ratio, dimensions["9:16"])

    async def get_status(self, video_id: str) -> AvatarStatus:
        """
        Check status of HeyGen video generation.

        Args:
            video_id: HeyGen video job ID

        Returns:
            Current status
        """
        url = f"{self.HEYGEN_STATUS_URL}/video_status.get"

        try:
            response = await self._get(
                url,
                headers={"X-Api-Key": settings.HEYGEN_API_KEY},
                params={"video_id": video_id}
            )

            result = response.json()
            data = result.get("data", {})

            return AvatarStatus(
                video_id=video_id,
                status=data.get("status", "unknown"),
                video_url=data.get("video_url"),
                error=data.get("error"),
                progress=data.get("progress")
            )

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise

    async def poll_until_complete(
        self,
        video_id: str,
        max_retries: Optional[int] = None,
        poll_interval: Optional[int] = None
    ) -> AvatarStatus:
        """
        Poll HeyGen until video is complete or failed.

        Args:
            video_id: HeyGen video job ID
            max_retries: Maximum polling attempts (default from settings)
            poll_interval: Seconds between polls (default from settings)

        Returns:
            Final status
        """
        max_retries = max_retries or settings.HEYGEN_MAX_RETRIES
        poll_interval = poll_interval or settings.HEYGEN_POLL_INTERVAL

        for attempt in range(max_retries):
            status = await self.get_status(video_id)

            if status.status == "completed":
                logger.info(f"HeyGen video {video_id} completed")
                return status

            if status.status == "failed":
                logger.error(f"HeyGen video {video_id} failed: {status.error}")
                raise Exception(f"HeyGen video generation failed: {status.error}")

            logger.info(f"HeyGen video {video_id} status: {status.status} (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"HeyGen video {video_id} timed out after {max_retries * poll_interval} seconds")

    async def download_video(
        self,
        video_url: str,
        script_id: int
    ) -> Path:
        """
        Download completed video from HeyGen.

        Args:
            video_url: URL to download video from
            script_id: Script ID for naming

        Returns:
            Path to downloaded video
        """
        try:
            response = await self._get(video_url, timeout=300.0)

            avatar_path = asset_paths.avatar_path(script_id)
            avatar_path.parent.mkdir(parents=True, exist_ok=True)

            with open(avatar_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Downloaded avatar video to {avatar_path}")
            return avatar_path

        except Exception as e:
            logger.error(f"Video download failed: {e}")
            raise

    async def generate_avatar(self, request: AvatarRequest) -> AvatarResult:
        """
        Full avatar generation pipeline.

        1. Create video job
        2. Poll until complete
        3. Download video

        Args:
            request: Avatar generation request

        Returns:
            Avatar result with local path
        """
        # Create video job
        video_id = await self.create_video(request)

        # Poll until complete
        status = await self.poll_until_complete(video_id)

        if not status.video_url:
            raise ValueError(f"Completed video has no URL: {video_id}")

        # Download video
        avatar_path = await self.download_video(status.video_url, request.script_id)

        # Get file info
        import subprocess
        duration_result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(avatar_path)
            ],
            capture_output=True,
            text=True
        )
        duration = float(duration_result.stdout.strip()) if duration_result.returncode == 0 else 0.0

        return AvatarResult(
            script_id=request.script_id,
            video_id=video_id,
            video_path=str(avatar_path),
            duration_seconds=duration,
            file_size_bytes=avatar_path.stat().st_size
        )

    def avatar_exists(self, script_id: int) -> bool:
        """
        Check if avatar video already exists.

        Args:
            script_id: Script ID to check

        Returns:
            True if avatar video exists
        """
        avatar_path = asset_paths.avatar_path(script_id)
        return avatar_path.exists() and avatar_path.stat().st_size > 0

    def get_avatar_path(self, script_id: int) -> Optional[Path]:
        """
        Get path to existing avatar video if it exists.

        Args:
            script_id: Script ID

        Returns:
            Path to avatar video or None
        """
        avatar_path = asset_paths.avatar_path(script_id)
        if avatar_path.exists():
            return avatar_path
        return None

    async def update_asset_status(
        self,
        script_id: int,
        db_session,
        video_id: str,
        video_path: str,
        status: str = "avatar_ready"
    ) -> None:
        """
        Update asset record with avatar information.

        Args:
            script_id: Script ID
            db_session: Database session
            video_id: HeyGen video ID
            video_path: Path to avatar video
            status: New status
        """
        from models import Asset

        try:
            result = await db_session.execute(
                Asset.__table__.select().where(Asset.script_id == script_id)
            )
            asset = result.fetchone()

            if asset:
                await db_session.execute(
                    Asset.__table__.update().where(Asset.script_id == script_id).values(
                        avatar_video_path=video_path,
                        heygen_video_id=video_id,
                        status=status
                    )
                )
            await db_session.commit()
            logger.info(f"Updated asset for script {script_id} with avatar status")

        except Exception as e:
            logger.error(f"Failed to update asset status: {e}")
            await db_session.rollback()
            raise
