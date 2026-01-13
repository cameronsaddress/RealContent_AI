"""
Video service for FFmpeg operations (combining, chromakey, etc).
Replaces n8n nodes: Build FFmpeg Command, Run FFmpeg, Success check.
"""

import subprocess
import shlex
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from .base import BaseService
from config import settings
from utils.paths import asset_paths
from utils.logging import get_logger

logger = get_logger(__name__)


class VideoComposeRequest(BaseModel):
    """Request parameters for video composition - maps to UI Settings."""
    script_id: int
    avatar_path: Optional[str] = None
    source_video_path: Optional[str] = None
    audio_path: Optional[str] = None

    # Video output settings (from UI: Video Settings)
    output_width: int = 1080
    output_height: int = 1920
    crf: int = 18  # Quality (15-28, lower = better)
    preset: str = "slow"  # ultrafast, fast, medium, slow, veryslow

    # Greenscreen settings (from UI: Video Settings)
    greenscreen_enabled: bool = True
    greenscreen_color: str = "#00FF00"

    # Avatar composition settings (from UI: Avatar Composition)
    avatar_position: str = "bottom-left"  # bottom-left, bottom-center, bottom-right
    avatar_scale: float = 0.35  # Scale relative to frame width (0.3-1.0)
    avatar_offset_x: int = 20  # Horizontal offset in pixels (positive = right)
    avatar_offset_y: int = -50  # Vertical offset in pixels (negative = up from bottom)

    # Audio settings (from UI: Audio Settings)
    original_volume: float = 0.7  # Source video audio volume
    avatar_volume: float = 1.0  # Avatar/voiceover volume
    music_volume: float = 0.3  # Background music volume
    music_path: Optional[str] = None  # Optional background music
    ducking_enabled: bool = True  # Lower source audio when avatar speaks
    avatar_delay_seconds: float = 3.0  # Delay before avatar starts
    duck_to_percent: float = 0.5  # Volume when ducked


class VideoResult(BaseModel):
    """Result of video composition."""
    script_id: int
    output_path: str
    duration_seconds: float
    file_size_bytes: int
    success: bool
    error: Optional[str] = None


class VideoService(BaseService):
    """
    Service for video composition using FFmpeg.

    Replaces n8n nodes:
    - Build FFmpeg Command
    - Run FFmpeg
    - Success?
    - Verify Download
    """

    def build_chromakey_command(
        self,
        request: VideoComposeRequest
    ) -> List[str]:
        """
        Build FFmpeg command for chromakey composition.

        Overlays avatar (with greenscreen removed) on source video,
        with full audio mixing support.

        Args:
            request: Composition parameters

        Returns:
            FFmpeg command as list
        """
        output_path = asset_paths.combined_path(request.script_id)

        # Convert hex color for FFmpeg chromakey
        hex_color = request.greenscreen_color.lstrip('#')

        # Calculate overlay position with offsets
        # Base positions + user offsets
        base_x = {
            "bottom-left": 10,
            "bottom-center": f"(W-w)/2",
            "bottom-right": "(W-w-10)",
        }
        base_y = "(H-h-10)"  # Bottom aligned

        pos = request.avatar_position
        if pos in base_x:
            if isinstance(base_x[pos], int):
                x_pos = f"{base_x[pos] + request.avatar_offset_x}"
            else:
                x_pos = f"({base_x[pos]})+{request.avatar_offset_x}"
            y_pos = f"({base_y})+{request.avatar_offset_y}"
        else:
            x_pos = f"10+{request.avatar_offset_x}"
            y_pos = f"(H-h-10)+{request.avatar_offset_y}"

        overlay_pos = f"{x_pos}:{y_pos}"

        # Scale avatar to percentage of output width
        avatar_width = int(request.output_width * request.avatar_scale)

        # Build video filter: chromakey, scale, overlay
        video_filter = (
            f"[1:v]chromakey=0x{hex_color}:0.1:0.2,scale={avatar_width}:-1[avatar];"
            f"[0:v]scale={request.output_width}:{request.output_height}:force_original_aspect_ratio=decrease,"
            f"pad={request.output_width}:{request.output_height}:(ow-iw)/2:(oh-ih)/2[bg];"
            f"[bg][avatar]overlay={overlay_pos}[out]"
        )

        # Build audio filter for mixing
        # [0:a] = source video audio
        # [2:a] = avatar voiceover (if separate audio file provided)
        if request.ducking_enabled and request.audio_path:
            # Audio ducking: lower source audio when avatar speaks
            # Use sidechaincompress for auto-ducking
            delay_ms = int(request.avatar_delay_seconds * 1000)
            audio_filter = (
                f"[0:a]volume={request.original_volume}[src];"
                f"[2:a]adelay={delay_ms}|{delay_ms},volume={request.avatar_volume}[voice];"
                f"[src][voice]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
            )
        elif request.audio_path:
            # Simple mix without ducking
            audio_filter = (
                f"[0:a]volume={request.original_volume}[src];"
                f"[2:a]volume={request.avatar_volume}[voice];"
                f"[src][voice]amix=inputs=2:duration=longest[aout]"
            )
        else:
            audio_filter = None

        # Combine filters
        if audio_filter:
            filter_complex = f"{video_filter};{audio_filter}"
        else:
            filter_complex = video_filter

        cmd = [
            "ffmpeg",
            "-y",
            "-i", request.source_video_path,  # [0] Background video
            "-i", request.avatar_path,  # [1] Avatar with greenscreen
        ]

        if request.audio_path:
            cmd.extend(["-i", request.audio_path])  # [2] Voiceover

        if request.music_path:
            cmd.extend(["-i", request.music_path])  # [3] Background music

        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[out]"])

        if audio_filter:
            cmd.extend(["-map", "[aout]"])
        elif request.audio_path:
            cmd.extend(["-map", "2:a"])
        else:
            cmd.extend(["-map", "1:a"])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", request.preset,
            "-crf", str(request.crf),
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path)
        ])

        return cmd

    def build_simple_compose_command(
        self,
        request: VideoComposeRequest
    ) -> List[str]:
        """
        Build FFmpeg command for simple video composition (no greenscreen).

        Args:
            request: Composition parameters

        Returns:
            FFmpeg command as list
        """
        output_path = asset_paths.combined_path(request.script_id)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", request.avatar_path,
        ]

        if request.audio_path and request.audio_path != request.avatar_path:
            # Replace audio track
            cmd.extend([
                "-i", request.audio_path,
                "-map", "0:v",
                "-map", "1:a",
            ])
        else:
            cmd.extend(["-map", "0"])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", request.preset,
            "-crf", str(request.crf),
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path)
        ])

        return cmd

    async def compose_video(
        self,
        request: VideoComposeRequest
    ) -> VideoResult:
        """
        Compose video using FFmpeg.

        Args:
            request: Composition parameters

        Returns:
            VideoResult with output path
        """
        output_path = asset_paths.combined_path(request.script_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Choose command based on whether we have source video and greenscreen
        if request.source_video_path and request.greenscreen_enabled:
            cmd = self.build_chromakey_command(request)
        else:
            cmd = self.build_simple_compose_command(request)

        logger.info(f"Running FFmpeg: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return VideoResult(
                    script_id=request.script_id,
                    output_path=str(output_path),
                    duration_seconds=0,
                    file_size_bytes=0,
                    success=False,
                    error=result.stderr[:500]  # Truncate error
                )

            # Get output file info
            duration = self.get_video_duration(output_path)
            file_size = output_path.stat().st_size

            logger.info(f"Composed video: {output_path} ({duration:.2f}s, {file_size} bytes)")

            return VideoResult(
                script_id=request.script_id,
                output_path=str(output_path),
                duration_seconds=duration,
                file_size_bytes=file_size,
                success=True
            )

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out")
            return VideoResult(
                script_id=request.script_id,
                output_path=str(output_path),
                duration_seconds=0,
                file_size_bytes=0,
                success=False,
                error="FFmpeg timed out after 10 minutes"
            )

        except Exception as e:
            logger.error(f"FFmpeg error: {e}")
            return VideoResult(
                script_id=request.script_id,
                output_path=str(output_path),
                duration_seconds=0,
                file_size_bytes=0,
                success=False,
                error=str(e)
            )

    def get_video_duration(self, video_path: Path) -> float:
        """
        Get video duration using ffprobe.

        Args:
            video_path: Path to video file

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
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Failed to get duration: {e}")
            return 0.0

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get detailed video information.

        Args:
            video_path: Path to video file

        Returns:
            Dict with video properties
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            import json
            return json.loads(result.stdout)
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {}

    def verify_video(self, video_path: Path) -> bool:
        """
        Verify video file is valid.

        Args:
            video_path: Path to video file

        Returns:
            True if video is valid
        """
        if not video_path.exists():
            return False

        if video_path.stat().st_size == 0:
            return False

        # Try to get duration - if this works, video is likely valid
        duration = self.get_video_duration(video_path)
        return duration > 0

    def combined_video_exists(self, script_id: int) -> bool:
        """
        Check if combined video already exists.

        Args:
            script_id: Script ID to check

        Returns:
            True if combined video exists and is valid
        """
        combined_path = asset_paths.combined_path(script_id)
        return self.verify_video(combined_path)

    def get_combined_path(self, script_id: int) -> Optional[Path]:
        """
        Get path to existing combined video if it exists.

        Args:
            script_id: Script ID

        Returns:
            Path to combined video or None
        """
        combined_path = asset_paths.combined_path(script_id)
        if combined_path.exists():
            return combined_path
        return None

    async def download_source_video(
        self,
        url: str,
        script_id: int
    ) -> Path:
        """
        Download source video from URL.

        Args:
            url: Video URL
            script_id: Script ID for naming

        Returns:
            Path to downloaded video
        """
        try:
            response = await self._get(url, timeout=300.0)

            source_path = asset_paths.source_video_path(script_id)
            source_path.parent.mkdir(parents=True, exist_ok=True)

            with open(source_path, "wb") as f:
                f.write(response.content)

            if not self.verify_video(source_path):
                raise ValueError("Downloaded video is invalid")

            logger.info(f"Downloaded source video: {source_path}")
            return source_path

        except Exception as e:
            logger.error(f"Source video download failed: {e}")
            raise

    async def update_asset_status(
        self,
        script_id: int,
        db_session,
        combined_path: str,
        status: str = "assembling"
    ) -> None:
        """
        Update asset record with combined video information.

        Args:
            script_id: Script ID
            db_session: Database session
            combined_path: Path to combined video
            status: New status
        """
        from models import Asset

        try:
            await db_session.execute(
                Asset.__table__.update().where(Asset.script_id == script_id).values(
                    combined_path=combined_path,
                    status=status
                )
            )
            await db_session.commit()
            logger.info(f"Updated asset for script {script_id} with combined video")

        except Exception as e:
            logger.error(f"Failed to update asset status: {e}")
            await db_session.rollback()
            raise
