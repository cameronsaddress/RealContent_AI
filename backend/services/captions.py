"""
Caption service for transcription and subtitle burning.
Replaces n8n nodes: Prepare Whisper, Whisper Transcribe, Parse Whisper Response, Save SRT, Build Caption Cmd, Burn Captions.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from .base import BaseService
from config import settings
from utils.paths import asset_paths
from utils.logging import get_logger

logger = get_logger(__name__)


class CaptionSegment(BaseModel):
    """A single caption segment with timing."""
    start: float  # seconds
    end: float  # seconds
    text: str


class TranscriptionResult(BaseModel):
    """Result of audio transcription."""
    script_id: int
    segments: List[CaptionSegment]
    full_text: str
    language: str = "en"
    duration: float


class CaptionRequest(BaseModel):
    """Request parameters for caption burning - maps to UI Settings."""
    script_id: int
    video_path: str
    srt_path: Optional[str] = None

    # Caption style settings (from UI: Caption Settings)
    caption_style: str = "karaoke"  # karaoke, static, none
    font_name: str = "Arial"
    font_size: int = 96  # UI default
    font_color: str = "#FFFFFF"  # caption_color
    highlight_color: str = "#FFFF00"  # caption_highlight_color (for karaoke)
    outline_color: str = "#000000"  # caption_outline_color
    outline_width: int = 5  # caption_outline_width
    position_y: int = 850  # caption_position_y

    # Video quality settings (for re-encode)
    crf: int = 18
    preset: str = "slow"


class CaptionResult(BaseModel):
    """Result of caption burning."""
    script_id: int
    output_path: str
    srt_path: str
    duration_seconds: float
    file_size_bytes: int
    success: bool
    error: Optional[str] = None


class CaptionService(BaseService):
    """
    Service for transcription and caption burning.

    Replaces n8n nodes:
    - Prepare Whisper
    - Read Audio for Whisper
    - Whisper Transcribe
    - Parse Whisper Response
    - Save SRT
    - Build Caption Cmd
    - Burn Captions
    - Caption Success?
    """

    OPENAI_WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe_audio(
        self,
        script_id: int,
        audio_path: Optional[Path] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio using OpenAI Whisper.

        Args:
            script_id: Script ID
            audio_path: Path to audio file (defaults to voice path)

        Returns:
            TranscriptionResult with segments
        """
        if audio_path is None:
            audio_path = asset_paths.voice_path(script_id)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

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
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment"
                },
                timeout=120.0
            )

            result = response.json()
            return self.parse_whisper_response(script_id, result)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def parse_whisper_response(
        self,
        script_id: int,
        response: Dict[str, Any]
    ) -> TranscriptionResult:
        """
        Parse Whisper API response into structured result.

        Args:
            script_id: Script ID
            response: Raw Whisper API response

        Returns:
            TranscriptionResult
        """
        segments = []
        for seg in response.get("segments", []):
            segments.append(CaptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip()
            ))

        return TranscriptionResult(
            script_id=script_id,
            segments=segments,
            full_text=response.get("text", ""),
            language=response.get("language", "en"),
            duration=response.get("duration", 0.0)
        )

    def generate_srt(
        self,
        result: TranscriptionResult,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate SRT file from transcription result.

        Args:
            result: Transcription result
            output_path: Output path (defaults to captions path)

        Returns:
            Path to SRT file
        """
        if output_path is None:
            output_path = asset_paths.srt_path(result.script_id)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        srt_content = []
        for i, segment in enumerate(result.segments, 1):
            start_time = self._format_srt_time(segment.start)
            end_time = self._format_srt_time(segment.end)
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(segment.text)
            srt_content.append("")  # Blank line

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

        logger.info(f"Generated SRT file: {output_path}")
        return output_path

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def generate_ass(
        self,
        result: TranscriptionResult,
        output_path: Optional[Path] = None,
        font_name: str = "Arial",
        font_size: int = 96,
        margin_v: int = 850,
        font_color: str = "#FFFFFF",
        highlight_color: str = "#FFFF00",
        outline_color: str = "#000000",
        outline_width: int = 5
    ) -> Path:
        """
        Generate ASS file with karaoke-style timing.

        Args:
            result: Transcription result
            output_path: Output path
            font_name: Font name
            font_size: Font size
            margin_v: Vertical margin
            font_color: Primary text color (hex)
            highlight_color: Karaoke highlight color (hex)
            outline_color: Text outline color (hex)
            outline_width: Outline thickness

        Returns:
            Path to ASS file
        """
        if output_path is None:
            output_path = asset_paths.ass_path(result.script_id)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert colors to ASS format
        primary = self._hex_to_ass_color(font_color)
        secondary = self._hex_to_ass_color(highlight_color)
        outline = self._hex_to_ass_color(outline_color)

        # ASS header - Alignment 5 = center of screen (horizontally and vertically centered)
        # For TikTok-style: white text, yellow highlight as words are spoken
        ass_content = [
            "[Script Info]",
            "Title: Karaoke Captions",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "WrapStyle: 2",  # No word wrapping, keep on one line
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            # Alignment 5 = center-center, PrimaryColour=white (before highlight), SecondaryColour=yellow (highlighted)
            f"Style: Default,{font_name},{font_size},{primary},{secondary},{outline},&H80000000,1,0,0,0,100,100,0,0,1,{outline_width},2,5,60,60,{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]

        # Collect all words with timing for single-line karaoke
        all_words = []
        for segment in result.segments:
            words = segment.text.split()
            if words:
                word_duration = (segment.end - segment.start) / len(words)
                word_start = segment.start
                for word in words:
                    all_words.append({
                        'word': word,
                        'start': word_start,
                        'duration': word_duration
                    })
                    word_start += word_duration

        # Group into short phrases (4-6 words per line) for single-line display
        words_per_line = 5
        for i in range(0, len(all_words), words_per_line):
            chunk = all_words[i:i + words_per_line]
            if not chunk:
                continue

            line_start = chunk[0]['start']
            line_end = chunk[-1]['start'] + chunk[-1]['duration']

            start = self._format_ass_time(line_start)
            end = self._format_ass_time(line_end)

            # Build karaoke text with \kf (fill effect) for yellow highlight
            karaoke_text = ""
            for w in chunk:
                duration_cs = int(w['duration'] * 100)  # centiseconds
                karaoke_text += f"{{\\kf{duration_cs}}}{w['word']} "
            karaoke_text = karaoke_text.strip()

            ass_content.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{karaoke_text}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ass_content))

        logger.info(f"Generated ASS file: {output_path}")
        return output_path

    def _format_ass_time(self, seconds: float) -> str:
        """Format seconds as ASS timestamp (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

    def _hex_to_ass_color(self, hex_color: str) -> str:
        """Convert hex color (#RRGGBB) to ASS format (&HBBGGRR)."""
        hex_color = hex_color.lstrip('#')
        r = hex_color[0:2]
        g = hex_color[2:4]
        b = hex_color[4:6]
        return f"&H00{b}{g}{r}"

    def build_burn_command(
        self,
        request: CaptionRequest
    ) -> List[str]:
        """
        Build FFmpeg command for burning captions.

        Args:
            request: Caption burn request

        Returns:
            FFmpeg command as list
        """
        output_path = asset_paths.final_path(request.script_id)
        srt_path = request.srt_path or str(asset_paths.srt_path(request.script_id))

        # Convert colors to ASS format
        primary_color = self._hex_to_ass_color(request.font_color)
        outline_color = self._hex_to_ass_color(request.outline_color)

        # Build subtitle filter with all settings from UI
        subtitle_filter = (
            f"subtitles={srt_path}:force_style='"
            f"FontName={request.font_name},"
            f"FontSize={request.font_size},"
            f"PrimaryColour={primary_color},"
            f"OutlineColour={outline_color},"
            f"Outline={request.outline_width},"
            f"Alignment=2,"  # Bottom center
            f"MarginV={1920 - request.position_y}'"  # Convert position_y to margin
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", request.video_path,
            "-vf", subtitle_filter,
            "-c:v", "libx264",
            "-preset", request.preset,
            "-crf", str(request.crf),
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path)
        ]

        return cmd

    def build_ass_burn_command(
        self,
        script_id: int,
        video_path: str,
        crf: int = 18,
        preset: str = "slow"
    ) -> List[str]:
        """
        Build FFmpeg command for burning ASS captions (karaoke style).

        Args:
            script_id: Script ID
            video_path: Path to input video
            crf: Video quality
            preset: Encoding preset

        Returns:
            FFmpeg command as list
        """
        output_path = asset_paths.final_path(script_id)
        ass_path = asset_paths.ass_path(script_id)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path)
        ]

        return cmd

    async def burn_captions(
        self,
        request: CaptionRequest
    ) -> CaptionResult:
        """
        Burn captions onto video using FFmpeg.

        Args:
            request: Caption burn request

        Returns:
            CaptionResult
        """
        output_path = asset_paths.final_path(request.script_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        srt_path = request.srt_path or str(asset_paths.srt_path(request.script_id))

        cmd = self.build_burn_command(request)

        logger.info(f"Burning captions: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                logger.error(f"Caption burn failed: {result.stderr}")
                return CaptionResult(
                    script_id=request.script_id,
                    output_path=str(output_path),
                    srt_path=srt_path,
                    duration_seconds=0,
                    file_size_bytes=0,
                    success=False,
                    error=result.stderr[:500]
                )

            # Get output file info
            duration = self._get_duration(output_path)
            file_size = output_path.stat().st_size

            logger.info(f"Burned captions: {output_path} ({duration:.2f}s)")

            return CaptionResult(
                script_id=request.script_id,
                output_path=str(output_path),
                srt_path=srt_path,
                duration_seconds=duration,
                file_size_bytes=file_size,
                success=True
            )

        except subprocess.TimeoutExpired:
            return CaptionResult(
                script_id=request.script_id,
                output_path=str(output_path),
                srt_path=srt_path,
                duration_seconds=0,
                file_size_bytes=0,
                success=False,
                error="Caption burning timed out"
            )

        except Exception as e:
            logger.error(f"Caption burn error: {e}")
            return CaptionResult(
                script_id=request.script_id,
                output_path=str(output_path),
                srt_path=srt_path,
                duration_seconds=0,
                file_size_bytes=0,
                success=False,
                error=str(e)
            )

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

    def srt_exists(self, script_id: int) -> bool:
        """Check if SRT file exists."""
        srt_path = asset_paths.srt_path(script_id)
        return srt_path.exists() and srt_path.stat().st_size > 0

    def final_video_exists(self, script_id: int) -> bool:
        """Check if final captioned video exists."""
        final_path = asset_paths.final_path(script_id)
        return final_path.exists() and final_path.stat().st_size > 0

    async def update_asset_status(
        self,
        script_id: int,
        db_session,
        srt_path: str,
        final_path: str,
        status: str = "ready_to_publish"
    ) -> None:
        """
        Update asset record with caption information.

        Args:
            script_id: Script ID
            db_session: Database session
            srt_path: Path to SRT file
            final_path: Path to final video
            status: New status
        """
        from models import Asset

        try:
            await db_session.execute(
                Asset.__table__.update().where(Asset.script_id == script_id).values(
                    srt_path=srt_path,
                    final_video_path=final_path,
                    status=status
                )
            )
            await db_session.commit()
            logger.info(f"Updated asset for script {script_id} with caption status")

        except Exception as e:
            logger.error(f"Failed to update asset status: {e}")
            await db_session.rollback()
            raise
