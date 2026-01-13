"""
Asset path management for the video pipeline.
Handles path resolution for different asset types.
"""

from pathlib import Path
from typing import Optional
from config import settings


class AssetPaths:
    """Manages asset file paths for the pipeline."""

    def __init__(self, base_path: Optional[str] = None):
        self.base = Path(base_path or settings.ASSETS_BASE_PATH)

    @property
    def audio_dir(self) -> Path:
        path = self.base / "audio"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def avatar_dir(self) -> Path:
        path = self.base / "avatar"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def videos_dir(self) -> Path:
        path = self.base / "videos"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def output_dir(self) -> Path:
        path = self.base / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def captions_dir(self) -> Path:
        path = self.base / "captions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def music_dir(self) -> Path:
        path = self.base / "music"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def voice_path(self, script_id: int) -> Path:
        return self.audio_dir / f"{script_id}_voice.mp3"

    def avatar_path(self, script_id: int) -> Path:
        return self.avatar_dir / f"{script_id}_avatar.mp4"

    def source_video_path(self, script_id: int) -> Path:
        return self.videos_dir / f"{script_id}_source.mp4"

    def combined_video_path(self, script_id: int) -> Path:
        return self.output_dir / f"{script_id}_combined.mp4"

    # Alias for combined_video_path
    def combined_path(self, script_id: int) -> Path:
        return self.combined_video_path(script_id)

    def final_video_path(self, script_id: int) -> Path:
        return self.output_dir / f"{script_id}_final.mp4"

    # Alias for final_video_path
    def final_path(self, script_id: int) -> Path:
        return self.final_video_path(script_id)

    def srt_path(self, script_id: int) -> Path:
        return self.captions_dir / f"{script_id}_captions.srt"

    def ass_path(self, script_id: int) -> Path:
        return self.captions_dir / f"{script_id}_captions.ass"

    def get_active_music(self) -> Optional[Path]:
        """Find the active music file in the music directory."""
        if not self.music_dir.exists():
            return None

        # Look for file starting with 'active_'
        for f in self.music_dir.iterdir():
            if f.name.startswith("active_") and f.suffix in (".mp3", ".wav"):
                return f

        # Fall back to first audio file
        for f in self.music_dir.iterdir():
            if f.suffix in (".mp3", ".wav"):
                return f

        return None

    def file_exists_and_valid(self, path: Path, min_size: int = 1000) -> bool:
        """Check if file exists and has minimum size."""
        return path.exists() and path.stat().st_size >= min_size


# Global instance
asset_paths = AssetPaths()
