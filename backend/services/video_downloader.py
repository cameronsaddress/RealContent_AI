"""
Video downloader service using Apify actors.

Downloads videos from TikTok, Instagram, and YouTube using Apify's
cloud-based scrapers to avoid IP restrictions.

Cost estimates:
- TikTok: ~$0.005-0.01 per video
- Instagram: ~$0.01 per video
- YouTube: ~$0.005 per video
"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel
from urllib.parse import urlparse

from .base import BaseService
from config import settings
from utils.logging import get_logger
from utils.paths import asset_paths

logger = get_logger(__name__)


class DownloadResult(BaseModel):
    """Result of video download."""
    success: bool
    script_id: int
    platform: str
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    duration_seconds: float = 0
    file_size_bytes: int = 0
    error: Optional[str] = None


class VideoDownloaderService(BaseService):
    """
    Service for downloading videos from social platforms using Apify.

    Uses platform-specific actors to download videos without watermarks
    and avoid IP blocking issues.
    """

    # Apify actors for video download
    APIFY_TIKTOK_DOWNLOADER = "wilcode~fast-tiktok-downloader-without-watermark"
    APIFY_INSTAGRAM_DOWNLOADER = "apify~instagram-scraper"
    APIFY_YOUTUBE_DOWNLOADER = "bernardo~youtube-video-downloader"

    APIFY_BASE_URL = "https://api.apify.com/v2"

    def detect_platform(self, url: str) -> str:
        """Detect platform from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if "tiktok" in domain:
            return "tiktok"
        elif "instagram" in domain:
            return "instagram"
        elif "youtube" in domain or "youtu.be" in domain:
            return "youtube"
        elif "twitter" in domain or "x.com" in domain:
            return "twitter"
        else:
            return "unknown"

    async def download_video(
        self,
        url: str,
        script_id: int
    ) -> DownloadResult:
        """
        Download video from URL using appropriate Apify actor.

        Args:
            url: Video URL (TikTok, Instagram, or YouTube)
            script_id: Script ID for file naming

        Returns:
            DownloadResult with local file path
        """
        platform = self.detect_platform(url)

        logger.info(f"Downloading {platform} video for script {script_id}: {url}")

        try:
            if platform == "tiktok":
                return await self._download_tiktok(url, script_id)
            elif platform == "instagram":
                return await self._download_instagram(url, script_id)
            elif platform == "youtube":
                return await self._download_youtube(url, script_id)
            else:
                # Fallback to local yt-dlp for unsupported platforms
                return await self._download_fallback(url, script_id, platform)

        except Exception as e:
            logger.error(f"Video download failed: {e}")
            return DownloadResult(
                success=False,
                script_id=script_id,
                platform=platform,
                error=str(e)
            )

    async def _download_tiktok(self, url: str, script_id: int) -> DownloadResult:
        """Download TikTok video using Apify actor (wilcode/fast-tiktok-downloader)."""
        actor_url = f"{self.APIFY_BASE_URL}/acts/{self.APIFY_TIKTOK_DOWNLOADER}/run-sync-get-dataset-items"

        # This actor expects "url" (singular), not "urls"
        response = await self._post(
            actor_url,
            params={"token": settings.APIFY_API_KEY},
            json={"url": url},
            timeout=120.0
        )

        items = response.json()
        if not items or not isinstance(items, list):
            raise ValueError("No video data returned from TikTok downloader")

        item = items[0]
        logger.info(f"TikTok downloader response keys: {item.keys()}")

        # Response format: {"status": "success", "result": {"video": {"playAddr": [...]}}}
        result_data = item.get("result", {})
        video_data = result_data.get("video", {})
        play_addrs = video_data.get("playAddr", [])

        # Also check for downloadAddr which may be more reliable
        download_addrs = video_data.get("downloadAddr", [])

        # Collect all possible video URLs to try
        video_urls = []
        if download_addrs:
            video_urls.extend(download_addrs if isinstance(download_addrs, list) else [download_addrs])
        if play_addrs:
            video_urls.extend(play_addrs if isinstance(play_addrs, list) else [play_addrs])

        if not video_urls:
            raise ValueError("No video URL in response")

        logger.info(f"Found {len(video_urls)} video URLs to try")

        # Try each URL until one works
        last_error = None
        for video_url in video_urls:
            try:
                logger.info(f"Trying video URL: {video_url[:80]}...")
                result = await self._download_and_save(video_url, script_id, "tiktok")
                if result.success and result.file_size_bytes > 1000:
                    return result
                else:
                    logger.warning(f"Download returned empty/small file, trying next URL")
            except Exception as e:
                logger.warning(f"URL failed: {e}")
                last_error = e
                continue

        raise ValueError(f"All video URLs failed. Last error: {last_error}")

    async def _download_instagram(self, url: str, script_id: int) -> DownloadResult:
        """Download Instagram video using Apify actor."""
        actor_url = f"{self.APIFY_BASE_URL}/acts/{self.APIFY_INSTAGRAM_DOWNLOADER}/run-sync-get-dataset-items"

        response = await self._post(
            actor_url,
            params={"token": settings.APIFY_API_KEY},
            json={
                "directUrls": [url],
                "resultsType": "details",
                "resultsLimit": 1
            },
            timeout=120.0
        )

        items = response.json()
        if not items or not isinstance(items, list):
            raise ValueError("No video data returned from Instagram downloader")

        item = items[0]
        video_url = item.get("videoUrl")

        if not video_url:
            raise ValueError("No video URL in response - may be an image post")

        return await self._download_and_save(video_url, script_id, "instagram")

    async def _download_youtube(self, url: str, script_id: int) -> DownloadResult:
        """Download YouTube video using Apify actor."""
        actor_url = f"{self.APIFY_BASE_URL}/acts/{self.APIFY_YOUTUBE_DOWNLOADER}/run-sync-get-dataset-items"

        response = await self._post(
            actor_url,
            params={"token": settings.APIFY_API_KEY},
            json={
                "startUrls": [{"url": url}],
                "downloadVideos": True,
                "maxResults": 1
            },
            timeout=180.0
        )

        items = response.json()
        if not items or not isinstance(items, list):
            raise ValueError("No video data returned from YouTube downloader")

        item = items[0]
        video_url = item.get("downloadUrl") or item.get("videoUrl")

        if not video_url:
            raise ValueError("No video URL in response")

        return await self._download_and_save(video_url, script_id, "youtube")

    async def _download_fallback(
        self,
        url: str,
        script_id: int,
        platform: str
    ) -> DownloadResult:
        """
        Fallback to local video-processor service for unsupported platforms.
        Uses yt-dlp running in the video-processor container.
        """
        import httpx

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                "http://video-processor:8080/download",
                json={"url": url}
            )

            if response.status_code != 200:
                raise ValueError(f"Video processor failed: {response.text}")

            result = response.json()
            downloaded_path = result.get("path")

            if not downloaded_path:
                raise ValueError("No path in response")

            # Copy to assets directory
            import shutil
            output_path = asset_paths.source_video_path(script_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(downloaded_path, output_path)

            return DownloadResult(
                success=True,
                script_id=script_id,
                platform=platform,
                video_path=str(output_path),
                file_size_bytes=output_path.stat().st_size
            )

    async def _download_and_save(
        self,
        video_url: str,
        script_id: int,
        platform: str
    ) -> DownloadResult:
        """Download video from URL and save to assets directory."""
        import httpx

        output_path = asset_paths.source_video_path(script_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Browser-like headers required for TikTok CDN and similar services
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.tiktok.com/",
            "Origin": "https://www.tiktok.com",
            "Sec-Fetch-Dest": "video",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
        }

        # Use a fresh client with browser headers and follow redirects
        async with httpx.AsyncClient(
            timeout=300.0,
            follow_redirects=True,
            headers=headers
        ) as client:
            response = await client.get(video_url)

            # Check for non-success status codes
            if response.status_code == 204:
                raise ValueError(f"CDN returned 204 No Content - URL may be expired or blocked")
            if response.status_code != 200:
                raise ValueError(f"Download failed with status {response.status_code}")

            content = response.content
            if len(content) < 1000:
                raise ValueError(f"Downloaded content too small ({len(content)} bytes)")

            with open(output_path, "wb") as f:
                f.write(content)

        file_size = output_path.stat().st_size

        # Get duration using ffprobe
        duration = self._get_video_duration(output_path)

        logger.info(f"Downloaded {platform} video: {output_path} ({file_size} bytes, {duration:.1f}s)")

        return DownloadResult(
            success=True,
            script_id=script_id,
            platform=platform,
            video_path=str(output_path),
            video_url=video_url,
            duration_seconds=duration,
            file_size_bytes=file_size
        )

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        import subprocess

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
                timeout=30
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def source_video_exists(self, script_id: int) -> bool:
        """Check if source video already exists."""
        path = asset_paths.source_video_path(script_id)
        return path.exists() and path.stat().st_size > 1000

    def get_source_video_path(self, script_id: int) -> Optional[Path]:
        """Get path to existing source video if it exists."""
        path = asset_paths.source_video_path(script_id)
        if path.exists() and path.stat().st_size > 1000:
            return path
        return None
