"""
Publisher service for multi-platform distribution using Blotato.
Replaces n8n nodes: Prepare Publish Data, Publish (Blotato), Parse Response, Save Publish Record.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from .base import BaseService
from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)


class PublishPlatform(BaseModel):
    """Configuration for a single platform."""
    platform: str  # tiktok, instagram, youtube, etc.
    account_id: str
    enabled: bool = True


class PublishRequest(BaseModel):
    """Request parameters for publishing."""
    script_id: int
    video_url: str
    caption: str
    platforms: List[str] = ["tiktok", "instagram", "youtube"]
    hashtags: List[str] = []
    schedule_time: Optional[str] = None  # ISO format or None for immediate


class PublishResult(BaseModel):
    """Result from publishing."""
    script_id: int
    platform: str
    success: bool
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    error: Optional[str] = None
    scheduled_time: Optional[str] = None


class PublisherService(BaseService):
    """
    Service for publishing videos to social platforms.

    Replaces n8n nodes:
    - Prepare Publish Data
    - Publish (Blotato)
    - Parse Response
    - Save Publish Record
    - Update Status Published
    """

    BLOTATO_BASE_URL = "https://backend.blotato.com/v2"

    async def publish(
        self,
        request: PublishRequest
    ) -> List[PublishResult]:
        """
        Publish video to multiple platforms via Blotato.

        Args:
            request: Publishing request with video and platforms

        Returns:
            List of results for each platform
        """
        results = []

        for platform in request.platforms:
            try:
                result = await self._publish_to_platform(request, platform)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to publish to {platform}: {e}")
                results.append(PublishResult(
                    script_id=request.script_id,
                    platform=platform,
                    success=False,
                    error=str(e)
                ))

        return results

    async def _publish_to_platform(
        self,
        request: PublishRequest,
        platform: str
    ) -> PublishResult:
        """
        Publish to a single platform.

        Args:
            request: Publishing request
            platform: Target platform

        Returns:
            PublishResult
        """
        url = f"{self.BLOTATO_BASE_URL}/posts"

        # Build caption with hashtags
        full_caption = self._build_caption(request.caption, request.hashtags)

        payload = {
            "platform": platform,
            "media_url": request.video_url,
            "caption": full_caption,
            "media_type": "video"
        }

        if request.schedule_time:
            payload["scheduled_time"] = request.schedule_time

        try:
            response = await self._post(
                url,
                headers={
                    "Authorization": f"Bearer {settings.BLOTATO_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60.0
            )

            result = response.json()

            return PublishResult(
                script_id=request.script_id,
                platform=platform,
                success=True,
                post_id=result.get("id"),
                post_url=result.get("url"),
                scheduled_time=request.schedule_time
            )

        except Exception as e:
            logger.error(f"Blotato API error for {platform}: {e}")
            raise

    def _build_caption(
        self,
        caption: str,
        hashtags: List[str]
    ) -> str:
        """
        Build full caption with hashtags.

        Args:
            caption: Base caption text
            hashtags: List of hashtags (without #)

        Returns:
            Full caption with hashtags appended
        """
        if not hashtags:
            return caption

        hashtag_str = " ".join(f"#{tag}" for tag in hashtags)
        return f"{caption}\n\n{hashtag_str}"

    async def get_post_status(
        self,
        post_id: str
    ) -> Dict[str, Any]:
        """
        Get status of a published or scheduled post.

        Args:
            post_id: Blotato post ID

        Returns:
            Post status information
        """
        url = f"{self.BLOTATO_BASE_URL}/posts/{post_id}"

        try:
            response = await self._get(
                url,
                headers={"Authorization": f"Bearer {settings.BLOTATO_API_KEY}"}
            )
            return response.json()

        except Exception as e:
            logger.error(f"Failed to get post status: {e}")
            raise

    async def cancel_scheduled_post(
        self,
        post_id: str
    ) -> bool:
        """
        Cancel a scheduled post.

        Args:
            post_id: Blotato post ID

        Returns:
            True if cancelled successfully
        """
        url = f"{self.BLOTATO_BASE_URL}/posts/{post_id}"

        try:
            response = await self._delete(
                url,
                headers={"Authorization": f"Bearer {settings.BLOTATO_API_KEY}"}
            )
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to cancel post: {e}")
            raise

    async def list_accounts(self) -> List[Dict[str, Any]]:
        """
        List connected social accounts.

        Returns:
            List of account information
        """
        url = f"{self.BLOTATO_BASE_URL}/accounts"

        try:
            response = await self._get(
                url,
                headers={"Authorization": f"Bearer {settings.BLOTATO_API_KEY}"}
            )
            return response.json().get("accounts", [])

        except Exception as e:
            logger.error(f"Failed to list accounts: {e}")
            raise

    def prepare_publish_data(
        self,
        script_id: int,
        video_url: str,
        script_text: str,
        pillar: str,
        niche: str = "real estate"
    ) -> PublishRequest:
        """
        Prepare data for publishing based on script and pillar.

        Args:
            script_id: Script ID
            video_url: Public video URL
            script_text: Script CTA or full text
            pillar: Content pillar
            niche: Content niche

        Returns:
            PublishRequest ready for publishing
        """
        # Generate hashtags based on pillar and niche
        base_hashtags = ["realestate", "realtor", "homebuying"]

        pillar_hashtags = {
            "market_intelligence": ["marketupdate", "realestateinvesting", "housingmarket"],
            "educational_tips": ["realestatetips", "homebuyertips", "firsttimehomebuyer"],
            "lifestyle_local": ["idahorealestate", "northidaho", "spokanerealestate"],
            "brand_humanization": ["realtorlife", "behindthescenes", "dayinthelife"]
        }

        hashtags = base_hashtags + pillar_hashtags.get(pillar, [])

        # Truncate caption for platform limits
        max_caption_length = 2200  # Instagram limit
        caption = script_text[:max_caption_length] if len(script_text) > max_caption_length else script_text

        return PublishRequest(
            script_id=script_id,
            video_url=video_url,
            caption=caption,
            platforms=["tiktok", "instagram", "youtube"],
            hashtags=hashtags
        )

    async def save_publish_record(
        self,
        results: List[PublishResult],
        db_session
    ) -> int:
        """
        Save publish records to database.

        Args:
            results: List of publish results
            db_session: Database session

        Returns:
            Number of records saved
        """
        from models import Published

        saved_count = 0

        for result in results:
            if not result.success:
                continue

            try:
                record = Published(
                    script_id=result.script_id,
                    platform=result.platform,
                    post_id=result.post_id,
                    post_url=result.post_url,
                    published_at=datetime.now(timezone.utc),
                    scheduled_time=result.scheduled_time
                )
                db_session.add(record)
                await db_session.commit()
                saved_count += 1

            except Exception as e:
                logger.error(f"Failed to save publish record: {e}")
                await db_session.rollback()
                continue

        return saved_count

    async def update_content_status(
        self,
        script_id: int,
        db_session,
        status: str = "published"
    ) -> None:
        """
        Update content idea status to published.

        Args:
            script_id: Script ID
            db_session: Database session
            status: New status
        """
        from models import ContentIdea, Script

        try:
            # Get content idea ID from script
            result = await db_session.execute(
                Script.__table__.select().where(Script.id == script_id)
            )
            script = result.fetchone()

            if script and script.content_idea_id:
                await db_session.execute(
                    ContentIdea.__table__.update().where(
                        ContentIdea.id == script.content_idea_id
                    ).values(status=status)
                )
                await db_session.commit()
                logger.info(f"Updated content idea status to {status}")

        except Exception as e:
            logger.error(f"Failed to update content status: {e}")
            await db_session.rollback()
            raise
