"""
Storage service for cloud storage operations (GCS, Dropbox).
Replaces n8n nodes: GCS Upload, Format GCS URL, Dropbox operations.
"""

import base64
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel

from .base import BaseService
from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)


class UploadResult(BaseModel):
    """Result of file upload."""
    script_id: int
    file_path: str
    public_url: str
    storage_provider: str
    file_size_bytes: int


class StorageService(BaseService):
    """
    Service for cloud storage operations.

    Replaces n8n nodes:
    - Read Final Video
    - GCS Upload
    - Format GCS URL
    - Dropbox OAuth operations
    """

    # GCS settings
    GCS_UPLOAD_URL = "https://storage.googleapis.com/upload/storage/v1/b"
    GCS_PUBLIC_URL_TEMPLATE = "https://storage.googleapis.com/{bucket}/{object}"

    # Dropbox settings
    DROPBOX_CONTENT_URL = "https://content.dropboxapi.com/2/files/upload"
    DROPBOX_API_URL = "https://api.dropboxapi.com/2"

    async def upload_to_gcs(
        self,
        script_id: int,
        file_path: Path,
        bucket: str,
        object_name: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> UploadResult:
        """
        Upload file to Google Cloud Storage.

        Args:
            script_id: Script ID for tracking
            file_path: Local file path
            bucket: GCS bucket name
            object_name: Object name in bucket (defaults to filename)
            credentials: GCS credentials dict

        Returns:
            UploadResult with public URL
        """
        if object_name is None:
            object_name = f"videos/{file_path.name}"

        url = f"{self.GCS_UPLOAD_URL}/{bucket}/o?uploadType=media&name={object_name}"

        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Get OAuth token (in n8n this comes from credential)
            auth_header = self._build_gcs_auth_header(credentials)

            response = await self._post(
                url,
                headers={
                    **auth_header,
                    "Content-Type": "video/mp4"
                },
                content=file_data,
                timeout=300.0
            )

            result = response.json()

            # Build public URL
            public_url = self.GCS_PUBLIC_URL_TEMPLATE.format(
                bucket=bucket,
                object=object_name
            )

            logger.info(f"Uploaded to GCS: {public_url}")

            return UploadResult(
                script_id=script_id,
                file_path=str(file_path),
                public_url=public_url,
                storage_provider="gcs",
                file_size_bytes=len(file_data)
            )

        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            raise

    def _build_gcs_auth_header(
        self,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Build authorization header for GCS."""
        # In production, this would use service account or OAuth
        # For now, we rely on credentials being passed
        if credentials and "access_token" in credentials:
            return {"Authorization": f"Bearer {credentials['access_token']}"}

        # Fall back to API key if available
        return {}

    async def upload_to_dropbox(
        self,
        script_id: int,
        file_path: Path,
        dropbox_path: Optional[str] = None
    ) -> UploadResult:
        """
        Upload file to Dropbox.

        Args:
            script_id: Script ID for tracking
            file_path: Local file path
            dropbox_path: Path in Dropbox (defaults to /videos/)

        Returns:
            UploadResult with shared link
        """
        if dropbox_path is None:
            dropbox_path = f"/videos/{file_path.name}"

        # First, get fresh access token using refresh token
        access_token = await self._refresh_dropbox_token()

        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Upload file
            import json
            response = await self._post(
                self.DROPBOX_CONTENT_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Dropbox-API-Arg": json.dumps({
                        "path": dropbox_path,
                        "mode": "overwrite",
                        "autorename": True
                    }),
                    "Content-Type": "application/octet-stream"
                },
                content=file_data,
                timeout=300.0
            )

            result = response.json()

            # Create shared link
            shared_url = await self._create_dropbox_shared_link(
                access_token,
                dropbox_path
            )

            logger.info(f"Uploaded to Dropbox: {shared_url}")

            return UploadResult(
                script_id=script_id,
                file_path=str(file_path),
                public_url=shared_url,
                storage_provider="dropbox",
                file_size_bytes=len(file_data)
            )

        except Exception as e:
            logger.error(f"Dropbox upload failed: {e}")
            raise

    async def _refresh_dropbox_token(self) -> str:
        """Refresh Dropbox access token using refresh token."""
        import base64

        auth_string = f"{settings.DROPBOX_APP_KEY}:{settings.DROPBOX_APP_SECRET}"
        auth_bytes = base64.b64encode(auth_string.encode()).decode()

        try:
            response = await self._post(
                "https://api.dropboxapi.com/oauth2/token",
                headers={
                    "Authorization": f"Basic {auth_bytes}",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": settings.DROPBOX_REFRESH_TOKEN
                }
            )

            result = response.json()
            return result["access_token"]

        except Exception as e:
            logger.error(f"Dropbox token refresh failed: {e}")
            raise

    async def _create_dropbox_shared_link(
        self,
        access_token: str,
        path: str
    ) -> str:
        """Create a shared link for a Dropbox file."""
        try:
            response = await self._post(
                f"{self.DROPBOX_API_URL}/sharing/create_shared_link_with_settings",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "path": path,
                    "settings": {
                        "requested_visibility": "public"
                    }
                }
            )

            result = response.json()
            return result.get("url", "")

        except Exception as e:
            # Link might already exist, try to get it
            try:
                response = await self._post(
                    f"{self.DROPBOX_API_URL}/sharing/list_shared_links",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    },
                    json={"path": path, "direct_only": True}
                )
                result = response.json()
                links = result.get("links", [])
                if links:
                    return links[0].get("url", "")
            except Exception:
                pass

            logger.error(f"Failed to create Dropbox shared link: {e}")
            raise

    def format_public_url(
        self,
        storage_provider: str,
        bucket_or_account: str,
        object_path: str
    ) -> str:
        """
        Format a public URL for a stored file.

        Args:
            storage_provider: gcs, dropbox, etc.
            bucket_or_account: Bucket name or account ID
            object_path: Path to object

        Returns:
            Public URL
        """
        if storage_provider == "gcs":
            return f"https://storage.googleapis.com/{bucket_or_account}/{object_path}"
        elif storage_provider == "dropbox":
            # Dropbox links are returned directly from API
            return object_path
        else:
            raise ValueError(f"Unknown storage provider: {storage_provider}")
