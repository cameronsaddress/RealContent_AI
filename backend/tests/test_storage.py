"""
Tests for StorageService - uses mocked responses, no real API calls.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from services.storage import (
    StorageService,
    UploadResult
)


class TestUploadResult:
    """Test UploadResult model."""

    def test_result(self):
        """Test result model."""
        result = UploadResult(
            script_id=22,
            file_path="/app/assets/output/22_final.mp4",
            public_url="https://storage.googleapis.com/bucket/videos/22_final.mp4",
            storage_provider="gcs",
            file_size_bytes=30000000
        )
        assert result.storage_provider == "gcs"
        assert "googleapis" in result.public_url


class TestStorageService:
    """Test StorageService methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return StorageService()

    def test_format_public_url_gcs(self, service):
        """Test GCS URL formatting."""
        url = service.format_public_url(
            "gcs",
            "my-bucket",
            "videos/22_final.mp4"
        )
        assert url == "https://storage.googleapis.com/my-bucket/videos/22_final.mp4"

    def test_format_public_url_dropbox(self, service):
        """Test Dropbox URL formatting (passthrough)."""
        dropbox_url = "https://dropbox.com/s/abc123/video.mp4"
        url = service.format_public_url(
            "dropbox",
            "ignored",
            dropbox_url
        )
        assert url == dropbox_url

    def test_format_public_url_unknown_provider(self, service):
        """Test unknown provider raises error."""
        with pytest.raises(ValueError) as exc_info:
            service.format_public_url("s3", "bucket", "path")
        assert "Unknown storage provider" in str(exc_info.value)

    def test_build_gcs_auth_header_with_token(self, service):
        """Test GCS auth header with access token."""
        credentials = {"access_token": "test_token_123"}
        header = service._build_gcs_auth_header(credentials)
        assert header["Authorization"] == "Bearer test_token_123"

    def test_build_gcs_auth_header_no_credentials(self, service):
        """Test GCS auth header without credentials."""
        header = service._build_gcs_auth_header(None)
        assert header == {}


class TestStorageServiceAsync:
    """Test async methods."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return StorageService()

    @pytest.mark.asyncio
    async def test_upload_to_gcs_mocked(self, service):
        """Test GCS upload with mocked response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "videos/22_final.mp4",
            "bucket": "my-bucket"
        }

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post, \
             patch('builtins.open', MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"video data"))),
                 __exit__=MagicMock()
             ))):

            mock_post.return_value = mock_response

            result = await service.upload_to_gcs(
                script_id=22,
                file_path=Path("/app/assets/output/22_final.mp4"),
                bucket="my-bucket"
            )

            assert result.script_id == 22
            assert result.storage_provider == "gcs"
            assert "my-bucket" in result.public_url

    @pytest.mark.asyncio
    async def test_refresh_dropbox_token(self, service):
        """Test Dropbox token refresh."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new_access_token_123"
        }

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            token = await service._refresh_dropbox_token()

            assert token == "new_access_token_123"
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_dropbox_shared_link(self, service):
        """Test creating Dropbox shared link."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "url": "https://dropbox.com/s/abc123/video.mp4"
        }

        with patch.object(service, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            url = await service._create_dropbox_shared_link(
                "test_token",
                "/videos/22_final.mp4"
            )

            assert "dropbox.com" in url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
