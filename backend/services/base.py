"""
Base service class with shared HTTP client and utilities.
"""

import httpx
from typing import Optional, Dict, Any, Union
from contextlib import asynccontextmanager

from utils.logging import get_logger
from utils.retry import async_retry

logger = get_logger(__name__)


class BaseService:
    """
    Base class for all services with shared HTTP client and retry logic.

    Usage:
        class MyService(BaseService):
            async def do_something(self):
                response = await self._request("GET", "https://api.example.com")
                return response.json()
    """

    def __init__(self, timeout: float = 60.0):
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialize HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @async_retry(max_attempts=3, min_wait=2, max_wait=10)
    async def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        json: Optional[Dict] = None,
        data: Optional[Any] = None,
        files: Optional[Dict] = None,
        content: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """
        Make HTTP request with automatic retry on failure.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            headers: Optional headers
            params: Optional query parameters
            json: Optional JSON body
            data: Optional form data
            files: Optional file uploads
            content: Optional raw bytes content
            timeout: Optional timeout override

        Returns:
            httpx.Response

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses
        """
        logger.info(f"HTTP {method} {url}")

        request_kwargs = {
            "method": method,
            "url": url,
        }

        if headers:
            request_kwargs["headers"] = headers
        if params is not None:
            request_kwargs["params"] = params
        if json is not None:
            request_kwargs["json"] = json
        if data is not None:
            request_kwargs["data"] = data
        if files is not None:
            request_kwargs["files"] = files
        if content is not None:
            request_kwargs["content"] = content
        if timeout is not None:
            request_kwargs["timeout"] = timeout

        response = await self.client.request(**request_kwargs)
        response.raise_for_status()

        return response

    async def _get(self, url: str, **kwargs) -> httpx.Response:
        """Convenience method for GET requests."""
        return await self._request("GET", url, **kwargs)

    async def _post(self, url: str, **kwargs) -> httpx.Response:
        """Convenience method for POST requests."""
        return await self._request("POST", url, **kwargs)

    async def _patch(self, url: str, **kwargs) -> httpx.Response:
        """Convenience method for PATCH requests."""
        return await self._request("PATCH", url, **kwargs)

    async def _delete(self, url: str, **kwargs) -> httpx.Response:
        """Convenience method for DELETE requests."""
        return await self._request("DELETE", url, **kwargs)
