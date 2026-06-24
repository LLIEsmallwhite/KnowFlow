"""
Rate Limiting Middleware

Token-bucket rate limiter using Redis.
Limits: 100 req/min per user, 300 req/min per IP.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory token bucket rate limiter (Redis-backed in production)."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict = {}  # key -> list of timestamps

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed. Returns True if under limit."""
        now = time.time()
        cutoff = now - self.window
        timestamps = self._buckets.get(key, [])
        # Prune expired
        timestamps = [t for t in timestamps if t > cutoff]
        self._buckets[key] = timestamps

        if len(timestamps) >= self.max_requests:
            return False
        timestamps.append(now)
        return True

    def remaining(self, key: str) -> int:
        """How many requests remaining in current window."""
        timestamps = self._buckets.get(key, [])
        cutoff = time.time() - self.window
        active = sum(1 for t in timestamps if t > cutoff)
        return max(0, self.max_requests - active)


# Global instances
user_limiter = RateLimiter(max_requests=100, window_seconds=60)   # 100/min per user
ip_limiter = RateLimiter(max_requests=300, window_seconds=60)     # 300/min per IP


async def check_rate_limit(user_id: Optional[str], client_ip: str) -> None:
    """Raise HTTPException if rate limit exceeded."""
    from fastapi import HTTPException, status

    if user_id and not user_limiter.is_allowed(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后再试 (100次/分钟)",
        )
    if not ip_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后再试 (300次/分钟)",
        )
