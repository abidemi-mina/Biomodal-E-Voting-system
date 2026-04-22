"""
security.py
===========
Security middleware and utilities:
  - RateLimiter: In-memory rate limiting for auth endpoints
  - AccountLockout: Lock voter after N failed auth attempts
  - SecurityHeadersMiddleware: Extra security headers on every response
"""

import time
import hashlib
import logging
from collections import defaultdict
from threading import Lock

from django.http import JsonResponse
from django.utils import timezone

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Thread-safe in-memory rate limiter
# For production use Redis (django-ratelimit or django-redis)
# ─────────────────────────────────────────────────────────────
class _RateLimiterStore:
    def __init__(self):
        self._lock = Lock()
        # {key: [timestamp, timestamp, ...]}
        self._requests = defaultdict(list)
        # {key: lockout_until_timestamp}
        self._lockouts = {}
        # {voter_id: fail_count}
        self._fail_counts = defaultdict(int)
        # {voter_id: lockout_until}
        self._voter_lockouts = {}

    def is_ip_limited(self, ip: str, max_requests: int = 10, window_seconds: int = 60) -> bool:
        with self._lock:
            now = time.time()
            key = f"ip:{ip}"
            # Remove expired timestamps
            self._requests[key] = [t for t in self._requests[key] if now - t < window_seconds]
            if len(self._requests[key]) >= max_requests:
                return True
            self._requests[key].append(now)
            return False

    def is_voter_locked(self, voter_id: str) -> tuple[bool, int]:
        """Returns (is_locked, seconds_remaining)"""
        with self._lock:
            until = self._voter_lockouts.get(voter_id, 0)
            now = time.time()
            if until > now:
                return True, int(until - now)
            return False, 0

    def record_fail(self, voter_id: str, max_fails: int = 5, lockout_minutes: int = 30):
        with self._lock:
            self._fail_counts[voter_id] += 1
            if self._fail_counts[voter_id] >= max_fails:
                self._voter_lockouts[voter_id] = time.time() + lockout_minutes * 60
                self._fail_counts[voter_id] = 0
                logger.warning(f"Voter {voter_id} locked out for {lockout_minutes} minutes after {max_fails} failures.")

    def reset_fails(self, voter_id: str):
        with self._lock:
            self._fail_counts.pop(voter_id, None)
            self._voter_lockouts.pop(voter_id, None)

    def fail_count(self, voter_id: str) -> int:
        with self._lock:
            return self._fail_counts.get(voter_id, 0)


# Singleton store
_store = _RateLimiterStore()


def check_auth_rate_limit(request, voter_id: str) -> JsonResponse | None:
    """
    Call at the start of /api/authenticate/.
    Returns a JsonResponse error if rate limited / locked out, else None.
    """
    ip = _get_ip(request)

    # 1. IP-level rate limit: max 20 attempts per minute across all voters
    if _store.is_ip_limited(ip, max_requests=20, window_seconds=60):
        logger.warning(f"IP rate limit hit: {ip}")
        return JsonResponse({
            'success': False,
            'message': 'Too many requests from your IP address. Please wait 1 minute.',
            'rate_limited': True,
        }, status=429)

    # 2. Voter-level lockout
    locked, secs = _store.is_voter_locked(voter_id)
    if locked:
        minutes = secs // 60
        return JsonResponse({
            'success': False,
            'message': f'Account temporarily locked due to multiple failed attempts. '
                       f'Try again in {minutes} minute(s).',
            'locked_out': True,
            'seconds_remaining': secs,
        }, status=429)

    return None


def record_auth_failure(voter_id: str, max_fails: int = 5, lockout_minutes: int = 30):
    _store.record_fail(voter_id, max_fails=max_fails, lockout_minutes=lockout_minutes)
    return _store.fail_count(voter_id)


def record_auth_success(voter_id: str):
    _store.reset_fails(voter_id)


def _get_ip(request) -> str:
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', '0.0.0.0')


# ─────────────────────────────────────────────────────────────
# Security Headers Middleware
# ─────────────────────────────────────────────────────────────
class SecurityHeadersMiddleware:
    """Adds security headers to every response."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response['Permissions-Policy'] = 'camera=(self), microphone=(), geolocation=()'
        return response
