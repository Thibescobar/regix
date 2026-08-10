"""Validated environment configuration used by the CLI, API and exporters.

Only names and non-secret fingerprints are exposed to manifests. Secret values are
never returned by :func:`environment_state`.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path

PSEUDONYM_SALT_ENV = "REGIX_PSEUDONYM_SALT"
DICOM_UID_ROOT_ENV = "REGIX_DICOM_UID_ROOT"
API_ALLOWED_ROOTS_ENV = "REGIX_API_ALLOWED_ROOTS"
API_TOKEN_ENV = "REGIX_API_TOKEN"

_EPHEMERAL_PSEUDONYM_KEY = secrets.token_bytes(32)
_WEAK_SALTS = {"regix", "changeme", "change-me", "default", "password", "secret", "test"}


def pseudonym_key(explicit: str | None = None) -> bytes:
    """Return a validated HMAC key; absence uses a safe process-local random key."""
    if explicit is not None:
        configured = explicit
    elif PSEUDONYM_SALT_ENV in os.environ:
        configured = os.environ[PSEUDONYM_SALT_ENV]
    else:
        configured = None
    if configured == "":
        raise ValueError(f"{PSEUDONYM_SALT_ENV} must not be empty")
    return configured.encode("utf-8") if configured is not None else _EPHEMERAL_PSEUDONYM_KEY


def pseudonymization_warning(explicit: str | None = None) -> str | None:
    """Describe an unsafe or non-reproducible pseudonymisation configuration."""
    if explicit is not None:
        configured = explicit
    elif PSEUDONYM_SALT_ENV in os.environ:
        configured = os.environ[PSEUDONYM_SALT_ENV]
    else:
        configured = None
    if configured == "":
        return f"{PSEUDONYM_SALT_ENV} is empty and invalid; remove it or configure a secret"
    if configured is None:
        return (
            f"{PSEUDONYM_SALT_ENV} is not set: a random process-local HMAC key is in use; "
            "pseudonyms are protected but will change after restart"
        )
    if len(configured) < 16 or configured.strip().lower() in _WEAK_SALTS:
        return (
            f"{PSEUDONYM_SALT_ENV} is weak: use at least 16 unpredictable characters "
            "from the deployment secret store"
        )
    return None


def pseudonym_salt_fingerprint(explicit: str | None = None) -> str:
    """Non-secret identifier allowing two runs to prove they used the same HMAC key."""
    return hashlib.sha256(pseudonym_key(explicit)).hexdigest()[:12]


def api_allowed_roots() -> tuple[Path, ...]:
    """Resolved API filesystem allowlist (the current directory when unspecified)."""
    configured = os.getenv(API_ALLOWED_ROOTS_ENV)
    values = configured.split(os.pathsep) if configured else [str(Path.cwd())]
    roots = tuple(Path(value).expanduser().resolve() for value in values if value.strip())
    if not roots:
        raise RuntimeError(f"{API_ALLOWED_ROOTS_ENV} does not contain a usable root")
    return roots


def environment_state() -> dict[str, object]:
    """Safe deployment state for diagnostics and manifests; contains no secrets or paths."""
    return {
        "pseudonym_salt_configured": PSEUDONYM_SALT_ENV in os.environ
        and bool(os.environ[PSEUDONYM_SALT_ENV]),
        "pseudonym_salt_fingerprint": pseudonym_salt_fingerprint(),
        "dicom_uid_root_configured": bool(os.getenv(DICOM_UID_ROOT_ENV)),
        "api_allowed_roots_configured": bool(os.getenv(API_ALLOWED_ROOTS_ENV)),
        "api_token_configured": bool(os.getenv(API_TOKEN_ENV)),
    }
