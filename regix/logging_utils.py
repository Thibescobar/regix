"""Logging and traceability.

Real-world constraints addressed here:

* no patient identifier in clear text in the logs by default (pseudonymisation by
  keyed HMAC, with no predictable built-in key);
* one JSON manifest per run: library versions, effective configuration, input
  hashes, duration of each step. That file is what you re-read six months later
  to know what actually ran;
* raw elastix logs are kept as they are (one elastix.log per stage).
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from regix.env import environment_state, pseudonym_key
from regix.env import pseudonymization_warning as _environment_pseudonymization_warning

LOGGER_NAME = "regix"
_PATH_RE = re.compile(r"(?<![\w.-])(?:[A-Za-z]:[\\/]|/)(?:[^\s\"'<>]|\\ )+")


def _disclaimer() -> str:
    """The regulatory statement, imported late to avoid a circular import."""
    from regix import DISCLAIMER

    return DISCLAIMER


def get_logger(name: str | None = None) -> logging.Logger:
    base = logging.getLogger(LOGGER_NAME)
    return base if name is None else base.getChild(name)


def setup_logging(
    level: str | int = "INFO",
    log_file: str | os.PathLike[str] | None = None,
    quiet: bool = False,
    redact_paths: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    base_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    class _RedactingFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            safe = copy(record)
            safe.msg = redact_text(record.getMessage())
            safe.args = ()
            return base_formatter.format(safe)

    fmt = _RedactingFormatter() if redact_paths else base_formatter
    resolved_level = getattr(level, "value", level)
    if not quiet:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(
            resolved_level
            if isinstance(resolved_level, int)
            else getattr(logging, str(resolved_level).upper())
        )
        stream.setFormatter(fmt)
        logger.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)  # the file keeps everything, even in quiet mode
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    elif quiet:
        # Without any handler, Python's ``lastResort`` handler still writes ERROR
        # records to stderr. A null handler is therefore required for strict JSON CLI
        # modes, where one stray log line corrupts the document.
        logger.addHandler(logging.NullHandler())
    return logger


def pseudonymize(value: str | None, salt: str | None = None, length: int = 32) -> str:
    """Return a keyed, non-enumerable pseudonym for an identifier.

    ``REGIX_PSEUDONYM_SALT`` is used as the HMAC key when ``salt`` is omitted.
    There is deliberately no public built-in key: without configuration a random
    process-local key is used, which is safe against enumeration but not stable
    across processes. :func:`pseudonymization_warning` exposes that condition to
    callers so it can be recorded in the manifest and by ``regix doctor``.
    """
    if not value:
        return "unknown"
    key = pseudonym_key(salt)
    digest = hmac.new(key, str(value).encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[: max(32, min(int(length), len(digest)))]


def pseudonymization_warning(salt: str | None = None) -> str | None:
    """Describe an unsafe or non-reproducible pseudonymisation configuration."""
    return _environment_pseudonymization_warning(salt)


def redact_path(value: str | os.PathLike[str] | None) -> str | None:
    """Replace an input path with a non-identifying representation."""
    if value is None:
        return None
    text = os.fspath(value)
    name = Path(text).name
    return f"<redacted>/{name}" if name else "<redacted-path>"


def redact_text(value: str) -> str:
    """Remove absolute filesystem paths from an exception or warning string."""
    return _PATH_RE.sub("<redacted-path>", value)


def _json_safe(value: Any, *, redact_paths: bool = False) -> Any:
    """Recursively produce strict JSON values, optionally removing absolute paths."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v, redact_paths=redact_paths) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, redact_paths=redact_paths) for v in value]
    if isinstance(value, Path):
        return redact_path(value) if redact_paths else str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and redact_paths:
        return redact_text(value)
    return value


def artifact_safe(value: Any, *, redact_paths: bool = True) -> Any:
    """Public wrapper for strict, path-safe JSON/YAML artifact values."""
    return _json_safe(value, redact_paths=redact_paths)


def file_digest(path: str | os.PathLike[str], chunk: int = 1 << 20, max_bytes: int | None = None) -> str:
    """SHA-256 of a file (truncatable for large volumes: then prefixed with 'partial:')."""
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            read += len(block)
            if max_bytes is not None and read >= max_bytes:
                return "partial:" + h.hexdigest()[:32]
    return h.hexdigest()[:32]


def environment_report() -> dict[str, Any]:
    """Versions of the dependencies that numerically influence the result."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    distributions = {
        "itk-elastix": "itk_elastix",
        "itk": "itk_python",
        "SimpleITK": "simpleitk",
        "numpy": "numpy",
        "pydicom": "pydicom",
        "PyYAML": "pyyaml",
        "matplotlib": "matplotlib",
        "torch": "torch",
        "monai": "monai",
        "TotalSegmentator": "totalsegmentator",
        "anatomix": "anatomix",
    }
    for distribution, key in distributions.items():
        try:  # pragma: no cover - depends on the installation
            report[key] = importlib.metadata.version(distribution)
        except (importlib.metadata.PackageNotFoundError, ValueError):
            report[key] = None
    try:  # pragma: no cover
        import itk

        report["itk_core"] = itk.Version.GetITKVersion()
    except Exception:
        pass
    # Do not import torch merely to build a manifest or run `doctor`: importing a GPU
    # stack can allocate hundreds of MB and can itself fail on a broken CUDA runtime.
    # If the feature path already loaded it, record the live state; otherwise keep the
    # distinction between "not probed" (None) and "probed, unavailable" (False).
    torch = sys.modules.get("torch")
    if torch is None:
        report["cuda_available"] = None
        report["cuda_device"] = None
    else:  # pragma: no cover - depends on optional feature execution
        try:
            report["cuda_available"] = bool(torch.cuda.is_available())
            report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            report["cuda_available"] = False
            report["cuda_device"] = None
    try:
        from regix import __version__

        report["regix"] = __version__
    except Exception:
        report["regix"] = None
    try:
        report["configuration"] = environment_state()
    except ValueError as exc:
        report["configuration"] = {"error": str(exc), "pseudonym_salt_configured": False}
    try:
        import shutil

        report["disk_free_mb"] = round(shutil.disk_usage(Path.cwd()).free / (1024 * 1024), 1)
    except OSError:
        report["disk_free_mb"] = None
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        report["memory_available_mb"] = round(pages * page_size / (1024 * 1024), 1)
    except (AttributeError, OSError, ValueError):
        report["memory_available_mb"] = None
    return report


@dataclass
class StepTiming:
    name: str
    seconds: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    """Run manifest, written as JSON at the end (and on exception)."""

    run_id: str
    output_dir: Path
    inputs: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=environment_report)
    steps: list[StepTiming] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    status: str = "running"
    redact_paths: bool = True
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    finished_at: str | None = None

    def warn(self, message: str) -> None:
        safe = redact_text(message) if self.redact_paths else message
        get_logger("manifest").warning(safe)
        self.warnings.append(safe)

    def degraded(self, message: str) -> None:
        """Record that requested behaviour fell back or could not be completed."""
        safe = redact_text(message) if self.redact_paths else message
        if safe not in self.degradations:
            self.degradations.append(safe)
        if safe not in self.warnings:
            self.warn(safe)

    @contextmanager
    def step(self, name: str, **details: Any) -> Iterator[dict[str, Any]]:
        log = get_logger("step")
        log.info("-> %s", name)
        t0 = time.perf_counter()
        extra: dict[str, Any] = dict(details)
        try:
            yield extra
        finally:
            dt = time.perf_counter() - t0
            self.steps.append(StepTiming(name=name, seconds=round(dt, 3), details=extra))
            log.info("<- %s (%.2f s)", name, dt)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_dir": str(self.output_dir),
            "inputs": self.inputs,
            "config": self.config,
            "environment": self.environment,
            "steps": [{"name": s.name, "seconds": s.seconds, **s.details} for s in self.steps],
            "metrics": self.metrics,
            "warnings": self.warnings,
            "degradations": self.degradations,
            "outputs": self.outputs,
            "disclaimer": _disclaimer(),
        }
        return _json_safe(payload, redact_paths=self.redact_paths)

    def save(self, path: str | os.PathLike[str] | None = None) -> Path:
        self.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            import resource

            peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # macOS reports bytes; Linux and the BSDs report KiB.
            self.environment["peak_rss_mb"] = round(
                peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024,
                1,
            )
        except (ImportError, OSError, ValueError):
            self.environment["peak_rss_mb"] = None
        target = Path(path) if path is not None else self.output_dir / "run_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str, allow_nan=False),
            encoding="utf-8",
        )
        return target
