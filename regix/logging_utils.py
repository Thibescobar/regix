"""Logging and traceability.

Real-world constraints addressed here:

* no patient identifier in clear text in the logs by default (pseudonymisation by
  truncated salted hash, configurable salt);
* one JSON manifest per run: library versions, effective configuration, input
  hashes, duration of each step. That file is what you re-read six months later
  to know what actually ran;
* raw elastix logs are kept as they are (one elastix.log per stage).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER_NAME = "regix"
_DEFAULT_SALT_ENV = "REGIX_PSEUDONYM_SALT"


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
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    if not quiet:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper()))
        stream.setFormatter(fmt)
        logger.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)  # the file keeps everything, even in quiet mode
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def pseudonymize(value: str | None, salt: str | None = None, length: int = 10) -> str:
    """Truncated salted hash of an identifier. Never reversible without the salt."""
    if not value:
        return "unknown"
    salt = salt if salt is not None else os.environ.get(_DEFAULT_SALT_ENV, "regix")
    digest = hashlib.sha256(f"{salt}::{value}".encode()).hexdigest()
    return digest[:length]


def file_digest(
    path: str | os.PathLike[str], chunk: int = 1 << 20, max_bytes: int | None = None
) -> str:
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
    for mod, key in (
        ("SimpleITK", "simpleitk"),
        ("itk", "itk"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("monai", "monai"),
        ("anatomix", "anatomix"),
    ):
        try:  # pragma: no cover - depends on the installation
            m = __import__(mod)
            report[key] = getattr(m, "__version__", "unknown")
        except Exception:
            report[key] = None
    try:  # pragma: no cover
        import itk

        report["itk"] = itk.Version.GetITKVersion()
    except Exception:
        pass
    try:  # pragma: no cover
        import torch

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
    status: str = "running"
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    finished_at: str | None = None

    def warn(self, message: str) -> None:
        get_logger("manifest").warning(message)
        self.warnings.append(message)

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
        return {
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
            "disclaimer": _disclaimer(),
        }

    def save(self, path: str | os.PathLike[str] | None = None) -> Path:
        self.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        target = Path(path) if path is not None else self.output_dir / "run_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return target
