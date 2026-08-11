"""Minimal HTTP service (FastAPI) to plug Regix into an existing workflow.

Designed for the reality of an imaging department: volumes do not travel as HTTP
attachments, they sit on a network share or on locally mounted object storage. The
API therefore takes **paths** and returns **paths**, not bytes. That also keeps
requests well under gateway size limits and avoids duplicating hundreds of
megabytes per call.

Work is asynchronous: a registration takes anywhere from seconds to minutes, well
beyond a reasonable request timeout. We therefore return a job identifier that can
be polled.

    uvicorn regix.api:app --host 127.0.0.1 --port 8000

Assumed limitations: in-memory execution in a single process and no job persistence.
Filesystem access is confined by ``REGIX_API_ALLOWED_ROOTS`` and an optional static
bearer token can be configured with ``REGIX_API_TOKEN``. A queue (Celery/RQ), TLS and
site identity management remain deployment responsibilities.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from regix import DISCLAIMER, __version__
from regix.env import API_TOKEN_ENV, api_allowed_roots
from regix.logging_utils import get_logger

log = get_logger("api")

try:  # pragma: no cover - optional dependency
    from fastapi import Depends, FastAPI, Header, HTTPException, Query
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("the API requires fastapi and uvicorn: pip install 'regix[api]'") from exc


# --------------------------------------------------------------------------- #
class RegisterRequest(BaseModel):
    fixed: str = Field(description="Path to the fixed volume (file or DICOM directory).")
    moving: str = Field(description="Path to the moving volume.")
    output_dir: str = Field(description="Output directory (must be writable).")
    fixed_series_uid: str | None = None
    moving_series_uid: str | None = None
    preset: str = Field(default="base", description="Bundled preset name or path to a YAML file.")
    organs: list[str] = Field(default_factory=list)
    fixed_labelmap: str | None = None
    moving_labelmap: str | None = None
    label_names: dict[int, str] | None = None
    landmarks_fixed: str | None = None
    landmarks_moving: str | None = None
    overwrite: bool = Field(default=False, description="Replace only known Regix artifacts in output_dir.")
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description='Nested configuration overrides, e.g. {"preprocess": {"working_spacing_mm": 1.5}}',
    )


class JobStatus(BaseModel):
    job_id: str
    state: Literal["queued", "running", "done", "error"]
    submitted_at: float
    finished_at: float | None = None
    seconds: float | None = None
    qc_status: str | None = None
    metrics: dict[str, Any] | None = None
    outputs: dict[str, str] | None = None
    warnings: list[str] | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
_MAX_JOBS = 1000


class _BoundedJobStore(dict[str, JobStatus]):
    """Insertion-bounded store; terminal jobs are evicted before active work."""

    def __setitem__(self, key: str, value: JobStatus) -> None:
        is_new = key not in self
        if is_new:
            while len(self) >= _MAX_JOBS:
                terminal = [
                    (job.submitted_at, job_id)
                    for job_id, job in self.items()
                    if job.state in {"done", "error"}
                ]
                victim = (
                    min(terminal)[1]
                    if terminal
                    else min(self.items(), key=lambda item: item[1].submitted_at)[0]
                )
                dict.__delitem__(self, victim)
        dict.__setitem__(self, key, value)


_JOBS = _BoundedJobStore()
_LOCK = threading.Lock()
_POOL: ThreadPoolExecutor | None = None


def _require_token(
    authorization: str | None = Header(default=None),
    x_regix_token: str | None = Header(default=None),
) -> None:
    expected = os.getenv(API_TOKEN_ENV)
    if not expected:
        return
    supplied = x_regix_token
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:]
    if supplied is None or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="authentication required")


@asynccontextmanager
async def _lifespan(_application: FastAPI):
    global _POOL
    _POOL = ThreadPoolExecutor(max_workers=1)  # elastix is already multi-threaded internally
    try:
        yield
    finally:
        pool, _POOL = _POOL, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)


app = FastAPI(
    title="Regix",
    description=f"Multimodal / multi-organ registration.\n\n**{DISCLAIMER}**",
    version=__version__,
    dependencies=[Depends(_require_token)],
    lifespan=_lifespan,
)


def _allowed_roots() -> tuple[Path, ...]:
    return api_allowed_roots()


def _confined_path(value: str, label: str, *, must_exist: bool) -> Path:
    candidate = Path(value).expanduser().resolve(strict=False)
    if not any(candidate == root or candidate.is_relative_to(root) for root in _allowed_roots()):
        raise HTTPException(status_code=400, detail=f"{label} is outside the allowed roots")
    if must_exist and not candidate.exists():
        raise HTTPException(status_code=400, detail=f"{label} volume not found")
    return candidate


def _validate_request_paths(request: RegisterRequest) -> None:
    for label, value, must_exist in (
        ("fixed", request.fixed, True),
        ("moving", request.moving, True),
        ("output_dir", request.output_dir, False),
        ("fixed_labelmap", request.fixed_labelmap, True),
        ("moving_labelmap", request.moving_labelmap, True),
        ("landmarks_fixed", request.landmarks_fixed, True),
        ("landmarks_moving", request.landmarks_moving, True),
    ):
        if value is not None:
            _confined_path(value, label, must_exist=must_exist)
    preset = Path(request.preset)
    if preset.suffix.lower() in {".yaml", ".yml"} or any(
        separator in request.preset for separator in ("/", "\\")
    ):
        _confined_path(request.preset, "preset", must_exist=True)


def _update(job_id: str, **fields: Any) -> None:
    with _LOCK:
        current = _JOBS[job_id]
        _JOBS[job_id] = current.model_copy(update=fields)


def _build_config(request: RegisterRequest):
    from regix.config import load_preset

    cfg = load_preset(request.preset)
    overrides: dict[str, Any] = dict(request.overrides)
    if request.fixed_series_uid:
        overrides["fixed_series_uid"] = request.fixed_series_uid
    if request.moving_series_uid:
        overrides["moving_series_uid"] = request.moving_series_uid
    if request.organs:
        overrides.setdefault("organs", {})["targets"] = request.organs
    if request.fixed_labelmap or request.moving_labelmap:
        organs = overrides.setdefault("organs", {})
        organs["backend"] = "external"
        if request.fixed_labelmap:
            organs["fixed_labelmap"] = request.fixed_labelmap
        if request.moving_labelmap:
            organs["moving_labelmap"] = request.moving_labelmap
        if request.label_names:
            organs["label_names"] = request.label_names
    if request.landmarks_fixed and request.landmarks_moving:
        qc = overrides.setdefault("qc", {})
        qc["landmarks_fixed"] = request.landmarks_fixed
        qc["landmarks_moving"] = request.landmarks_moving
    overrides.setdefault("output", {})["overwrite"] = request.overwrite
    return cfg.with_overrides(**overrides)


def _run_job(job_id: str, request: RegisterRequest) -> None:
    from regix.pipeline import RegistrationPipeline

    _update(job_id, state="running")
    try:
        _validate_request_paths(request)
        cfg = _build_config(request)
        result = RegistrationPipeline(cfg).run(request.fixed, request.moving, request.output_dir)
        _update(
            job_id,
            state="done",
            finished_at=time.time(),
            seconds=round(result.seconds, 2),
            qc_status=result.status,
            metrics=result.metrics,
            outputs={k: str(v) for k, v in result.outputs.items()},
            warnings=result.warnings,
        )
    except Exception:  # details stay server-side; paths can identify a patient
        trace_id = uuid.uuid4().hex[:12]
        log.exception("job %s failed (trace %s)", job_id, trace_id)
        _update(
            job_id,
            state="error",
            finished_at=time.time(),
            error=f"ProcessingError (trace_id={trace_id})",
        )


# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, Any]:
    """Service state and engine availability: poll this before submitting."""
    from regix.logging_utils import environment_report
    from regix.registration.itk_bridge import engine_available

    ok, detail = engine_available()
    return {
        "status": "ok" if ok else "degraded",
        "engine": detail,
        "environment": environment_report(),
        "jobs": len(_JOBS),
        "disclaimer": DISCLAIMER,
    }


@app.get("/presets")
def presets() -> list[dict[str, Any]]:
    from regix.config import available_presets, load_preset

    out = []
    for name in available_presets():
        cfg = load_preset(name)
        out.append(
            {
                "name": name,
                "fixed_modality": cfg.fixed_modality,
                "moving_modality": cfg.moving_modality,
                "stages": [s.type.value for s in cfg.stages],
                "description": (cfg.description or "").strip(),
            }
        )
    return out


@app.post("/register", response_model=JobStatus, status_code=202)
def register(request: RegisterRequest) -> JobStatus:
    """Submit a registration. Returns a job identifier immediately."""
    _validate_request_paths(request)
    try:
        _build_config(request)  # validate the configuration before accepting the job
    except Exception as exc:
        trace_id = uuid.uuid4().hex[:12]
        log.warning("invalid job configuration (trace %s): %s", trace_id, exc)
        raise HTTPException(
            status_code=400,
            detail=f"invalid configuration (trace_id={trace_id})",
        ) from exc

    if _POOL is None:  # primarily protects direct ASGI use without a lifespan manager
        raise HTTPException(status_code=503, detail="service is not ready")
    job_id = uuid.uuid4().hex[:12]
    status = JobStatus(job_id=job_id, state="queued", submitted_at=time.time())
    with _LOCK:
        _JOBS[job_id] = status
    _POOL.submit(_run_job, job_id, request)
    log.info("job %s submitted", job_id)
    return status


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job(job_id: str) -> JobStatus:
    with _LOCK:
        if job_id not in _JOBS:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        return _JOBS[job_id]


@app.get("/jobs", response_model=list[JobStatus])
def jobs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[JobStatus]:
    with _LOCK:
        ordered = sorted(_JOBS.values(), key=lambda j: j.submitted_at, reverse=True)
        return ordered[offset : offset + limit]
