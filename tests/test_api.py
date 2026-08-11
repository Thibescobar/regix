"""HTTP service tests.

Audit I-01: `regix/api.py` had zero lines of coverage and no test file at all. The README
files it under the paths "that need hardware or third-party weights this project does not
redistribute", alongside anatomix and the GPU stage -- but it is the only one of the four
that needs neither. `fastapi` and `httpx` install on CPU in seconds, and `TestClient`
exercises every route in-process without a network.

That matters more than the coverage figure: the API is the most exposed surface of the
project. Its filesystem allowlist, optional bearer token, bounded job history and opaque
errors are therefore executable contracts rather than deployment prose.
"""

from __future__ import annotations

import time

import pytest
import SimpleITK as sitk

from tests.conftest import known_rigid, make_phantom, warp

fastapi = pytest.importorskip("fastapi", reason="the HTTP service needs `pip install regix[api]`")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from regix.api import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _clear_jobs(monkeypatch, tmp_path):
    """The service keeps jobs in a module-level dict; tests must not see each other's."""
    from regix import api

    with api._LOCK:
        api._JOBS.clear()
    monkeypatch.setenv("REGIX_API_ALLOWED_ROOTS", str(tmp_path))
    yield


@pytest.fixture
def pair(tmp_path):
    """A small CT-CT pair on disk, sized so a full job stays under a few seconds."""
    image, labels = make_phantom("CT", shape=(28, 32, 32), noise=2.0)
    truth = known_rigid(image, rotation_deg=(2.0, 0.0, 1.0), translation_mm=(3.0, -2.0, 1.0))
    paths = {}
    for name, img, dtype in (
        ("fixed", image, sitk.sitkFloat32),
        ("moving", warp(image, truth), sitk.sitkFloat32),
        ("labels", labels, sitk.sitkUInt16),
    ):
        path = tmp_path / f"{name}.nii.gz"
        sitk.WriteImage(sitk.Cast(img, dtype), str(path), True)
        paths[name] = path
    return paths


def _wait(client, job_id: str, timeout: float = 240.0) -> dict:
    """Poll a job to completion. The service is asynchronous by design."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/jobs/{job_id}").json()
        if payload["state"] in ("done", "error"):
            return payload
        time.sleep(0.25)
    pytest.fail(f"job {job_id} did not finish within {timeout} s")


# --------------------------------------------------------------------------- #
# Informational routes
# --------------------------------------------------------------------------- #
def test_health_reports_the_engine_and_the_disclaimer(client):
    """`poll this before submitting`, says the docstring. It must say something useful."""
    from regix import DISCLAIMER

    payload = client.get("/health").json()
    assert payload["status"] in ("ok", "degraded")
    assert payload["disclaimer"] == DISCLAIMER, "the regulatory statement must not drift"
    assert payload["environment"]["python"]
    assert isinstance(payload["jobs"], int)


def test_presets_route_lists_every_bundled_preset(client):
    from regix.config import available_presets

    payload = client.get("/presets").json()
    assert {entry["name"] for entry in payload} == set(available_presets())
    for entry in payload:
        assert entry["stages"], f"{entry['name']} reports no stage"
        assert entry["description"].strip(), f"{entry['name']} reports no description"


# --------------------------------------------------------------------------- #
# Submission and validation
# --------------------------------------------------------------------------- #
def test_a_missing_volume_is_refused_before_the_job_is_queued(client, tmp_path, pair):
    response = client.post(
        "/register",
        json={
            "fixed": str(tmp_path / "nope.nii.gz"),
            "moving": str(pair["moving"]),
            "output_dir": str(tmp_path / "out"),
        },
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]
    assert client.get("/jobs").json() == [], "a rejected submission must not create a job"


def test_an_invalid_configuration_is_refused_before_the_job_is_queued(client, tmp_path, pair):
    """The configuration is validated at submission, not discovered when the job runs."""
    response = client.post(
        "/register",
        json={
            "fixed": str(pair["fixed"]),
            "moving": str(pair["moving"]),
            "output_dir": str(tmp_path / "out"),
            "preset": "no_such_preset",
        },
    )
    assert response.status_code == 400
    assert "invalid configuration" in response.json()["detail"]


def test_an_unknown_job_is_a_404(client):
    assert client.get("/jobs/deadbeef").status_code == 404


def test_overrides_reach_the_configuration(pair, tmp_path):
    """`_build_config` is where organs, label maps and landmarks are wired in."""
    from regix.api import RegisterRequest, _build_config

    cfg = _build_config(
        RegisterRequest(
            fixed=str(pair["fixed"]),
            moving=str(pair["moving"]),
            output_dir=str(tmp_path / "out"),
            organs=["liver"],
            fixed_labelmap=str(pair["labels"]),
            moving_labelmap=str(pair["labels"]),
            label_names={1: "liver"},
            overrides={"preprocess": {"working_spacing_mm": 3.0}},
        )
    )
    assert cfg.organs.targets == ["liver"]
    assert cfg.organs.backend.value == "external"
    assert cfg.organs.label_names == {1: "liver"}
    assert cfg.preprocess.working_spacing_mm == 3.0


def test_submission_protects_existing_outputs_by_default(pair, tmp_path):
    from regix.api import RegisterRequest, _build_config

    cfg = _build_config(
        RegisterRequest(
            fixed=str(pair["fixed"]), moving=str(pair["moving"]), output_dir=str(tmp_path / "out")
        )
    )
    assert cfg.output.overwrite is False


# --------------------------------------------------------------------------- #
# A full job
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_a_submitted_job_runs_and_reports_its_outputs(client, tmp_path, pair):
    out = tmp_path / "out"
    response = client.post(
        "/register",
        json={
            "fixed": str(pair["fixed"]),
            "moving": str(pair["moving"]),
            "output_dir": str(out),
            "overrides": {
                "preprocess": {"working_spacing_mm": 3.0},
                "qc": {"report_html": False},
                "runtime": {"log_level": "WARNING"},
            },
        },
    )
    assert response.status_code == 202
    submitted = response.json()
    assert submitted["state"] == "queued"

    payload = _wait(client, submitted["job_id"])
    assert payload["state"] == "done", payload.get("error")
    assert payload["qc_status"] in ("PASS", "WARN", "FAIL", "NOT_EVALUATED")
    assert payload["seconds"] and payload["seconds"] > 0
    assert (out / "run_manifest.json").exists()
    assert "registered" in payload["outputs"]
    assert client.get("/jobs").json()[0]["job_id"] == submitted["job_id"]


@pytest.mark.slow
def test_a_failing_job_is_reported_as_an_error_not_a_crash(client, tmp_path, pair):
    """A job that dies must leave the service healthy and say so through the API."""
    broken = tmp_path / "broken.nii.gz"
    broken.write_bytes(b"this is not a volume")
    response = client.post(
        "/register",
        json={"fixed": str(pair["fixed"]), "moving": str(broken), "output_dir": str(tmp_path / "out")},
    )
    assert response.status_code == 202
    payload = _wait(client, response.json()["job_id"])
    assert payload["state"] == "error"
    assert payload["error"], "an error state must carry a reason"
    assert client.get("/health").json()["status"] in ("ok", "degraded")


# --------------------------------------------------------------------------- #
# Security expectations (audit C-03, F-11)
# --------------------------------------------------------------------------- #
def test_a_path_outside_the_allowlist_is_refused(client, tmp_path, pair, monkeypatch):
    monkeypatch.setenv("REGIX_API_ALLOWED_ROOTS", str(tmp_path / "allowed"))
    response = client.post(
        "/register",
        json={
            "fixed": str(pair["fixed"]),  # outside the allowed root
            "moving": str(pair["moving"]),
            "output_dir": str(tmp_path / "elsewhere"),
        },
    )
    assert response.status_code == 400, "a path outside the allowlist must be refused"


def test_a_symlink_cannot_escape_the_allowlist(client, tmp_path, pair, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.nii.gz"
    outside.write_bytes(pair["fixed"].read_bytes())
    link = allowed / "escape.nii.gz"
    link.symlink_to(outside)
    monkeypatch.setenv("REGIX_API_ALLOWED_ROOTS", str(allowed))

    response = client.post(
        "/register",
        json={
            "fixed": str(link),
            "moving": str(link),
            "output_dir": str(allowed / "out"),
        },
    )

    assert response.status_code == 400


def test_configured_api_token_is_required(client, monkeypatch):
    monkeypatch.setenv("REGIX_API_TOKEN", "correct-horse-battery-staple")

    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert (
        client.get(
            "/health",
            headers={"Authorization": "Bearer correct-horse-battery-staple"},
        ).status_code
        == 200
    )


def test_a_qc_gate_can_be_disabled_by_an_api_override(pair, tmp_path):
    from regix.api import RegisterRequest, _build_config

    cfg = _build_config(
        RegisterRequest(
            fixed=str(pair["fixed"]),
            moving=str(pair["moving"]),
            output_dir=str(tmp_path / "out"),
            overrides={"qc": {"gates": {"max_tre_mm": None}}},
        )
    )

    assert cfg.qc.gates.max_tre_mm is None


@pytest.mark.slow
def test_a_job_error_does_not_leak_a_filesystem_path(client, tmp_path, pair):
    canary = tmp_path / "PATIENTNAMECANARY.nii.gz"
    canary.write_bytes(b"this is not a volume")
    response = client.post(
        "/register",
        json={"fixed": str(pair["fixed"]), "moving": str(canary), "output_dir": str(tmp_path / "out")},
    )
    payload = _wait(client, response.json()["job_id"])
    assert payload["state"] == "error"
    assert "PATIENTNAMECANARY" not in (payload["error"] or ""), (
        "the error returned to an unauthenticated client names the input file"
    )


def test_the_job_history_is_bounded(client, tmp_path, pair):
    from regix import api
    from regix.api import JobStatus

    with api._LOCK:
        for index in range(2000):
            job_id = f"synthetic{index:05d}"
            api._JOBS[job_id] = JobStatus(job_id=job_id, state="done", submitted_at=float(index))

    assert len(api._JOBS) <= 1000, f"{len(api._JOBS)} jobs retained with no eviction policy"
    assert len(client.get("/jobs").json()) <= 100, "GET /jobs returns the whole history at once"
