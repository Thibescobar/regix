"""Acceptance gates for a registration.

Philosophy: a registration is never deleted, it is **labelled**. A user who
receives a FAIL knows to look; a result silently replaced by a fallback is a
trap. Every check reports its measured value, its threshold and its verdict, so
that it can be re-read later.

Three verdicts: PASS (nothing to report), WARN (verify visually, often because a
measurement is unavailable), FAIL (do not use as is).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from regix.config import QCGates
from regix.logging_utils import get_logger

log = get_logger("qc.gates")

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_SEVERITY = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class Check:
    name: str
    status: str
    measured: Any = None
    threshold: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "measured": self.measured,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class GateResult:
    status: str = PASS
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)
        if _SEVERITY[check.status] > _SEVERITY[self.status]:
            self.status = check.status

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_failures": len(self.failures),
            "n_warnings": len(self.warnings),
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary_line(self) -> str:
        if self.status == PASS:
            return f"QC: PASS ({len(self.checks)} checks)"
        details = "; ".join(f"{c.name}={c.measured}" for c in (self.failures or self.warnings))
        return f"QC: {self.status} -> {details}"


def evaluate_gates(
    gates: QCGates,
    similarity: dict[str, Any] | None = None,
    organ_overlap: dict[str, dict[str, float]] | None = None,
    jacobian: dict[str, Any] | None = None,
    linear_analysis: dict[str, Any] | None = None,
    landmarks: dict[str, Any] | None = None,
    deformable: bool = False,
) -> GateResult:
    """Compare the QC measurements against the configured thresholds."""
    result = GateResult()

    # --- 1. similarity gain ---------------------------------------------- #
    # NCC for monomodal, NMI for multimodal: comparing CT and MR intensities by
    # correlation is meaningless, so the threshold would be arbitrary.
    for key, threshold in (("ncc", gates.min_ncc_gain), ("nmi", gates.min_nmi_gain)):
        if threshold is None:
            continue
        gain = (similarity or {}).get(f"{key}_gain")
        name = f"{key}_gain"
        if gain is None or not np.isfinite(gain):
            result.add(
                Check(name, WARN, gain, threshold,
                      f"{key.upper()} gain not computable (no comparable initial state)")
            )
        elif gain < threshold:
            result.add(
                Check(name, FAIL, gain, threshold,
                      "registration did not improve similarity: initialization or metric "
                      "is inadequate")
            )
        else:
            result.add(Check(name, PASS, gain, threshold))

    # --- 2. per-organ Dice ------------------------------------------------ #
    for organ, threshold in (gates.min_dice or {}).items():
        entry = (organ_overlap or {}).get(organ)
        if entry is None or entry.get("dice") is None or not np.isfinite(entry.get("dice", np.nan)):
            result.add(
                Check(f"dice[{organ}]", WARN, None, threshold,
                      f"Dice for {organ} unavailable (organ missing from one of the volumes)")
            )
            continue
        value = float(entry["dice"])
        status = PASS if value >= threshold else FAIL
        result.add(
            Check(f"dice[{organ}]", status, round(value, 4), threshold,
                  "" if status == PASS else "insufficient overlap after registration")
        )

    # --- 3. field folding ------------------------------------------------- #
    if jacobian and jacobian.get("available"):
        fraction = float(jacobian.get("folding_fraction", 0.0))
        status = PASS if fraction <= gates.max_folding_fraction else FAIL
        result.add(
            Check(
                "folding_fraction",
                status,
                fraction,
                gates.max_folding_fraction,
                ""
                if status == PASS
                else "folded field: the deformation turns anatomy inside out locally. "
                "Increase bending_energy_weight or widen the B-spline grid.",
            )
        )
        det_min, det_max = jacobian.get("det_min"), jacobian.get("det_max")
        if det_min is not None and det_max is not None and (det_min < 0.2 or det_max > 5.0):
            result.add(
                Check(
                    "jacobian_range",
                    WARN,
                    [det_min, det_max],
                    [0.2, 5.0],
                    "extreme local compression/expansion: plausible during breathing, "
                    "suspicious on a solid organ",
                )
            )
    elif deformable:
        result.add(
            Check("folding_fraction", WARN, None, gates.max_folding_fraction,
                  "Jacobian not computed although the transform is deformable")
        )

    # --- 4. plausibility of the linear part ------------------------------ #
    if linear_analysis:
        if gates.max_translation_mm is not None:
            value = float(linear_analysis.get("translation_norm_mm", 0.0))
            status = PASS if value <= gates.max_translation_mm else FAIL
            result.add(
                Check("translation_mm", status, value, gates.max_translation_mm,
                      "" if status == PASS else "implausible translation: likely divergence")
            )
        if gates.max_scale_deviation is not None:
            value = float(linear_analysis.get("max_scale_deviation", 0.0))
            status = PASS if value <= gates.max_scale_deviation else FAIL
            result.add(
                Check("scale_deviation", status, value, gates.max_scale_deviation,
                      "" if status == PASS else "abnormal scale change for an intra-patient "
                      "registration: check the DICOM spacings")
            )
        determinant = linear_analysis.get("determinant")
        if determinant is not None and determinant <= 0:
            result.add(
                Check("determinant", FAIL, determinant, "> 0",
                      "negative determinant: the transform includes a mirror flip")
            )

    # --- 5. TRE ----------------------------------------------------------- #
    if gates.max_tre_mm is not None:
        if not landmarks:
            result.add(Check("tre_mm", WARN, None, gates.max_tre_mm, "no landmarks provided"))
        else:
            value = float(landmarks.get("tre_mean_mm", np.nan))
            if not np.isfinite(value):
                result.add(Check("tre_mm", WARN, None, gates.max_tre_mm, "TRE not computable"))
            else:
                status = PASS if value <= gates.max_tre_mm else FAIL
                result.add(
                    Check("tre_mm", status, value, gates.max_tre_mm,
                          "" if status == PASS else "landmark error above the clinical threshold")
                )

    if not result.checks:
        result.add(Check("configuration", WARN, None, None, "no QC gate configured"))

    log.info(result.summary_line())
    for check in result.failures:
        log.error(
            "QC FAIL %s: measured=%s threshold=%s | %s",
            check.name, check.measured, check.threshold, check.message,
        )
    for check in result.warnings:
        log.warning("QC WARN %s: %s", check.name, check.message)
    return result
