"""One registration, run in its own process, printing 400 transformed points as JSON.

Used by ``tests/test_contract.py::test_two_runs_of_the_same_configuration_agree``. It has
to be a separate process: elastix keeps global state, and two runs inside one interpreter
would prove less than two runs from a clean start -- which is the situation a deploying
site actually cares about.

The leading underscore keeps pytest from collecting it as a test module.

    python tests/data/_determinism_probe.py FIXED MOVING OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main(fixed: str, moving: str, out_dir: str) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from regix.config import load_preset
    from regix.pipeline import RegistrationPipeline

    out = Path(out_dir)
    cfg = load_preset("base").with_overrides(
        preprocess={"working_spacing_mm": 2.5},
        output={"dir": str(out), "overwrite": True},
        runtime={"log_level": "ERROR"},
        qc={"report_html": False, "enabled": False},
    )
    result = RegistrationPipeline(cfg).run(fixed, moving, out)
    transform = result.applied_transform.as_sitk_transform()
    if transform is None:
        print("no usable sitk transform", file=sys.stderr)
        return 1

    reference = sitk.ReadImage(fixed)
    rng = np.random.default_rng(20250101)
    size = np.asarray(reference.GetSize(), dtype=float)
    indices = rng.uniform(0.1, 0.9, size=(400, 3)) * (size - 1)
    mapped = [
        list(map(float, transform.TransformPoint(reference.TransformContinuousIndexToPhysicalPoint(list(i)))))
        for i in indices
    ]
    print(json.dumps(mapped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
