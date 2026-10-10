"""Fail-closed deployment gate for the optional V23 response predictor."""
from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path


REQUIRED_ONLINE_CHECKS = (
    "original_entry_off_equivalence",
    "original_entry_refusal_equivalence",
    "original_entry_failure_equivalence",
    "sequential_candidates",
    "self_generated_candidates",
    "untouched_holdout",
    "original_8_levels_closed_loop",
    "new_scene_closed_loop",
    "safety_contract",
    "latency_contract",
)


class QualificationGate:
    """A model is controllable only after offline and online contracts pass."""

    def __init__(self, *, mode: str = "off", qualification: dict | None = None):
        if mode not in ("off", "shadow", "guarded"):
            raise ValueError("mode must be off, shadow or guarded")
        self.mode = mode
        self.qualification = qualification if qualification is not None else {}
        if not isinstance(self.qualification, dict) or not isinstance(self.qualification.get('online_checks', {}), dict):
            raise ValueError('Qualification/checks must be mappings')
        self.offline_eligible = self.qualification.get("research_eligible", False) is True
        self.online_checks = {
            key: self.qualification.get("online_checks", {}).get(key, False) is True
            for key in REQUIRED_ONLINE_CHECKS
        }
        self.online_eligible = all(self.online_checks.values())
        self.control_eligible = mode == "guarded" and self.offline_eligible and self.online_eligible
        if mode == "off":
            self.status = "off"
        elif not self.offline_eligible:
            self.status = "qualification_failed"
        elif not self.online_eligible:
            self.status = "online_contract_pending"
        elif mode == "shadow":
            self.status = "shadow_only"
        else:
            self.status = "guarded_ready"

    @classmethod
    def from_json(cls, path: Path, *, mode: str = "off") -> "QualificationGate":
        return cls(mode=mode, qualification=json.loads(path.read_text(encoding="utf8")))

    def refusal(self) -> dict:
        return {
            "mode": self.mode,
            "status": self.status,
            "offline_eligible": self.offline_eligible,
            "online_eligible": self.online_eligible,
            "control_eligible": self.control_eligible,
            "online_checks": dict(self.online_checks),
        }


class OptionalResponseAdapter:
    """Lazy public TARGET forecast boundary, never an action-selection function.

    Empty forecast means the caller keeps the original controller route. Shadow
    may forecast a failed model, but never acquires control eligibility.
    """

    def __init__(self, gate: QualificationGate, loader=None):
        self.gate, self.loader = gate, loader
        self.predictor = None
        self.status = gate.status
        self.load_calls = self.prediction_calls = 0

    def forecast(self, history, relative, proposed, anchor, backbone, cv):
        import numpy as np
        if (self.gate.mode == 'off' or (self.gate.mode == 'guarded' and not self.gate.control_eligible) or
                self.status in ('load_failed', 'prediction_failed')):
            return None, {'forecast_available': False, 'control_eligible': False, 'reason': self.status}
        try:
            h, r, p, a, b, v = [_finite_array(x, name).copy() for x, name in zip(
                (history, relative, proposed, anchor, backbone, cv), ('history', 'relative', 'proposed', 'anchor', 'backbone', 'cv'))]
            if (h.shape != (8, 252) or r.shape != (4, 6) or p.ndim != 4 or p.shape[1:] != (8, 4, 3) or
                not len(p) or a.shape != (8, 4, 3) or b.shape != (8, 3) or v.shape != (8, 3) or
                b.dtype != np.float64 or v.dtype != b.dtype or
                any(not np.issubdtype(x.dtype, np.floating) for x in (h, r, p, a, b, v)) or
                np.linalg.norm(p, axis=-1).max() > 5. + 1e-8 or np.linalg.norm(a, axis=-1).max() > 5. + 1e-8):
                raise ValueError('Frozen original public forecast input contract differs')
            original_p, original_a, target_dtype = p.copy(), a.copy(), b.dtype
            if self.predictor is None:
                self.load_calls += 1
                try:
                    with preserved_cpu_rng():
                        self.predictor = self.loader()
                    if not callable(self.predictor):
                        raise ValueError('Loader must return a public predictor')
                except Exception:
                    self.predictor = None
                    self.status = 'load_failed'
                    return None, {'forecast_available': False, 'control_eligible': False, 'reason': self.status}
            self.prediction_calls += 1
            with preserved_cpu_rng():
                result = self.predictor(h, r, p, a, b, v)
            if not isinstance(result, dict) or set(result) != {'prediction', 'reference', 'response'}:
                raise ValueError('Explicit target prediction/reference/response required')
            result = {k: _finite_array(x, k).copy() for k, x in result.items()}
            prediction, reference, response = (result[k] for k in ('prediction', 'reference', 'response'))
            target_shape = (len(original_p), 8, 3)  # Target paths, NOT joint UAV actions.
            if any(x.shape != target_shape or not np.issubdtype(x.dtype, np.floating) for x in result.values()):
                raise ValueError('Target prediction output shape/type differs')
            if prediction.dtype != target_dtype or reference.dtype != target_dtype:
                raise ValueError('Original target path precision differs')
            if any(row.tobytes() != reference[0].tobytes() for row in reference):
                raise ValueError('Common motion must not depend on proposed actions')
            anchor_rows = (original_p == original_a[None]).all(axis=(1, 2, 3))
            if (response[anchor_rows] != 0).any():
                raise ValueError('Anchor response is not exactly zero')
            expected = np.where(response == 0, reference, reference + response.astype(reference.dtype))
            if prediction.tobytes() != expected.tobytes():
                raise ValueError('Prediction does not equal anchored common plus response')
            self.status = 'shadow_ready' if self.gate.mode == 'shadow' else 'guarded_ready'
            return result, {'forecast_available': True, 'control_eligible': self.gate.control_eligible, 'reason': self.status}
        except Exception as error:
            self.status = 'prediction_failed'
            return None, {'forecast_available': False, 'control_eligible': False,
                          'reason': self.status, 'error_type': type(error).__name__}


@contextmanager
def preserved_cpu_rng():
    """No CUDA initialization; original CPU caller global RNG is restored."""
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _finite_array(value, name):
    import numpy as np

    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result
