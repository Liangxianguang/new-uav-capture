"""Default-off/offline-shadow isolation. Guarded control is deliberately closed."""
from __future__ import annotations

import copy
import time

import numpy as np


class ResponsePlugin:
    def __init__(self, mode="off", predictor=None, budget_ms=10.):
        if mode not in ("off", "shadow"):
            raise ValueError("Guarded mode requires independent promotion; not implemented in this pilot")
        self.mode, self.predictor, self.budget_ms = mode, predictor, float(budget_ms)
        self.last_diagnostics = {"status": "off"}

    def observe(self, baseline_action, public_inputs):
        action = np.asarray(baseline_action).copy()
        if self.mode == "off":
            self.last_diagnostics = {"status": "off", "predictor_called": False}
            return action
        started = time.perf_counter()
        try:
            if self.predictor is None:
                raise ValueError("Missing predictor")
            prediction = np.asarray(self.predictor(copy.deepcopy(public_inputs)), dtype=float)
            if prediction.ndim != 3 or prediction.shape[-1] != 3 or not np.isfinite(prediction).all():
                raise ValueError("Invalid response prediction")
            elapsed = (time.perf_counter() - started) * 1000
            self.last_diagnostics = {"status": "shadow_over_budget" if elapsed > self.budget_ms else "shadow",
                                     "predictor_called": True, "elapsed_ms": elapsed}
        except Exception as error:
            self.last_diagnostics = {"status": "shadow_error", "predictor_called": True, "reason": str(error)}
        # Offline shadow never chooses actions. Its synchronous timer is a
        # diagnostic, NOT a preemptive real-time deadline guarantee.
        return action
