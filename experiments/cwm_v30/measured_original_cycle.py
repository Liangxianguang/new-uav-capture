"""Read-only original cycle timestamp observer; no clocks or deadlines replaced."""
import math
import sys
import time


class MeasuredOriginalCycle:
    def __init__(self, base):
        self.base = base
        self.parent = base.evaluator.CaptureRadiusPursuit3DEnv
        self.code = base.evaluator.run_episode.__code__
        self.samples = []
        owner = self

        class MeasuredEnv(self.parent):
            def step(self, actions, *args, **kwargs):
                caller = sys._getframe(1)
                try:
                    if caller.f_code is not owner.code:
                        raise ValueError('Cycle observer requires actual unchanged evaluator run_episode')
                    started = caller.f_locals['control_started']
                finally:
                    del caller  # Never retain simulator/frame/truth as model input.
                before = time.perf_counter()
                if type(started) is not float or not math.isfinite(started) or not 0 < started <= before:
                    raise ValueError('Original actual cycle timestamp unavailable')
                # Includes original inference/proposal/scoring/CBF, V29 flush IO,
                # numeric command observer and native environment step/history.
                result = super().step(actions, *args, **kwargs)
                ended = time.perf_counter()
                owner.samples.append((ended-started)*1000.)
                return result

        self.measured = MeasuredEnv

    def install(self):
        self.samples = []
        self.base.evaluator.CaptureRadiusPursuit3DEnv = self.measured

    def reset(self):
        self.samples = []

    def close(self):
        self.base.evaluator.CaptureRadiusPursuit3DEnv = self.parent

    def validate(self, steps):
        if len(self.samples) != len(steps) or not self.samples:
            raise ValueError('Complete first/warmup native cycle timings required')
        native = [float(row['total_control_latency_ms']) for row in steps]
        if any(not math.isfinite(a) or a <= 0 or not math.isfinite(b) or b+1e-6 < a for a, b in zip(native, self.samples)):
            raise ValueError('Actual full cycle time must contain original native control time')
        failures = [float(row['planner_local_solver_failures']) for row in steps]
        if any(not math.isfinite(v) or v < 0 or int(v) != v for v in failures):
            raise ValueError('Finite actual integer local-failure counters required')
        return {'control_steps': len(steps), 'native_control_latency_ms': native,
            'end_to_end_control_latency_ms': list(self.samples),
            'local_solver_failure_steps': sum(v > 0 for v in failures),
            'planner_local_solver_failures': int(sum(failures))}
