"""Exercise the optional boundary at the real historical run_episode entry.

The planner and local CBF remain authoritative. The boundary runs after the
unchanged DN-MPC plan is produced, so refusal or prediction failure can only
return an empty forecast and leave the original action route untouched.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "cwm_v1"))
sys.path.insert(0, str(ROOT / "experiments" / "cwm_v25"))

from baseline import Baseline
from qualification_gate import OptionalResponseAdapter, QualificationGate
from freeze_baseline import REFERENCE, sha, verify

BASELINE_SHA = 'c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329'


def failed_record() -> dict:
    report = json.loads((ROOT / 'experiments/cwm_v23/reports/independent_model_validation.json').read_text())
    if report['primary_research_eligible'] is not False or report['enhanced_control_enabled'] is not False:
        raise ValueError('Expected failed, disabled V23 research model')
    return {'research_eligible': False}


def _valid_forecast(*args):
    proposed, backbone = np.asarray(args[2]), np.asarray(args[4])
    reference = np.repeat(backbone[None], len(proposed), axis=0)
    response = np.zeros(reference.shape, dtype=np.float32)
    return {"prediction": reference.copy(), "reference": reference.copy(), "response": response}


def fault_loader(failure):
    """Synthetic fault injection only; never loads or qualifies a real model."""
    import torch
    def perturb():
        random.random(); np.random.rand(); torch.rand(1)
    def load():
        perturb()
        if failure == 'load':
            raise OSError('controlled loader failure')
        def predict(*args):
            perturb()
            if failure == 'predict':
                for value in args:
                    value[...] = 100.
                raise RuntimeError('controlled inference failure after input mutation')
            return _valid_forecast(*args)
        return predict
    return load


class OriginalEntryBoundary:
    """Run the original entry while observing optional forecast behavior."""

    def __init__(self, capsule: Path, restored: Path, *, mode: str, failure=None):
        self.base = Baseline(capsule, restored)
        self.configure(mode, failure=failure)
        self.events: list[dict] = []
        parent = self.base.evaluator.DistributedMinimaxDNMPC
        boundary = self

        class BoundaryPlanner(parent):
            def plan(self, observation, scenarios, **kwargs):
                plan = super().plan(observation, scenarios, **kwargs)
                if boundary.plain:
                    return plan
                if boundary.adapter.gate.mode == "off" or (boundary.adapter.gate.mode == 'guarded' and not boundary.adapter.gate.control_eligible):
                    inputs = [object()] * 6
                else:
                    proposed = np.asarray(plan.action_sequence, dtype=np.float64)[None]
                    inputs = [
                        np.zeros((8, 252), dtype=np.float64), np.zeros((4, 6), dtype=np.float64),
                        proposed, proposed[0].copy(),
                        np.asarray(scenarios.trajectories[0][:8], dtype=np.float64),
                        np.zeros((8, 3), dtype=np.float64),
                    ]
                _, info = boundary.adapter.forecast(*inputs)
                boundary.events.append(info)
                return plan

        self.base.evaluator.DistributedMinimaxDNMPC = BoundaryPlanner
        self._boundary_class = BoundaryPlanner
        self._parent = parent

    def run(self, record: dict, trajectory: Path):
        return self.base.run(record, trajectory)

    def configure(self, mode: str, *, failure=None):
        self.plain = mode == 'plain'
        self.adapter = OptionalResponseAdapter(
            QualificationGate(mode="guarded" if mode == "refusal" else 'off' if self.plain else mode,
                              qualification=failed_record()),
            fault_loader(failure),
        )
        self.events = []

    def close(self):
        self.base.evaluator.DistributedMinimaxDNMPC = self._parent


def select_records(base: Baseline, count: int = 2) -> list[dict]:
    records = base.records()
    level_six = [row for row in records if row.get("level") == 6]
    return (records[:max(1, count - 1)] + level_six[:1])[:count]


def compare_rows(reference: dict, actual: dict) -> dict:
    fields = ("safe_capture_success", "capture_event", "collision", "boundary_violation",
              "timeout", "target_invalid_episode", 'target_boundary_violation', 'target_obstacle_violation', "termination_reason")
    mismatches = {field: [reference[field], actual[field]] for field in fields
                  if reference[field] != actual[field]}
    for field in ("capture_time_seconds", "min_clearance_m"):
        left, right = reference[field], actual[field]
        same = ((left is None and right is None) if left is None or right is None
                else np.isclose(left, right, atol=1e-9, rtol=0))
        if not same:
            mismatches[field] = [left, right]
    return mismatches


def run_boundary_audit(capsule: Path, output: Path, count: int = 2) -> dict:
    if not 1 <= count <= 128 or sha(capsule) != BASELINE_SHA:
        raise ValueError('Pinned capsule and count in [1,128] required')
    output.mkdir(parents=True, exist_ok=False)
    reference_runner = OriginalEntryBoundary(capsule, output / "reference_restored", mode="off")
    records = select_records(reference_runner.base, count)
    modes = {'plain': dict(mode='plain'), "off": dict(mode="off"), "refusal": dict(mode="refusal"),
             "load_failure": dict(mode="shadow", failure='load'),
             'prediction_failure': dict(mode='shadow', failure='predict'),
             'synthetic_shadow': dict(mode='shadow')}
    results = {}
    for name, options in modes.items():
        runner = reference_runner
        runner.configure(**options)
        runner.base.evaluator.DistributedMinimaxDNMPC = runner._parent if name == 'plain' else runner._boundary_class
        rows = []
        for record in records:
            trajectory = output / name / f"level{record['level']}_{record['episode_index']}.npz"
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            commanded, planned = [], []
            def observe(env, observation, actions, sequence):
                commanded.append(actions.copy())
                planned.append(sequence.copy())
            row, _ = runner.base.run(record, trajectory, observe)
            np.savez_compressed(trajectory.with_suffix('.commands.npz'), commanded=np.stack(commanded), planned=np.stack(planned))
            rows.append({"episode_index": record["episode_index"], "trajectory_file": str(trajectory), **row})
        results[name] = {"rows": rows, "events": list(runner.events),
                         "load_calls": runner.adapter.load_calls,
                         "prediction_calls": runner.adapter.prediction_calls,
                         "status": runner.adapter.status}
    reference_runner.close()
    historical = {row['episode_index']: row for row in map(json.loads,
        (reference_runner.base.root / REFERENCE / 'episodes.jsonl').read_text().splitlines())}
    reference_rows = {row["episode_index"]: row for row in results["off"]["rows"]}
    for name, result in results.items():
        trajectory_equal = True
        for row in result["rows"]:
            mismatches = compare_rows(reference_rows[row["episode_index"]], row)
            mismatches.update(compare_rows(historical[row['episode_index']], row))
            if mismatches:
                raise AssertionError(f"{name} changed original outcome: {mismatches}")
            with np.load(reference_rows[row["episode_index"]]["trajectory_file"]) as reference, np.load(row["trajectory_file"]) as actual:
                for field in ("defender_positions", "target_positions"):
                    equal = reference[field].shape == actual[field].shape and np.array_equal(reference[field], actual[field])
                    trajectory_equal &= equal
                    if not equal:
                        raise AssertionError(f"{name} changed original {field} trajectory")
            archived = reference_runner.base.root / REFERENCE / 'trajectories' / Path(row['trajectory_file']).name
            with np.load(archived) as reference, np.load(row['trajectory_file']) as actual:
                for field in ('defender_positions', 'target_positions'):
                    if not np.array_equal(reference[field], actual[field]):
                        raise AssertionError(f'{name} differs from historical capsule {field}')
            plain_file = output / 'plain' / Path(row['trajectory_file']).name
            with np.load(plain_file.with_suffix('.commands.npz')) as plain, np.load(Path(row['trajectory_file']).with_suffix('.commands.npz')) as actual:
                if any(not np.array_equal(plain[key], actual[key]) for key in ('commanded', 'planned')):
                    raise AssertionError(f'{name} changed commanded actions or DN-MPC plans')
        result["trajectory_byte_equal_to_off"] = trajectory_equal
    if results['off']['load_calls'] or results['refusal']['load_calls'] or results['load_failure']['load_calls'] != 1 or results['prediction_failure']['prediction_calls'] != 1:
        raise AssertionError('Fault paths were not exercised as expected')
    if any(info['control_eligible'] for result in results.values() for info in result['events']):
        raise AssertionError('Audit must never qualify a controller')
    verify(reference_runner.base.root, reference_runner.base.capsule_manifest)
    summary = {"status": "passed_original_entry_optional_boundary_fallback_audit",
               "episodes": len(records), "modes": results,
               "all_outcomes_equal_to_off": True, "all_trajectories_byte_equal_to_off": all(
                   result["trajectory_byte_equal_to_off"] for result in results.values()),
               "enhanced_control_enabled": False,
               'historical_capsule_trajectories_equal': True, 'commands_and_plans_equal_to_plain': True,
               'source_and_artifact_integrity_after_replay': True,
               'source_hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in (
                   Path(__file__), Path(__file__).with_name('qualification_gate.py'),
                   ROOT / 'experiments/cwm_v1/baseline.py', ROOT / 'experiments/cwm_v1/freeze_baseline.py',
                   ROOT / 'experiments/cwm_v23/reports/independent_model_validation.json')},
               "baseline_capsule_sha256": BASELINE_SHA,
               "scope": "Post-plan fault-injection harness at the real historical run_episode. Load/inference fixtures have zero history/relative/CV placeholders and are NOT real learned-model public contexts. All fault injection uses noncontrolling shadow, refusal uses failed V23 status. No active local-selector integration, sequential/self-generated support or real learned-model failure coverage is proved. CPU/runtime representative regression only, not 8-level or latency qualification."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = run_boundary_audit(args.capsule, args.output, args.episodes)
    print(json.dumps({key: result[key] for key in ('status', 'episodes', 'commands_and_plans_equal_to_plain',
        'historical_capsule_trajectories_equal', 'enhanced_control_enabled')}), flush=True)
