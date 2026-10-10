"""Actual original-entry cycle instrumentation on two archived episodes ONLY.

No optional model load, active research, new scenes, holdout, training or latency
qualification. This checks the real evaluator frame/timestamps and preserves all
numeric trajectories/plans/CBF/native non-timing diagnostics on fixed records.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paired_closed_loop import Baseline, CompleteSequentialEntry, capture_sources, published_sources, sha, verify, write_json, json_diagnostics
from measured_original_cycle import MeasuredOriginalCycle
from audit_full_baseline import array_equal, require_equal, STEP_TIMING, EPISODE_TIMING
from freeze_baseline import REFERENCE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(output):
    prereg = published_sources()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preregistration.json', prereg)
    capsule = HERE.parent/'cwm_v1/baseline/capsule.zip'
    base = Baseline(capsule, output/'r')
    ids = json.loads((HERE.parent/'cwm_v29/protocol.json').read_bytes())['fallback_record_ids']
    records = {r['episode_index']: r for r in base.records()}
    historical = {r['episode_index']: r for r in map(json.loads, (base.root/REFERENCE/'episodes.jsonl').read_text().splitlines())}
    entry, measurement = CompleteSequentialEntry(base), MeasuredOriginalCycle(base)
    sources = capture_sources(output)
    results, original_rows, original_steps = {}, {}, {}
    try:
        for mode in ('plain', 'off', 'refusal'):
            rows = []
            folder = output/mode
            folder.mkdir()
            for identifier in ids:
                entry.configure(mode)  # No bundle/certificate/checkpoint loader.
                entry.episode = identifier
                if mode == 'plain':
                    measurement.close()
                else:
                    measurement.install()
                path = folder/f'{identifier}.npz'
                commands, plans = [], []
                def observe(env, observation, action, sequence):
                    commands.append(action.copy()); plans.append(sequence.copy())
                row, steps = base.run(records[identifier], path, observe)
                np.savez_compressed(path.with_suffix('.commands.npz'), commanded=np.stack(commands), planned=np.stack(plans))
                array_equal(base.root/REFERENCE/'trajectories'/f"level{row['level']}_{identifier}.npz", path)
                for field in ('safe_capture_success', 'capture_event', 'collision', 'boundary_violation', 'timeout',
                    'target_invalid_episode', 'termination_reason', 'capture_time_seconds'):
                    if row[field] != historical[identifier][field]:
                        raise ValueError('Original cycle instrumentation changed historical outcome')
                if mode == 'plain':
                    original_rows[identifier], original_steps[identifier] = row, steps
                    times = None
                else:
                    times = measurement.validate(steps)
                    require_equal(json_diagnostics(original_rows[identifier]), row, EPISODE_TIMING, 'Original measured episode')
                    if len(steps) != len(original_steps[identifier]):
                        raise ValueError('Original measurement changed native step support')
                    for a, b in zip(original_steps[identifier], steps):
                        require_equal(json_diagnostics(a), b, STEP_TIMING, 'Original measured step')
                    array_equal(output/'plain'/path.name, path)
                    array_equal(output/'plain'/path.with_suffix('.commands.npz').name, path.with_suffix('.commands.npz'))
                if entry.history or entry.events or entry.pending:
                    raise ValueError('Original measurement read optional context/model')
                evidence = {'episode_index': identifier, 'row': row, 'steps': steps, 'timing_observation': times,
                    'trajectory_sha256': sha(path), 'commands_sha256': sha(path.with_suffix('.commands.npz'))}
                write_json(folder/f'{identifier}.json', evidence)
                rows.append(evidence)
            results[mode] = rows
            print(json.dumps({'mode': mode, 'actual_archived_episodes': len(rows)}), flush=True)
    finally:
        entry.close(); measurement.close()
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT/name) != digest for name, digest in sources.items()):
        raise ValueError('Actual measurement used-source changed')
    write_json(output/'summary.json', {'status': 'actual_original_cycle_measurement_smoke_passed_not_latency_qualification',
        'preregistration': prereg, 'source_hashes': sources, 'record_ids': ids, 'modes': results,
        'historical_full_numeric_trajectories_equal': True, 'plain_off_refusal_plan_cbf_bytes_equal': True,
        'all_native_nontiming_diagnostics_equal': True, 'actual_original_control_started_observed': True,
        'optional_context_or_model_read': False, 'research_actions_exercised': False,
        'enhanced_control_enabled': False, 'holdout_used': False, 'new_model_trained': False,
        'full_population_exercised': False, 'active_model_faults_exercised': False,
        'latency_qualified': False, 'artifact_packaged_and_replayed': False,
        'scope': 'Two fixed archived episodes, each plain/off/refusal. Actual native timestamp observation and full cycle timing with numeric equality; NOT complete8Level, active inference/faults, latency gate, fresh/holdout capture benefit or formal safety proof.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    run(args.output.resolve())
