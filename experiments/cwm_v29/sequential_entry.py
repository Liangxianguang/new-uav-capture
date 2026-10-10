"""Research-only sequential selector at the unchanged original planner entry."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / 'cwm_v1'), str(HERE.parent / 'cwm_v25'),
               str(HERE.parent / 'cwm_v9')]
from local_selector import select_research
from research_bundle import ResearchBundle
from baseline import Baseline
from freeze_baseline import sha, verify, REFERENCE
from original_entry_boundary import BASELINE_SHA, compare_rows
from local_shadow import public_call_snapshot


class SequentialResearchEntry:
    def __init__(self, base):
        self.base = base
        self.parent_planner = base.evaluator.DistributedMinimaxDNMPC
        self.parent_runtime = base.evaluator.PredictionRuntime
        self.history, self.events, self.pending = [], [], []
        owner = self
        class Runtime(self.parent_runtime):
            def _append_current_frame(self, observation):
                super()._append_current_frame(observation)
                owner.history.append(self.history[-1].reshape(-1).copy())
                owner.history = owner.history[-8:]
        class Planner(self.parent_planner):
            def plan(self, observation, scenarios, **kwargs):
                owner.ordinal = 0
                owner.step = int(kwargs.get('step_index', 0))
                owner.pending = []
                plan = super().plan(observation, scenarios, **kwargs)
                owner.flush()  # File IO outside original timed planner, logged separately by later evaluation.
                return plan
            def _select_local_sequence(self, observation, scenarios, agent_id, local_candidates, known, peer_sequences):
                selected, costs = super()._select_local_sequence(observation, scenarios, agent_id, local_candidates, known, peer_sequences)
                ordinal = owner.ordinal
                owner.ordinal += 1
                if owner.bundle.authorized is not True:
                    decision = select_research({'selected': selected, 'selected_costs': costs},
                        bundle=owner.bundle, mode=owner.mode)
                    owner.pending.append((ordinal, {'agent': int(agent_id)}, decision))
                    return selected, costs
                try:
                    call = {'observation': copy.deepcopy(observation), 'scenarios': copy.deepcopy(scenarios),
                        'planner': copy.deepcopy(self), 'agent': int(agent_id),
                        'local_candidates': np.stack(local_candidates).copy(), 'selected': selected.copy(),
                        'selected_costs': costs.copy(), 'known': copy.deepcopy(known), 'peer_sequences': copy.deepcopy(peer_sequences)}
                    history = (np.stack([owner.history[0]]*max(0, 8-len(owner.history))+owner.history[-8:])
                               if owner.history else None)
                    decision = select_research(call, history, owner.base.evaluator.belief_reference,
                        owner.bundle, mode=owner.mode, score=owner.score)
                except Exception as error:
                    owner.bundle.authorized, owner.bundle.status = False, 'public_capture_failed'
                    call = {'agent': int(agent_id)}
                    decision = {'selected': selected.copy(), 'selected_costs': costs.copy(), 'evidence': None,
                        'info': {'status': 'public_capture_failed', 'error_type': type(error).__name__,
                                 'research_action_selected': False, 'deployment_control_eligible': False}}
                # All best-response calls use NEWLY received peer plans; no reuse
                # of proposals/forecast/scoring across calls or agents.
                owner.pending.append((ordinal, call, decision))
                return decision['selected'], decision['selected_costs']
        self.runtime_class, self.planner_class = Runtime, Planner

    def configure(self, mode='off', *, bundle=None, score='cv_rank_l2', output=None):
        if mode not in ('plain', 'off', 'refusal', 'research_eval', 'shadow'):
            raise ValueError('Unknown research entry mode')
        self.mode, self.bundle, self.score, self.output = mode, bundle, score, output
        self.history, self.events, self.pending = [], [], []
        active = (mode in ('research_eval', 'shadow') and isinstance(bundle, ResearchBundle) and
                  bundle.authorized is True and bundle.deployment_control_eligible is False)
        # Off/refusal uses EXACT parent classes: no model/context/history IO and
        # no timed hook overhead, not merely a zero-valued learned adjustment.
        self.base.evaluator.DistributedMinimaxDNMPC = self.planner_class if active else self.parent_planner
        self.base.evaluator.PredictionRuntime = self.runtime_class if active else self.parent_runtime
        self.status = 'research_ready_not_deployed' if active else 'off' if mode in ('plain', 'off') else 'research_refused'

    def flush(self):
        for ordinal, call, decision in self.pending:
            identity = {'episode_index': self.episode, 'step': self.step, 'ordinal': ordinal, 'agent': call['agent']}
            event = {**identity, **decision['info']}
            if decision['evidence'] is not None and self.output is not None:
                path = self.output / 'calls' / f"{self.episode}_{self.step}_{ordinal}_{call['agent']}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **decision['evidence'])
                context = path.with_suffix('.json')
                context.write_text(json.dumps(public_call_snapshot(call)), encoding='utf8')
                event.update(arrays_path=path.relative_to(self.output).as_posix(), arrays_sha256=sha(path),
                    context_path=context.relative_to(self.output).as_posix(), context_sha256=sha(context))
            self.events.append(event)
        self.pending = []

    def close(self):
        self.base.evaluator.DistributedMinimaxDNMPC = self.parent_planner
        self.base.evaluator.PredictionRuntime = self.parent_runtime


def fallback_regression(output):
    """Two actual fixed historical records; NO active or mock qualified model."""
    protocol = json.loads((HERE / 'protocol.json').read_bytes())
    capsule = HERE.parent / 'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA or protocol['baseline_capsule_sha256'] != BASELINE_SHA:
        raise ValueError('Pinned original capsule required')
    output.mkdir(parents=True, exist_ok=False)
    base = Baseline(capsule, output / 'restored')
    entries = {r['episode_index']: r for r in base.records()}
    historical = {r['episode_index']: r for r in map(json.loads, (base.root / REFERENCE / 'episodes.jsonl').read_text().splitlines())}
    entry, modes = SequentialResearchEntry(base), {}
    try:
        for mode in ('plain', 'off', 'refusal'):
            entry.configure(mode)  # No artifact/certificate/model loader is created.
            rows = []
            for identifier in protocol['fallback_record_ids']:
                entry.episode = identifier
                record = entries[identifier]
                path = output / mode / f"level{record['level']}_{identifier}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                commands, plans = [], []
                def observe(env, observation, action, sequence):
                    commands.append(action.copy()); plans.append(sequence.copy())
                row, _ = base.run(record, path, observe)
                np.savez_compressed(path.with_suffix('.commands.npz'), commanded=np.stack(commands), planned=np.stack(plans))
                if compare_rows(historical[identifier], row):
                    raise ValueError('Default-off/refusal changed original historical outcome')
                for reference in (base.root / REFERENCE / 'trajectories' / path.name, output / 'plain' / path.name):
                    with np.load(reference) as a, np.load(path) as b:
                        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                            raise ValueError('Default-off/refusal changed original full trajectory')
                with np.load(output / 'plain' / path.with_suffix('.commands.npz').name) as a, np.load(path.with_suffix('.commands.npz')) as b:
                    if any(not np.array_equal(a[k], b[k]) for k in ('commanded', 'planned')):
                        raise ValueError('Default-off/refusal changed original plan/CBF commands')
                fields = ('episode_index', 'level', 'safe_capture_success', 'collision', 'boundary_violation',
                          'timeout', 'target_invalid_episode', 'termination_reason')
                rows.append({k: row[k] for k in fields})
            if entry.history or entry.events or entry.pending:
                raise ValueError('Off/refusal inspected/captured optional context')
            modes[mode] = {'rows': rows, 'original_parent_classes_used': True, 'optional_context_or_model_read': False,
                           'historical_full_trajectory_equal': True, 'plain_plan_and_cbf_commands_equal': True}
            print(json.dumps({'mode': mode, 'real_historical_episodes': len(rows)}), flush=True)
    finally:
        entry.close()
    verify(base.root, base.capsule_manifest)
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)
               and Path(m.__file__).resolve().is_relative_to(ROOT / 'experiments')}
    sources |= {HERE / 'protocol.json', HERE / 'sequential_entry.py'}
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        path = output / 'source' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        hashes[relative.as_posix()] = sha(source)
    report = {'status': 'original_sequential_entry_off_refusal_regression_passed', 'protocol': protocol,
        'source_hashes': hashes, 'modes': modes, 'enhanced_control_enabled': False, 'holdout_used': False,
        'new_model_trained': False, 'active_research_selector_exercised': False,
        'actual_v28_checkpoint_load_failure_exercised': False, 'actual_v28_forward_failure_exercised': False,
        'scope': 'Two fixed original historical records; no full-Level, active selector, failure-timing, capture-gain or latency qualification.'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fallback-regression', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    print(json.dumps({'status': fallback_regression(args.output)['status']}), flush=True)
