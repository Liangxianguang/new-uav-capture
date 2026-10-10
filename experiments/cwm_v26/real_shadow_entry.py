"""Real V23 checkpoint shadow at every supported historical sequential call."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / 'cwm_v25'), str(HERE.parent / 'cwm_v9'),
               str(HERE.parent / 'cwm_v1')]
from baseline import Baseline
from freeze_baseline import REFERENCE, sha, verify
from qualification_gate import OptionalResponseAdapter, QualificationGate
from public_provider import delayed_joint_context
from checkpoint_provider import PinnedV23Loader, ARCHIVE_SHA
from original_entry_boundary import compare_rows, BASELINE_SHA


def build_public_inputs(call, history, belief_reference):
    observation, config = call['observation'], call['planner'].config
    reference, velocity = belief_reference(observation, config)
    context = delayed_joint_context(observation, call['known'], call['peer_sequences'], call['agent'],
        call['local_candidates'], call['selected'], reference, velocity)
    if context is None:
        return None
    proposed, anchor, relative = context
    backbone = np.asarray(call['scenarios'].trajectories[0], dtype=np.float64)
    cv = reference[None] + .1 * np.arange(1, 9)[:, None] * velocity[None]
    return tuple(x.copy() for x in (history, relative, proposed, anchor, backbone, cv))


class SequentialShadow:
    """Copy public local contexts, then forecast only after original plan."""
    def __init__(self, base):
        self.base = base
        self.parent_planner = base.evaluator.DistributedMinimaxDNMPC
        self.parent_runtime = base.evaluator.PredictionRuntime
        self.history, self.events, self.snapshots = [], [], []
        owner = self
        parent_planner, parent_runtime = self.parent_planner, self.parent_runtime
        class Runtime(parent_runtime):
            def _append_current_frame(self, observation):
                super()._append_current_frame(observation)
                owner.history.append(self.history[-1].reshape(-1).copy())
                owner.history = owner.history[-8:]
        class Planner(parent_planner):
            def plan(self, observation, scenarios, **kwargs):
                owner.snapshots = []
                owner.step = int(kwargs.get('step_index', 0))
                plan = super().plan(observation, scenarios, **kwargs)
                # No checkpoint IO, model forward or public score replay in the
                # original local timeout window. Snapshot copying IS timed.
                owner.after_plan()
                return plan
            def _select_local_sequence(self, observation, scenarios, agent_id, local_candidates, known, peer_sequences):
                selected, costs = super()._select_local_sequence(observation, scenarios, agent_id, local_candidates, known, peer_sequences)
                if owner.mode not in ('plain', 'off', 'refusal'):
                    # All invocations, including subsequent best-response rounds.
                    from types import SimpleNamespace
                    owner.snapshots.append({'observation': copy.deepcopy(observation),
                        'scenarios': copy.deepcopy(scenarios), 'agent': int(agent_id),
                        'ordinal': len(owner.snapshots), 'planner': SimpleNamespace(config=self.config),
                        'local_candidates': np.stack(local_candidates).copy(),
                        'known': copy.deepcopy(known), 'peer_sequences': copy.deepcopy(peer_sequences),
                        'selected': selected.copy(), 'selected_costs': costs.copy()})
                return selected, costs
        self.planner_class, self.runtime_class = Planner, Runtime

    def configure(self, mode, output, artifact):
        self.mode, self.output = mode, output
        self.history, self.events = [], []
        fault = {'load_failure': 'load', 'prediction_failure': 'predict'}.get(mode)
        self.loader = PinnedV23Loader(artifact, output.parent / 'assembled', fault=fault)
        gate_mode = 'off' if mode in ('plain', 'off') else 'guarded' if mode == 'refusal' else 'shadow'
        # Frozen V23 primary is failed. No online booleans can override it.
        self.adapter = OptionalResponseAdapter(QualificationGate(mode=gate_mode,
            qualification={'research_eligible': False}), self.loader)
        self.base.evaluator.DistributedMinimaxDNMPC = self.parent_planner if mode == 'plain' else self.planner_class
        self.base.evaluator.PredictionRuntime = self.parent_runtime if mode in ('plain', 'off', 'refusal') else self.runtime_class

    def after_plan(self):
        if self.mode in ('off', 'refusal'):
            result, info = self.adapter.forecast(*([object()] * 6))
            if result is not None:
                raise AssertionError('Off/refusal must not forecast')
            self.events.append({'episode_index': self.episode, 'step': self.step, **info})
            return
        if not self.history:
            raise AssertionError('Actual original public history was not captured')
        history = np.stack([self.history[0]] * max(0, 8-len(self.history)) + self.history[-8:])
        from local_shadow import public_call_snapshot
        for call in self.snapshots:
            identity = {'episode_index': self.episode, 'step': self.step, 'agent': call['agent'], 'ordinal': call['ordinal']}
            values = build_public_inputs(call, history, self.base.evaluator.belief_reference)
            if values is None:
                self.events.append({**identity, 'status': 'missing_received_peer_context',
                                    'forecast_available': False, 'control_eligible': False})
                continue
            started = time.perf_counter()
            result, info = self.adapter.forecast(*values)
            elapsed = (time.perf_counter()-started)*1000
            path = self.output / 'calls' / f"{self.episode}_{self.step}_{call['ordinal']}_{call['agent']}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            saved = dict(zip(('history', 'relative', 'proposed', 'anchor', 'backbone', 'cv'), values))
            if result is not None:
                saved.update(result)
            np.savez_compressed(path, **saved)
            context = path.with_suffix('.json')
            context.write_text(json.dumps(public_call_snapshot(call)), encoding='utf8')
            self.events.append({**identity, **info, 'status': info['reason'], 'latency_ms': elapsed,
                'arrays_path': path.relative_to(self.output).as_posix(), 'arrays_sha256': sha(path),
                'context_path': context.relative_to(self.output).as_posix(), 'context_sha256': sha(context),
                'raw_candidate_count': len(values[2]), 'known_sent_steps': {str(k): int(v.sent_step) for k,v in call['known'].items()}})

    def close(self):
        self.base.evaluator.DistributedMinimaxDNMPC = self.parent_planner
        self.base.evaluator.PredictionRuntime = self.parent_runtime


def run(artifact, output):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    if any(protocol[k] for k in ('enhanced_control_enabled', 'new_model_trained', 'holdout_used')):
        raise ValueError('V26 cannot train, control or access holdout')
    capsule = HERE.parent / 'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA or protocol['model_archive_sha256'] != ARCHIVE_SHA:
        raise ValueError('Original/model capsule differs')
    output.mkdir(parents=True, exist_ok=False)
    base = Baseline(capsule, output / 'restored')
    records = {row['episode_index']: row for row in base.records()}
    records = [records[key] for key in protocol['record_ids']]
    historical = {row['episode_index']: row for row in map(json.loads,
        (base.root / REFERENCE / 'episodes.jsonl').read_text().splitlines())}
    shadow = SequentialShadow(base)
    results = {}
    try:
        for mode in protocol['modes']:
            destination = output / mode
            shadow.configure(mode, destination, artifact)
            rows = []
            for record in records:
                shadow.history = []
                shadow.episode = record['episode_index']
                path = destination / f"level{record['level']}_{shadow.episode}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                commands, plans = [], []
                def observe(env, observation, actions, sequence):
                    commands.append(actions.copy()); plans.append(sequence.copy())
                row, _ = base.run(record, path, observe)
                np.savez_compressed(path.with_suffix('.commands.npz'), commanded=np.stack(commands), planned=np.stack(plans))
                if compare_rows(historical[shadow.episode], row):
                    raise AssertionError('Shadow changed historical physical outcomes')
                with np.load(base.root / REFERENCE / 'trajectories' / path.name) as archived, np.load(path) as actual:
                    if any(not np.array_equal(archived[k],actual[k]) for k in ('defender_positions','target_positions')):
                        raise AssertionError('Shadow changed historical trajectory')
                plain_file = output / 'plain' / path.with_suffix('.commands.npz').name
                with np.load(plain_file) as plain, np.load(path.with_suffix('.commands.npz')) as actual:
                    if any(not np.array_equal(plain[k], actual[k]) for k in ('commanded','planned')):
                        raise AssertionError('Shadow changed DN-MPC plan or CBF command')
                fields = ('episode_index','level','safe_capture_success','collision','boundary_violation','timeout',
                    'capture_event','capture_time_seconds','min_clearance_m','target_invalid_episode',
                    'target_boundary_violation','target_obstacle_violation','termination_reason')
                rows.append({k:row[k] for k in fields})
            events = shadow.events
            supported = [e for e in events if 'arrays_path' in e]
            if any(e['control_eligible'] for e in events):
                raise AssertionError('Failed V23 cannot control')
            if mode in ('off','refusal') and (shadow.adapter.load_calls or shadow.adapter.prediction_calls):
                raise AssertionError('Off/refusal read a model')
            if mode == 'real_shadow' and (not supported or not all(e['forecast_available'] for e in supported)):
                raise AssertionError('Real model did not forecast all supported sequential calls')
            expected = {'load_failure': (1,0,'load_failed'), 'prediction_failure': (1,1,'prediction_failed')}
            if mode in expected and (shadow.adapter.load_calls, shadow.adapter.prediction_calls, shadow.adapter.status) != expected[mode]:
                raise AssertionError('Actual-checkpoint fault path not exercised')
            results[mode] = {'rows':rows, 'events':events, 'supported_calls':len(supported),
                'forecast_calls':sum(e['forecast_available'] for e in supported),
                'missing_peer_calls':sum(e.get('status') == 'missing_received_peer_context' for e in events),
                'load_calls':shadow.adapter.load_calls,'prediction_calls':shadow.adapter.prediction_calls,
                'checkpoint_sha256':shadow.loader.checkpoint_sha256, 'status':shadow.adapter.status,
                'historical_trajectory_equal':True,'plain_commands_and_plans_equal':True}
            print(json.dumps({'mode':mode,'supported':len(supported),'forecast':results[mode]['forecast_calls']}),flush=True)
    finally:
        shadow.close()
    verify(base.root, base.capsule_manifest)
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and Path(m.__file__).resolve().is_relative_to(ROOT / 'experiments')}
    sources |= {HERE/'protocol.json', HERE/'real_shadow_entry.py', HERE/'checkpoint_provider.py'}
    hashes = {}
    for source in sorted(sources):
        relative = source.relative_to(ROOT)
        path = output / 'source' / relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(source.read_bytes())
        hashes[relative.as_posix()] = sha(source)
    summary = {'status':'real_checkpoint_sequential_shadow_finished_not_control_qualification',
        'protocol':protocol,'source_hashes':hashes,'modes':results,'historical_capsule_integrity_after_replay':True,
        'enhanced_control_enabled':False,'holdout_used':False,'new_model_trained':False,
        'scope':'Every supported sequential local call forecast after original plan. Copy overhead is inside original timeout; latency not qualified. No self-generated candidate or active selector integration. Failed model remains failed.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=HERE.parent/'cwm_v23/artifacts/deployed_cost_contrast_20261010.parts.json')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps({'status':run(args.artifact,args.output)['status']}),flush=True)
