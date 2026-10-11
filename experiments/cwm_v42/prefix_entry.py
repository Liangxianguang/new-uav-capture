"""Distinct benchmark entry; old V29 bundle/selector remain unchanged."""
import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v29')]
from prefix_selector import select_prefix, ready, revoke, SCORES
from sequential_entry import SequentialResearchEntry


class PrefixSequentialEntry(SequentialResearchEntry):
    """Reuse evidence serialization/close only; own frozen planner/runtime hook.

    Every best-response call rebuilds proposals from its newly received peers.
    IO stays after parent timed plan, and within actual end-to-end cycle timing.
    """
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
                owner.ordinal = 0; owner.step = int(kwargs.get('step_index', 0)); owner.pending = []
                result = super().plan(observation, scenarios, **kwargs)
                owner.flush()
                return result

            def _select_local_sequence(self, observation, scenarios, agent_id, local_candidates, known, peer_sequences):
                selected, costs = super()._select_local_sequence(observation, scenarios, agent_id,
                    local_candidates, known, peer_sequences)
                ordinal = owner.ordinal; owner.ordinal += 1
                if not ready(owner.bundle):
                    decision = select_prefix({'selected': selected, 'selected_costs': costs},
                        bundle=owner.bundle, mode=owner.mode)
                    owner.pending.append((ordinal, {'agent': int(agent_id), 'ordinal': ordinal}, decision))
                    return selected, costs
                try:
                    call = {'observation': copy.deepcopy(observation), 'scenarios': copy.deepcopy(scenarios),
                        'planner': copy.deepcopy(self), 'agent': int(agent_id), 'ordinal': ordinal,
                        'local_candidates': np.stack(local_candidates).copy(), 'selected': selected.copy(),
                        'selected_costs': costs.copy(), 'known': copy.deepcopy(known),
                        'peer_sequences': copy.deepcopy(peer_sequences)}
                    history = (np.stack([owner.history[0]]*max(0, 8-len(owner.history))+owner.history[-8:])
                               if owner.history else None)
                    decision = select_prefix(call, history, owner.base.evaluator.belief_reference,
                        owner.bundle, mode=owner.mode, score=owner.score)
                except Exception as error:
                    revoke(owner.bundle, 'public_capture_failed')
                    call = {'agent': int(agent_id), 'ordinal': ordinal}
                    decision = {'selected': selected.copy(), 'selected_costs': costs.copy(), 'evidence': None,
                        'info': {'status': 'public_capture_failed', 'error_type': type(error).__name__,
                            'research_action_selected': False, 'deployment_control_eligible': False}}
                owner.pending.append((ordinal, call, decision))
                return decision['selected'], decision['selected_costs']

        self.runtime_class, self.planner_class = Runtime, Planner

    def configure(self, mode='off', *, bundle=None, score='cv_prefix_rank_l2', output=None):
        if mode not in ('plain', 'off', 'refusal', 'research_eval', 'shadow'):
            raise ValueError('No deployment or holdout entry mode')
        active = mode in ('research_eval', 'shadow') and ready(bundle)
        if active and score not in SCORES:
            raise ValueError('Fixed prefix/L2/public scoring controls required')
        self.mode, self.bundle, self.score, self.output = mode, bundle, score, output
        self.history, self.events, self.pending = [], [], []
        self.base.evaluator.DistributedMinimaxDNMPC = self.planner_class if active else self.parent_planner
        self.base.evaluator.PredictionRuntime = self.runtime_class if active else self.parent_runtime
        self.status = 'research_ready_not_deployed' if active else 'off' if mode in ('plain', 'off') else 'research_refused'
