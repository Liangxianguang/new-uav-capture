import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mediated_release import audit, frozen_configs, arrays


@pytest.fixture(scope='module')
def evidence():
    torch.set_num_threads(1)
    with zipfile.ZipFile(ROOT/'artifacts/mediated_response_training_20261010.zip') as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    with zipfile.ZipFile(ROOT.parent/'cwm_v7/artifacts/paired_training_20261010.zip') as archive:
        source = {n: archive.read(n) for n in archive.namelist()}
    restored = ROOT.parent.parent/'results/cwm_v15/test_audit_restored'
    configs = frozen_configs(ROOT.parent/'cwm_v1/baseline/capsule.zip', restored)
    mediator = ROOT.parent/'cwm_v14/artifacts/public_mechanism_training_20261010.zip'
    return members, source, configs, restored, mediator


def run_audit(evidence, changed=None):
    members, source, configs, restored, mediator = evidence
    return audit((members if changed is None else changed).__getitem__, source.__getitem__, configs, restored, mediator)


def test_public_mediator_and_eighteen_core_recomputation(evidence):
    result = run_audit(evidence)
    assert result['train_calls'] == 1152 and result['development_calls'] == 384
    assert result['weights_optimizer_rng_history_equal']
    assert result['public_mediator_estimates_byte_equal']
    assert result['all_learned_full_cost_rankings_recomputed']
    assert not result['enhanced_control_enabled'] and not result['holdout_used']


def test_cached_estimate_cannot_be_substituted_with_private_command(evidence):
    members = evidence[0]
    changed = dict(members)
    values = arrays(changed['primary/public_estimates.npz'])
    values['call0'] = values['call0'].copy()
    values['call0'][0, 0, 0, 0] += .5
    raw = io.BytesIO()
    np.savez_compressed(raw, **values)
    changed['primary/public_estimates.npz'] = raw.getvalue()
    with pytest.raises(ValueError, match='public command estimates'):
        run_audit(evidence, changed)


def test_optimistic_gate_claim_must_be_recomputed(evidence):
    members = evidence[0]
    changed = dict(members)
    report = json.loads(changed['primary/summary.json'])
    gate = report['qualification']['mediated_plain']
    gate['research_eligible'] = not gate['research_eligible']
    changed['primary/summary.json'] = json.dumps(report).encode()
    with pytest.raises(ValueError, match='development gate'):
        run_audit(evidence, changed)
