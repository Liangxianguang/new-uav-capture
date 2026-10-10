"""Recompute all18 models and reject checkpoint/gate/population forgery."""
import copy
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import public_mechanism_release as release


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    release.torch.set_num_threads(1)
    with zipfile.ZipFile(ROOT/'artifacts/public_mechanism_training_20261010.zip') as archive:
        members = {name:archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(ROOT.parent/'cwm_v7/artifacts/paired_training_20261010.zip') as source:
        original = {name:source.read(name) for name in source.namelist()}
    restored = tmp_path_factory.mktemp('v14audit')/'restored'
    configs = release.frozen_configs(ROOT.parent/'cwm_v1/baseline/capsule.zip',restored)
    return members, original, configs, restored


def run_audit(evidence):
    members, original, configs, restored = evidence
    return release.audit(members.__getitem__,original.__getitem__,configs,restored)


def test_all_eighteen_models_reload_and_independent_runs_match(evidence):
    result = run_audit(evidence)
    assert result['models_per_run'] == 9 and result['training_runs'] == 2
    assert result['weights_optimizer_rng_history_equal']
    assert result['enhanced_control_enabled'] is False


def test_qualification_forgery_rejected(evidence):
    members = dict(evidence[0])
    report = json.loads(members['primary/summary.json'])
    gate = report['qualification']['branch_history']
    gate['mechanism_research_eligible'] = not gate['mechanism_research_eligible']
    members['primary/summary.json'] = json.dumps(report).encode()
    with pytest.raises(ValueError,match='gate recomputation'):
        run_audit((members,*evidence[1:]))


def test_checkpoint_forgery_rejected(evidence):
    members = dict(evidence[0])
    members['primary/branch_snapshot_seed976101.pt'] += b'forged'
    with pytest.raises(ValueError,match='checkpoint digest'):
        run_audit((members,*evidence[1:]))


def test_partial_population_rejected(evidence):
    members = dict(evidence[0])
    report = json.loads(members['primary/summary.json'])
    report['models'].pop()
    members['primary/summary.json'] = json.dumps(report).encode()
    with pytest.raises(ValueError,match='Nine-model population'):
        run_audit((members,*evidence[1:]))
