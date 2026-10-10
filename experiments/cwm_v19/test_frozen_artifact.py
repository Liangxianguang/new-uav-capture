"""Actual artifact recomputation and semantic tamper rejection, not hash-only."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from frozen_training_release import audit, frozen_configs
from geometry_release import arrays


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    torch.set_num_threads(1)
    with zipfile.ZipFile(ROOT/'artifacts/frozen_common_motion_training_20261010.zip') as archive:
        members = {name:archive.read(name) for name in archive.namelist()}
    restored = tmp_path_factory.mktemp('cwm_v19_artifact')/'restored'
    configs = frozen_configs(ROOT.parent/'cwm_v1/baseline/capsule.zip',restored)
    return members,configs,restored


def run(evidence,changed=None):
    members,configs,restored = evidence
    return audit((members if changed is None else changed).__getitem__,configs,restored)


def test_full_model_and_original_cost_recomputation(evidence):
    result = run(evidence)
    assert result['split_calls'] == {'train':1152,'development_validation':384}
    assert result['full_development_calls'] == 271
    assert result['weights_optimizer_rng_history_equal']
    assert result['public252_frame_reencoding_verified']
    assert result['frozen_shared_common_motion_weights_equal']
    assert result['all_original_full_cost_decisions_recomputed']
    assert all(not g['research_eligible'] for g in result['qualification'].values())
    assert not result['enhanced_control_enabled'] and not result['holdout_used']


def test_fake_forecast_cannot_pass_semantic_reloading(evidence):
    changed = dict(evidence[0])
    name = 'primary/raw_response_seed989101_predictions.npz'
    values = arrays(changed[name])
    values['call0_response'][0,0,0] += .01
    stream = io.BytesIO()
    np.savez_compressed(stream,**values)
    changed[name] = stream.getvalue()
    with pytest.raises(ValueError,match='predictions/metrics'):
        run(evidence,changed)


def test_fake_qualification_cannot_promote_failed_model(evidence):
    changed = dict(evidence[0])
    report = json.loads(changed['primary/summary.json'])
    report['qualification']['raw_response']['research_eligible'] = True
    changed['primary/summary.json'] = json.dumps(report).encode()
    with pytest.raises(ValueError,match='gates/fixed ADE'):
        run(evidence,changed)


def test_modified_checkpoint_rng_cannot_fake_exact_retraining(evidence):
    changed = dict(evidence[0])
    name = 'primary/raw_response_seed989101.pt'
    checkpoint = torch.load(io.BytesIO(changed[name]),map_location='cpu',weights_only=True)
    checkpoint['torch_rng_state'][0] ^= 1
    stream = io.BytesIO()
    torch.save(checkpoint,stream)
    changed[name] = stream.getvalue()
    report = json.loads(changed['primary/summary.json'])
    row = next(r for r in report['models'] if r['configuration'] == 'raw_response' and r['seed'] == 989101)
    row['checkpoint_sha256'] = hashlib.sha256(changed[name]).hexdigest()
    changed['primary/summary.json'] = json.dumps(report).encode()
    with pytest.raises(ValueError,match='weights/optimizer/RNG'):
        run(evidence,changed)
