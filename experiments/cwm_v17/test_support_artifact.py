import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from support_release import audit,frozen_configs


@pytest.fixture(scope='module')
def evidence():
    torch.set_num_threads(1)
    with zipfile.ZipFile(ROOT/'artifacts/actual_local_information_20261010.zip') as archive:
        members = {n:archive.read(n) for n in archive.namelist()}
    restored = ROOT.parent.parent/'results/cwm_v17/test_audit_restored'
    configs = frozen_configs(ROOT.parent/'cwm_v1/baseline/capsule.zip',restored)
    return members,configs,restored


def run(evidence,changed=None):
    members,configs,restored = evidence
    with zipfile.ZipFile(ROOT.parent/'cwm_v15/artifacts/mediated_response_training_20261010.zip') as source, \
         zipfile.ZipFile(ROOT.parent/'cwm_v7/artifacts/paired_training_20261010.zip') as candidate:
        return audit((members if changed is None else changed).__getitem__,source,candidate,configs,restored,
                     ROOT.parent/'cwm_v14/artifacts/public_mechanism_training_20261010.zip',
                     json.loads((ROOT/'protocol.json').read_text()))


def test_actual_candidate_mapping_and_all_original_costs_recompute(evidence):
    result = run(evidence)
    assert result['candidate_mapping_recomputed'] and result['actual_original_solver_costs_reproduced']
    assert result['all_full_cost_rankings_recomputed'] and result['oracle_not_model_input']
    assert result['split_calls'] == {'train':1152,'development_validation':384}
    dev = result['populations']['development_validation']
    assert dev['actual_unique']['response_support']['full_support_calls'] == 272
    assert dev['expanded_union']['response_support']['full_support_calls'] == 259
    assert not result['enhanced_control_enabled']


def test_fake_oracle_gain_is_not_accepted(evidence):
    changed = dict(evidence[0])
    report = json.loads(changed['run/summary.json'])
    gain = report['populations']['development_validation']['actual_unique']['decisions']['comparisons']['original_gru__to__gru_plus_exact_response']['gain']
    gain['group_equal_mean'] = 10.
    changed['run/summary.json'] = json.dumps(report).encode()
    with pytest.raises(ValueError,match='information summaries'):
        run(evidence,changed)


def test_actual_indices_cannot_be_replaced_by_expanded_population(evidence):
    changed = dict(evidence[0])
    rows = json.loads(changed['run/support.json'])
    rows[0]['indices'].append(999)
    changed['run/support.json'] = json.dumps(rows).encode()
    report = json.loads(changed['run/summary.json'])
    report['support_sha256'] = hashlib.sha256(changed['run/support.json']).hexdigest()
    changed['run/summary.json'] = json.dumps(report).encode()
    with pytest.raises(ValueError,match='support recomputation'):
        run(evidence,changed)
