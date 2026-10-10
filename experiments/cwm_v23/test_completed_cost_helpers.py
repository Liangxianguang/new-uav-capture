"""Unit checks of semantic-forgery helpers, not completed scientific audits."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from check_completed_cost_evidence import (npz_bytes, checkpoint_bytes, json_bytes,
    cache_forgery, run_completed_checks)
from geometry_release import arrays


def test_npz_and_torch_roundtrips_preserve_numeric_evidence():
    original = {'call0_label_effect': np.array([0., .2, .3]), 'call0_reference': np.ones((3, 8, 3))}
    rebuilt = arrays(npz_bytes(original))
    assert all(np.array_equal(v, rebuilt[k]) for k, v in original.items())
    value = {'tensor': torch.tensor([3., 4.]), 'seed': 993101}
    rebuilt = torch.load(io.BytesIO(checkpoint_bytes(value)), weights_only=True)
    assert torch.equal(value['tensor'], rebuilt['tensor']) and rebuilt['seed'] == 993101


def test_cost_cache_forgery_updates_all_linked_hashes_but_not_weights():
    protocol = json.loads((ROOT / 'training_protocol.json').read_text(encoding='utf8'))
    seed = protocol['training']['seeds'][0]
    report = {'models': [{'configuration': name, 'seed': seed, 'checkpoint_sha256': 'old'} for name in protocol['models']]}
    dataset = {'primary/summary.json': json_bytes(report),
               f'primary/cv_seed{seed}_deployed_cost_cache.npz': npz_bytes({'call0_label_effect': np.zeros(3)})}
    for name in protocol['models']:
        dataset[f'primary/{name}_seed{seed}.pt'] = checkpoint_bytes({'deployed_cost_cache_sha256': 'old', 'weight': torch.ones(2)})
    changed = cache_forgery(dataset.__getitem__, protocol)
    assert len(changed) == 5  # NPZ, three origin checkpoints, summary
    measured = json.loads(changed['primary/summary.json'])
    for row in measured['models']:
        if row['configuration'].startswith('cv_'):
            assert row['checkpoint_sha256'] != 'old'
            cp = torch.load(io.BytesIO(changed[f"primary/{row['configuration']}_seed{seed}.pt"]), weights_only=True)
            assert cp['deployed_cost_cache_sha256'] != 'old' and torch.equal(cp['weight'], torch.ones(2))
        else:
            assert row['checkpoint_sha256'] == 'old'
    assert arrays(changed[f'primary/cv_seed{seed}_deployed_cost_cache.npz'])['call0_label_effect'][-1] == .01


def test_unfinished_run_cannot_pass_completed_evidence_check(tmp_path):
    roots = {stage: tmp_path / stage for stage in ('data', 'public', 'primary', 'retrained')}
    # Must reject before restoring an engine or suggesting any research success.
    with pytest.raises(ValueError, match='BOTH full18-model'):
        run_completed_checks(roots, tmp_path / 'restored')
