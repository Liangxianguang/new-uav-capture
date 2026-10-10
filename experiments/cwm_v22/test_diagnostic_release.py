import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from diagnostic_release import evidence_names, check_record, verify_manifest
from decision_diagnostics import repair_contributions


def protocol():
    return json.loads((ROOT / 'diagnostic_protocol.json').read_text(encoding='utf8'))


def record():
    costs = {name: [8., 9.] for name in protocol()['paths']}
    costs['action_specific_truth'] = [6., 5.]
    costs['motion_plus_exact_response'] = [8., 7.]
    metrics = {}
    for name, cost in costs.items():
        choice = int(np.argmin(cost))
        truth = costs['action_specific_truth']
        regret = truth[choice] - min(truth)
        metrics[name] = {'choice': choice, 'regret_in_diagnostic_cost': regret,
                         'realized_cost_at_choice': truth[choice], 'realized_tie_optimal': regret <= 1e-9}
    return {'costs': costs, 'metrics': metrics, 'diagnostics': {'repairs': repair_contributions(1., 1., 0., 0.)}}


def test_all_sixty_record_sets_and_summary_required():
    names = evidence_names(protocol())
    assert len(names) == len(set(names)) == 61
    assert 'raw_point_l2_seed990103_development_validation_actual_unique_decisions.json' in names
    assert 'raw_call_mse_seed990101_train_geometry_union_decisions.json' in names


def test_saved_costs_metrics_and_repairs_consistent():
    check_record(record(), protocol())


@pytest.mark.parametrize('field', ['choice', 'tie', 'regret', 'cost_at_choice', 'repair', 'nonfinite', 'missing_path'])
def test_forged_record_semantics_rejected(field):
    row = copy.deepcopy(record())
    if field == 'choice':
        row['metrics']['model']['choice'] = 1
    elif field == 'tie':
        row['metrics']['model']['realized_tie_optimal'] = True
    elif field == 'regret':
        row['metrics']['model']['regret_in_diagnostic_cost'] = 0.
    elif field == 'cost_at_choice':
        row['metrics']['model']['realized_cost_at_choice'] += 1
    elif field == 'repair':
        row['diagnostics']['repairs']['motion_repair'] += 1
    elif field == 'nonfinite':
        row['costs']['model'][0] = float('nan')
    else:
        row['costs'].pop('own_motion')
    with pytest.raises(ValueError):
        check_record(row, protocol())


def test_changed_member_rejected_even_with_same_archive_layout(tmp_path):
    artifact = tmp_path / 'archive.zip'
    with zipfile.ZipFile(artifact, 'x') as archive:
        archive.writestr('primary/summary.json', b'forged')
        archive.writestr('ARTIFACT_MANIFEST.json', json.dumps({'primary/summary.json': hashlib.sha256(b'original').hexdigest()}))
    with pytest.raises(ValueError, match='Changed artifact'):
        verify_manifest(artifact)


def test_manifest_must_cover_all_members(tmp_path):
    artifact = tmp_path / 'archive.zip'
    with zipfile.ZipFile(artifact, 'x') as archive:
        archive.writestr('primary/summary.json', b'original')
        archive.writestr('ARTIFACT_MANIFEST.json', '{}')
    with pytest.raises(ValueError, match='Incomplete artifact'):
        verify_manifest(artifact)
