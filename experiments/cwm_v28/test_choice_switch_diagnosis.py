"""Saved-vector diagnostic fixtures; no real model/control execution claim."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnose_selected_response as diagnosis
from decision_value import rank_metrics


@pytest.fixture
def saved(tmp_path):
    protocol = json.loads((HERE/'training_protocol.json').read_bytes())
    identifier = 'cv_rank_l2_seed994101'
    checkpoint = tmp_path/(identifier+'.pt')
    checkpoint.write_bytes(b'fixture-not-reinferred')
    rows = []
    for i, truth in enumerate(([0., 2.], [0., 1.])):
        costs = {'model': [0., 1.] if i == 0 else [1., 0.],
            'own_motion_component': [1., 0.] if i == 0 else [0., 1.],
            'action_specific_truth': truth}
        rows.append({'episode_index': i, 'step': 1, 'ordinal': i, 'agent': 0, 'group': 'g'+str(i),
            'population': 'bounded_shared', 'indices': [0, 1], 'costs': costs,
            'metrics': {name: rank_metrics(v, truth, 1e-9) for name, v in costs.items()}})
    report = {'status': 'offline_fixed_fresh_ranking_training_finished_pending_two_run_release',
        'protocol': protocol, 'enhanced_control_enabled': False, 'holdout_used': False,
        'primary_research_eligible': False, 'selected_median_seeds': {'cv_rank_l2': 994101},
        'qualification': {'cv_rank_l2': {'selected_seed': 994101, 'selected_gains': {
            'own_motion': {'group_equal_mean': .5}}}},
        'models': [{'configuration': 'cv_rank_l2', 'seed': 994101, 'checkpoint_sha256': diagnosis.digest(checkpoint)}],
        'full_cost_calls': {'train': 2, 'development_validation': 2},
        'split_groups': {'train': ['g0', 'g1'], 'development_validation': ['g0', 'g1']}}
    (tmp_path/'summary.json').write_text(json.dumps(report))
    (tmp_path/(identifier+'_decisions.json')).write_text(json.dumps({
        split: {'bounded_shared': rows} for split in ('train', 'development_validation')}))
    return tmp_path


def test_full_fixed_saved_vector_diagnosis_without_new_gate(saved):
    result = diagnosis.diagnose(saved)
    for split in result['splits'].values():
        assert split['calls'] == 2 and split['groups'] == 2
        assert split['group_equal']['true_original_cost_gain'] == .5
        assert split['group_equal']['beneficial_choice_fraction'] == .5
        assert split['group_equal']['harmful_choice_fraction'] == .5
        assert split['group_equal']['choice_switch_fraction'] == 1.
    assert result['gate_or_model_selected_by_diagnosis'] is False
    assert result['enhanced_control_enabled'] is False


@pytest.mark.parametrize('change', ['missing_call', 'duplicate', 'different_population', 'forged_metric',
    'group', 'selection', 'file_hash', 'fixed_gain', 'holdout', 'enabled'])
def test_no_selection_support_or_saved_cost_relaxation(saved, change):
    summary = saved/'summary.json'
    path = saved/'cv_rank_l2_seed994101_decisions.json'
    report = json.loads(summary.read_bytes()); decisions = json.loads(path.read_bytes())
    rows = decisions['development_validation']['bounded_shared']
    if change == 'missing_call': rows.pop()
    elif change == 'duplicate': rows[1] = rows[0]
    elif change == 'different_population': rows[0]['population'] = 'actual_unique'
    elif change == 'forged_metric': rows[0]['metrics']['model']['choice'] = 1
    elif change == 'group': rows[0]['group'] = 'new'
    elif change == 'selection': report['qualification']['cv_rank_l2']['selected_seed'] = 994102
    elif change == 'file_hash': report['models'][0]['checkpoint_sha256'] = '0'*64
    elif change == 'fixed_gain': report['qualification']['cv_rank_l2']['selected_gains']['own_motion']['group_equal_mean'] = .6
    elif change == 'holdout': report['holdout_used'] = True
    else: report['enhanced_control_enabled'] = True
    summary.write_text(json.dumps(report)); path.write_text(json.dumps(decisions))
    with pytest.raises(ValueError): diagnosis.diagnose(saved)


def test_input_changed_during_diagnosis_is_rejected(saved, monkeypatch):
    actual = diagnosis.validated_choice
    def mutate(call, method):
        path = saved/'cv_rank_l2_seed994101_decisions.json'
        path.write_bytes(path.read_bytes()+b' ')
        return actual(call, method)
    monkeypatch.setattr(diagnosis, 'validated_choice', mutate)
    with pytest.raises(ValueError, match='changed during diagnosis'):
        diagnosis.diagnose(saved)
