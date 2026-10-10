import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fresh_cost_data import validate_protocol, assigned_split, assign_scenes


def protocol():
    return json.loads((ROOT / 'data_protocol.json').read_text(encoding='utf8'))


def test_new_data_preassigned_disjoint_from_inspected_and_reserved_groups():
    p = protocol()
    validate_protocol(p)
    episodes = set(range(p['seed_start'], p['seed_start'] + 2 * p['groups']))
    layouts = set(range(p['layout_seed_start'], p['layout_seed_start'] + p['groups']))
    assert episodes == set(range(993010, 993074))
    assert layouts == set(range(1993010, 1993042))
    reserved = p['reserved_holdout_not_collected']
    assert not episodes.intersection(range(reserved['seed_start'], reserved['seed_start'] + 2 * reserved['groups']))
    assert not layouts.intersection(range(reserved['layout_seed_start'], reserved['layout_seed_start'] + reserved['groups']))
    for a, b in p['prior_episodes_not_reused']:
        assert not episodes.intersection(range(a, b + 1))
    for a, b in p['prior_layouts_not_reused']:
        assert not layouts.intersection(range(a, b + 1))
    scenes = assign_scenes([{'member': i % 2} for i in range(64)], p)
    assert sum(s['model_split'] == 'train' for s in scenes) == 48
    assert all(scenes[i]['model_split'] == scenes[i + 1]['model_split'] for i in range(0, 64, 2))
    for split in ('train', 'development_validation'):
        assert {i % 4 for i in range(32) if assigned_split(i, p) == split} == {0, 1, 2, 3}


@pytest.mark.parametrize('key,value', [('seed_start', 990010), ('layout_seed_start', 1966010),
    ('holdout_used', True), ('enhanced_control_enabled', True), ('horizon_steps', 80),
    ('train_groups', 28), ('noise_seed', 990001), ('target_branch_rule_overrides', {'speed': 0})])
def test_changed_published_cost_data_contract_rejected(key, value):
    p = protocol()
    p[key] = value
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_fixed_primary_and_matched_two_by_two_protocol():
    p = json.loads((ROOT / 'training_protocol.json').read_text(encoding='utf8'))
    assert p['primary_configuration'] == 'cv_cost_l2'
    assert p['training']['seeds'] == [993101, 993102, 993103]
    assert p['training']['cost_contrast_weight'] == .1
    assert set((v['motion_reference'], v['auxiliary_cost_contrast']) for v in p['response_configurations'].values()) == {
        ('frozen_learned', False), ('frozen_learned', True), ('public_cv', False), ('public_cv', True)}
    assert not any(p[k] for k in ('enhanced_control_enabled', 'original_gru_in_optimizer',
                                 'common_motion_in_response_optimizer', 'private_labels_model_inputs',
                                 'mediator_in_optimizer', 'holdout_used', 'prior_gate_override_allowed'))
    assert p['development_gate']['maximum_median_ade_ratio_vs_cv'] == .95
    assert p['development_gate']['maximum_median_response_error_ratio_vs_zero'] == .95


def test_cost_data_scripts_use_their_own_protocol_not_v21_sources():
    for name in ('fresh_cost_data.py', 'replay_cost_public.py', 'cost_data_audit.py'):
        text = (ROOT / name).read_text(encoding='utf8')
        assert 'cwm_v21' not in text
        assert '990010' not in text
        assert 'fresh_geometry_data' not in text
    assert "from fresh_cost_data import" in (ROOT / 'cost_data_audit.py').read_text(encoding='utf8')


def test_outside_preassigned_population_rejected():
    for i in (-1, 32):
        with pytest.raises(ValueError):
            assigned_split(i, protocol())
