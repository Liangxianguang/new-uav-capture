import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cost_data_release import data_only_summary


def example():
    calls = [{'record': {'split': 'train' if i < 1152 else 'development_validation',
                         'group': 'g' + str(i // 48)},
              'values': {'valid': np.ones((2, 8), dtype=bool), 'anchor_valid': np.ones(8, dtype=bool)}} for i in range(1536)]
    report = {'status': 'fresh_cost_contrast_data_finished_not_promoted', 'training_eligible': True,
              'skipped_calls': 0, 'enhanced_control_enabled': False, 'new_model_trained': False, 'holdout_used': False,
              'episodes': [{'safe_capture_success': True, 'collision': False, 'boundary_violation': False,
                            'timeout': False, 'target_invalid_episode': False} for _ in range(64)]}
    public = {'status': 'independent_original_public_cost_data_replay_frames_equal',
              'enhanced_control_enabled': False, 'new_model_trained': False, 'holdout_used': False,
              'episodes': [{'trajectory_byte_equal': True} for _ in range(64)]}
    return calls, report, public


def test_complete_data_summary_is_never_enhanced_model_performance():
    calls, report, public = example()
    result = data_only_summary(calls, report, public)
    assert result['calls'] == 1536 and result['episodes'] == 64
    assert result['split_calls'] == {'train': 1152, 'development_validation': 384}
    assert len(result['split_groups']['train']) == 24 and len(result['split_groups']['development_validation']) == 8
    assert result['original_controller_outcomes']['safe_capture_success'] == 64
    assert not result['model_training_artifacts_included'] and not result['enhanced_control_enabled']
    assert 'No enhanced-controller performance' in result['scope']


@pytest.mark.parametrize('key', ['enhanced_control_enabled', 'new_model_trained', 'holdout_used'])
def test_non_original_data_contract_rejected(key):
    calls, report, public = example()
    public[key] = True
    with pytest.raises(ValueError, match='DATA status'):
        data_only_summary(calls, report, public)


def test_missing_data_or_failed_gate_cannot_be_published_as_qualified():
    calls, report, public = example()
    with pytest.raises(ValueError, match='DATA population'):
        data_only_summary(calls[:-1], report, public)
    report['training_eligible'] = False
    with pytest.raises(ValueError, match='DATA status'):
        data_only_summary(calls, report, public)
