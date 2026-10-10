import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mediated_release import validate_report


def valid_report():
    protocol = json.loads((ROOT/'training_protocol.json').read_text())
    report = {'protocol': protocol, 'status': 'offline_mediated_response_finished_not_promoted',
              'mediator_unchanged': True, 'models': [
                  {'configuration': name, 'seed': seed}
                  for name in protocol['models'] for seed in protocol['training']['seeds']]}
    report.update({key: False for key in ('baseline_weights_included', 'enhanced_control_enabled',
                                         'holdout_used', 'prior_gate_overridden', 'mediator_optimized',
                                         'private_labels_model_inputs')})
    return report, protocol


@pytest.mark.parametrize('key', ['baseline_weights_included', 'enhanced_control_enabled',
    'holdout_used', 'prior_gate_overridden', 'mediator_optimized', 'private_labels_model_inputs'])
def test_offline_contract_rejects_unauthorized_promotion(key):
    report, protocol = valid_report()
    validate_report(report, protocol)
    report[key] = True
    with pytest.raises(ValueError, match='offline contract'):
        validate_report(report, protocol)


def test_failed_seed_cannot_be_silently_removed_or_duplicated():
    report, protocol = valid_report()
    report['models'][-1] = copy.deepcopy(report['models'][0])
    with pytest.raises(ValueError, match='nine-core population'):
        validate_report(report, protocol)


def test_protocol_cannot_be_relaxed_after_results():
    report, protocol = valid_report()
    report = copy.deepcopy(report)
    report['protocol']['development_gate']['maximum_median_response_error_ratio_vs_zero'] = 2.
    with pytest.raises(ValueError, match='protocol/status'):
        validate_report(report, protocol)
