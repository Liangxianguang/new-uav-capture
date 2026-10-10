import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from support_release import validate_report


def report():
    p = json.loads((ROOT/'protocol.json').read_text())
    row = {'status':'actual_local_information_diagnostic_finished_not_promoted',
           'protocol':copy.deepcopy(p),'source_sha256':p['source_archive_sha256'],
           'enhanced_control_enabled':False,'new_model_trained':False,
           'holdout_used':False,'prior_gate_overridden':False}
    return row,p


@pytest.mark.parametrize('key',['enhanced_control_enabled','new_model_trained','holdout_used','prior_gate_overridden'])
def test_reanalysis_report_cannot_promote_failed_model(key):
    row,p = report()
    validate_report(row,p)
    row[key] = True
    with pytest.raises(ValueError,match='Diagnostic-only'):
        validate_report(row,p)


def test_reanalysis_source_cannot_be_swapped():
    row,p = report()
    row['source_sha256'] = '0'*64
    with pytest.raises(ValueError,match='V15 diagnostic source'):
        validate_report(row,p)


def test_observed_response_cannot_change_fixed_population_contract():
    row,p = report()
    row['protocol']['candidate_populations'] = ['expanded_union']
    with pytest.raises(ValueError,match='Diagnostic-only'):
        validate_report(row,p)
