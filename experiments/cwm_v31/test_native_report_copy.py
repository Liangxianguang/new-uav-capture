"""Regression for publication-only JSON type drift; no scientific claims."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_native_report_copy import verify, digest


@pytest.fixture
def report(tmp_path):
    native = tmp_path/'native.json'; wrapper = tmp_path/'wrapper.json'
    payload = {'fraction': 0.0, 'gain': -0.0, 'enabled': False, 'calls': 2304,
        'groups': ['g0', 'g1'], 'scope': 'Synthetic fixture, not model results'}
    native.write_text(json.dumps(payload, indent=2), encoding='utf8')
    wrapped = {'native_summary_sha256': digest(native), 'native_summary': payload,
        'native_payload_semantic_copy_not_same_file_bytes': True}
    wrapper.write_text(json.dumps(wrapped, indent=2), encoding='utf8')
    return wrapper, native


def test_complete_typed_copy_preserves_native_and_wrapper(report):
    wrapper, native = report
    before = (wrapper.read_bytes(), native.read_bytes())
    result = verify(wrapper, native)
    assert result['all_typed_native_fields_equal'] is True
    assert result['native_reexecuted_by_this_checker'] is False
    assert before == (wrapper.read_bytes(), native.read_bytes())


@pytest.mark.parametrize('change', ['float_zero_to_int', 'signed_zero', 'boolean_to_int',
    'int_count_to_float', 'gain', 'missing_field', 'extra_field', 'support', 'hash', 'copy_marker'])
def test_type_value_support_and_native_identity_drift_rejected(report, change):
    wrapper, native = report
    w = json.loads(wrapper.read_bytes()); p = w['native_summary']
    if change == 'float_zero_to_int': p['fraction'] = 0
    elif change == 'signed_zero': p['gain'] = 0.0
    elif change == 'boolean_to_int': p['enabled'] = 0
    elif change == 'int_count_to_float': p['calls'] = 2304.0
    elif change == 'gain': p['gain'] = .1
    elif change == 'missing_field': p.pop('scope')
    elif change == 'extra_field': p['new'] = True
    elif change == 'support': p['groups'].pop()
    elif change == 'hash': w['native_summary_sha256'] = '0'*64
    else: w['native_payload_semantic_copy_not_same_file_bytes'] = 1
    wrapper.write_text(json.dumps(w), encoding='utf8')
    with pytest.raises(ValueError): verify(wrapper, native)


def test_published_v31_zero_fields_remain_native_float_zeros():
    report = json.loads((Path(__file__).resolve().parent/'reports/completed_native_input_audit_20261011.json').read_bytes())
    native = report['native_summary']
    assert type(native['input_tables']['response_core']['single_observation_point_fraction']) is float
    assert type(native['input_tables']['whole_optional_model']['single_observation_point_fraction']) is float
    assert type(native['float32_anchor_collapse']['maximum_response_m']) is float
