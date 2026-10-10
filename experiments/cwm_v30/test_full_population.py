import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from closed_loop_populations import original_population, decode_original_population, validate_protocol, restricted_safe_capture_time
from baseline_full_levels import summarize, json_diagnostics


def test_actual_pinned_original_corpus_all_records_levels_variants_and_existing_singletons():
    protocol = json.loads((HERE/'closed_loop_protocol.json').read_bytes())
    records = original_population(ROOT/'experiments/cwm_v1/baseline/capsule.zip', protocol)
    assert len(records) == 1280
    assert Counter(r['level'] for r in records) == Counter({i: 256 if i == 7 else 128 for i in range(9)})
    assert len({r['variant'] for r in records}) == 10
    assert len({r['episode_index'] for r in records}) == 1280
    counts = Counter(r['mirror_group_id'] for r in records if r['level'] == 5)
    assert any(count == 1 for count in counts.values())
    assert {r['variant'] for r in records if r['level'] == 7} == {'fast_target', 'agile_target'}
    assert not {r['episode_index'] for r in records}.intersection(range(966010, 966026))


@pytest.mark.parametrize('key,value', [('max_steps', 500), ('maximum_command_speed_mps', 6),
    ('holdout_used', True), ('active_research_experiment_started', True), ('prior_gate_override_allowed', True)])
def test_original_contract_or_holdout_changes_refused(key, value):
    protocol = json.loads((HERE/'closed_loop_protocol.json').read_bytes())
    protocol[key] = value
    with pytest.raises(ValueError):
        validate_protocol(protocol)


def fixture_records():
    curriculum = {'curriculum': {'variants_by_level': [{'level': i, 'variants': [f'v{i}']} for i in range(9)]}}
    records = [{'episode_index': 200+i, 'episode_seed': 300+i, 'layout_seed': 400+i,
        'variant': f'v{i}', 'scene_block': 'development', 'mirror_group_id': f'g{i}', 'mirror_pair_member': 'upper'} for i in range(9)]
    def values(records=records, curriculum=curriculum):
        raw = ''.join(json.dumps(r)+'\n' for r in records).encode()
        source = yaml.safe_dump(curriculum).encode()
        spec = {'scenes_sha256': hashlib.sha256(raw).hexdigest(), 'curriculum_sha256': hashlib.sha256(source).hexdigest(),
                'level_variants': {str(i): [f'v{i}'] for i in range(9)}, 'total_episodes': 9, 'episodes_each_variant': 1}
        return raw, source, spec
    return records, curriculum, values


@pytest.mark.parametrize('fault', ['missing_level', 'duplicate_id', 'wrong_split', 'holdout_episode', 'holdout_layout', 'wrong_level', 'missing_record'])
def test_honestly_rehashed_incomplete_or_wrong_population_refused(fault):
    records, curriculum, values = fixture_records()
    records, curriculum = copy.deepcopy(records), copy.deepcopy(curriculum)
    if fault == 'missing_level':
        curriculum['curriculum']['variants_by_level'].pop()
    elif fault == 'duplicate_id':
        records[1]['episode_index'] = records[0]['episode_index']
    elif fault == 'wrong_split':
        records[0]['scene_block'] = 'training'
    elif fault == 'holdout_episode':
        records[0]['episode_seed'] = 966010
    elif fault == 'holdout_layout':
        records[0]['layout_seed'] = 1966010
    elif fault == 'wrong_level':
        records[0]['level'] = 7
    else:
        records.pop()
    with pytest.raises(ValueError):
        decode_original_population(*values(records, curriculum))


def test_original_file_hash_required_not_replaced_by_valid_record_counts():
    records, curriculum, values = fixture_records()
    raw, source, spec = values()
    with pytest.raises(ValueError, match='source differs'):
        decode_original_population(raw+b'\n', source, spec)


@pytest.mark.parametrize('success,time,result', [(True, 8., 8.), (False, None, 25.), (False, 3., 25.)])
def test_failed_unsafe_or_timeout_episodes_not_dropped_from_capped_timing(success, time, result):
    row = {'safe_capture_success': success, 'capture_time_seconds': time,
           'collision': False, 'boundary_violation': False, 'target_invalid_episode': False}
    assert restricted_safe_capture_time(row) == result


@pytest.mark.parametrize('key,value', [('collision', True), ('boundary_violation', True), ('target_invalid_episode', True),
    ('capture_time_seconds', float('nan')), ('capture_time_seconds', 26.)])
def test_inconsistent_safe_capture_or_invalid_required_time_refused(key, value):
    row = {'safe_capture_success': True, 'capture_time_seconds': 4., 'collision': False,
           'boundary_violation': False, 'target_invalid_episode': False}
    row[key] = value
    with pytest.raises(ValueError):
        restricted_safe_capture_time(row)


def test_existing_groups_and_two_variants_not_window_weighted_or_silent_dropped():
    def row(group, variant, time, success=True):
        return {'level': 7, 'variant': variant, 'mirror_group_id': group,
            'restricted_safe_capture_time_seconds': time, 'safe_capture_success': success,
            'collision': False, 'boundary_violation': False, 'timeout': not success, 'target_invalid_episode': False}
    rows = [row('singleton', 'fast_target', 2.), row('mirror', 'fast_target', 10.),
            row('mirror', 'fast_target', 10.), row('agile', 'agile_target', 25., False)]
    result = summarize(rows)['7']
    assert result['episodes'] == 4
    assert result['variants']['fast_target']['groups'] == 2
    assert result['variants']['fast_target']['group_equal_restricted_safe_capture_time_seconds'] == 6.
    assert result['variant_equal_restricted_safe_capture_time_seconds'] == 15.5
    assert result['variants']['agile_target']['timeout_rate'] == 1.


def test_nonfinite_native_diagnostics_are_preserved_as_tags_not_zero():
    values = {'diagnostics': [float('inf'), float('-inf'), float('nan')], 'actual_time': 1.2, 'native': np.int64(3)}
    encoded = json_diagnostics(values)
    assert encoded['diagnostics'] == [{'native_nonfinite_float': 'positive_infinity'},
        {'native_nonfinite_float': 'negative_infinity'}, {'native_nonfinite_float': 'nan'}]
    assert encoded['actual_time'] == 1.2 and encoded['native'] == 3
    json.dumps(encoded, allow_nan=False)
