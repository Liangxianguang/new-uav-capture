"""Exact-key/conditional-variance implementation tests, not identifiability."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_input_ambiguity import feature_bytes, ConditionalBuckets


def public():
    return {'history': np.zeros((8, 252)), 'relative': np.zeros((4, 6)),
        'proposed': np.ones((2, 8, 4, 3)), 'anchor': np.zeros((8, 4, 3)),
        'backbone': np.ones((8, 3)), 'reference': np.array([1., 2., 3.]),
        'velocity': np.array([.1, .2, .3])}


def keys(values):
    return feature_bytes(values, 0, np.zeros((1, 1, 252)), np.ones((1, 1, 252)))


def test_only_public_features_construct_the_key():
    values = public(); first = keys(values)
    for name in ('target', 'anchor_target', 'commanded', 'valid', 'branch', 'cost', 'executed'):
        values[name] = object()
    assert keys(values) == first


@pytest.mark.parametrize('name', ['history', 'relative', 'proposed', 'anchor', 'backbone'])
def test_each_actual_neural_public_tensor_changes_the_key(name):
    values = public(); before = keys(values)
    values[name].flat[0] += 1.
    assert keys(values)[0]['response_core'] != before[0]['response_core']


def test_public_cv_is_only_additional_full_model_information():
    values = public(); before = keys(values)
    values['velocity'][0] += .1
    after = keys(values)
    assert before[0]['response_core'] == after[0]['response_core']
    assert before[0]['whole_optional_model'] != after[0]['whole_optional_model']


def test_only_existing_float32_conversion_not_new_rounding_bins():
    values = public(); values['anchor'].fill(1.)
    values['proposed'][0].fill(1.+1e-10)
    assert not np.array_equal(values['proposed'][0], values['anchor'])
    assert keys(values)[1] is True
    values['proposed'][0].fill(1.+1e-5)
    assert keys(values)[1] is False


def test_feature_bytes_equal_actual_torch_plain_core_construction():
    values = public(); mean = np.ones((1, 1, 252))*.01; scale = np.ones_like(mean)*.1
    actual = feature_bytes(values, 0, mean, scale)[0]['response_core']
    tensors = [torch.as_tensor((values['history']-mean)/scale, dtype=torch.float32),
        torch.as_tensor(values['relative'], dtype=torch.float32)/torch.tensor([10., 10., 10., 5., 5., 5.]),
        torch.as_tensor(values['proposed'][0], dtype=torch.float32)/5.,
        torch.as_tensor(values['anchor'], dtype=torch.float32)/5.,
        torch.as_tensor(values['backbone'], dtype=torch.float32)/5.]
    assert actual == b''.join(v.numpy().tobytes() for v in tensors)+b'\x00'


def add(table, raw, label, mask=None, weight=.5):
    mask = np.ones(8, dtype=bool) if mask is None else mask
    table.add(raw, np.asarray(label, dtype=np.float64), mask, weight, raw, lambda origin: origin)


def test_repeated_conflicting_inputs_give_exact_empirical_mse_floor():
    table = ConditionalBuckets()
    add(table, b'same', np.zeros((8, 3)))
    add(table, b'same', np.ones((8, 3))*2.)
    report = table.result(.05)
    assert report['empirical_minimum_deterministic_coordinate_mse_m2'] == 1.
    assert report['duplicate_input_buckets'] == 1
    assert report['duplicate_buckets_with_distance_from_first_over_5cm'] == 1


def test_weighted_floor_matches_direct_conditional_least_squares():
    table = ConditionalBuckets()
    first = np.zeros((8, 3)); second = np.ones((8, 3))*4
    add(table, b'same', first, weight=.75); add(table, b'same', second, weight=.25)
    mean = .75*first+.25*second
    expected = (.75*((first-mean)**2).sum()+.25*((second-mean)**2).sum())/(3*8)
    assert table.result(.05)['empirical_minimum_deterministic_coordinate_mse_m2'] == expected


def test_distinct_inputs_do_not_create_a_spurious_conflict():
    table = ConditionalBuckets()
    add(table, b'a', np.zeros((8, 3))); add(table, b'b', np.ones((8, 3))*10.)
    report = table.result(.05)
    assert report['empirical_minimum_deterministic_coordinate_mse_m2'] == 0.
    assert report['single_observation_point_fraction'] == 1.


def test_terminal_padding_is_label_only_and_does_not_enter_variance():
    table = ConditionalBuckets(); mask = np.array([True, True, False, False, False, False, False, False])
    first = np.zeros((8, 3)); second = first.copy(); second[2:] = 100.
    add(table, b'a', first, mask); add(table, b'a', second, mask)
    report = table.result(.05)
    assert report['observed_common_prefix_points'] == 4
    assert report['empirical_minimum_deterministic_coordinate_mse_m2'] == 0.
    assert report['maximum_distance_from_first_response_m'] == 0.


def test_hash_bucket_is_not_permission_to_merge_different_feature_bytes():
    table = ConditionalBuckets(); add(table, b'same', np.zeros((8, 3)))
    with pytest.raises(ValueError, match='Hash collision'):
        table.add(b'same', np.zeros((8, 3)), np.ones(8, bool), .5, b'same', lambda origin: b'changed')


@pytest.mark.parametrize('field', ['history', 'relative', 'proposed', 'anchor', 'backbone', 'reference', 'velocity'])
def test_nonfinite_public_features_are_rejected(field):
    values = public(); values[field].flat[0] = np.nan
    with pytest.raises(ValueError): keys(values)


def test_empty_observed_support_does_not_emit_floor():
    with pytest.raises(ValueError): ConditionalBuckets().result(.05)
