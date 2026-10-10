"""Historical TRAIN fixture tests implementation only, never V23 fitting/data."""
import copy
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_cost_response import frozen_configs, BASELINE_SHA, sha
from deployed_cost_loss import FrozenLocalScore, original_scores, contrast_cache, deployed_contrast, audit_deployed_score
from local_shadow import restore_public_call
from geometry_release import arrays


@pytest.fixture(scope='module')
def engine_case(tmp_path_factory):
    torch.set_num_threads(1)
    capsule = ROOT.parent / 'cwm_v1/baseline/capsule.zip'
    assert sha(capsule) == BASELINE_SHA
    configs = frozen_configs(capsule, tmp_path_factory.mktemp('v23_engine_test') / 'restored')
    with zipfile.ZipFile(ROOT.parent / 'cwm_v10/artifacts/two_head_training_20261010.zip') as archive:
        records = json.loads(archive.read('data/calls.json'))
        record = next(r for r in records if r.get('all_candidates_full_horizon_valid') and
                      r.get('split') == 'train' and 'arrays_path' in r)
        values = arrays(archive.read('data/' + record['arrays_path']))
        context = restore_public_call(json.loads(archive.read('data/' + record['context_path'])),
                                      *configs, values['backbone'])
    return context, values


@pytest.mark.parametrize('origin', ['gru', 'cv'])
def test_deployed_effect_full_original_cost_and_piecewise_gradient(engine_case, origin):
    context, values = engine_case
    actions = values['proposed'][:, :, context['agent']]
    reference = values['backbone'] if origin == 'gru' else values['reference'][None] + .1 * np.arange(1, 9)[:, None] * values['velocity'][None]
    reference = np.repeat(reference[None], len(actions), 0)
    score = FrozenLocalScore(context, actions)
    cache = contrast_cache(score, reference, values)
    report = audit_deployed_score(context, values, reference, score, cache)
    assert report['gradient_checked_coordinates'] > 0
    assert max(report['score_max_absolute_errors'].values()) < 1e-7
    response = torch.full_like(cache['reference'], .04, requires_grad=True)
    measured = deployed_contrast(cache, response)
    predicted = original_scores(context, actions, reference + response.detach().numpy()) - original_scores(context, actions, reference)
    target = original_scores(context, actions, values['target'])
    anchor = original_scores(context, actions, np.repeat(values['anchor_target'][None], len(actions), 0))
    expected = (np.abs(predicted - (target - anchor)) / cache['scale'].numpy())[cache['nonanchor'].numpy()].mean()
    assert float(measured.detach()) == pytest.approx(float(expected), rel=1e-10, abs=1e-10)
    measured.backward()
    assert torch.isfinite(response.grad).all() and response.grad.abs().sum() > 0
    assert (response.grad[~cache['nonanchor']] == 0).all()


@pytest.mark.parametrize('flag', ['fc_dbf_enabled', 'escape_gap_cost_enabled', 'reachability_normalized_cost_enabled'])
def test_new_loss_does_not_bypass_unsupported_original_objectives(engine_case, flag):
    context, values = copy.deepcopy(engine_case)
    context['planner'].config = replace(context['planner'].config, **{flag: True})
    with pytest.raises(ValueError, match='Unsupported'):
        FrozenLocalScore(context, values['proposed'][:, :, context['agent']])


def test_changed_score_or_label_cache_is_rejected_by_original_engine(engine_case):
    context, values = engine_case
    actions = values['proposed'][:, :, context['agent']]
    ref = np.repeat(values['backbone'][None], len(actions), 0)
    score = FrozenLocalScore(context, actions)
    cache = contrast_cache(score, ref, values)
    cache['label_effect'] += .1
    with pytest.raises(ValueError, match='detached label'):
        audit_deployed_score(context, values, ref, score, cache)
    cache = contrast_cache(score, ref, values)
    score.offset += 1.
    with pytest.raises(ValueError, match='full original cost'):
        audit_deployed_score(context, values, ref, score, cache)
