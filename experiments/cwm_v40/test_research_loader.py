"""Synthetic certificate/weights/forecast boundaries; no native qualification."""
import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_loader as loader
from archive_io import write_json
from test_checkpoint_reload import toy_checkpoints


class Unreadable:
    def __fspath__(self): pytest.fail('Disabled preload must not touch paths')


@pytest.mark.parametrize('mode', ['off', 'refused'])
def test_off_and_refusal_read_nothing_and_construct_no_models(mode, monkeypatch):
    monkeypatch.setattr(loader, 'load_plan', lambda: pytest.fail('No plan IO on fallback'))
    bundle = loader.load_bundle(Unreadable(), Unreadable(), Unreadable(), mode=mode)
    assert bundle.authorized is False and not bundle.scorers and bundle.proposer is None
    assert bundle.deployment_control_eligible is False and bundle.full_v39_runner_ready is False


def proof(eligible=False):
    return {'status': 'complete_streamed_prefix_archive_independently_replayed',
        'artifact': {'sha256': 'a'*64, 'bytes': 123, 'members': 10},
        'all512_original_trajectories_plans_CBF_commands_replayed': True,
        'all_public_contexts_augmented_physical_branches_original_costs_replayed': True,
        'all18_final_models_optimizer_RNG_predictions_original_costs_gates_reloaded': True,
        'two_complete_fixed_runs_equal': True, 'artifact_packaged_and_replayed': True,
        'enhanced_control_enabled': False, 'holdout_used': False, 'new_training_performed': False,
        'closed_loop_capture_safety_latency_qualified': False, 'active_model_fault_fallback_qualified': False,
        'primary_research_eligible': eligible, 'qualification': {loader.PRIMARY:
            {'research_eligible': eligible, 'online_promoted': False, 'checks': {'own_motion': eligible, 'matched_full': eligible}}},
        'prior_complete_reload_summary_sha256': hashlib.sha256(b'{}').hexdigest()}


@pytest.mark.parametrize('kind', ['partial', 'artifact', 'numeric_flag', 'promotion', 'holdout', 'training', 'override', 'failed_check'])
def test_certificate_tamper_refused(kind):
    original = proof(True); changed = copy.deepcopy(original)
    loader.validate_certificate(original, original['artifact'])
    if kind == 'partial': changed['all18_final_models_optimizer_RNG_predictions_original_costs_gates_reloaded'] = False
    elif kind == 'artifact': changed['artifact']['sha256'] = 'b'*64
    elif kind == 'numeric_flag': changed['two_complete_fixed_runs_equal'] = 1
    elif kind == 'promotion': changed['enhanced_control_enabled'] = True
    elif kind == 'holdout': changed['holdout_used'] = True
    elif kind == 'training': changed['new_training_performed'] = True
    elif kind == 'override': changed['qualification'][loader.PRIMARY]['research_eligible'] = False
    else: changed['qualification'][loader.PRIMARY]['checks']['matched_full'] = False
    with pytest.raises(ValueError): loader.validate_certificate(changed, original['artifact'])


def report_rows():
    contract, plan = loader.load_contract()
    models = [{'configuration': m, 'seed': s, 'development': {'group_equal_ade_m': float(i+1)}}
              for m in plan['models'] for i, s in enumerate(plan['training']['seeds'])]
    return {'v34_plan': plan, 'v36_contract': contract, 'models': models,
            'selected_median_seeds': {m: plan['training']['seeds'][1] for m in plan['models']}}


@pytest.mark.parametrize('kind', ['best_seed', 'missing', 'nonfinite', 'bool_metric', 'budget'])
def test_seed_selection_never_switches_for_gain(kind):
    report = report_rows(); loader.selected_rows(report)
    if kind == 'best_seed': report['selected_median_seeds'][loader.PRIMARY] = 1034101
    elif kind == 'missing': report['models'].pop()
    elif kind == 'nonfinite': report['models'][0]['development']['group_equal_ade_m'] = float('nan')
    elif kind == 'bool_metric': report['models'][0]['development']['group_equal_ade_m'] = True
    else: report['v34_plan']['training']['response_epochs'] = 1
    with pytest.raises(ValueError): loader.selected_rows(report)


def test_tied_ade_median_uses_fixed_seed_order():
    report = report_rows()
    for row in report['models']: row['development']['group_equal_ade_m'] = 1.
    assert {r['seed'] for r in loader.selected_rows(report).values()} == {1034102}


def mocked_archive(tmp_path, monkeypatch, eligible):
    cert = tmp_path/'certificate.json'; p = proof(eligible); write_json(cert, p)
    artifact = tmp_path/'toy.zip'; artifact.write_bytes(b'not a scientific archive')
    prior = {'qualification': p['qualification'], 'primary_research_eligible': eligible}
    monkeypatch.setattr(loader, 'verify_archive', lambda path: p['artifact'])
    monkeypatch.setattr(loader, 'extract_archive', lambda *a: None)
    monkeypatch.setattr(loader.release, 'check_extracted', lambda *a: None)
    report = report_rows()
    def read(path):
        if path == cert: return loader.read_json_original(cert)
        if path.name == loader.MANIFEST: return {'members': {'fixture': {}, 'release_contract.json': {}}}
        if path.name == 'release_contract.json': return loader.release.release_contract(prior)
        if path.name == 'records.json': return [{'arrays_path': 'toy', 'split': 'train'}, {'arrays_path': 'toy2', 'split': 'development_validation'}]
        if path.parent.name == 'data': return {'protocol': {}}
        return report
    monkeypatch.setattr(loader, 'read_json_original', loader.read_json, raising=False)
    monkeypatch.setattr(loader, 'read_json', read)
    def inventory(**roots):
        # Only provenance plumbing: this authority is mocked, not native V34.
        (roots['reload']).mkdir(parents=True, exist_ok=True)
        (roots['reload']/'summary.json').write_bytes(b'{}')
        return {'fixture': artifact}, prior
    monkeypatch.setattr(loader.release, 'inventory', inventory)
    return artifact, cert, p


def test_failed_primary_refuses_before_any_checkpoint_or_proposer(tmp_path, monkeypatch):
    artifact, cert, _ = mocked_archive(tmp_path, monkeypatch, False)
    monkeypatch.setattr(loader, 'load_selected', lambda *a: pytest.fail('Failed primary cannot load any control'))
    monkeypatch.setattr(loader, 'load_proposer', lambda *a: pytest.fail('Failed primary cannot load proposer'))
    bundle = loader.load_bundle(artifact, cert, tmp_path/'cache', mode='research_prepare')
    assert bundle.authorized is False and bundle.status == 'offline_qualification_failed'
    assert not bundle.scorers and bundle.proposer is None


@pytest.mark.parametrize('kind', ['success', 'changed_certificate', 'failed_model'])
def test_preload_authorization_last_and_never_full_runner_or_deployment(tmp_path, monkeypatch, kind):
    artifact, cert, _ = mocked_archive(tmp_path, monkeypatch, True)
    events = []
    def selected(directory, report, rows, indices):
        events.append('selected')
        assert indices == [0] and set(rows) == set(report['v34_plan']['models'])
        if kind == 'failed_model': raise ValueError('Toy reconstruction failure')
        if kind == 'changed_certificate': cert.write_bytes(b'{}')
        return {name: object() for name in rows}, {'toy_restoration': True}
    monkeypatch.setattr(loader, 'load_selected', selected)
    monkeypatch.setattr(loader, 'load_proposer', lambda *a: events.append('proposer') or object())
    if kind == 'success':
        bundle = loader.load_bundle(artifact, cert, tmp_path/'cache', mode='research_prepare')
        assert bundle.authorized is True and events == ['selected', 'proposer']
        assert bundle.deployment_control_eligible is False and bundle.legacy_l2_control_loaded is False
        assert bundle.full_v39_runner_ready is False
    else:
        with pytest.raises(ValueError): loader.load_bundle(artifact, cert, tmp_path/'cache', mode='research_prepare')


def test_unsupported_deploy_mode_or_existing_cache_refused(tmp_path):
    with pytest.raises(ValueError): loader.load_bundle(mode='deploy')
    with pytest.raises(ValueError): loader.load_bundle(None, None, tmp_path, mode='research_prepare')


def inputs():
    return [np.zeros((8, 252)), np.zeros((4, 6)), np.zeros((3, 8, 4, 3)),
            np.zeros((8, 4, 3)), np.zeros((8, 3)), np.ones((8, 3))]


@pytest.mark.parametrize('index', [0, 1])
def test_actual_toy80epoch_reconstruction_public_predictor_preserves_rng_and_outputs(toy_checkpoints, index):
    checkpoint = toy_checkpoints[index]; before = torch.get_rng_state().clone()
    model, restored = loader.reconstruct(checkpoint, checkpoint['configuration'], 1034101, [0, 1])
    predict = loader.public_predictor(model, np.zeros((1, 1, 252)), np.ones((1, 1, 252)))
    arrays = inputs(); original = [a.copy() for a in arrays]
    adapter = loader.OptionalResponseAdapter(loader.QualificationGate(mode='shadow'), lambda: predict)
    output, info = adapter.forecast(*arrays)
    assert output is not None and info['control_eligible'] is False
    assert output['prediction'].dtype == np.float64 and output['reference'].dtype == np.float64
    assert (output['response'] == 0).all() and output['prediction'].tobytes() == output['reference'].tobytes()
    assert torch.equal(before, torch.get_rng_state()) and restored['optimizer_updated_by_reload'] is False
    assert all(a.tobytes() == b.tobytes() for a, b in zip(arrays, original))


@pytest.mark.parametrize('kind', ['collapse', 'overspeed', 'private_extra', 'too_many', 'nonfinite', 'bad_shape'])
def test_public_predictor_refuses_bad_inputs_before_forward(kind):
    class Model:
        def eval(self): return self
        def requires_grad_(self, value): return self
        def __call__(self, *a): pytest.fail('Bad public input must be refused before neural forward')
    predict = loader.public_predictor(Model(), np.zeros((1, 1, 252)), np.ones((1, 1, 252)))
    arrays = inputs()
    if kind == 'collapse':
        arrays[3][:, 0, 0] = 1.; arrays[2][:, :, 0, 0] = 1.+1e-9
    elif kind == 'overspeed': arrays[2][0, 0, 0, 0] = 6.
    elif kind == 'private_extra': arrays[0] = np.zeros((8, 253))
    elif kind == 'too_many': arrays[2] = np.zeros((25, 8, 4, 3))
    elif kind == 'nonfinite': arrays[4][0, 0] = float('nan')
    else: arrays[1] = np.zeros((4, 7))
    with pytest.raises(ValueError): predict(*arrays)


@pytest.mark.parametrize('tamper', [False, True])
def test_selected_actual_toy_weights_cache_normalizer_and_stage_provenance(tmp_path, toy_checkpoints, tamper):
    seed = 1034101; name = 'cv_prefix_rank_l2'
    cp = copy.deepcopy(toy_checkpoints[1]); report = {'source_hashes': {}, 'data_summary_sha256': 'a'*64, 'data_audit_summary_sha256': 'b'*64}
    mean = np.zeros((1, 1, 252)); scale = np.ones_like(mean)
    np.savez_compressed(tmp_path/'normalization.npz', mean=mean, scale=scale)
    write_json(tmp_path/'population_weights.json', {'fixture': True})
    (tmp_path/f'cache{seed}').mkdir(); write_json(tmp_path/f'cache{seed}/manifest.json', {'fixture': True})
    cp.update(report); cp.update(normalizer_mean=torch.from_numpy(mean.copy()), normalizer_scale=torch.from_numpy(scale.copy()),
        population_weights_sha256=loader.sha(tmp_path/'population_weights.json'),
        canonical_cache_manifest_sha256=loader.sha(tmp_path/f'cache{seed}/manifest.json'))
    if tamper: cp['normalizer_mean'][0, 0, 0] = 1.
    path = tmp_path/f'{name}_seed{seed}.pt'; torch.save(cp, path)
    torch.save(toy_checkpoints[0], tmp_path/f'cv_seed{seed}_motion_stage.pt')
    rows = {name: {'seed': seed, 'checkpoint_sha256': loader.sha(path)}}
    if tamper:
        with pytest.raises(ValueError, match='normalizer'): loader.load_selected(tmp_path, report, rows, [0, 1])
    else:
        scorers, restored = loader.load_selected(tmp_path, report, rows, [0, 1])
        output, info = scorers[name].forecast(*inputs())
        assert output is not None and info['control_eligible'] is False
        assert restored[name]['actual_optimizer_state_exact'] is True
        assert restored[name]['optimizer_updated_by_reload'] is False
