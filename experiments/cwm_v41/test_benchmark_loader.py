"""Scorer preparation boundaries, with explicitly synthetic archive authorities.

Initialized legacy neural weights exercise cold state/public forward only.
They are not scientifically trained or qualified, despite fixture history labels.
"""
import copy
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_loader as loader
import release_ranking_training_v2 as legacy_types


class Unreadable:
    def __fspath__(self):
        pytest.fail('Disabled/failed prefix must never inspect legacy paths')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def legacy_proof():
    published = loader.read_json(loader.HERE.parent/'cwm_v28/reports/completed_native_archive_replay_failed_primary_20261011.json')
    return copy.deepcopy(published['native_summary'])


@pytest.mark.parametrize('mode', ['off', 'refused'])
def test_disabled_does_not_read_plan_paths_or_models(mode, monkeypatch):
    monkeypatch.setattr(loader.prefix, 'load_plan', lambda: pytest.fail('Disabled must not read plan'))
    monkeypatch.setattr(loader.prefix, 'load_bundle', lambda *a, **k: pytest.fail('Disabled must not load prefix'))
    result = loader.load_benchmark(*[Unreadable()]*5, mode=mode)
    assert result.status == mode and result.authorized is False
    assert not result.scorers and result.proposer is None and result.prefix_bundle is None
    assert result.deployment_control_eligible is False and result.full_v39_runner_ready is False


@pytest.mark.parametrize('mode', ['deploy', 'guarded', 'holdout'])
def test_no_deployment_or_holdout_entry(mode):
    with pytest.raises(ValueError):
        loader.load_benchmark(*[Unreadable()]*5, mode=mode)


def test_failed_prefix_never_reads_legacy_or_creates_legacy_cache(tmp_path, monkeypatch):
    bundle = loader.prefix.PrefixResearchBundle(status='offline_qualification_failed')
    monkeypatch.setattr(loader.prefix, 'load_bundle', lambda *a, **k: bundle)
    monkeypatch.setattr(loader, '_load_legacy', lambda *a: pytest.fail('Failed prefix cannot load failed legacy'))
    result = loader.load_benchmark(None, None, Unreadable(), Unreadable(), tmp_path/'cache', mode='research_prepare')
    assert result.authorized is False and result.status == 'prefix_primary_failed_no_legacy_access'
    assert not (tmp_path/'cache'/'legacy').exists() and not result.scorers


@pytest.mark.parametrize('kind', ['certificate', 'archive', 'partial', 'numeric_flag', 'promoted', 'holdout', 'l2_qualified', 'rank_qualified'])
def test_legacy_pinned_native_certificate_and_failed_gates_required(kind):
    plan = loader.prefix.load_plan(); proof = legacy_proof()
    integrity = copy.deepcopy(proof['artifact'])
    digest = plan['legacy_l2_control']['native_replay_certificate_sha256']
    loader.validate_legacy_certificate(proof, integrity, digest, plan)
    if kind == 'certificate': digest = '0'*64
    elif kind == 'archive': integrity['sha256'] = '0'*64
    elif kind == 'partial': proof['all18_final_models_original_costs_reloaded'] = False
    elif kind == 'numeric_flag': proof['two_complete_runs_equal'] = 1
    elif kind == 'promoted': proof['enhanced_control_enabled'] = True
    elif kind == 'holdout': proof['holdout_used'] = True
    elif kind == 'l2_qualified': proof['qualification']['cv_l2']['research_eligible'] = True
    else: proof['qualification']['cv_rank_l2']['research_eligible'] = True
    with pytest.raises(ValueError):
        loader.validate_legacy_certificate(proof, integrity, digest, plan)


def toy_report():
    protocol = loader.read_json(loader.HERE.parent/'cwm_v28/training_protocol.json')
    proof = legacy_proof()
    models = [{'configuration': m, 'seed': s, 'development': {'group_equal_ade_m': float([2, 1, 3][i])}}
              for m in protocol['models'] for i, s in enumerate(protocol['training']['seeds'])]
    return {'protocol': protocol, 'qualification': copy.deepcopy(proof['qualification']), 'models': models,
        'selected_median_seeds': {m: 994101 for m in protocol['models']},
        'primary_research_eligible': False, 'enhanced_control_enabled': False, 'holdout_used': False,
        'prior_gate_overridden': False, 'common_motion_in_response_optimizer': False}, proof


@pytest.mark.parametrize('kind', ['best_seed', 'missing', 'nonfinite', 'bool_ade', 'budget', 'override', 'gate', 'identity'])
def test_fixed_final80_median_never_switches_for_gain(kind):
    report, proof = toy_report()
    assert loader.selected_legacy_row(report, proof)['seed'] == 994101
    if kind == 'best_seed': report['selected_median_seeds']['cv_l2'] = 994102
    elif kind == 'missing': report['models'].pop()
    elif kind == 'nonfinite': report['models'][0]['development']['group_equal_ade_m'] = float('nan')
    elif kind == 'bool_ade': report['models'][0]['development']['group_equal_ade_m'] = True
    elif kind == 'budget': report['protocol']['training']['response_epochs'] = 1
    elif kind == 'override': report['prior_gate_overridden'] = True
    elif kind == 'gate': report['qualification']['cv_l2']['research_eligible'] = True
    else: report['models'][0]['seed'] = 994100
    with pytest.raises(ValueError):
        loader.selected_legacy_row(report, proof)


def initialized_checkpoint(directory, mutation=None):
    """Untrained fixture with 80 synthetic history rows; no optimizer step."""
    torch.set_num_threads(1)
    seed = 994101; protocol = loader.read_json(loader.HERE.parent/'cwm_v28/training_protocol.json')
    with loader.preserved_cpu_rng():
        torch.manual_seed(seed)
        common = legacy_types.CalibratedOrigin(legacy_types.LocalTwoHead('motion_only', 2.5, 2.5), 'cv')
        model = legacy_types.FrozenOriginResponse(common, legacy_types.LocalTwoHead('plain', 2.5, 2.5))
    mean = np.zeros((1, 1, 252)); scale = np.ones_like(mean)
    report = {'source_hashes': {}, 'data_summary_sha256': 'a'*64, 'data_audit_summary_sha256': 'b'*64}
    checkpoint = {**report, 'protocol': protocol, 'configuration': 'cv_l2', 'seed': seed,
        'online_promoted': False, 'baseline_weights_included': False, 'common_motion_in_response_optimizer': False,
        'model_state': copy.deepcopy(model.state_dict()), 'common_motion_state_digest': legacy_types.state_digest(common),
        'normalizer_mean': torch.from_numpy(mean.copy()), 'normalizer_scale': torch.from_numpy(scale.copy())}
    write(directory/'population_weights.json', {'synthetic': True})
    np.savez_compressed(directory/f'cv_seed{seed}_ranking_cache.npz', synthetic=np.zeros(1))
    checkpoint['population_weights_sha256'] = loader.sha(directory/'population_weights.json')
    checkpoint['ranking_cache_sha256'] = loader.sha(directory/f'cv_seed{seed}_ranking_cache.npz')
    np.savez_compressed(directory/'normalization.npz', mean=mean, scale=scale)
    history = [{'epoch': epoch, 'mean_batch_loss_terms': [0.]} for epoch in range(1, 81)]
    stage = {'model_state': copy.deepcopy(common.state_dict())}
    if mutation == 'normalizer': checkpoint['normalizer_mean'][0, 0, 0] = 1.
    elif mutation == 'cache': checkpoint['ranking_cache_sha256'] = 'c'*64
    elif mutation == 'provenance': checkpoint['data_summary_sha256'] = 'c'*64
    elif mutation == 'dtype':
        key = next(iter(checkpoint['model_state'])); checkpoint['model_state'][key] = checkpoint['model_state'][key].double()
    elif mutation == 'stage': next(iter(stage['model_state'].values())).flatten()[0] += 1.
    elif mutation == 'history': history.pop()
    elif mutation == 'promoted': checkpoint['online_promoted'] = True
    elif mutation == 'nonfinite': next(iter(checkpoint['model_state'].values())).flatten()[0] = float('nan')
    path = directory/f'cv_l2_seed{seed}.pt'; torch.save(checkpoint, path)
    torch.save(stage, directory/f'cv_seed{seed}_motion_stage.pt')
    write(directory/f'cv_l2_seed{seed}_history.json', history)
    return report, {'configuration': 'cv_l2', 'seed': seed, 'checkpoint_sha256': loader.sha(path)}


def public_arrays():
    values = [np.zeros((8, 252)), np.zeros((4, 6)), np.zeros((3, 8, 4, 3)),
              np.zeros((8, 4, 3)), np.zeros((8, 3)), np.ones((8, 3))]
    values[2][1, :, 0, 0] = .2; values[2][2, :, 0, 0] = -.2
    return values


def test_actual_initialized_legacy_model_cold_reload_public_forward_and_rng_preservation(tmp_path):
    report, row = initialized_checkpoint(tmp_path)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    adapter, evidence = loader.load_legacy_checkpoint(tmp_path, report, row)
    arrays = public_arrays(); original = [a.copy() for a in arrays]
    forecast, info = adapter.forecast(*arrays)
    assert forecast is not None and info['control_eligible'] is False and adapter.load_calls == 0
    assert evidence['model_state_exact'] is True and evidence['optimizer_loaded_here'] is False
    assert evidence['research_eligible'] is False and evidence['matched_prefix_control'] is False
    assert forecast['prediction'].shape == (3, 8, 3) and forecast['prediction'].dtype == np.float64
    assert (forecast['response'][0] == 0).all()
    assert all(a.tobytes() == b.tobytes() for a, b in zip(arrays, original))
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    after = np.random.get_state()
    assert np.array_equal(numpy_rng[1], after[1]) and numpy_rng[2:] == after[2:]


@pytest.mark.parametrize('kind', ['normalizer', 'cache', 'provenance', 'dtype', 'stage', 'history', 'promoted', 'nonfinite'])
def test_cold_model_provenance_precision_budget_and_original_motion_stage_refuse_changes(tmp_path, kind):
    report, row = initialized_checkpoint(tmp_path, kind)
    with pytest.raises(ValueError):
        loader.load_legacy_checkpoint(tmp_path, report, row)


def qualified_fixture():
    names = ['cv_motion_only', 'cv_full_query_rank_l2', 'cv_prefix_rank_l2']
    return loader.prefix.PrefixResearchBundle(authorized=True, scorers={n: object() for n in names},
        proposer=object(), selected_median_seeds={n: 1034102 for n in names})


@pytest.mark.parametrize('kind', ['success', 'legacy_failure', 'prefix_mutated', 'missing_comparator', 'partial_prefix'])
def test_benchmark_preload_grants_no_control_or_runner_and_revokes_on_failure(tmp_path, monkeypatch, kind):
    bundle = qualified_fixture(); events = []
    monkeypatch.setattr(loader.prefix, 'load_bundle', lambda *a, **k: events.append('prefix') or bundle)
    if kind == 'partial_prefix': bundle.scorers.pop('cv_full_query_rank_l2')
    def legacy(*args):
        events.append('legacy')
        if kind == 'legacy_failure': raise ValueError('Synthetic legacy load failure')
        return object(), {'selected_seed': 994101, 'research_eligible': False}
    def recheck(*args):
        events.append('prefix_recheck')
        if kind == 'prefix_mutated': raise ValueError('Synthetic prefix archive mutation')
    monkeypatch.setattr(loader, '_load_legacy', legacy)
    monkeypatch.setattr(loader, 'recheck_prefix', recheck)
    args = ('prefix.zip', 'prefix.json', None if kind == 'missing_comparator' else 'legacy.zip', 'legacy.json', tmp_path/'cache')
    if kind == 'success':
        result = loader.load_benchmark(*args, mode='research_prepare')
        assert result.authorized is True and events == ['prefix', 'legacy', 'prefix_recheck']
        assert set(result.scorers) == set(bundle.scorers) | {'cv_l2'}
        assert result.full_v39_runner_ready is False and result.deployment_control_eligible is False
        assert result.legacy_l2_research_eligible is False
        assert bundle.legacy_l2_control_loaded is False  # Keep V40's distinct proof unchanged.
    else:
        with pytest.raises(ValueError): loader.load_benchmark(*args, mode='research_prepare')
        assert bundle.authorized is False and bundle.status == 'benchmark_preload_failed'
        if kind in ('missing_comparator', 'partial_prefix'): assert events == ['prefix']


@pytest.mark.parametrize('exists', [True, False])
def test_invalid_cache_refused_before_prefix_or_legacy_io(tmp_path, monkeypatch, exists):
    monkeypatch.setattr(loader.prefix, 'load_bundle', lambda *a, **k: pytest.fail('Invalid cache must fail first'))
    with pytest.raises(ValueError, match='NEW benchmark cache'):
        loader.load_benchmark(cache=tmp_path if exists else None, mode='research_prepare')


def test_non_native_legacy_certificate_rejected_before_archive_access(tmp_path, monkeypatch):
    certificate = tmp_path/'fake.json'; write(certificate, legacy_proof())
    monkeypatch.setattr(loader.prefix, 'resolve_artifact', lambda *a: pytest.fail('Reject wrong certificate bytes first'))
    with pytest.raises(ValueError, match='native legacy certificate bytes'):
        loader._load_legacy(Unreadable(), certificate, tmp_path/'cache', loader.prefix.load_plan())


@pytest.mark.parametrize('kind', ['success', 'incomplete_inventory', 'contract', 'checkpoint_failure', 'mutation'])
def test_legacy_full_inventory_contract_cold_load_and_final_rechecks(tmp_path, monkeypatch, kind):
    """All archive/replay authorities mocked: orchestration, not native proof."""
    plan = loader.prefix.load_plan(); proof = legacy_proof(); report, _ = toy_report()
    artifact = tmp_path/'toy.zip'; artifact.write_bytes(b'synthetic authority')
    certificate = tmp_path/'toy.json'; write(certificate, proof)
    original_sha = loader.sha
    digest = plan['legacy_l2_control']['native_replay_certificate_sha256']
    def digest_fixture(path):
        if path == certificate: return digest
        if path == artifact: return plan['legacy_l2_control']['archive_sha256']
        return original_sha(path)
    monkeypatch.setattr(loader, 'sha', digest_fixture)
    monkeypatch.setattr(loader.legacy_release, 'verify_archive', lambda *a: proof['artifact'])
    monkeypatch.setattr(loader.legacy_release, 'extract_archive', lambda *a: proof['artifact'])
    events = []
    def check(*args):
        events.append('evidence')
        if kind == 'mutation' and events.count('evidence') == 2: raise ValueError('Synthetic extracted mutation')
    monkeypatch.setattr(loader.legacy_release, 'check_extracted_evidence', check)
    prior = {'qualification': proof['qualification'], 'primary_research_eligible': False}
    monkeypatch.setattr(loader.legacy_release, 'inventory', lambda **k: events.append('inventory') or ({'fixture': artifact}, prior))
    contract = loader.legacy_release.release_contract(prior)
    original_read = loader.read_json
    def read(path):
        if path == certificate: return original_read(path)
        if path == loader.HERE.parent/'cwm_v28/training_protocol.json': return original_read(path)
        if path.name == loader.MANIFEST:
            names = {'fixture': {}, 'release_contract.json': {}}
            if kind == 'incomplete_inventory': names.pop('fixture')
            return {'members': names}
        if path.name == 'release_contract.json':
            return {**contract, 'holdout_used': True} if kind == 'contract' else contract
        return report
    monkeypatch.setattr(loader, 'read_json', read)
    monkeypatch.setattr(loader, 'bound_manifest', lambda archive, directory: read(directory/loader.MANIFEST))
    def selected(*args):
        events.append('checkpoint')
        if kind == 'checkpoint_failure': raise ValueError('Synthetic checkpoint failure')
        return object(), {'selected_seed': 994101}
    monkeypatch.setattr(loader, 'load_legacy_checkpoint', selected)
    if kind == 'success':
        _, evidence = loader._load_legacy(artifact, certificate, tmp_path/'cache', plan)
        assert events == ['evidence', 'inventory', 'checkpoint', 'evidence', 'inventory']
        assert evidence['archive_sha256'] == proof['artifact']['sha256']
        assert evidence['certificate_sha256'] == digest
    else:
        with pytest.raises(ValueError): loader._load_legacy(artifact, certificate, tmp_path/'cache', plan)
        if kind in ('incomplete_inventory', 'contract'): assert 'checkpoint' not in events


def test_prefix_certificate_rechecked_after_legacy_preload(tmp_path, monkeypatch):
    artifact = tmp_path/'prefix.zip'; artifact.write_bytes(b'changed archive')
    cert = tmp_path/'prefix.json'; cert.write_bytes(b'changed certificate')
    bundle = qualified_fixture(); bundle.archive_sha256 = 'a'*64; bundle.certificate_sha256 = 'b'*64
    monkeypatch.setattr(loader.prefix, 'verify_archive', lambda *a: {'sha256': 'a'*64})
    monkeypatch.setattr(loader.prefix.release, 'inventory', lambda **k: pytest.fail('Refuse mutated certificate before evidence checks'))
    with pytest.raises(ValueError, match='changed while loading legacy'):
        loader.recheck_prefix(bundle, artifact, cert, tmp_path/'cache')


@pytest.mark.parametrize('mutation', ['changed_hash', 'omitted_evidence', 'duplicate_key'])
def test_extracted_manifest_cannot_redefine_pinned_archive_evidence(tmp_path, mutation):
    artifact = tmp_path/'toy.zip'; extracted = tmp_path/'extracted'; extracted.mkdir()
    original = {'schema': 'SYNTHETIC_ONLY', 'members': {'evidence.json': {'bytes': 2, 'sha256': 'a'*64}}}
    with zipfile.ZipFile(artifact, 'x') as archive:
        archive.writestr(loader.MANIFEST, json.dumps(original))
    changed = copy.deepcopy(original)
    if mutation == 'changed_hash': changed['members']['evidence.json']['sha256'] = 'b'*64
    elif mutation == 'omitted_evidence': changed['members'] = {}
    if mutation == 'duplicate_key':
        (extracted/loader.MANIFEST).write_text('{"members":{},"members":{}}', encoding='utf8')
    else:
        write(extracted/loader.MANIFEST, changed)
    with pytest.raises(ValueError): loader.bound_manifest(artifact, extracted)


def test_bound_manifest_accepts_identical_json_types_values_and_rejects_bool_length(tmp_path):
    artifact = tmp_path/'toy.zip'; extracted = tmp_path/'extracted'; extracted.mkdir()
    original = {'schema': 'SYNTHETIC_ONLY', 'members': {'evidence.json': {'bytes': 1, 'sha256': 'a'*64}}}
    with zipfile.ZipFile(artifact, 'x') as archive: archive.writestr(loader.MANIFEST, json.dumps(original))
    write(extracted/loader.MANIFEST, original)
    assert loader.bound_manifest(artifact, extracted) == original
    original['members']['evidence.json']['bytes'] = True; write(extracted/loader.MANIFEST, original)
    with pytest.raises(ValueError): loader.bound_manifest(artifact, extracted)
