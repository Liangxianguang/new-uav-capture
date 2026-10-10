"""Preload the mandatory failed V28 L2 comparator for qualified prefix research.

Calls the unchanged V40 loader itself; callers cannot supply a qualified bundle.
Default off and failed prefix qualification never inspect legacy evidence.
This prepares scorers only, with no planner, rollout, holdout or deployment entry.
"""
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent/n) for n in ('cwm_v40', 'cwm_v28', 'cwm_v38', 'cwm_v37')]
import research_loader as prefix
import package_ranking_release_v2 as legacy_release
from release_ranking_training_v2 import model_from_checkpoint
from checkpoint_reload import finite_state
from archive_io import sha, regular, read_json, decode_json, MANIFEST
from repeatability_identity import exact_tree
from qualification_gate import OptionalResponseAdapter, QualificationGate, preserved_cpu_rng


@dataclass
class BenchmarkScorers:
    authorized: bool = False
    status: str = 'off'
    scorers: dict = field(default_factory=dict)
    proposer: object = None
    prefix_bundle: object = None
    legacy_evidence: dict = field(default_factory=dict)
    selected_median_seeds: dict = field(default_factory=dict)
    deployment_control_eligible: bool = False
    full_v39_runner_ready: bool = False
    legacy_l2_research_eligible: bool = False


def validate_legacy_certificate(proof, integrity, digest, plan):
    control = plan['legacy_l2_control']
    if (digest != control['native_replay_certificate_sha256'] or
        integrity['sha256'] != control['archive_sha256'] or
        proof['status'] != 'complete_fixed_ranking_archive_independently_replayed' or
        not exact_tree(proof['artifact'], integrity) or
        any(proof[k] is not True for k in ('real_public_context_all_branch_replay_passed',
            'all64_original_trajectories_plans_commands_equal', 'all18_final_models_original_costs_reloaded',
            'two_complete_runs_equal', 'artifact_packaged_and_replayed')) or
        any(proof[k] is not False for k in ('primary_research_eligible', 'enhanced_control_enabled', 'holdout_used'))):
        raise ValueError('Pinned complete native failed V28 archive/certificate required')
    for name in ('cv_l2', 'cv_rank_l2'):
        gate = proof['qualification'][name]
        if (gate['research_eligible'] is not False or gate['online_promoted'] is not False or
            not gate['checks'] or any(type(v) is not bool for v in gate['checks'].values()) or
            all(gate['checks'].values())):
            raise ValueError('Legacy efficacy failures must remain failed, never promoted')


def selected_legacy_row(report, proof):
    protocol = read_json(HERE.parent/'cwm_v28/training_protocol.json')
    if (not exact_tree(report['protocol'], protocol) or
        not exact_tree(report['qualification'], proof['qualification']) or
        any(report['qualification'][name][key] is not False
            for name in ('cv_l2', 'cv_rank_l2') for key in ('research_eligible', 'online_promoted')) or
        any(report[k] is not False for k in ('primary_research_eligible', 'enhanced_control_enabled',
            'holdout_used', 'prior_gate_overridden', 'common_motion_in_response_optimizer'))):
        raise ValueError('Unchanged failed legacy training/qualification contract required')
    expected = {(m, s) for m in protocol['models'] for s in protocol['training']['seeds']}
    if (len(report['models']) != 9 or
        any(type(r['seed']) is not int for r in report['models']) or
        {(r['configuration'], r['seed']) for r in report['models']} != expected):
        raise ValueError('All nine legacy final80 models required')
    selected = {}
    for name in protocol['models']:
        rows = [r for r in report['models'] if r['configuration'] == name]
        if any(type(r['development']['group_equal_ade_m']) not in (int, float) or
               not np.isfinite(r['development']['group_equal_ade_m']) for r in rows):
            raise ValueError('Finite legacy ADE for all three seeds required')
        chosen = sorted(rows, key=lambda r: (r['development']['group_equal_ade_m'], r['seed']))[1]
        if (type(report['selected_median_seeds'][name]) is not int or
            chosen['seed'] != report['selected_median_seeds'][name]):
            raise ValueError('Legacy final80 ADE-median selection differs; no gain-based switch')
        selected[name] = chosen
    row = selected['cv_l2']
    if proof['qualification']['cv_l2']['selected_seed'] != row['seed']:
        raise ValueError('Native legacy qualification seed differs')
    return row


def load_legacy_checkpoint(directory, report, row):
    """Cold model reload; prior pinned replay owns full training/Adam/RNG proof."""
    name, seed = row['configuration'], row['seed']
    if name != 'cv_l2':
        raise ValueError('Only the fixed failed L2 benchmark may be loaded')
    path = regular(directory, f'{name}_seed{seed}.pt')
    if sha(path) != row['checkpoint_sha256']:
        raise ValueError('Legacy selected checkpoint bytes differ')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    finite_state(checkpoint)
    if checkpoint['configuration'] != name or type(checkpoint['seed']) is not int or checkpoint['seed'] != seed:
        raise ValueError('Legacy selected checkpoint identity differs')
    for key in ('source_hashes', 'data_summary_sha256', 'data_audit_summary_sha256'):
        if not exact_tree(checkpoint[key], report[key]):
            raise ValueError('Legacy selected checkpoint provenance differs: '+key)
    for key, filename in (('population_weights_sha256', 'population_weights.json'),
                          ('ranking_cache_sha256', f'cv_seed{seed}_ranking_cache.npz')):
        if checkpoint[key] != sha(regular(directory, filename)):
            raise ValueError('Legacy full training weights/cache identity differs')
    with np.load(regular(directory, 'normalization.npz'), allow_pickle=False) as norm:
        if set(norm.files) != {'mean', 'scale'}:
            raise ValueError('Complete legacy TRAIN-only normalizer required')
        mean, scale = norm['mean'].copy(), norm['scale'].copy()
    for key, expected in (('normalizer_mean', mean), ('normalizer_scale', scale)):
        actual = checkpoint[key].numpy()
        if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
            raise ValueError('Legacy selected checkpoint/full TRAIN normalizer differs')
    history = read_json(regular(directory, f'{name}_seed{seed}_history.json'))
    if ([r['epoch'] for r in history] != list(range(1, 81)) or
        any(type(r['epoch']) is not int for r in history) or
        not np.isfinite([r['mean_batch_loss_terms'] for r in history]).all()):
        raise ValueError('Complete legacy final80 history required')
    with preserved_cpu_rng():
        model = model_from_checkpoint(checkpoint)
        if not exact_tree(model.state_dict(), checkpoint['model_state']):
            raise ValueError('Legacy cold state changed dtype/shape/bytes')
        stage = torch.load(regular(directory, f'cv_seed{seed}_motion_stage.pt'), map_location='cpu', weights_only=True)
        finite_state(stage)
        if not exact_tree(model.common.state_dict(), stage['model_state']):
            raise ValueError('Legacy frozen common differs from original motion stage')
        predictor = prefix.public_predictor(model, mean, scale)
    adapter = OptionalResponseAdapter(QualificationGate(mode='shadow'), lambda: predictor)
    adapter.predictor = predictor
    return adapter, {'configuration': name, 'selected_seed': seed, 'checkpoint_sha256': row['checkpoint_sha256'],
        'model_state_exact': True, 'frozen_common_stage_exact': True, 'normalizer_exact': True,
        'model_cold_reloaded': True, 'optimizer_loaded_here': False, 'optimizer_updated_here': False,
        'full_training_proof_requires_pinned_native_archive_certificate': True,
        'research_eligible': False, 'deployment_control_eligible': False, 'matched_prefix_control': False}


def bound_manifest(archive_path, extracted):
    with zipfile.ZipFile(archive_path) as archive:
        manifest = decode_json(archive.read(MANIFEST))
    if not exact_tree(read_json(regular(extracted, MANIFEST)), manifest):
        raise ValueError('Extracted manifest differs from the bound original archive')
    return manifest


def _load_legacy(artifact, certificate, cache, plan):
    certificate = Path(certificate)
    digest = sha(certificate)
    if digest != plan['legacy_l2_control']['native_replay_certificate_sha256']:
        raise ValueError('Exact published native legacy certificate bytes required')
    proof = read_json(certificate)
    path = prefix.resolve_artifact(Path(artifact), cache/'assembled')
    if sha(path) != plan['legacy_l2_control']['archive_sha256']:
        raise ValueError('Exact published complete V28 v2 archive required')
    integrity = legacy_release.verify_archive(path)
    validate_legacy_certificate(proof, integrity, digest, plan)
    extracted = cache/'extracted'
    if not exact_tree(legacy_release.extract_archive(path, extracted), integrity):
        raise ValueError('Legacy archive changed before extraction')
    manifest = bound_manifest(path, extracted)
    legacy_release.check_extracted_evidence(extracted, manifest)
    roots = {n: extracted/n for n in ('data', 'audit', 'primary', 'repeated', 'reload')}
    members, prior = legacy_release.inventory(**roots)
    if set(manifest['members']) != set(members) | {'release_contract.json'}:
        raise ValueError('Complete legacy native evidence inventory required')
    if (not exact_tree(read_json(extracted/'release_contract.json'), legacy_release.release_contract(prior)) or
        not exact_tree(prior['qualification'], proof['qualification']) or prior['primary_research_eligible'] is not False):
        raise ValueError('Legacy native replay/contract qualification binding differs')
    report = read_json(roots['primary']/'summary.json')
    row = selected_legacy_row(report, proof)
    adapter, evidence = load_legacy_checkpoint(roots['primary'], report, row)
    if sha(certificate) != digest or not exact_tree(legacy_release.verify_archive(path), integrity):
        raise ValueError('Legacy artifact/certificate changed during preload')
    legacy_release.check_extracted_evidence(extracted, manifest)
    legacy_release.inventory(**roots)
    evidence.update(archive_sha256=integrity['sha256'], certificate_sha256=digest,
                    interpretation=plan['legacy_l2_control']['interpretation'])
    return adapter, evidence


def recheck_prefix(bundle, artifact, certificate, cache):
    """Legacy loading must not leave a previously checked prefix archive stale."""
    path = prefix.resolve_artifact(Path(artifact), cache/'assembled')
    integrity = prefix.verify_archive(path)
    if sha(Path(certificate)) != bundle.certificate_sha256 or integrity['sha256'] != bundle.archive_sha256:
        raise ValueError('Prefix artifact/certificate changed while loading legacy control')
    proof = read_json(Path(certificate))
    prefix.validate_certificate(proof, integrity)
    if proof['primary_research_eligible'] is not True:
        raise ValueError('Fixed prefix primary lost qualification')
    extracted = cache/'extracted'
    manifest = bound_manifest(path, extracted)
    prefix.release.check_extracted(extracted, manifest)
    roots = {n: extracted/n for n in ('data', 'audit', 'primary', 'repeated', 'reload')}
    members, prior = prefix.release.inventory(**roots)
    if (set(manifest['members']) != set(members) | {'release_contract.json'} or
        not exact_tree(read_json(extracted/'release_contract.json'), prefix.release.release_contract(prior)) or
        not exact_tree(prior['qualification'], proof['qualification']) or prior['primary_research_eligible'] is not True or
        sha(roots['reload']/'summary.json') != proof['prior_complete_reload_summary_sha256']):
        raise ValueError('Prefix full inventory/contract/reload qualification binding changed')


def load_benchmark(prefix_artifact=None, prefix_certificate=None, legacy_artifact=None,
                   legacy_certificate=None, cache=None, *, mode='off'):
    if mode in ('off', 'refused'):
        return BenchmarkScorers(status=mode)
    if mode != 'research_prepare':
        raise ValueError('Research scorer preparation only; no deployment or holdout mode')
    if cache is None:
        raise ValueError('Exclusive NEW benchmark cache required')
    cache = Path(cache)
    if cache.exists():
        raise ValueError('Exclusive NEW benchmark cache required')
    plan = prefix.load_plan()
    prefix_bundle = prefix.load_bundle(prefix_artifact, prefix_certificate, cache/'prefix', mode='research_prepare')
    if prefix_bundle.authorized is not True:
        return BenchmarkScorers(status='prefix_primary_failed_no_legacy_access')
    try:
        if (not isinstance(prefix_bundle, prefix.PrefixResearchBundle) or
            prefix_bundle.deployment_control_eligible is not False or
            set(prefix_bundle.scorers) != set(plan['new_model_configurations']) or prefix_bundle.proposer is None or
            set(prefix_bundle.selected_median_seeds) != set(plan['new_model_configurations'])):
            raise ValueError('Complete independently qualified fixed prefix preload required')
        if legacy_artifact is None or legacy_certificate is None:
            raise ValueError('Mandatory legacy comparator archive/certificate required; no control omission')
        adapter, evidence = _load_legacy(legacy_artifact, legacy_certificate, cache/'legacy', plan)
        recheck_prefix(prefix_bundle, prefix_artifact, prefix_certificate, cache/'prefix')
        result = BenchmarkScorers(status='all_fixed_neural_scorers_preloaded_research_only',
            scorers={**prefix_bundle.scorers, 'cv_l2': adapter}, proposer=prefix_bundle.proposer,
            prefix_bundle=prefix_bundle, legacy_evidence=evidence,
            selected_median_seeds={**prefix_bundle.selected_median_seeds, 'cv_l2': evidence['selected_seed']})
        result.authorized = True  # Preload only; no native runner or control eligibility.
        return result
    except Exception:
        prefix_bundle.authorized, prefix_bundle.status = False, 'benchmark_preload_failed'
        raise
