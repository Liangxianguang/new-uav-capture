"""Preload V38-bound fixed V36 scorers, never actions or deployment.

Off/refusal reads nothing. Failed primary reconstructs no neural checkpoint.
Full archive evidence is verified before and after loading outside planner time.
This is not the V39 eight-method runner: legacy L2 still needs a separate loader.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent/n) for n in ('cwm_v38', 'cwm_v37', 'cwm_v36', 'cwm_v39', 'cwm_v25', 'cwm_v26')]
import package_streamed_release as release
from archive_io import sha, regular, read_json, verify_archive, extract_archive, MANIFEST
from artifact_transport import resolve_artifact
from checkpoint_reload import reconstruct
from contract import load_contract
from development_statistics import load_plan
from repeatability_identity import exact_tree
from qualification_gate import OptionalResponseAdapter, QualificationGate, preserved_cpu_rng

PRIMARY = 'cv_prefix_rank_l2'


@dataclass
class PrefixResearchBundle:
    authorized: bool = False
    status: str = 'off'
    scorers: dict = field(default_factory=dict)
    proposer: object = None
    selected_median_seeds: dict = field(default_factory=dict)
    restoration_reports: dict = field(default_factory=dict)
    archive_sha256: str | None = None
    certificate_sha256: str | None = None
    deployment_control_eligible: bool = False
    legacy_l2_control_loaded: bool = False
    full_v39_runner_ready: bool = False


def validate_certificate(proof, integrity):
    if (proof['status'] != 'complete_streamed_prefix_archive_independently_replayed' or
        not exact_tree(proof['artifact'], integrity) or
        any(proof[k] is not True for k in ('all512_original_trajectories_plans_CBF_commands_replayed',
            'all_public_contexts_augmented_physical_branches_original_costs_replayed',
            'all18_final_models_optimizer_RNG_predictions_original_costs_gates_reloaded',
            'two_complete_fixed_runs_equal', 'artifact_packaged_and_replayed')) or
        any(proof[k] is not False for k in ('enhanced_control_enabled', 'holdout_used', 'new_training_performed',
            'closed_loop_capture_safety_latency_qualified', 'active_model_fault_fallback_qualified')) or
        type(proof['primary_research_eligible']) is not bool):
        raise ValueError('Complete unchanged artifact-bound native V38 replay certificate required')
    gate = proof['qualification'][PRIMARY]
    if gate['research_eligible'] is not proof['primary_research_eligible'] or gate['online_promoted'] is not False:
        raise ValueError('Fixed prefix primary qualification differs')
    if proof['primary_research_eligible'] and (not gate['checks'] or any(v is not True for v in gate['checks'].values())):
        raise ValueError('Every strict prefix primary offline gate must pass')


def selected_rows(report):
    contract, plan = load_contract()
    release.require_equal(report['v34_plan'], plan, 'fixed model plan')
    release.require_equal(report['v36_contract'], contract, 'fixed training contract')
    expected = {(m, s) for m in plan['models'] for s in plan['training']['seeds']}
    if len(report['models']) != 9 or {(r['configuration'], r['seed']) for r in report['models']} != expected:
        raise ValueError('All nine fixed final80 models required')
    result = {}
    for name in plan['models']:
        rows = [r for r in report['models'] if r['configuration'] == name]
        if any(type(r['development']['group_equal_ade_m']) not in (int, float) or
               not np.isfinite(r['development']['group_equal_ade_m']) for r in rows):
            raise ValueError('Finite BASE ADE required for all fixed seeds')
        row = sorted(rows, key=lambda r: (r['development']['group_equal_ade_m'], r['seed']))[1]
        if type(report['selected_median_seeds'][name]) is not int or row['seed'] != report['selected_median_seeds'][name]:
            raise ValueError('Fixed final BASE ADE-median seed differs; never choose best gain')
        result[name] = row
    return result


def public_predictor(model, mean, scale):
    if (mean.shape != (1, 1, 252) or scale.shape != mean.shape or mean.dtype != np.float64 or
        scale.dtype != np.float64 or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale < .01).any()):
        raise ValueError('Exact finite TRAIN-only float64 normalizer required')
    mean, scale = mean.copy().reshape(1, 252), scale.copy().reshape(1, 252)
    model.eval().requires_grad_(False)

    def predict(history, relative, proposed, anchor, backbone, cv):
        h, r, p, a, b, v = [np.asarray(x).copy() for x in (history, relative, proposed, anchor, backbone, cv)]
        if (h.shape != (8, 252) or r.shape != (4, 6) or p.ndim != 4 or p.shape[1:] != (8, 4, 3) or
            not 1 <= len(p) <= 24 or a.shape != (8, 4, 3) or b.shape != (8, 3) or v.shape != b.shape or
            b.dtype != np.float64 or v.dtype != np.float64 or
            any(not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all() for x in (h, r, p, a, b, v)) or
            max(np.linalg.norm(p, axis=-1).max(), np.linalg.norm(a, axis=-1).max()) > 5.+1e-8):
            raise ValueError('Original six public arrays and bounded BASE command library required')
        identity = (p == a[None]).reshape(len(p), 8, -1).all(2).cumprod(1).astype(bool)
        cast_identity = (p.astype(np.float32) == a.astype(np.float32)[None]).reshape(len(p), 8, -1).all(2).cumprod(1).astype(bool)
        if not np.array_equal(identity, cast_identity):
            raise ValueError('Online float32 conversion changes public prefix-anchor identity')
        n = len(p)
        tensors = (torch.as_tensor((h-mean)/scale, dtype=torch.float32)[None].repeat(n, 1, 1),
            torch.as_tensor(r, dtype=torch.float32)[None].repeat(n, 1, 1), torch.as_tensor(p, dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.float32)[None].repeat(n, 1, 1, 1),
            torch.as_tensor(b, dtype=torch.float64)[None].repeat(n, 1, 1),
            torch.as_tensor(v, dtype=torch.float64)[None].repeat(n, 1, 1))
        with preserved_cpu_rng(), torch.no_grad():
            prediction, _, response = model(*tensors)
            reference = model.common(*tensors)[0] if hasattr(model, 'common') else prediction
        return {k: x.numpy().copy() for k, x in (('prediction', prediction), ('reference', reference), ('response', response))}
    return predict


def load_selected(directory, report, rows, train_indices):
    scorers, restored = {}, {}
    with np.load(regular(directory, 'normalization.npz'), allow_pickle=False) as norm:
        mean, scale = norm['mean'].copy(), norm['scale'].copy()
    for name, row in rows.items():
        seed = row['seed']; path = regular(directory, f'{name}_seed{seed}.pt')
        if sha(path) != row['checkpoint_sha256']:
            raise ValueError('Individual selected checkpoint identity differs')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('source_hashes', 'data_summary_sha256', 'data_audit_summary_sha256'):
            release.require_equal(checkpoint[key], report[key], 'selected provenance '+key)
        for key, value in (('population_weights_sha256', sha(regular(directory, 'population_weights.json'))),
            ('canonical_cache_manifest_sha256', sha(regular(directory, f'cache{seed}/manifest.json')))):
            if checkpoint[key] != value: raise ValueError('Selected normalization/cache provenance differs')
        for key, expected in (('normalizer_mean', mean), ('normalizer_scale', scale)):
            actual = checkpoint[key].numpy()
            if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
                raise ValueError('Selected checkpoint/full TRAIN normalizer differs')
        model, evidence = reconstruct(checkpoint, name, seed, train_indices)
        stage = torch.load(regular(directory, f'cv_seed{seed}_motion_stage.pt'), map_location='cpu', weights_only=True)
        common = model.common if hasattr(model, 'common') else model
        release.require_equal(common.state_dict(), stage['model_state'], 'selected fixed common motion stage')
        predictor = public_predictor(model, mean, scale)
        adapter = OptionalResponseAdapter(QualificationGate(mode='shadow'), lambda p=predictor: p)
        adapter.predictor = predictor  # No checkpoint IO inside a timed planner.
        scorers[name], restored[name] = adapter, evidence
    return scorers, restored


def load_proposer(protocol):
    from checkpoint_provider import checkpoint_predictor
    import zipfile
    path = regular(ROOT, 'experiments/cwm_v26/artifacts/real_sequential_shadow_20261010.zip')
    if sha(path) != protocol['proposal_archive_sha256']:
        raise ValueError('Frozen shared public proposer differs')
    with zipfile.ZipFile(path) as archive, preserved_cpu_rng():
        predictor = checkpoint_predictor(archive.read('model/primary.pt'), protocol['proposal_checkpoint_sha256'])
    adapter = OptionalResponseAdapter(QualificationGate(mode='shadow'), lambda: predictor)
    adapter.predictor = predictor
    return adapter


def load_bundle(artifact=None, certificate=None, cache=None, *, mode='off'):
    if mode in ('off', 'refused'):
        return PrefixResearchBundle(status=mode)  # Do not even resolve supplied paths.
    if mode != 'research_prepare':
        raise ValueError('Explicit isolated research preload only; no deployment mode')
    load_plan()
    cache = Path(cache)
    if cache.exists(): raise ValueError('Exclusive NEW preload cache required')
    if artifact is None or certificate is None:
        raise ValueError('Complete artifact/certificate paths required')
    certificate = Path(certificate)
    certificate_digest = sha(certificate); proof = read_json(certificate)
    cache.mkdir(parents=True, exist_ok=False)
    path = resolve_artifact(Path(artifact), cache/'assembled')
    integrity = verify_archive(path); validate_certificate(proof, integrity)
    extracted = cache/'extracted'; extract_archive(path, extracted)
    manifest = read_json(extracted/MANIFEST); release.check_extracted(extracted, manifest)
    roots = {name: extracted/name for name in ('data', 'audit', 'primary', 'repeated', 'reload')}
    members, prior = release.inventory(**roots)
    if set(manifest['members']) != set(members) | {'release_contract.json'}:
        raise ValueError('Complete scientific archive evidence required')
    release.require_equal(read_json(extracted/'release_contract.json'), release.release_contract(prior), 'archive contract')
    release.require_equal(prior['qualification'], proof['qualification'], 'artifact/certificate qualification')
    if (prior['primary_research_eligible'] is not proof['primary_research_eligible'] or
        proof['prior_complete_reload_summary_sha256'] != sha(roots['reload']/'summary.json')):
        raise ValueError('Exact independent reload certificate binding differs')
    report = read_json(roots['primary']/'summary.json')
    rows = selected_rows(report)
    bundle = PrefixResearchBundle(status='offline_qualification_failed', archive_sha256=integrity['sha256'],
        certificate_sha256=certificate_digest)
    if proof['primary_research_eligible']:
        records = read_json(roots['data']/'records.json')
        supported = [r for r in records if 'arrays_path' in r]
        train_indices = [i for i, r in enumerate(supported) if r['split'] == 'train']
        with preserved_cpu_rng():
            scorers, restoration = load_selected(roots['primary'], report, rows, train_indices)
            proposer = load_proposer(read_json(roots['data']/'summary.json')['protocol'])
        bundle.scorers, bundle.proposer, bundle.restoration_reports = scorers, proposer, restoration
        bundle.selected_median_seeds = {name: row['seed'] for name, row in rows.items()}
        bundle.status = 'offline_qualified_prefix_scorers_preloaded_research_only'
    if sha(certificate) != certificate_digest or verify_archive(path) != integrity:
        raise ValueError('Archive/certificate changed during preload')
    release.check_extracted(extracted, manifest); release.inventory(**roots)
    # Authorization is assigned last; this does not grant a full runner or deploy.
    bundle.authorized = proof['primary_research_eligible']
    return bundle
