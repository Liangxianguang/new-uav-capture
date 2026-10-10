"""Preloaded, artifact-bound V28 models for RESEARCH evaluation, not deployment."""
import hashlib
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent / 'cwm_v28'), str(HERE.parent / 'cwm_v26'),
               str(HERE.parent / 'cwm_v25'), str(HERE.parent / 'cwm_v21')]
from package_ranking_release import verify_archive
from artifact_transport import resolve_artifact, file_digest
from qualification_gate import OptionalResponseAdapter, QualificationGate, preserved_cpu_rng


@dataclass
class ResearchBundle:
    authorized: bool = False
    status: str = 'offline_release_pending'
    proposer: object = None
    scorers: object = None
    archive_sha256: str | None = None
    certificate_sha256: str | None = None
    deployment_control_eligible: bool = False


def predictor_from_checkpoint(raw, row, report):
    import torch
    from release_ranking_training import model_from_checkpoint
    if hashlib.sha256(raw).hexdigest() != row['checkpoint_sha256']:
        raise ValueError('Selected final checkpoint identity differs')
    cp = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if (cp['seed'] != row['seed'] or cp['configuration'] != row['configuration'] or
        any(cp[k] != report[k] for k in ('source_hashes', 'data_summary_sha256', 'data_audit_summary_sha256'))):
        raise ValueError('Selected final checkpoint provenance differs')
    model = model_from_checkpoint(cp)
    mean, scale = (cp[key].numpy().reshape(1, 252) for key in ('normalizer_mean', 'normalizer_scale'))
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale < .01).any():
        raise ValueError('Train-only public normalizer differs')
    def predict(history, relative, proposed, anchor, backbone, cv):
        n = len(proposed)
        tensors = (torch.as_tensor((history-mean)/scale, dtype=torch.float32)[None].repeat(n, 1, 1),
            torch.as_tensor(relative, dtype=torch.float32)[None].repeat(n, 1, 1),
            torch.as_tensor(proposed, dtype=torch.float32),
            torch.as_tensor(anchor, dtype=torch.float32)[None].repeat(n, 1, 1, 1),
            torch.as_tensor(backbone, dtype=torch.float64)[None].repeat(n, 1, 1),
            torch.as_tensor(cv, dtype=torch.float64)[None].repeat(n, 1, 1))
        with torch.no_grad():
            prediction, _, response = model(*tensors)
            reference = model.common(*tensors)[0] if hasattr(model, 'common') else prediction
        return {k: v.numpy().copy() for k, v in (('prediction', prediction), ('reference', reference), ('response', response))}
    return predict


def load_bundle(artifact, certificate, cache):
    """Called BEFORE the original timed plan; no checkpoint IO inside selection.

    The certificate is local audited provenance, not a cryptographic signature.
    Qualified offline models may be tested, never deployed by this entry. Failed
    qualification refuses before any checkpoint reconstruction/forward.
    """
    proof = json.loads(Path(certificate).read_bytes())
    if (proof['status'] != 'complete_fixed_ranking_archive_independently_replayed' or
        any(proof[k] is not True for k in ('real_public_context_all_branch_replay_passed',
            'all64_original_trajectories_plans_commands_equal', 'all18_final_models_original_costs_reloaded',
            'two_complete_runs_equal', 'artifact_packaged_and_replayed')) or
        proof['enhanced_control_enabled'] is not False or proof['holdout_used'] is not False):
        raise ValueError('Complete real extracted-release certificate required')
    path = resolve_artifact(Path(artifact), Path(cache))
    integrity = verify_archive(path)
    if proof['artifact'] != integrity:
        raise ValueError('Certificate is bound to a different complete artifact')
    bundle = ResearchBundle(archive_sha256=integrity['sha256'], certificate_sha256=file_digest(Path(certificate)))
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('ARTIFACT_MANIFEST.json'))
        for name in manifest['members']:
            if name.startswith(('verification_source/', 'dependencies/')):
                relative = name.split('/', 1)[1]
                if file_digest(ROOT / relative) != manifest['members'][name]['sha256']:
                    raise ValueError('Matching release Git checkout/source required')
        report = json.loads(archive.read('primary/summary.json'))
        prior = json.loads(archive.read('reload/summary.json'))
        protocol = json.loads((HERE.parent / 'cwm_v28/training_protocol.json').read_bytes())
        if (report['protocol'] != protocol or report['qualification'] != proof['qualification'] or
            report['qualification'] != prior['qualification'] or
            report['primary_research_eligible'] != proof['primary_research_eligible'] or
            prior['primary_research_eligible'] != proof['primary_research_eligible'] or
            hashlib.sha256(archive.read('primary/summary.json')).hexdigest() != prior['primary_summary_sha256']):
            raise ValueError('Offline qualification/provenance differs')
        gate = report['qualification']['cv_rank_l2']
        if proof['primary_research_eligible'] is not True:
            bundle.status = 'offline_qualification_failed'
            return bundle
        if (gate['research_eligible'] is not True or gate['online_promoted'] is not False or
            not gate['checks'] or any(value is not True for value in gate['checks'].values()) or
            report['enhanced_control_enabled'] is not False or report['holdout_used'] is not False or
            report['prior_gate_overridden'] is not False):
            raise ValueError('Fixed primary qualified/offline/default-off contract differs')
        rows = report['models']
        expected = {(m, s) for m in protocol['models'] for s in protocol['training']['seeds']}
        if len(rows) != 9 or {(r['configuration'], r['seed']) for r in rows} != expected:
            raise ValueError('Complete fixed three-seed population required')
        scorers = {}
        for name in protocol['models']:
            chosen = sorted([r for r in rows if r['configuration'] == name],
                            key=lambda r: (r['development']['group_equal_ade_m'], r['seed']))[1]
            if chosen['seed'] != report['selected_median_seeds'][name]:
                raise ValueError('Fixed ADE-median seed differs')
            raw = archive.read(f"primary/{name}_seed{chosen['seed']}.pt")
            with preserved_cpu_rng():
                predictor = predictor_from_checkpoint(raw, chosen, report)
            scorers[name] = OptionalResponseAdapter(QualificationGate(mode='shadow'), lambda p=predictor: p)
            scorers[name].predictor = predictor  # Already reconstructed outside original timeout.
        from checkpoint_provider import checkpoint_predictor
        data_protocol = json.loads((HERE.parent / 'cwm_v28/data_protocol.json').read_bytes())
        teacher = ROOT / 'experiments/cwm_v26/artifacts/real_sequential_shadow_20261010.zip'
        if file_digest(teacher) != data_protocol['proposal_archive_sha256']:
            raise ValueError('Fixed common public proposer differs')
        with zipfile.ZipFile(teacher) as original:
            with preserved_cpu_rng():
                proposer = checkpoint_predictor(original.read('model/primary.pt'), data_protocol['proposal_checkpoint_sha256'])
        bundle.proposer = OptionalResponseAdapter(QualificationGate(mode='shadow'), lambda: proposer)
        bundle.proposer.predictor = proposer
        bundle.scorers, bundle.authorized, bundle.status = scorers, True, 'offline_qualified_research_only'
    return bundle
