"""Actual ALL1280 original episodes in BOTH V42 off/refusal modes.

Bind complete V30 archive and native certificate, then reuse unchanged full
native auditor with the distinct prefix entry. No model, newscene or holdout.
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), *[str(HERE.parent/n) for n in ('cwm_v42', 'cwm_v41', 'cwm_v30', 'cwm_v38')]]
from prefix_entry import PrefixSequentialEntry
from paired_prefix import published_sources, validate_output
from benchmark_loader import bound_manifest
import audit_full_baseline as audit
import package_full_baseline as original_release
from archive_io import read_json, write_json, sha, regular, MANIFEST
from repeatability_identity import exact_tree

ARCHIVE_SHA = 'a6995520623e2fe311f06777ca97bae0d7a763eec6fd0cd1a821d863a28173e3'
CERTIFICATE_SHA = '8179de04eae22b3a7d8e0c5ad290e2f4c8a15dfba7205d56281b7cd340728983'
ENTRY_SOURCES = ('experiments/cwm_v42/prefix_entry.py', 'experiments/cwm_v42/prefix_selector.py',
                 'experiments/cwm_v43/replay_prefix_fallback.py')


class FallbackOnlyEntry(PrefixSequentialEntry):
    def configure(self, mode='off', **kwargs):
        if mode not in ('off', 'refusal') or kwargs:
            raise ValueError('Original off/refusal only; no optional model or research configuration')
        super().configure(mode)


def validate_certificate(proof, integrity):
    if (proof['status'] != 'complete_original_population_archive_independently_replayed' or
        not exact_tree(proof['artifact'], integrity) or integrity['sha256'] != ARCHIVE_SHA or
        type(proof['episodes_each_mode']) is not int or proof['episodes_each_mode'] != 1280 or
        proof['modes'] != ['off', 'refusal'] or
        any(proof[k] is not True for k in ('all_full_original_trajectories_plans_cbf_commands_equal',
            'all_nontiming_native_episode_step_diagnostics_equal', 'artifact_packaged_and_replayed')) or
        any(proof[k] is not False for k in ('enhanced_control_enabled', 'holdout_used', 'new_model_trained',
            'active_model_faults_exercised', 'latency_qualified'))):
        raise ValueError('Pinned complete original native1280 off/refusal certificate required')


def check_reference(archive, certificate, extracted):
    archive, certificate, extracted = Path(archive), Path(certificate), Path(extracted)
    if sha(certificate) != CERTIFICATE_SHA or sha(archive) != ARCHIVE_SHA:
        raise ValueError('Exact published original archive/certificate bytes required')
    integrity = original_release.verify_archive(archive)
    proof = read_json(certificate); validate_certificate(proof, integrity)
    manifest = bound_manifest(archive, extracted)
    for name, row in manifest['members'].items():
        path = regular(extracted, name)
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Complete extracted original evidence differs: '+name)
    members, baseline, prior = original_release.inventory(extracted/'baseline', extracted/'audit')
    if (set(manifest['members']) != set(members) or
        sha(extracted/'baseline/summary.json') != proof['baseline_summary_sha256'] or
        sha(extracted/'audit/summary.json') != proof['prior_full_audit_summary_sha256'] or
        not exact_tree(baseline['original8Level_equal_macro'], proof['original8Level_equal_macro'])):
        raise ValueError('Complete extracted original inventory/native certificate binding differs')
    return integrity, proof


def validate_result(reference, output):
    capsule = ROOT/'experiments/cwm_v1/baseline/capsule.zip'
    _, report, _, _ = original_release.audit_inventory(reference/'baseline', output/'native', capsule)
    if not set(ENTRY_SOURCES) <= set(report['source_hashes']):
        raise ValueError('Native auditor did not include actual V42/V43 used source closure')
    for name in ENTRY_SOURCES:
        if (sha(ROOT/name) != report['source_hashes'][name] or
            sha(regular(output/'native', 'source/'+name)) != report['source_hashes'][name]):
            raise ValueError('Actual prefix fallback entry bytes changed during full replay')
    return report


def run(archive, certificate, reference, output):
    output = validate_output(output, (archive, certificate, reference))
    prereg = published_sources()  # Includes imported V43 source before native outputs.
    integrity, proof = check_reference(archive, certificate, reference)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preregistration.json', prereg)
    write_json(output/'reference_binding.json', {'artifact': integrity, 'certificate_sha256': CERTIFICATE_SHA,
        'native_prefix_fallback_replay_completed': False, 'enhanced_control_enabled': False, 'holdout_used': False})
    parent_entry = audit.SequentialResearchEntry
    try:
        audit.SequentialResearchEntry = FallbackOnlyEntry
        audit.run(Path(reference)/'baseline', output/'native')
    finally:
        audit.SequentialResearchEntry = parent_entry
    result = validate_result(Path(reference), output)
    after, _ = check_reference(archive, certificate, reference)
    if not exact_tree(after, integrity):
        raise ValueError('Complete original archive changed during native replay')
    write_json(output/'summary.json', {'status': 'complete_native_prefix_entry_original1280_off_refusal_replayed',
        'preregistration': prereg, 'original_archive': integrity, 'original_certificate_sha256': CERTIFICATE_SHA,
        'native_audit_summary_sha256': sha(output/'native/summary.json'),
        'native_used_source_hashes': result['source_hashes'], 'modes': result['modes'],
        'original8Level_equal_macro': proof['original8Level_equal_macro'],
        'all1280_each_mode_trajectories_plans_CBF_commands_equal': True,
        'all_nontiming_episode_step_diagnostics_equal': True,
        'new_model_trained': False, 'enhanced_control_enabled': False, 'holdout_used': False,
        'active_model_faults_exercised': False, 'latency_qualified': False,
        'artifact_packaged_and_replayed': False,
        'scope': 'Actual V42 entry in off/refusal EACH1280 original development episodes. Complete native numeric and '
                 'non-timing replay only; not active model faults, causal efficacy, newscene/holdout, latency, '
                 'cross-platform or formal safety qualification. Complete new proof packaging/publication still required.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'certificate', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    existed = args.output.exists()
    try:
        run(args.archive, args.certificate, args.reference, args.output)
    except Exception as error:
        if not existed and args.output.is_dir() and not (args.output/'summary.json').exists():
            write_json(args.output/'failure.json', {'status': 'incomplete_prefix_native_fallback_replay_not_qualification',
                'exception_type': type(error).__name__, 'exception': str(error)[:300],
                'enhanced_control_enabled': False, 'holdout_used': False})
        raise


if __name__ == '__main__':
    main()
