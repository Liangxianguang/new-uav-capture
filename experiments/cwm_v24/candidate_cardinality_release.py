"""Archive both public-only diagnoses; verification fully reruns the diagnosis."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from candidate_cardinality import run, summarize_records


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def audit_records(read):
    protocol = json.loads((ROOT / 'diagnostic_protocol.json').read_text(encoding='utf8'))
    summaries, records_raw = [], []
    for stage in ('primary', 'repeated'):
        summary = json.loads(read(stage + '/summary.json'))
        raw = read(stage + '/records.json')
        rows = json.loads(raw)
        if (summary['status'] != 'posthoc_public_candidate_cardinality_finished_not_promoted' or
            summary['protocol'] != protocol or summary['data_archive_sha256'] != protocol['data_archive_sha256'] or
            summary['records_sha256'] != digest(raw) or summary['calls'] != 1536 or len(rows) != 1536 or
            any(summary[k] for k in ('enhanced_control_enabled', 'new_model_trained', 'holdout_used')) or
            not summary['full_original_data_public_cost_audit_passed'] or not summary['all_independent_original_candidate_bytes_equal']):
            raise ValueError('Full fixed original-only diagnosis status/provenance required')
        identities = [tuple(r[k] for k in ('episode_index', 'step', 'agent')) for r in rows]
        if (len(set(identities)) != len(rows) or
            {r['episode_index'] for r in rows} != set(range(993010, 993074)) or
            {r['step'] for r in rows} != {2, 3, 4, 5, 6, 8} or
            {r['agent'] for r in rows} != set(range(4))):
            raise ValueError('Complete original1,536call coverage required')
        for row in rows:
            reason = row['duplicate_reason']
            if (reason not in ('distinct', 'interceptor_perimeter_invariant', 'speed_clipping_induced_duplicate', 'other_public_geometry_duplicate') or
                row['raw_candidates'] != 2 or row['unique_candidates'] != (2 if reason == 'distinct' else 1) or
                not row['independent_original_candidate_bytes_equal'] or
                not isinstance(row['complete_union'], bool) or not isinstance(row['unclipped_diagnostic_pair_equal'], bool)):
                raise ValueError('Saved candidate diagnosis contract differs')
            if (reason == 'interceptor_perimeter_invariant') != (row['agent'] == row['public_interceptor']):
                raise ValueError('Saved public role invariance differs')
            if reason == 'speed_clipping_induced_duplicate' and row['unclipped_diagnostic_pair_equal']:
                raise ValueError('Saved unclipped duplicate diagnosis differs')
        measured = summarize_records(rows)
        if measured != summary['supports'] or measured['train']['all_collected']['calls'] != 1152 or measured['development_validation']['all_collected']['calls'] != 384:
            raise ValueError('Saved diagnosis support/group summaries differ')
        sources = ('candidate_cardinality.py', 'diagnostic_protocol.json')
        if set(summary['source_hashes']) != set(sources):
            raise ValueError('Diagnosis source population differs')
        for name in sources:
            source = read(stage + '/source/' + name)
            if source != (ROOT / name).read_bytes() or digest(source) != summary['source_hashes'][name]:
                raise ValueError('Diagnosis source/current bytes differ')
        summaries.append(summary)
        records_raw.append(raw)
    if summaries[0] != summaries[1] or records_raw[0] != records_raw[1] or read('primary/summary.json') != read('repeated/summary.json'):
        raise ValueError('Two independent complete diagnoses differ')
    return {'status': 'two_complete_saved_public_candidate_diagnoses_equal',
            'summary_sha256': digest(read('primary/summary.json')), 'records_sha256': digest(records_raw[0]),
            'calls_per_run': 1536, 'independent_runs': 2, 'supports': summaries[0]['supports'],
            'data_archive_sha256': protocol['data_archive_sha256'], 'enhanced_control_enabled': False,
            'holdout_used': False, 'scope': protocol['scope']}


def archive_integrity(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive members')
        manifest = json.loads(archive.read('ARTIFACT_MANIFEST.json'))
        expected = {f'{stage}/{name}' for stage in ('primary', 'repeated')
                    for name in ('summary.json', 'records.json', 'source/candidate_cardinality.py', 'source/diagnostic_protocol.json')}
        expected |= {'release_summary.json', 'verification_source/candidate_cardinality.py',
                     'verification_source/candidate_cardinality_release.py', 'verification_source/diagnostic_protocol.json'}
        if set(manifest) != expected or set(names) != expected | {'ARTIFACT_MANIFEST.json'}:
            raise ValueError('Incomplete or unexpected diagnostic archive members')
        if any(digest(archive.read(name)) != value for name, value in manifest.items()):
            raise ValueError('Diagnostic archive member digest differs')
    return {'sha256': digest(path.read_bytes()), 'bytes': path.stat().st_size, 'members': len(manifest)}


def verify_archive(path, artifact, recompute_output):
    integrity = archive_integrity(path)
    with zipfile.ZipFile(path) as archive:
        recorded = audit_records(archive.read)
        if json.loads(archive.read('release_summary.json')) != recorded:
            raise ValueError('Saved release summary differs')
        for name in ('candidate_cardinality.py', 'candidate_cardinality_release.py', 'diagnostic_protocol.json'):
            if archive.read('verification_source/' + name) != (ROOT / name).read_bytes():
                raise ValueError('Archived/current verification source differs')
        # Not checksum-only: independently execute all original data/cost checks
        # and every original command reconstruction in a NEW output directory.
        measured = run(artifact, recompute_output)
        if (measured != json.loads(archive.read('primary/summary.json')) or
            (recompute_output / 'summary.json').read_bytes() != archive.read('primary/summary.json') or
            (recompute_output / 'records.json').read_bytes() != archive.read('primary/records.json')):
            raise ValueError('Independent full original-engine diagnosis recomputation differs')
    return {'status': 'passed_independent_full_public_candidate_archive_recomputation',
            'artifact': integrity, 'data_archive_sha256': recorded['data_archive_sha256'],
            'all1536_records_and_summary_recomputed_exact': True,
            'full_original_data_public_cost_audit_reexecuted': True,
            'enhanced_control_enabled': False, 'holdout_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary', 'repeated', 'output', 'verify', 'artifact', 'recompute-output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.artifact is None or args.recompute_output is None:
            parser.error('Verification requires pinned DATA artifact and new recompute output')
        print(json.dumps(verify_archive(args.verify, args.artifact, args.recompute_output)), flush=True)
        return
    if any(getattr(args, key) is None for key in ('primary', 'repeated', 'output')):
        parser.error('Both complete diagnoses and exclusive archive output required')
    roots = {'primary': args.primary, 'repeated': args.repeated}
    def read(name):
        stage, relative = name.split('/', 1)
        return (roots[stage] / relative).read_bytes()
    report = audit_records(read)
    members = {f'{stage}/{name}': read(f'{stage}/{name}') for stage in roots
               for name in ('summary.json', 'records.json', 'source/candidate_cardinality.py', 'source/diagnostic_protocol.json')}
    for name in ('candidate_cardinality.py', 'candidate_cardinality_release.py', 'diagnostic_protocol.json'):
        members['verification_source/' + name] = (ROOT / name).read_bytes()
    members['release_summary.json'] = json.dumps(report, indent=2, allow_nan=False).encode('utf8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr('ARTIFACT_MANIFEST.json', json.dumps({name: digest(raw) for name, raw in members.items()}, indent=2))
    integrity = archive_integrity(args.output)
    with zipfile.ZipFile(args.output) as archive:
        if audit_records(archive.read) != report:
            raise ValueError('Archived saved diagnosis differs')
    print(json.dumps({'status': report['status'], 'artifact': integrity,
                      'independent_full_archive_recomputation': 'not_yet_executed'}), flush=True)


if __name__ == '__main__':
    main()
