"""Archive all V22 records; verification re-executes the full pinned diagnosis."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from decision_diagnostics import run, sha, validate_protocol, summarize, paired_summary


def evidence_names(protocol):
    return ['summary.json'] + [f'{name}_seed{seed}_{split}_{population}_decisions.json'
                              for name in protocol['models'] for seed in protocol['seeds']
                              for split in protocol['splits'] for population in protocol['candidate_populations']]


def verify_manifest(artifact):
    with zipfile.ZipFile(artifact) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate artifact members')
        manifest = json.loads(archive.read('ARTIFACT_MANIFEST.json'))
        if set(manifest) != set(names) - {'ARTIFACT_MANIFEST.json'}:
            raise ValueError('Incomplete artifact member coverage')
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('Changed artifact member')
    return {'sha256': sha(artifact), 'bytes': artifact.stat().st_size, 'members': len(manifest)}


def check_record(row, protocol):
    costs = {k: np.asarray(v, dtype=np.float64) for k, v in row['costs'].items()}
    if set(costs) != set(protocol['paths']) or set(row['metrics']) != set(costs):
        raise ValueError('Fixed full-cost diagnostic paths required')
    truth = costs['action_specific_truth']
    if truth.ndim != 1 or not len(truth) or any(c.shape != truth.shape or not np.isfinite(c).all() for c in costs.values()):
        raise ValueError('Finite equal complete costs required')
    for name, cost in costs.items():
        choice = int(np.argmin(cost))
        regret = float(truth[choice] - truth.min())
        expected = {'choice': choice, 'regret_in_diagnostic_cost': regret,
                    'realized_cost_at_choice': float(truth[choice]),
                    'realized_tie_optimal': bool(regret <= protocol['tie_absolute_cost_tolerance'])}
        if row['metrics'][name] != expected:
            raise ValueError('Saved full-cost metrics differ')
    repairs = row['diagnostics']['repairs']
    from decision_diagnostics import repair_contributions
    regret = lambda key: row['metrics'][key]['regret_in_diagnostic_cost']
    if repairs != repair_contributions(regret('model'), regret('reference_truth_plus_learned_response'),
                                      regret('motion_plus_exact_response'), regret('action_specific_truth')):
        raise ValueError('Saved repair attribution differs')


def audit(read):
    protocol = json.loads((ROOT / 'diagnostic_protocol.json').read_text(encoding='utf8'))
    validate_protocol(protocol)
    reports = [json.loads(read(stage + '/summary.json')) for stage in ('primary', 'repeated')]
    expected_sources = {p.name: sha(p) for p in (ROOT / 'decision_diagnostics.py', ROOT / 'diagnostic_protocol.json')}
    sets = []
    for stage, report in zip(('primary', 'repeated'), reports):
        if (report['status'] != 'posthoc_v21_decision_failure_diagnosis_not_new_validation'
                or report['protocol'] != protocol or report['source_hashes'] != expected_sources
                or report['source_artifact']['sha256'] != protocol['source_artifact_sha256']
                or not report['published_v21_full_audit_passed']
                or not report['published_v21_dev_decisions_exactly_reproduced']
                or report['complete_union_support'] != {'train': 833, 'development_validation': 272}
                or any(report[key] for key in ('enhanced_control_enabled', 'new_model_trained', 'holdout_used', 'prior_gates_overridden'))):
            raise ValueError('Pinned descriptive diagnostic contract differs')
        if any(gate['research_eligible'] or gate['online_promoted'] for gate in report['prior_qualification'].values()):
            raise ValueError('Prior V21 failure must be retained')
        expected_models = {f'{name}_seed{seed}' for name in protocol['models'] for seed in protocol['seeds']}
        if set(report['models']) != expected_models:
            raise ValueError('Incomplete final model population')
        records = {}
        for identifier in sorted(expected_models):
            if set(report['models'][identifier]) != set(protocol['splits']):
                raise ValueError('Incomplete diagnostic splits')
            for split in protocol['splits']:
                if set(report['models'][identifier][split]) != set(protocol['candidate_populations']):
                    raise ValueError('Incomplete candidate populations')
                for population in protocol['candidate_populations']:
                    rows = json.loads(read(f'{stage}/{identifier}_{split}_{population}_decisions.json'))
                    if len(rows) != report['complete_union_support'][split] or len({r['group'] for r in rows}) != (24 if split == 'train' else 8):
                        raise ValueError('Same complete-union call/group support required')
                    if any(r['split'] != split or r['population'] != population for r in rows):
                        raise ValueError('Diagnostic support mislabeled')
                    for row in rows:
                        check_record(row, protocol)
                    if summarize(rows, protocol) != report['models'][identifier][split][population]:
                        raise ValueError('Saved group-equal diagnostic summary differs')
                    records[(split, population, identifier)] = rows
        measured = {}
        for split in protocol['splits']:
            measured[split] = {}
            for population in protocol['candidate_populations']:
                pairs = {}
                # V21's five matched comparisons, not newly selected best pairs.
                for changed, control in [('raw_point_mse', 'raw_call_mse'), ('raw_call_l2', 'raw_call_mse'),
                                         ('raw_point_l2', 'raw_point_mse'), ('raw_point_l2', 'raw_call_l2'),
                                         ('raw_point_l2', 'raw_call_mse')]:
                    pairs[f'{changed}_vs_{control}'] = {str(seed): paired_summary(
                        records[(split, population, f'{changed}_seed{seed}')],
                        records[(split, population, f'{control}_seed{seed}')], protocol) for seed in protocol['seeds']}
                measured[split][population] = pairs
        if measured != report['paired_factorial_decision_gains']:
            raise ValueError('Saved paired descriptive bootstrap differs')
        sets.append(records)
    for name in evidence_names(protocol):
        if read('primary/' + name) != read('repeated/' + name):
            raise ValueError('Independent full diagnostic record bytes differ')
    return {'status': 'passed_v22_record_audit_not_new_validation', 'models': 15, 'independent_runs': 2,
            'files_per_run': len(evidence_names(protocol)), 'records_per_run': 33150,
            'complete_union_support': reports[0]['complete_union_support'],
            'full_record_bytes_equal': True, 'source_artifact': reports[0]['source_artifact'],
            'enhanced_control_enabled': False, 'holdout_used': False, 'new_model_trained': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary', 'repeated', 'output', 'verify', 'recompute-output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'diagnostic_protocol.json').read_text(encoding='utf8'))
    if args.verify:
        if args.recompute_output is None:
            parser.error('Exclusive --recompute-output required for full source/model/engine verification')
        integrity = verify_manifest(args.verify)
        with zipfile.ZipFile(args.verify) as archive:
            result = audit(archive.read)
            if json.loads(archive.read('release_summary.json')) != result:
                raise ValueError('Archived record audit differs')
            for name in ('decision_diagnostics.py', 'diagnostic_protocol.json'):
                if archive.read('source/' + name) != (ROOT / name).read_bytes():
                    raise ValueError('Archived diagnostic source differs')
            # Full semantic verification, not just checksums of two copied summaries.
            run(ROOT.parent.parent / protocol['source_artifact'], args.recompute_output)
            for name in evidence_names(protocol):
                if archive.read('primary/' + name) != (args.recompute_output / name).read_bytes():
                    raise ValueError('Full recomputed source/model/oracle costs differ')
        print(json.dumps({'status': 'passed_v22_full_recomputed_release_audit', **integrity}), flush=True)
        return
    if any(getattr(args, name) is None for name in ('primary', 'repeated', 'output')):
        parser.error('Both completed diagnoses and exclusive output required')
    roots = {'primary': args.primary, 'repeated': args.repeated}
    names = evidence_names(protocol)
    for root in roots.values():
        if {p.name for p in root.iterdir() if p.is_file()} != set(names):
            raise ValueError('Incomplete or unexpected diagnostic evidence files')
    def read(name):
        stage, relative = name.split('/', 1)
        return (roots[stage] / relative).read_bytes()
    result = audit(read)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for stage, root in roots.items():
            for name in names:
                member = stage + '/' + name
                archive.write(root / name, member)
                manifest[member] = sha(root / name)
        for source in (ROOT / 'decision_diagnostics.py', ROOT / 'diagnostic_protocol.json', Path(__file__)):
            member = 'source/' + source.name
            archive.write(source, member)
            manifest[member] = sha(source)
        raw = json.dumps(result, indent=2, allow_nan=False).encode('utf8')
        archive.writestr('release_summary.json', raw)
        manifest['release_summary.json'] = hashlib.sha256(raw).hexdigest()
        archive.writestr('ARTIFACT_MANIFEST.json', json.dumps(manifest, indent=2))
    integrity = verify_manifest(args.output)
    with zipfile.ZipFile(args.output) as archive:
        if audit(archive.read) != result:
            raise ValueError('Postarchive diagnostic record audit differs')
    print(json.dumps({**result, **integrity, 'full_recompute_verify_still_required': True}), flush=True)


if __name__ == '__main__':
    main()
