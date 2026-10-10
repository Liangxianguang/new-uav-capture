"""Bind two independently repeated diagnostic reports to their source evidence."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def release(primary, repeated, destination):
    first, second = (p.read_bytes() for p in (primary, repeated))
    if first != second:
        raise ValueError('Independent diagnostic reports are not byte-identical')
    report = json.loads(first)
    protocol = json.loads((ROOT / 'diagnostic_protocol.json').read_text())
    if (report['protocol'] != protocol or report['status'] != 'descriptive_v19_failure_diagnosis_not_new_validation'
            or any(report[k] for k in ('enhanced_control_enabled', 'new_model_trained', 'holdout_used', 'prior_gates_overridden'))
            or not report['published_v19_independent_audit_passed']
            or not report['published_development_predictions_exactly_reproduced']):
        raise ValueError('Diagnostic/no-promotion contract changed')
    original = ROOT.parent.parent / protocol['source_artifact']
    if sha(original.read_bytes()) != protocol['source_artifact_sha256']:
        raise ValueError('Published V19 evidence changed')
    with zipfile.ZipFile(original) as archive:
        published = json.loads(archive.read('primary/summary.json'))
    if report['prior_qualification'] != published['qualification']:
        raise ValueError('Previously failed V19 qualification changed')
    for row in published['models']:
        identifier = f"{row['configuration']}_seed{row['seed']}"
        if report['models'][identifier]['development_validation']['v19_metrics'] != row['development']:
            raise ValueError('Published V19 development metrics changed')
    if any(sha((ROOT / name).read_bytes()) != digest for name, digest in report['source_hashes'].items()):
        raise ValueError('Diagnostic source changed after runs')
    expected = {f'{name}_seed{seed}' for name in protocol['models'] for seed in protocol['seeds']}
    if set(report['models']) != expected or any(set(value) != set(protocol['splits']) for value in report['models'].values()):
        raise ValueError('Incomplete diagnostic model/split population')
    manifest = {'status': 'two_full_independent_diagnostic_runs_byte_equal', 'summary_sha256': sha(first),
                'source_artifact_sha256': protocol['source_artifact_sha256'],
                'models': 9, 'splits': protocol['splits'], 'source_hashes': report['source_hashes'],
                'interpretation': protocol['scope'], 'enhanced_control_enabled': False, 'holdout_used': False,
                'prior_gates_overridden': False}
    members = {'primary/summary.json': first, 'repeated/summary.json': second,
               'repeat_manifest.json': json.dumps(manifest, indent=2).encode()}
    for p in sorted(ROOT.glob('*.py')) + [ROOT / 'diagnostic_protocol.json', ROOT / 'README.md']:
        members['source/' + p.name] = p.read_bytes()
    members['ARTIFACT_MANIFEST.json'] = json.dumps({n: sha(v) for n, v in members.items()}, indent=2).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with zipfile.ZipFile(destination) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or any(archive.read(n) != v for n, v in members.items()):
            raise ValueError('Archived diagnostic evidence differs')
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    with (reports / 'summary.json').open('xb') as file:
        file.write(first)
    with (reports / 'release_manifest.json').open('x') as file:
        json.dump({**manifest, 'artifact_sha256': sha(destination.read_bytes()), 'artifact_bytes': destination.stat().st_size}, file, indent=2)
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--repeated', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    release(args.primary, args.repeated, args.output)


if __name__ == '__main__':
    main()
