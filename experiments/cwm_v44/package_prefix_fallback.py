"""Archive ALL V43 evidence and independently rerun the actual V42 fallback entry.

Not a causal efficacy, active-model fault, holdout or latency qualification.
Requires matching Git history/source and original Python runtime. New outputs
only. The pinned complete original ZIP/certificate are included without edits.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), *[str(HERE.parent/n) for n in ('cwm_v43', 'cwm_v38', 'cwm_v21')]]
import replay_prefix_fallback as native
from archive_io import (MANIFEST, decode_json, read_json, write_json, sha,
                        safe_name, regular, stream_digest)
from repeatability_identity import exact_tree
from artifact_transport import resolve_artifact, split_archive

SCHEMA = 'cwm_v44.complete_prefix_entry_original_fallback_archive.v1'
STATUS = 'complete_native_prefix_entry_original1280_off_refusal_replayed'
ENTRY = 'experiments/cwm_v43/replay_prefix_fallback.py'
ORIGINAL = 'dependencies/original.zip'
CERTIFICATE = 'dependencies/original_certificate.json'
FALSE_FLAGS = ('new_model_trained', 'enhanced_control_enabled', 'holdout_used',
               'active_model_faults_exercised', 'latency_qualified', 'artifact_packaged_and_replayed')


def equal(left, right, label):
    if not exact_tree(left, right):
        raise ValueError('Exact typed complete evidence differs: '+label)


def validate_wrapper(folder, integrity, certificate, audit):
    report = read_json(regular(folder, 'summary.json'))
    prereg = read_json(regular(folder, 'preregistration.json'))
    if (set(audit['modes']) != {'off', 'refusal'} or
        any(type(row['episodes']) is not int or row['episodes'] != 1280 for row in audit['modes'].values())):
        raise ValueError('Complete integer1280 native episodes in BOTH modes required')
    if (report['status'] != STATUS or any(report[k] is not False for k in FALSE_FLAGS) or
        any(report[k] is not True for k in ('all1280_each_mode_trajectories_plans_CBF_commands_equal',
                                          'all_nontiming_episode_step_diagnostics_equal'))):
        raise ValueError('Complete actual prefix-entry native fallback proof required')
    equal(report['original_archive'], integrity, 'original archive')
    equal(report['original8Level_equal_macro'], certificate['original8Level_equal_macro'], 'original macro')
    equal(report['modes'], audit['modes'], 'all native mode evidence')
    equal(report['native_used_source_hashes'], audit['source_hashes'], 'actual native source closure')
    equal(report['preregistration'], prereg, 'published preregistration')
    if (report['original_certificate_sha256'] != native.CERTIFICATE_SHA or
        report['native_audit_summary_sha256'] != sha(regular(folder, 'native/summary.json')) or
        prereg.get('exact_pushed_sources_verified') is not True or
        prereg.get('remote') != 'https://github.com/Liangxianguang/new-uav-capture.git' or
        prereg.get('branch') != 'refs/heads/causal-world-model-v1-20261009' or
        not isinstance(prereg.get('commit'), str) or len(prereg['commit']) != 40 or
        any(c not in '0123456789abcdef' for c in prereg['commit'])):
        raise ValueError('Actual published native prefix evidence identity differs')
    equal(read_json(regular(folder, 'reference_binding.json')), {'artifact': integrity,
          'certificate_sha256': native.CERTIFICATE_SHA, 'native_prefix_fallback_replay_completed': False,
          'enhanced_control_enabled': False, 'holdout_used': False}, 'pre-run reference binding')
    if not set(native.ENTRY_SOURCES) <= set(audit['source_hashes']):
        raise ValueError('Actual V42/V43 used source closure missing')
    for name, digest in audit['source_hashes'].items():
        raw = subprocess.check_output(['git', 'show', prereg['commit']+':'+safe_name(name)], cwd=ROOT)
        if (hashlib.sha256(raw).hexdigest() != digest or sha(regular(ROOT, name)) != digest or
            sha(regular(folder/'native', 'source/'+name)) != digest):
            raise ValueError('Native source differs from retained published commit: '+name)
    return report


def inventory(archive, certificate, reference, proof):
    # Partial live jobs cannot authorize packaging or an expensive reference audit.
    regular(proof, 'summary.json')
    regular(proof, 'native/summary.json')
    integrity, original = native.check_reference(archive, certificate, reference)
    _, audit, _, hashes = native.original_release.audit_inventory(
        reference/'baseline', proof/'native', ROOT/'experiments/cwm_v1/baseline/capsule.zip')
    report = validate_wrapper(proof, integrity, original, audit)
    members = {'proof/native/'+name: regular(proof/'native', name) for name in hashes}
    for name in ('summary.json', 'preregistration.json', 'reference_binding.json'):
        members['proof/'+name] = regular(proof, name)
    members[ORIGINAL] = archive
    members[CERTIFICATE] = certificate
    sources = set(audit['source_hashes']) | {
        'experiments/cwm_v38/archive_io.py', 'experiments/cwm_v21/artifact_transport.py',
        *[p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py')]}
    members.update({'verification_source/'+name: regular(ROOT, name) for name in sources})
    return members, report


def contract(report):
    return {'schema': SCHEMA, 'prior_complete_native_proof': report,
        'episodes_each_mode': 1280, 'modes': ['off', 'refusal'],
        'independent_actual_V42_V43_native_replay_required': True,
        'enhanced_control_enabled': False, 'holdout_used': False,
        'scope': 'Complete original development population with actual V42 entry off/refusal. '
                 'No active causal efficacy/faults, newscene/holdout, latency, cross-platform or formal safety claim. '
                 'Matching Git history/source and original Python environment required.'}


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [safe_name(i.filename) for i in infos]
        if (MANIFEST not in names or len({n.casefold() for n in names}) != len(names) or
            any(i.is_dir() or (i.external_attr >> 16)&0o170000 not in (0, 0o100000) for i in infos)):
            raise ValueError('Nonregular or duplicate complete fallback ZIP inventory')
        manifest = decode_json(archive.read(MANIFEST))
        if (set(manifest) != {'schema', 'members'} or manifest['schema'] != SCHEMA or
            set(manifest['members']) != set(names)-{MANIFEST}):
            raise ValueError('Exact complete prefix fallback manifest differs')
        for info in infos:
            if info.filename == MANIFEST:
                continue
            row = manifest['members'][info.filename]
            if (set(row) != {'bytes', 'sha256'} or type(row['bytes']) is not int or
                row['bytes'] != info.file_size):
                raise ValueError('Fallback evidence length differs')
            with archive.open(info) as stream:
                if stream_digest(stream) != row['sha256']:
                    raise ValueError('Fallback evidence bytes differ: '+info.filename)
    return {'sha256': sha(path), 'bytes': path.stat().st_size, 'members': len(names)-1}


def extract(path, destination, verifier=verify_archive):
    if destination.exists():
        raise ValueError('Exclusive NEW extraction directory required')
    before = verifier(path)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = destination.joinpath(*PurePosixPath(safe_name(info.filename)).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open('xb') as output:
                for block in iter(lambda: source.read(1 << 20), b''):
                    output.write(block)
    equal(verifier(path), before, 'archive after extraction')
    return before


def check_extracted(extracted, manifest):
    for name, row in manifest['members'].items():
        path = regular(extracted, name)
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Complete extracted evidence changed: '+name)
        if name.startswith('verification_source/'):
            if sha(regular(ROOT, name.split('/', 1)[1])) != row['sha256']:
                raise ValueError('Matching checkout source required: '+name)


def compare_native(reference, prior, fresh, integrity, certificate):
    new_audit = native.validate_result(reference, fresh)
    new = validate_wrapper(fresh, integrity, certificate, new_audit)
    old_audit = native.validate_result(reference, prior)
    old = validate_wrapper(prior, integrity, certificate, old_audit)
    native.original_release.compare_fresh(reference/'baseline', prior/'native', fresh/'native')
    # New preregistration commit and retained nondeterministic timings/ZIP hashes
    # can differ. BOTH complete evidence sets and every source are checked above.
    excluded = {'preregistration', 'native_audit_summary_sha256', 'modes', 'native_used_source_hashes'}
    equal({k: v for k, v in old.items() if k not in excluded},
          {k: v for k, v in new.items() if k not in excluded}, 'complete actual prefix-entry replay scope')
    for name in native.ENTRY_SOURCES:
        equal(old['native_used_source_hashes'][name], new['native_used_source_hashes'][name], 'actual entry')
    return new


def run_child(archive, certificate, reference, output, logs):
    command = [sys.executable, str(regular(ROOT, ENTRY)), '--archive', str(archive),
               '--certificate', str(certificate), '--reference', str(reference), '--output', str(output)]
    with (logs/'stdout.log').open('x', encoding='utf8') as stdout, \
         (logs/'stderr.log').open('x', encoding='utf8') as stderr:
        print(json.dumps({'status': 'actual_extracted_prefix_fallback_replay_started'}), flush=True)
        subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)


def replay_archive(artifact, output, input_recheck=None):
    output = native.validate_output(output, (artifact,))
    output.mkdir(parents=True, exist_ok=False)
    path = resolve_artifact(artifact, output/'transport')
    extracted = output/'e'
    integrity = extract(path, extracted)
    manifest = read_json(extracted/MANIFEST)
    check_extracted(extracted, manifest)
    original, certificate = extracted/ORIGINAL, extracted/CERTIFICATE
    reference = output/'reference'
    extract(original, reference, native.original_release.verify_archive)
    prior = extracted/'proof'
    members, previous = inventory(original, certificate, reference, prior)
    if set(manifest['members']) != set(members) | {'release_contract.json'}:
        raise ValueError('Omitted/added complete native fallback evidence')
    equal(read_json(extracted/'release_contract.json'), contract(previous), 'release contract')
    fresh = output/'fresh'
    run_child(original, certificate, reference, fresh, output)
    compare_native(reference, prior, fresh, previous['original_archive'], read_json(certificate))
    equal(verify_archive(path), integrity, 'full archive after native replay')
    check_extracted(extracted, manifest)
    inventory(original, certificate, reference, prior)
    if input_recheck is not None:
        input_recheck()
    result = {'status': 'complete_prefix_entry_original_fallback_archive_independently_replayed',
        'artifact': integrity, 'episodes_each_mode': 1280, 'modes': ['off', 'refusal'],
        'actual_V42_V43_entry_independently_rerun': True,
        'all_original_trajectories_plans_CBF_commands_equal': True,
        'all_nontiming_episode_step_diagnostics_equal': True,
        'original8Level_equal_macro': previous['original8Level_equal_macro'],
        'prior_prefix_proof_summary_sha256': sha(prior/'summary.json'),
        'fresh_prefix_proof_summary_sha256': sha(fresh/'summary.json'),
        'original_archive_sha256': native.ARCHIVE_SHA, 'original_certificate_sha256': native.CERTIFICATE_SHA,
        'artifact_packaged_and_replayed': True,
        **{k: False for k in FALSE_FLAGS if k != 'artifact_packaged_and_replayed'},
        'scope': contract(previous)['scope']}
    write_json(output/'summary.json', result)
    return result


def validate_outputs(outputs, inputs):
    resolved = [p.resolve() for p in outputs]
    if any(p.exists() for p in outputs):
        raise ValueError('Exclusive NEW release outputs required')
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(resolved) for b in resolved[i+1:]):
        raise ValueError('Distinct nonoverlapping release outputs required')
    for output in resolved:
        if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in inputs):
            raise ValueError('Release outputs cannot overlap complete input evidence')
    for output in outputs[:2]:
        native.validate_output(output, inputs)
    if len(outputs) == 3:
        parts = outputs[2]
        prefix = parts.name.removesuffix('.parts.json')
        if not parts.name.endswith('.parts.json') or any(parts.parent.glob(prefix+'.part[0-9]*')):
            raise ValueError('Exclusive canonical lossless transport outputs required')


def package(archive, certificate, reference, proof, artifact, replay_output, parts_output=None):
    validate_outputs([artifact, replay_output, *([parts_output] if parts_output is not None else [])],
                     (archive, certificate, reference, proof))
    native.published_sources()  # Exact published packaging/replay source before outputs.
    members, report = inventory(archive, certificate, reference, proof)
    expected = {n: {'bytes': p.stat().st_size, 'sha256': sha(p)} for n, p in members.items()}
    raw = json.dumps(contract(report), indent=2, allow_nan=False).encode('utf8')
    expected['release_contract.json'] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as out:
        for name, path in sorted(members.items()):
            out.write(path, safe_name(name))
        out.writestr('release_contract.json', raw)
        out.writestr(MANIFEST, json.dumps({'schema': SCHEMA, 'members': expected}, indent=2, allow_nan=False))
    # Recheck original inputs BEFORE issuing the native replay certificate too.
    result = replay_archive(artifact, replay_output,
                            lambda: inventory(archive, certificate, reference, proof))
    if parts_output is not None:
        split_archive(artifact, parts_output)
        joined = resolve_artifact(parts_output, replay_output/'transport_roundtrip')
        if sha(joined) != result['artifact']['sha256']:
            raise ValueError('Lossless published transport differs from native replay archive')
        write_json(replay_output/'transport_verified.json', {'parts_manifest_sha256': sha(parts_output),
            'full_archive_sha256': sha(joined), 'transport_only_not_second_native_replay': True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    for name in ('archive', 'certificate', 'reference', 'proof', 'artifact', 'parts-output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--replay-output', type=Path, required=True)
    args = parser.parse_args()
    fields = ('archive', 'certificate', 'reference', 'proof', 'artifact', 'parts_output')
    existed = args.replay_output.exists()
    try:
        if args.verify is not None:
            if any(getattr(args, k) is not None for k in fields):
                parser.error('Verification cannot also package')
            result = replay_archive(args.verify, args.replay_output)
        else:
            if any(getattr(args, k) is None for k in fields[:-1]):
                parser.error('Complete original archive/certificate/reference and V43 proof required')
            result = package(*[getattr(args, k) for k in fields[:-1]], args.replay_output, args.parts_output)
    except Exception as error:
        if not existed and args.replay_output.is_dir() and not (args.replay_output/'summary.json').exists():
            write_json(args.replay_output/'failure.json', {'status': 'incomplete_prefix_fallback_archive_not_qualification',
                'exception_type': type(error).__name__, 'exception': str(error)[:300],
                'enhanced_control_enabled': False, 'holdout_used': False})
        raise
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
