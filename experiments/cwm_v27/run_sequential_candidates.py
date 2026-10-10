"""Reobserve real sequential public calls and expand only after original plan."""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v26'), str(HERE.parent/'cwm_v25'),
               str(HERE.parent/'cwm_v9'), str(HERE.parent/'cwm_v1')]
from bounded_candidates import expand_candidates, SOURCES, MAX_ROUNDS, MAX_CANDIDATES
from release_real_shadow import independent_public_replay, CHECKPOINT_SHA, equal_arrays, arrays
from checkpoint_provider import checkpoint_predictor
from qualification_gate import OptionalResponseAdapter, QualificationGate
from baseline import Baseline
from freeze_baseline import sha, verify
from local_shadow import restore_public_call


def run(output):
    protocol = json.loads((HERE/'protocol.json').read_text())
    if (protocol['maximum_unique_candidates'] != MAX_CANDIDATES or
        protocol['maximum_expansion_rounds'] != MAX_ROUNDS or protocol['proposal_sources'] != list(SOURCES) or
        any(protocol[k] for k in ('enhanced_control_enabled','new_model_trained','holdout_used','prior_gate_override_allowed'))):
        raise ValueError('Published noncontrolling proposal protocol differs')
    artifact = HERE.parent/'cwm_v26/artifacts/real_sequential_shadow_20261010.zip'
    capsule = HERE.parent/'cwm_v1/baseline/capsule.zip'
    if sha(artifact) != protocol['source_shadow_sha256'] or sha(capsule) != protocol['baseline_capsule_sha256']:
        raise ValueError('Pinned original/public shadow archive differs')
    output.mkdir(parents=True, exist_ok=False)
    base = Baseline(capsule, output/'restored')
    original_protocol = json.loads((HERE.parent/'cwm_v26/protocol.json').read_text())
    if original_protocol['record_ids'] != protocol['record_ids']:
        raise ValueError('Original sequential records differ')
    expected = independent_public_replay(base, original_protocol, output/'original_replay')
    records, saved_keys = [], []
    with zipfile.ZipFile(artifact) as archive:
        summary = json.loads(archive.read('primary/summary.json'))
        old_events = summary['modes']['real_shadow']['events']
        if len(expected) != len(old_events):
            raise ValueError('Sequential invocation population changed')
        raw = archive.read('model/primary.pt')
        if hashlib.sha256(raw).hexdigest() != protocol['checkpoint_sha256'] or protocol['checkpoint_sha256'] != CHECKPOINT_SHA:
            raise ValueError('Fixed failed checkpoint differs')
        predictor = checkpoint_predictor(raw, CHECKPOINT_SHA)
        adapters = {s: OptionalResponseAdapter(QualificationGate(mode='shadow', qualification={'research_eligible': False}),
                        lambda: predictor) for s in SOURCES}
        for index, ((identity, snapshot, values), event) in enumerate(zip(expected, old_events)):
            if any(event[k] != v for k, v in identity.items()):
                raise ValueError('Sequential call identity differs from audited V26')
            if values is None:
                if event['status'] != 'missing_received_peer_context':
                    raise ValueError('Original missing received context differs')
                records.append({**identity, 'status': 'missing_received_peer_context', 'control_eligible': False})
                continue
            if snapshot != json.loads(archive.read('primary/real_shadow/'+event['context_path'])):
                raise ValueError('Fresh original public snapshot differs')
            old = arrays(archive.read('primary/real_shadow/'+event['arrays_path']))
            if not equal_arrays(old, values, tuple(values)):
                raise ValueError('Fresh public history/peer plans differ')
            call = restore_public_call(snapshot, base.planner_config, base.distributed_config, values['backbone'])
            generated = np.stack(call['planner']._local_candidate_sequences(call['observation'],call['scenarios'],
                call['agent'],call['observation']['defender_positions'][call['agent']],call['known']))
            if generated.tobytes() != call['local_candidates'].tobytes():
                raise ValueError('Original candidate generator changed')
            off = expand_candidates({'local_candidates': call['local_candidates']}, object(), object(), object())
            if off['actions'].tobytes() != call['local_candidates'].tobytes():
                raise ValueError('Off changed original order/multiplicity')
            results = {}
            for source, adapter in adapters.items():
                result = expand_candidates(call, values['history'], base.evaluator.belief_reference, adapter,
                                          mode='shadow', source=source)
                if result['info']['status'] != 'shadow_candidates_ready':
                    raise ValueError('Real expansion failed: '+str(result['info']))
                fields = dict(zip(('history','relative','proposed','anchor','backbone','cv'),result['inputs']))
                fields.update(result['forecast']); fields['actions'] = result['actions']
                for key in ('history','relative','anchor','backbone','cv'):
                    if fields[key].tobytes() != values[key].tobytes():
                        raise ValueError('Expansion changed frozen original public context: '+key)
                for peer in set(range(4)) - {call['agent']}:
                    if any(row[:,peer].tobytes() != values['anchor'][:,peer].tobytes() for row in fields['proposed']):
                        raise ValueError('Expanded joint context changed received peer plan')
                path = output/'calls'/f'{index}_{source}.npz'
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, **fields)
                results[source] = {**result['info'], 'arrays_path': path.relative_to(output).as_posix()}
                saved_keys.append(results[source]['arrays_path'])
            records.append({**identity, 'status':'supported', 'actual_raw_count':len(call['local_candidates']),
                            'actual_unique_count':len(np.unique(call['local_candidates'], axis=0)), 'sources':results,
                            'control_eligible':False})
            if (index+1) % 64 == 0:
                print(json.dumps({'sequential_calls_processed':index+1}), flush=True)
    verify(base.root, base.capsule_manifest)
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
               and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments')} | {HERE/'protocol.json'}
    hashes = {}
    for source in sorted(sources):
        relative = source.relative_to(ROOT)
        target = output/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(source.read_bytes())
        hashes[relative.as_posix()] = sha(source)
    supported = [r for r in records if r['status'] == 'supported']
    counts = {s: {'minimum':min(r['sources'][s]['candidate_count'] for r in supported),
                  'maximum':max(r['sources'][s]['candidate_count'] for r in supported),
                  'calls_added_beyond_geometry':sum(r['sources'][s]['candidate_count'] > r['sources'][s]['seed_count'] for r in supported),
                  'total_added_beyond_geometry':sum(r['sources'][s]['candidate_count']-r['sources'][s]['seed_count'] for r in supported)} for s in SOURCES}
    report = {'status':'bounded_public_sequential_candidates_finished_not_efficacy', 'protocol':protocol,
              'source_hashes':hashes,'records':records, 'supported_calls':len(supported),
              'missing_peer_calls':len(records)-len(supported), 'counts':counts,
              'forecast_calls':{s:a.prediction_calls for s,a in adapters.items()},
              'enhanced_control_enabled':False,'holdout_used':False,'new_model_trained':False,
              'qualification':'V23 remains failed. New candidates are research outputs, never executed. No capture/ranking/latency benefit measured.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:report[k] for k in ('status','supported_calls','missing_peer_calls','counts')}),flush=True)
    return report


def validate_completed(report):
    protocol = json.loads((HERE/'protocol.json').read_text())
    if (report['protocol'] != protocol or report['status'] != 'bounded_public_sequential_candidates_finished_not_efficacy' or
        any(report[k] is not False for k in ('enhanced_control_enabled','new_model_trained','holdout_used'))):
        raise ValueError('Noncontrolling complete candidate replay contract differs')
    with zipfile.ZipFile(HERE.parent/'cwm_v26/artifacts/real_sequential_shadow_20261010.zip') as z:
        original = json.loads(z.read('primary/summary.json'))['modes']['real_shadow']['events']
    if len(report['records']) != len(original):
        raise ValueError('Incomplete sequential invocation population')
    supported, missing = 0,0
    for row,event in zip(report['records'],original):
        if row['control_eligible'] is not False or any(row[k] != event[k] for k in ('episode_index','step','agent','ordinal')):
            raise ValueError('Sequential identity/control differs')
        if event['status'] == 'missing_received_peer_context':
            if row['status'] != event['status'] or row.get('sources'):
                raise ValueError('Missing peer call filled')
            missing += 1
        else:
            if row['status'] != 'supported' or set(row['sources']) != set(SOURCES):
                raise ValueError('Supported source population differs')
            for source,r in row['sources'].items():
                if (r['status'] != 'shadow_candidates_ready' or r['control_eligible'] is not False or
                    not 1 <= len(r['rounds']) <= MAX_ROUNDS or not r['seed_count'] <= r['candidate_count'] <= MAX_CANDIDATES):
                    raise ValueError('Bounded noncontrolling proposal differs')
            supported += 1
    if (report['supported_calls'],report['missing_peer_calls']) != (supported,missing):
        raise ValueError('Sequential support counters differ')
    for name,digest in report['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Run-used source changed: '+name)


def compare(primary, repeated, artifact):
    first, second = [json.loads((p/'summary.json').read_text()) for p in (primary,repeated)]
    for report in (first,second):
        validate_completed(report)
    if first != second:
        raise ValueError('Independent process summaries/proposal provenance differ')
    for record in first['records']:
        for source in record.get('sources',{}).values():
            name = source['arrays_path']
            a,b = [arrays((p/name).read_bytes()) for p in (primary,repeated)]
            if set(a) != set(b) or not equal_arrays(a,b,tuple(a)):
                raise ValueError('Independent expanded pool/input/forecast differs')
    if artifact.exists():
        raise FileExistsError('Do not overwrite prior evidence')
    artifact.parent.mkdir(parents=True,exist_ok=True)
    # Includes public research outputs and original replay plans/commands, not restored checkout.
    with zipfile.ZipFile(artifact,'w',zipfile.ZIP_DEFLATED) as archive:
        for stage, directory in (('primary',primary),('repeated',repeated)):
            for path in sorted(directory.rglob('*')):
                relative = path.relative_to(directory)
                if path.is_file() and relative.parts[0] in ('calls','source','original_replay','summary.json'):
                    archive.write(path, stage+'/'+relative.as_posix())
        archive.writestr('release_summary.json',json.dumps({'status':'two_process_exact_candidate_replay_passed',
            'summary':first,'artifact_scope':'No new training, branch labels, active selector or efficacy evidence.'},indent=2))
    # Validate every packaged member against its original bytes after writing.
    with zipfile.ZipFile(artifact) as archive:
        for name in archive.namelist():
            if name == 'release_summary.json':
                continue
            stage,relative = name.split('/',1)
            directory = primary if stage == 'primary' else repeated
            if archive.read(name) != (directory/relative).read_bytes():
                raise ValueError('Packaged evidence bytes differ')
    return {'status':'two_process_exact_candidate_replay_passed','artifact_sha256':sha(artifact),
            'artifact_bytes':artifact.stat().st_size,'supported_calls':first['supported_calls'],
            'missing_peer_calls':first['missing_peer_calls'],'counts':first['counts'],
            'enhanced_control_enabled':False,'new_model_trained':False,'holdout_used':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--primary', type=Path)
    parser.add_argument('--repeated', type=Path)
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.output is not None:
        run(args.output)
    else:
        result = compare(args.primary,args.repeated,args.artifact)
        if args.report is not None:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            with args.report.open('x',encoding='utf8') as file:
                json.dump(result,file,indent=2,allow_nan=False)
        print(json.dumps(result),flush=True)
