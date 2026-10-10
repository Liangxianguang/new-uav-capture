"""Reaudit original data, true-label replays and all18 public pilot models."""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_public_mechanisms import (audit_fresh, audit_labels, normalization, prior_probabilities, evaluate,
                                    qualification, PublicMechanismModel, frozen_configs, audit_sources,
                                    arrays, compare_arrays, check_archive, sha, BASELINE_SHA)
from training_release import tensor_tree_equal
from mechanism_replay import call_metrics as mechanism_call_metrics, summarize as mechanism_summary


def audit(read, candidate_read, configs, restored):
    protocol = json.loads((ROOT / 'training_protocol.json').read_text())
    calls, data_report = audit_fresh(read, candidate_read, configs, restored)
    labels = audit_labels(read, calls, data_report)
    repeated_labels = json.loads(read('mechanism_repeated/summary.json'))
    for key in labels.keys() - {'elapsed_seconds'}:
        if labels[key] != repeated_labels[key]:
            raise ValueError('Independent true-label replay summaries differ')
    for row in json.loads(read('mechanism/records.json')):
        name = row['mechanism_path']
        if read('mechanism/' + name) != read('mechanism_repeated/' + name):
            raise ValueError('Independent true-label replay bytes differ')
    if read('mechanism/records.json') != read('mechanism_repeated/records.json'):
        raise ValueError('Independent true-label record populations differ')
    audit_sources(read, 'mechanism_repeated', repeated_labels)
    for row in labels['episodes']:
        name = f"trajectories/{row['episode_index']}.npz"
        if not compare_arrays(arrays(read('mechanism/' + name)), arrays(read('mechanism_repeated/' + name))):
            raise ValueError('Independent true-label original trajectories differ')
    indices = {split: [i for i,c in enumerate(calls) if c['record']['split'] == split] for split in ('train', 'development_validation')}
    mean, scale = normalization(calls, indices['train'])
    prior = prior_probabilities(calls, indices['train'], protocol['training']['branch_prior_smoothing'])
    sets, summaries = [], []
    for stage in ('primary', 'retrained'):
        summary = json.loads(read(stage + '/summary.json'))
        if summary['protocol'] != protocol or summary['status'] != 'public_mechanism_pilot_finished_not_promoted':
            raise ValueError('Preregistered public mechanism training mismatch')
        audit_sources(read, stage, summary)
        if json.loads(read(stage + '/source/cwm_v14/training_protocol.json')) != protocol:
            raise ValueError('Recorded public mechanism protocol differs')
        if any(summary[k] for k in ('baseline_weights_included','enhanced_control_enabled','holdout_used','private_labels_model_inputs','prior_gate_overridden')):
            raise ValueError('Public-input original/offline contract violated')
        for key, name in (('data_summary_sha256','data'),('public_summary_sha256','public'),('mechanism_summary_sha256','mechanism')):
            if summary[key] != hashlib.sha256(read(name + '/summary.json')).hexdigest():
                raise ValueError('Training input provenance differs')
        if summary['split_calls'] != {k:len(v) for k,v in indices.items()} or summary['split_groups'] != {k:sorted({calls[i]['record']['group'] for i in v}) for k,v in indices.items()}:
            raise ValueError('Training split/group mismatch')
        normalizer = arrays(read(stage + '/normalization.npz'))
        if not np.array_equal(normalizer['mean'],mean) or not np.array_equal(normalizer['scale'],scale) or json.loads(read(stage + '/training_prior.json')) != prior:
            raise ValueError('Train-only normalization/prior mismatch')
        expected = {(kind,seed) for kind in protocol['models'] for seed in protocol['training']['seeds']}
        if len(summary['models']) != 9 or {(r['kind'],r['seed']) for r in summary['models']} != expected:
            raise ValueError('Nine-model population incomplete')
        checkpoints = {}
        for row in summary['models']:
            name = f"{row['kind']}_seed{row['seed']}"
            raw = read(stage + '/' + name + '.pt')
            if hashlib.sha256(raw).hexdigest() != row['checkpoint_sha256']:
                raise ValueError('Pilot checkpoint digest mismatch')
            checkpoint = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            if checkpoint['kind'] != row['kind'] or checkpoint['seed'] != row['seed'] or checkpoint['protocol'] != protocol or checkpoint['source_hashes'] != summary['source_hashes'] or checkpoint['online_promoted'] or checkpoint['baseline_weights_included']:
                raise ValueError('Pilot checkpoint contract mismatch')
            if (checkpoint['data_summary_sha256'] != summary['data_summary_sha256'] or checkpoint['mechanism_summary_sha256'] != summary['mechanism_summary_sha256']
                    or not np.array_equal(checkpoint['normalizer_mean'].numpy(),mean) or not np.array_equal(checkpoint['normalizer_scale'].numpy(),scale) or checkpoint['prior'] != prior):
                raise ValueError('Pilot checkpoint provenance mismatch')
            with torch.random.fork_rng(devices=[]):
                model = PublicMechanismModel(row['kind'])
            model.load_state_dict(checkpoint['model_state'],strict=True)
            measured, predictions, records = evaluate(model,calls,indices['development_validation'],mean,scale,prior)
            if measured != row['development'] or records != json.loads(read(f'{stage}/{name}_records.json')) or not compare_arrays(predictions,arrays(read(f'{stage}/{name}_predictions.npz'))):
                raise ValueError('Independent pilot predictions/metrics differ')
            if row['parameters'] != sum(p.numel() for p in model.parameters()):
                raise ValueError('Parameter budget mismatch')
            history = json.loads(read(f'{stage}/{name}_history.json'))
            if [r['epoch'] for r in history] != list(range(1,41)) or not np.isfinite([r['mean_batch_loss'] for r in history]).all():
                raise ValueError('Fixed forty-epoch budget mismatch')
            checkpoints[name] = checkpoint
        if qualification(summary['models'],protocol) != summary['qualification']:
            raise ValueError('Independent mechanism gate recomputation differs')
        sets.append(checkpoints)
        summaries.append(summary)
    for key in summaries[0].keys() - {'models'}:
        if summaries[0][key] != summaries[1][key]:
            raise ValueError('Independent pilot training summaries differ')
    for name, checkpoint in sets[0].items():
        if not tensor_tree_equal(checkpoint,sets[1][name]) or read(f'primary/{name}_history.json') != read(f'retrained/{name}_history.json'):
            raise ValueError('Independent pilot weights/optimizer/RNG/history differ')
    diagnostic_protocol = json.loads((ROOT.parent/'cwm_v13/protocol.json').read_text())
    diagnostic = {}
    for split, selected in indices.items():
        measured = mechanism_summary([{**{k:calls[i]['record'][k] for k in ('group','step')},
                                      'metrics':mechanism_call_metrics(calls[i]['values'],calls[i]['private'],diagnostic_protocol)} for i in selected],diagnostic_protocol)
        # Do not inherit V13 TRAIN-only status/limits for a two-split new pilot.
        diagnostic[split] = {key:measured[key] for key in ('calls','groups','nonanchor_candidates','true_branch_flip_pairs','nonanchor_common_points',
                                                        'cbf_changed_command_points','observed_command_points','cbf_correction_max_mps','execution_error_max_mps',
                                                        'response_by_true_branch','by_step','support_points_by_offset')}
    return {'status':'passed_independent_public_mechanism_pilot_audit','episodes':64,'groups':32,
            'split_calls':summaries[0]['split_calls'],'models_per_run':9,'training_runs':2,
            'original_public_and_two_label_replays_equal':True,'weights_optimizer_rng_history_equal':True,
            'qualification':summaries[0]['qualification'],'mechanism_diagnostics':diagnostic,'enhanced_control_enabled':False,'holdout_used':False,
            'scope':protocol['limitations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ('data','public','mechanism','mechanism_repeated','primary','retrained'):
        parser.add_argument('--'+stage.replace('_','-'),type=Path)
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--source',type=Path,default=ROOT.parent/'cwm_v7/artifacts/paired_training_20261010.zip')
    parser.add_argument('--capsule',type=Path,default=ROOT.parent/'cwm_v1/baseline/capsule.zip')
    args = parser.parse_args()
    protocol = json.loads((ROOT/'data_protocol.json').read_text())
    if sha(args.source) != protocol['source_archive_sha256'] or sha(args.capsule) != BASELINE_SHA:
        raise ValueError('Protected original/candidate sources changed')
    check_archive(args.source,'ARTIFACT_MANIFEST.json',False)
    torch.set_num_threads(1)
    configs = frozen_configs(args.capsule,ROOT.parent.parent/'results/cwm_v14/audit_restored')
    with zipfile.ZipFile(args.source) as source:
        if args.verify:
            integrity = check_archive(args.verify,'ARTIFACT_MANIFEST.json',False)
            with zipfile.ZipFile(args.verify) as archive:
                result = audit(archive.read,source.read,configs,ROOT.parent.parent/'results/cwm_v14/audit_restored')
            print(json.dumps({**result,'artifact_sha256':integrity['sha256']}),flush=True)
            return
        members = {}
        for stage in ('data','public','mechanism','mechanism_repeated','primary','retrained'):
            run = getattr(args,stage)
            if run is None:
                parser.error('All completed collection/public/label/training runs required')
            for path in run.rglob('*'):
                relative = path.relative_to(run)
                if path.is_file() and 'restored' not in relative.parts:
                    members[stage+'/'+relative.as_posix()] = path.read_bytes()
        for path in ROOT.glob('*.py'):
            members['verification_source/'+path.name] = path.read_bytes()
        result = audit(members.__getitem__,source.read,configs,ROOT.parent.parent/'results/cwm_v14/audit_restored')
        path = ROOT/'artifacts/public_mechanism_training_20261010.zip'
        path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,raw in members.items():
                archive.writestr(name,raw)
            archive.writestr('ARTIFACT_MANIFEST.json',json.dumps({name:hashlib.sha256(raw).hexdigest() for name,raw in members.items()},indent=2))
        integrity = check_archive(path,'ARTIFACT_MANIFEST.json',False)
        with zipfile.ZipFile(path) as archive:
            audit(archive.read,source.read,configs,ROOT.parent.parent/'results/cwm_v14/audit_restored')
    report_dir = ROOT/'reports'
    report_dir.mkdir(exist_ok=True)
    with (report_dir/'release_manifest.json').open('x') as file:
        json.dump({**result,'artifact_sha256':integrity['sha256'],'artifact_bytes':path.stat().st_size},file,indent=2)
    for stage in ('data','mechanism','primary'):
        with (report_dir/(stage+'_summary.json')).open('xb') as file:
            file.write(members[stage+'/summary.json'])
    print(json.dumps({**result,'artifact_sha256':integrity['sha256'],'artifact_bytes':path.stat().st_size}),flush=True)


if __name__ == '__main__':
    main()
