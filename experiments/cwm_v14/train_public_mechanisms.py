"""Fixed-budget, public-input branch/CBF identifiability pilot, no control."""
import argparse
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'cwm_v10'), str(ROOT.parent / 'cwm_v13')]
from public_mechanism_model import PublicMechanismModel, public_inputs, batch_call_loss, branch_labels, cbf_labels
from public_mechanism_data import audit_fresh
from local_shadow_release import frozen_configs
from geometry_release import arrays, compare_arrays, sha, BASELINE_SHA
from mechanism_release import _validate_private_labels
from two_head_release import audit_sources
from verify_release import check_archive


def audit_labels(read, calls, report):
    summary = json.loads(read('mechanism/summary.json'))
    if summary['status'] != 'original_public_and_true_label_replay_equal' or summary['protocol'] != report['protocol']:
        raise ValueError('True-label replay protocol/status mismatch')
    if any(summary[k] for k in ('enhanced_control_enabled', 'holdout_used', 'private_labels_model_inputs')):
        raise ValueError('True-label public/offline contract mismatch')
    for name in ('summary', 'calls'):
        if summary['data_' + name + '_sha256'] != hashlib.sha256(read('data/' + name + '.json')).hexdigest():
            raise ValueError('True-label replay data provenance mismatch')
    audit_sources(read, 'mechanism', summary)
    records = json.loads(read('mechanism/records.json'))
    key = lambda row: (row['episode_index'], row['step'], row['agent'])
    keyed = {key(r): r for r in records}
    if len(keyed) != len(records) or set(keyed) != {key(c['record']) for c in calls} or summary['eligible_calls'] != len(records):
        raise ValueError('True-label eligible population mismatch')
    for call in calls:
        row = keyed[key(call['record'])]
        if any(row[k] != call['record'][k] for k in ('group', 'split')) or not all(row[k] for k in ('public_history_and_backbone_equal', 'original_branch_fields_replay_equal', 'anchor_repeat_equal')):
            raise ValueError('True-label call identity mismatch')
        private = arrays(read('mechanism/' + row['mechanism_path']))
        _validate_private_labels(read, 'mechanism', row, private, call['values'])
        call['private'] = private
    if summary['public_calls'] != len(json.loads(read('data/calls.json'))):
        raise ValueError('Incomplete public-label replay')
    episodes = summary['episodes']
    expected = {r['episode_index'] for r in report['episodes']}
    if len(episodes) != 64 or {r['episode_index'] for r in episodes} != expected:
        raise ValueError('Incomplete true-label original episodes')
    for row in episodes:
        raw = read(f"mechanism/trajectories/{row['episode_index']}.npz")
        if hashlib.sha256(raw).hexdigest() != row['trajectory_sha256'] or not row['original_arrays_equal'] or not compare_arrays(arrays(raw), arrays(read(f"data/observed/{row['episode_index']}.npz"))):
            raise ValueError('True-label original trajectory mismatch')
    return summary


def normalization(calls, indices):
    history = np.concatenate([calls[i]['values']['history'] for i in indices], axis=0)
    return history.mean(0), np.maximum(history.std(0), 1e-6)


def prior_probabilities(calls, indices, smoothing):
    counts = collections.defaultdict(lambda: np.full((8, 3), float(smoothing)))
    groups = collections.Counter(calls[i]['record']['group'] for i in indices)
    for i in indices:
        call = calls[i]
        labels, mask = branch_labels(call['values'], call['private'])
        onehot = np.eye(3)[labels.numpy()]
        mask = mask.numpy()
        fraction = (onehot * mask[..., None]).sum(0) / np.maximum(mask.sum(0)[:, None], 1)
        counts[call['record']['step']] += fraction / groups[call['record']['group']]
    return {str(step): (value/value.sum(-1, keepdims=True)).tolist() for step, value in sorted(counts.items())}


def call_metrics(values, private, branch_probability, prior_probability, command):
    label, branch_support = branch_labels(values, private)
    truth = np.eye(3)[label.numpy()]
    support = branch_support.numpy()
    mean = lambda data, mask: float(data[mask].mean()) if mask.any() else None
    probabilities = {'model': branch_probability, 'prior': prior_probability}
    result = {'calibration': {}}
    common = support[1:] & support[:1]
    nonanchor = ~(values['proposed'] == values['anchor'][None]).all((1, 2, 3))
    common &= nonanchor[:, None]
    true_effect = truth[1:] - truth[:1]
    for name, probability in probabilities.items():
        if probability is None:
            continue
        if probability.shape != truth.shape or not np.isfinite(probability).all() or not np.allclose(probability.sum(-1), 1., atol=1e-6) or (probability < 0).any():
            raise ValueError('Invalid branch probability')
        actual = np.take_along_axis(probability, label.numpy()[..., None], -1)[..., 0]
        effect = probability[1:] - probability[:1]
        result[name + '_nll'] = mean(-np.log(np.clip(actual, 1e-12, 1.)), support)
        result[name + '_brier'] = mean(((probability - truth)**2).sum(-1), support)
        result[name + '_accuracy'] = mean((probability.argmax(-1) == label.numpy()).astype(float), support)
        result[name + '_paired_brier'] = mean(((effect - true_effect)**2).sum(-1), common)
        confidence = probability.max(-1)[support]
        correct = (probability.argmax(-1) == label.numpy())[support]
        bins = np.minimum((confidence * 10).astype(int), 9)
        result['calibration'][name] = [{'fraction': float(np.mean(bins == j)),
                                       'confidence_sum_per_point': float(confidence[bins == j].sum()/len(confidence)),
                                       'correct_sum_per_point': float(correct[bins == j].sum()/len(confidence))} for j in range(10)]
    result['zero_paired_brier'] = mean((true_effect**2).sum(-1), common)
    result['branch_points'] = int(support.sum())
    result['paired_points'] = int(common.sum())
    truth_command, observed = cbf_labels(values)
    truth_command = np.concatenate([values['anchor_commanded'][None], values['commanded']], axis=0)
    proposed = np.concatenate([values['anchor'][None], values['proposed']], axis=0)
    command_error = np.linalg.norm(proposed-truth_command, axis=-1).mean(-1)
    result['proposed_command_error_mps'] = mean(command_error, observed.numpy())
    if command is not None:
        if command.shape != truth_command.shape or not np.isfinite(command).all():
            raise ValueError('Invalid predicted CBF command')
        result['model_command_error_mps'] = mean(np.linalg.norm(command-truth_command, axis=-1).mean(-1), observed.numpy())
    result['command_points'] = int(observed.sum()) * 4
    return result


def summarize(records):
    groups = sorted({r['group'] for r in records})
    names = sorted(set().union(*(r['metrics'].keys() for r in records)) - {'calibration'})
    result = {'calls': len(records), 'groups': len(groups), 'group_metrics': {}}
    for group in groups:
        chosen = [r['metrics'] for r in records if r['group'] == group]
        result['group_metrics'][group] = {key: float(np.mean([r[key] for r in chosen if key in r and r[key] is not None])) for key in names
                                         if not key.endswith('points') and any(key in r and r[key] is not None for r in chosen)}
    for name in names:
        if name.endswith('points'):
            result[name] = sum(r['metrics'].get(name, 0) for r in records)
        else:
            supported = [r[name] for r in result['group_metrics'].values() if name in r]
            result[name] = float(np.mean(supported)) if supported else None
    result['calibration'] = {}
    for name in ('model', 'prior'):
        group_bins = []
        for group in groups:
            chosen = [r['metrics']['calibration'][name] for r in records if r['group'] == group and name in r['metrics']['calibration']]
            if chosen:
                group_bins.append(np.mean([[[bin[k] for k in ('fraction','confidence_sum_per_point','correct_sum_per_point')] for bin in bins] for bins in chosen],axis=0))
        if group_bins:
            bins = np.mean(group_bins,axis=0)
            result['calibration'][name] = {'group_equal_ece_10_bins': float(np.abs(bins[:,1]-bins[:,2]).sum()),
                                          'bins': [{'fraction':float(b[0]), 'mean_confidence':float(b[1]/b[0]) if b[0]>0 else None,
                                                    'mean_accuracy':float(b[2]/b[0]) if b[0]>0 else None} for b in bins]}
    result['by_step'] = {}
    for step in sorted({r['step'] for r in records}):
        chosen = [r for r in records if r['step'] == step]
        step_groups = sorted({r['group'] for r in chosen})
        fields = {}
        for name in names:
            if name.endswith('points'):
                fields[name] = sum(r['metrics'].get(name,0) for r in chosen)
            else:
                grouped = [[r['metrics'][name] for r in chosen if r['group'] == g and name in r['metrics'] and r['metrics'][name] is not None] for g in step_groups]
                fields[name] = float(np.mean([np.mean(g) for g in grouped if g])) if any(grouped) else None
        result['by_step'][str(step)] = {'calls':len(chosen),'groups':len(step_groups),**fields}
    return result


def evaluate(model, calls, indices, mean, scale, prior):
    predictions, records = {}, []
    model.eval()
    with torch.no_grad():
        for index in indices:
            call = calls[index]
            output = model(*public_inputs(call['values'], mean, scale))
            probability = torch.softmax(output, -1).numpy() if model.kind != 'cbf_history' else None
            command = output.numpy() if model.kind == 'cbf_history' else None
            prior_probability = np.repeat(np.asarray(prior[str(call['record']['step'])])[None], len(call['values']['proposed'])+1, 0)
            measured = call_metrics(call['values'], call['private'], probability, prior_probability, command)
            records.append({**{key: call['record'][key] for key in ('episode_index', 'step', 'agent', 'group')}, 'metrics': measured})
            predictions[f'call{index}'] = probability if probability is not None else command
    return summarize(records), predictions, records


def gain_ci(group_metrics, base, model, protocol):
    gains = np.asarray([r[base] - r[model] for r in group_metrics.values() if base in r and model in r], float)
    if not len(gains):
        return {'groups': 0, 'mean': None, 'lower': None, 'upper': None}
    rng = np.random.default_rng(protocol['gates']['bootstrap_seed'])
    samples = gains[rng.integers(0, len(gains), (protocol['gates']['bootstrap_draws'], len(gains)))].mean(1)
    return {'groups': len(gains), 'mean': float(gains.mean()), 'lower': float(np.quantile(samples, .025)), 'upper': float(np.quantile(samples, .975))}


def qualification(models, protocol):
    result = {}
    for kind in protocol['models']:
        rows = [r for r in models if r['kind'] == kind]
        primary = 'model_command_error_mps' if kind == 'cbf_history' else 'model_nll'
        ordered = sorted(rows, key=lambda r: (r['development'][primary], r['seed']))
        chosen = ordered[len(ordered)//2]
        metrics = chosen['development']
        pairs = [('proposed_command_error_mps', 'model_command_error_mps', 'cbf_vector_error_relative_improvement_vs_proposed')] if kind == 'cbf_history' else [
            ('prior_nll', 'model_nll', 'branch_nll_relative_improvement_vs_train_prior'), ('zero_paired_brier', 'model_paired_brier', 'paired_branch_brier_relative_improvement_vs_zero')]
        checks, gains, ratios = {}, {}, {}
        for base, model, gate in pairs:
            baseline = np.median([r['development'][base] for r in rows])
            predicted = np.median([r['development'][model] for r in rows])
            ratio = float(1.-predicted/baseline) if baseline is not None and baseline > 0 else None
            ratios[gate] = ratio
            checks[gate] = ratio is not None and ratio >= protocol['gates'][gate]
            checks[gate + '_all_seeds_improve'] = all(r['development'][model] < r['development'][base] for r in rows)
            ci = gain_ci(metrics['group_metrics'], base, model, protocol)
            gains[gate] = ci
            checks[gate + '_positive_group_lower_bound'] = ci['lower'] is not None and ci['lower'] > 0.
        result[kind] = {'selected_median_seed': chosen['seed'], 'median_relative_improvements': ratios, 'gains': gains, 'checks': checks,
                        'mechanism_research_eligible': all(checks.values()), 'online_promoted': False}
    return result


def dataset_read(data, public, mechanism):
    mapping = {'data': data, 'public': public, 'mechanism': mechanism}
    def read(name):
        stage, relative = name.split('/', 1)
        return (mapping[stage] / relative).read_bytes()
    return read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'public', 'mechanism', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'cwm_v7/artifacts/paired_training_20261010.zip')
    parser.add_argument('--capsule', type=Path, default=ROOT.parent / 'cwm_v1/baseline/capsule.zip')
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'training_protocol.json').read_text())
    data_protocol = json.loads((ROOT / 'data_protocol.json').read_text())
    if any(protocol[k] for k in ('enhanced_control_enabled', 'original_gru_in_optimizer', 'holdout_used', 'private_labels_model_inputs', 'prior_gate_override_allowed')):
        raise ValueError('Offline public-input pilot contract mismatch')
    if sha(args.source) != data_protocol['source_archive_sha256'] or sha(args.capsule) != BASELINE_SHA:
        raise ValueError('Original/candidate source changed')
    check_archive(args.source, 'ARTIFACT_MANIFEST.json', False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    configs = frozen_configs(args.capsule, args.output / 'restored')
    read = dataset_read(args.data, args.public, args.mechanism)
    with zipfile.ZipFile(args.source) as source:
        calls, data_report = audit_fresh(read, source.read, configs, args.output / 'restored')
    labels_report = audit_labels(read, calls, data_report)
    indices = {split: [i for i, call in enumerate(calls) if call['record']['split'] == split] for split in ('train', 'development_validation')}
    print(json.dumps({'status':'independent_data_public_and_label_audit_passed','split_calls':{k:len(v) for k,v in indices.items()}}),flush=True)
    mean, scale = normalization(calls, indices['train'])
    prior = prior_probabilities(calls, indices['train'], protocol['training']['branch_prior_smoothing'])
    np.savez_compressed(args.output / 'normalization.npz', mean=mean, scale=scale)
    (args.output / 'training_prior.json').write_text(json.dumps(prior, indent=2))
    sources = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)
                      and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} | {ROOT / 'data_protocol.json', ROOT / 'training_protocol.json'})
    hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sources}
    for p in sources:
        dest = args.output / 'source' / p.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    counts = collections.Counter(calls[i]['record']['group'] for i in indices['train'])
    models = []
    config = protocol['training']
    for kind in protocol['models']:
        for seed in config['seeds']:
            torch.manual_seed(seed)
            model = PublicMechanismModel(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            sampler = np.random.default_rng(seed)
            history = []
            for epoch in range(1, config['epochs']+1):
                model.train()
                order = sampler.permutation(indices['train'])
                losses = []
                for start in range(0, len(order), config['batch_calls']):
                    optimizer.zero_grad(set_to_none=True)
                    batch = order[start:start+config['batch_calls']]
                    loss = batch_call_loss(model, calls, batch, mean, scale, counts, config['cbf_energy_penalty'])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite mechanism loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                    optimizer.step()
                    losses.append(float(loss.detach()))
                history.append({'epoch': epoch, 'mean_batch_loss': float(np.mean(losses))})
                if epoch % 10 == 0:
                    print(json.dumps({'kind': kind, 'seed': seed, 'epoch': epoch, 'loss': history[-1]['mean_batch_loss']}), flush=True)
            metrics, predictions, records = evaluate(model, calls, indices['development_validation'], mean, scale, prior)
            name = f'{kind}_seed{seed}'
            checkpoint = {'kind': kind, 'seed': seed, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                          'torch_rng': torch.get_rng_state(), 'sampler_rng_state': sampler.bit_generator.state,
                          'normalizer_mean': torch.from_numpy(mean), 'normalizer_scale': torch.from_numpy(scale), 'prior': prior,
                          'protocol': protocol, 'source_hashes': hashes, 'data_summary_sha256': sha(args.data / 'summary.json'),
                          'mechanism_summary_sha256': sha(args.mechanism / 'summary.json'), 'online_promoted': False, 'baseline_weights_included': False}
            torch.save(checkpoint, args.output / f'{name}.pt')
            np.savez_compressed(args.output / f'{name}_predictions.npz', **predictions)
            (args.output / f'{name}_records.json').write_text(json.dumps(records, indent=2))
            (args.output / f'{name}_history.json').write_text(json.dumps(history, indent=2))
            models.append({'kind': kind, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()), 'checkpoint_sha256': sha(args.output / f'{name}.pt'), 'development': metrics})
    gates = qualification(models, protocol)
    if any(sha(ROOT.parent / name) != digest for name, digest in hashes.items()):
        raise ValueError('Run-used training sources changed')
    summary = {'status': 'public_mechanism_pilot_finished_not_promoted', 'protocol': protocol, 'source_hashes': hashes,
               'models': models, 'qualification': gates, 'split_calls': {k: len(v) for k, v in indices.items()},
               'split_groups': {k: sorted({calls[i]['record']['group'] for i in v}) for k, v in indices.items()},
               'data_summary_sha256': sha(args.data / 'summary.json'), 'public_summary_sha256': sha(args.public / 'summary.json'),
               'mechanism_summary_sha256': sha(args.mechanism / 'summary.json'), 'baseline_weights_included': False,
               'enhanced_control_enabled': False, 'holdout_used': False, 'private_labels_model_inputs': False,
               'prior_gate_overridden': False, 'limitations': protocol['limitations']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
