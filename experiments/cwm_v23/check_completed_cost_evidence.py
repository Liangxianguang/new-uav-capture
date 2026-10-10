"""Full completed-evidence positive/forgery checks, never pass unfinished runs."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cost_training_release import audit_cost_training, frozen_configs, arrays, sha, BASELINE_SHA


def json_bytes(value):
    return json.dumps(value, indent=2, allow_nan=False).encode('utf8')


def npz_bytes(value):
    stream = io.BytesIO()
    np.savez_compressed(stream, **value)
    return stream.getvalue()


def checkpoint_bytes(checkpoint):
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    return stream.getvalue()


def replace_checkpoint(read, changes, report, name, seed, mutate):
    path = f'primary/{name}_seed{seed}.pt'
    checkpoint = torch.load(io.BytesIO(read(path)), map_location='cpu', weights_only=True)
    mutate(checkpoint)
    raw = checkpoint_bytes(checkpoint)
    changes[path] = raw
    row = next(r for r in report['models'] if r['configuration'] == name and r['seed'] == seed)
    row['checkpoint_sha256'] = hashlib.sha256(raw).hexdigest()


def cache_forgery(read, protocol):
    """Update ALL corresponding hashes so numeric cache audit is necessary."""
    seed = protocol['training']['seeds'][0]
    origin = protocol['primary_configuration'].split('_')[0]
    path = f'primary/{origin}_seed{seed}_deployed_cost_cache.npz'
    cache = arrays(read(path))
    key = next(k for k in cache if k.endswith('_label_effect'))
    cache[key].flat[-1] += .01
    raw = npz_bytes(cache)
    digest = hashlib.sha256(raw).hexdigest()
    changes = {path: raw}
    report = json.loads(read('primary/summary.json'))
    for name in protocol['models']:
        if name.split('_')[0] == origin:
            replace_checkpoint(read, changes, report, name, seed,
                               lambda checkpoint: checkpoint.__setitem__('deployed_cost_cache_sha256', digest))
    changes['primary/summary.json'] = json_bytes(report)
    return changes


def expect_cost_rejection(read, configs, restored, changes, message):
    def altered(name):
        return changes[name] if name in changes else read(name)
    try:
        audit_cost_training(altered, configs, restored)
    except ValueError as error:
        if message not in str(error):
            raise AssertionError(f'Unexpected semantic rejection: {error}') from error
        return
    raise AssertionError('Forged completed research evidence was accepted')


def run_completed_checks(roots, restored):
    for stage in ('data', 'public', 'primary', 'retrained'):
        if not (roots[stage] / 'summary.json').is_file():
            raise ValueError('Complete data/public and BOTH full18-model research runs required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    capsule = ROOT.parent / 'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Frozen original capsule changed')
    configs = frozen_configs(capsule, restored)
    def read(name):
        stage, relative = name.split('/', 1)
        return (roots[stage] / relative).read_bytes()
    positive = audit_cost_training(read, configs, restored, True)
    print(json.dumps({'test': 'full_positive_completed_evidence_audit_passed'}), flush=True)
    protocol = json.loads((ROOT / 'training_protocol.json').read_text(encoding='utf8'))
    primary, seed = protocol['primary_configuration'], protocol['training']['seeds'][0]
    prediction_path = f'primary/{primary}_seed{seed}_development_validation_predictions.npz'
    prediction = arrays(read(prediction_path))
    prediction['call0_prediction'][0, 0, 0] += .01
    expect_cost_rejection(read, configs, restored, {prediction_path: npz_bytes(prediction)}, 'forecast or metrics')
    print(json.dumps({'test': 'forged_development_prediction_rejected'}), flush=True)
    report = json.loads(read('primary/summary.json'))
    report['primary_research_eligible'] = not report['primary_research_eligible']
    expect_cost_rejection(read, configs, restored, {'primary/summary.json': json_bytes(report)}, 'selection/qualification')
    print(json.dumps({'test': 'forged_primary_qualification_rejected'}), flush=True)
    report, changes = json.loads(read('primary/summary.json')), {}
    def corrupt_rng(checkpoint):
        checkpoint['torch_rng_state'][0] ^= 1
    replace_checkpoint(read, changes, report, primary, seed, corrupt_rng)
    changes['primary/summary.json'] = json_bytes(report)
    expect_cost_rejection(read, configs, restored, changes, 'Final Torch RNG')
    print(json.dumps({'test': 'forged_rng_rejected_even_with_updated_checkpoint_hash'}), flush=True)
    expect_cost_rejection(read, configs, restored, cache_forgery(read, protocol),
                          'reference/scale/label/gradient cache')
    print(json.dumps({'test': 'forged_private_cost_effect_label_rejected_even_with_all_updated_hashes'}), flush=True)
    report = json.loads(read('primary/summary.json'))
    report['models'].pop()
    expect_cost_rejection(read, configs, restored, {'primary/summary.json': json_bytes(report)}, 'final18-model population')
    print(json.dumps({'test': 'missing_final_model_rejected'}), flush=True)
    result = {'status': 'six_completed_cost_evidence_semantic_checks_passed',
              'positive_status': positive['status'], 'enhanced_control_enabled': False, 'holdout_used': False}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'public', 'primary', 'retrained', 'restored'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run_completed_checks({name: getattr(args, name) for name in ('data', 'public', 'primary', 'retrained')}, args.restored)


if __name__ == '__main__':
    main()
