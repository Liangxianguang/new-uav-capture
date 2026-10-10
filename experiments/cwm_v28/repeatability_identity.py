"""Exact V28 checkpoint contents; preserve each run's serialized file identity.

Only models[*].checkpoint_sha256 may differ between complete run summaries.
Those identities are verified against the actual files BEFORE any comparison.
No checkpoint, history, training source, metric, or qualification is rewritten.
"""
import hashlib
import json
import struct

import torch


IDENTITY_SCHEMA = 'cwm_v28.exact_contents_individual_file_identity.v1'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def exact_tree(a, b):
    """Typed bitwise tensor/scalar equality, ignoring only dictionary order."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return False
        if a.dtype != b.dtype or a.shape != b.shape or a.layout != b.layout:
            return False
        if a.layout != torch.strided:
            raise ValueError('Unsupported checkpoint tensor layout')
        left = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        right = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return torch.equal(left, right)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if {(type(k), k) for k in a} != {(type(k), k) for k in b}:
            return False
        return all(exact_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact_tree(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return struct.pack('!d', a) == struct.pack('!d', b)
    if a is None or isinstance(a, (bool, int, str, bytes)):
        return a == b
    raise ValueError('Unsupported checkpoint field type: ' + type(a).__name__)


def summary_semantics(report):
    """Detach the report; remove only the per-model serialized file digest."""
    result = json.loads(json.dumps(report, allow_nan=False))
    for row in result['models']:
        value = row.pop('checkpoint_sha256')
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Canonical checkpoint SHA256 required')
    return result


def verify_pair(primary, repeated, p, r):
    """Read ALL final/stage contents and histories; never infer equality from SHA."""
    summary_hashes = [digest(folder/'summary.json') for folder in (primary, repeated)]
    for folder, report in ((primary, p), (repeated, r)):
        if not exact_tree(json.loads((folder/'summary.json').read_bytes()), report):
            raise ValueError('Saved training summary differs')
        expected = {(m, s) for m in report['protocol']['models'] for s in report['protocol']['training']['seeds']}
        if len(report['models']) != len(expected) or {(v['configuration'], v['seed']) for v in report['models']} != expected:
            raise ValueError('Complete fixed model support required')
        for row in report['models']:
            path = folder/f"{row['configuration']}_seed{row['seed']}.pt"
            if digest(path) != row['checkpoint_sha256']:
                raise ValueError('Individual checkpoint file digest differs')
    if not exact_tree(summary_semantics(p), summary_semantics(r)):
        raise ValueError('Independent complete training summary semantics differ')
    paths = [f"{row['configuration']}_seed{row['seed']}" for row in p['models']]
    paths += [f'cv_seed{seed}_motion_stage' for seed in p['protocol']['training']['seeds']]
    equal_files = 0
    for name in paths:
        left, right = primary/(name+'.pt'), repeated/(name+'.pt')
        before = (digest(left), digest(right))
        a = torch.load(left, map_location='cpu', weights_only=True)
        b = torch.load(right, map_location='cpu', weights_only=True)
        if not exact_tree(a, b):
            raise ValueError('Independent weights/optimizer/RNG/provenance contents differ: ' + name)
        if before != (digest(left), digest(right)):
            raise ValueError('Checkpoint changed during cross-run content comparison')
        if (primary/(name+'_history.json')).read_bytes() != (repeated/(name+'_history.json')).read_bytes():
            raise ValueError('Independent complete history bytes differ: ' + name)
        equal_files += before[0] == before[1]
    if summary_hashes != [digest(folder/'summary.json') for folder in (primary, repeated)]:
        raise ValueError('Training summaries changed during cross-run comparison')
    # Verify final file identities again; the stage files are preserved by the
    # full release inventory and actual extracted replay, not newly relabeled.
    for folder, report in ((primary, p), (repeated, r)):
        for row in report['models']:
            if digest(folder/f"{row['configuration']}_seed{row['seed']}.pt") != row['checkpoint_sha256']:
                raise ValueError('Individual checkpoint file changed during comparison')
    return {'schema': IDENTITY_SCHEMA, 'only_summary_field_excluded': 'models[*].checkpoint_sha256',
        'individual_final_file_hashes_verified': True, 'all_checkpoint_contents_bitwise_equal': True,
        'all_final_and_motion_stage_history_bytes_equal': True, 'checkpoints_each_run': len(paths),
        'identical_serialized_checkpoint_pairs': equal_files,
        'different_serialized_checkpoint_pairs': len(paths)-equal_files,
        'training_summaries_preserved': True, 'checkpoint_files_rewritten': False}
