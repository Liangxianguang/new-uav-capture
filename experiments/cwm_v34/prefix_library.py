"""Public-only fixed own-action prefix augmentation and exact physical checks."""
import numpy as np


def augment(joint, anchor, agent, prefixes=(2, 5)):
    joint, anchor = np.asarray(joint), np.asarray(anchor)
    if (joint.ndim != 4 or joint.shape[1:] != (8, 4, 3) or anchor.shape != (8, 4, 3) or
        joint.dtype != np.float64 or anchor.dtype != np.float64 or not 1 <= len(joint) <= 24 or
        type(agent) is not int or not 0 <= agent < 4 or tuple(prefixes) != (2, 5) or
        not np.isfinite(joint).all() or not np.isfinite(anchor).all() or
        max(np.linalg.norm(joint, axis=-1).max(), np.linalg.norm(anchor, axis=-1).max()) > 5.+1e-8):
        raise ValueError('Unchanged finite float64 public bounded24/8step/4agent/5mps contract required')
    peers = [p for p in range(4) if p != agent]
    if not np.array_equal(np.take(joint, peers, axis=2), np.repeat(np.take(anchor, peers, axis=1)[None], len(joint), 0)):
        raise ValueError('Only own-agent plans vary; received peers cannot be changed')
    if sum(np.array_equal(p, anchor) for p in joint) != 1:
        raise ValueError('Unique public original anchor required')
    if any(np.array_equal(joint[i], joint[j]) for i in range(len(joint)) for j in range(i)):
        raise ValueError('Unique unchanged base library required')
    rows = [p.copy() for p in joint]; pairs = []
    for k in prefixes:
        for base_index, proposed in enumerate(joint):
            if np.array_equal(proposed, anchor): continue
            probe = proposed.copy(); probe[k:, agent] = anchor[k:, agent]
            existing = next((i for i, p in enumerate(rows) if np.array_equal(p, probe)), None)
            if existing is None:
                existing = len(rows); rows.append(probe)
            pairs.append((base_index, k, existing))
    return np.stack(rows), np.asarray(pairs, dtype=np.int64).reshape(-1, 3)


def check_native_prefix(values):
    pairs = values['prefix_pairs']; declared = np.asarray(values['base_candidate_count'])
    if declared.shape != () or declared.dtype != np.int64 or not 1 <= int(declared) <= 24:
        raise ValueError('Exact scalar int64 base candidate count required')
    count = int(declared)
    if pairs.dtype != np.int64 or pairs.ndim != 2 or pairs.shape[1:] != (3,):
        raise ValueError('Exact public prefix mapping required')
    checked = 0
    for original, length, probe in pairs:
        if not (0 <= original < count and 0 <= probe < len(values['proposed']) and length in (2, 5)):
            raise ValueError('Prefix mapping outside fixed public population')
        if not np.array_equal(values['proposed'][original, :length], values['proposed'][probe, :length]):
            raise ValueError('Proposed physical prefixes differ')
        for field in ('target', 'defenders', 'commanded', 'executed', 'branch_sign_label_only', 'valid', 'termination'):
            left, right = values[field][original, :length], values[field][probe, :length]
            if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
                raise ValueError('Actual native paired prefix differs: '+field)
        checked += 1
    return checked
