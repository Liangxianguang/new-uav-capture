"""ALL native public command prefix identities before float32 optimization."""
import numpy as np


def command_precision(call):
    values = call['values']; proposed = values['proposed']; anchor = values['anchor']
    public = (proposed == anchor[None]).reshape(len(proposed), 8, -1).all(2).cumprod(1).astype(bool)
    cast = (proposed.astype(np.float32) == anchor.astype(np.float32)[None]).reshape(len(proposed), 8, -1).all(2).cumprod(1).astype(bool)
    if not np.array_equal(public, cast):
        raise ValueError('Float32 command conversion changes hard public prefix-anchor identity')
    for base, length, probe in values['prefix_pairs']:
        if not np.array_equal(proposed[base, :length].astype(np.float32), proposed[probe, :length].astype(np.float32)):
            raise ValueError('Native matched command prefix differs at actual neural precision')
    return {'candidates': len(proposed), 'native_prefix_pairs': len(values['prefix_pairs']),
        'native_float32_prefix_anchor_identity_equal': True}


def audit_command_precision(corpus):
    total = {'calls': 0, 'candidates': 0, 'native_prefix_pairs': 0}
    for index in range(len(corpus)):
        result = command_precision(corpus.load(index)); total['calls'] += 1
        for key in ('candidates', 'native_prefix_pairs'): total[key] += result[key]
    return {'status': 'all_native_float32_command_prefix_anchor_identities_equal', **total,
        'scope': 'Input identity audit only, no causal/predictive efficacy or observability claim.'}
