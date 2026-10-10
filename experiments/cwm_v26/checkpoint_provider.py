"""Lazy read-only V23 primary checkpoint provider for noncontrolling shadow."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'cwm_v25'), str(HERE.parent / 'cwm_v21')]
from qualification_gate import preserved_cpu_rng
from artifact_transport import resolve_artifact, file_digest

ARCHIVE_SHA = '2ec8d4165e2b56c99f1781c38e54122a481ef63dc3f8aacf1069d53ace11ca78'
SUMMARY_SHA = 'ea519ccb76c714224672ac09ecc9b68029cdd80540bb5e0095ba8d544faf5b5a'
CONFIGURATION = 'cv_cost_l2'
SEED = 993102  # V23's fixed ADE-median primary, NOT a new best-seed search.


def checkpoint_predictor(raw, expected_sha):
    """Only public six-array inference; no labels, costs or future truth input."""
    import torch
    sys.path.insert(0, str(HERE.parent / 'cwm_v23'))
    from cost_origin_model import CalibratedOrigin, FrozenOriginResponse, LocalTwoHead
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Pinned checkpoint bytes differ')
    cp = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if (cp['configuration'] != CONFIGURATION or cp['seed'] != SEED or cp['origin'] != 'cv' or
            any(cp[k] is not False for k in ('online_promoted', 'baseline_weights_included',
                'mediator_optimized', 'common_motion_in_response_optimizer'))):
        raise ValueError('V23 primary checkpoint contract differs')
    protocol = json.loads((HERE.parent / 'cwm_v23/training_protocol.json').read_text())
    if cp['protocol'] != protocol:
        raise ValueError('Original V23 training protocol differs')
    config = protocol['training']
    with preserved_cpu_rng():
        common = CalibratedOrigin(LocalTwoHead('motion_only', config['motion_scale_m'], config['response_scale_m']), 'cv')
        model = FrozenOriginResponse(common, LocalTwoHead('plain', config['motion_scale_m'], config['response_scale_m']))
        model.load_state_dict(cp['model_state'], strict=True)
    model.eval().requires_grad_(False)
    mean, scale = (cp[key].numpy().reshape(1, 252) for key in ('normalizer_mean', 'normalizer_scale'))
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale < .01).any():
        raise ValueError('Train-only normalizer contract differs')
    def predict(history, relative, proposed, anchor, backbone, cv):
        n = len(proposed)
        inputs = (
            torch.as_tensor((history - mean) / scale, dtype=torch.float32)[None].repeat(n, 1, 1),
            torch.as_tensor(relative, dtype=torch.float32)[None].repeat(n, 1, 1),
            torch.as_tensor(proposed, dtype=torch.float32),
            torch.as_tensor(anchor, dtype=torch.float32)[None].repeat(n, 1, 1, 1),
            torch.as_tensor(backbone, dtype=torch.float64)[None].repeat(n, 1, 1),
            torch.as_tensor(cv, dtype=torch.float64)[None].repeat(n, 1, 1))
        with torch.no_grad():
            prediction, _, response = model(*inputs)
            reference = model.common(*inputs)[0]
        return {k: v.numpy().copy() for k, v in
                (('prediction', prediction), ('reference', reference), ('response', response))}
    return predict


class PinnedV23Loader:
    """Constructing the loader does not read a manifest, archive or checkpoint."""
    def __init__(self, artifact, cache, fault=None):
        if fault not in (None, 'load', 'predict'):
            raise ValueError('Unknown controlled fault')
        self.artifact, self.cache, self.fault = Path(artifact), Path(cache), fault
        self.checkpoint_sha256 = None

    def __call__(self):
        archive = resolve_artifact(self.artifact, self.cache)
        if file_digest(archive) != ARCHIVE_SHA:
            raise ValueError('Audited V23 model archive differs')
        with zipfile.ZipFile(archive) as z:
            summary_raw = z.read('primary/summary.json')
            if hashlib.sha256(summary_raw).hexdigest() != SUMMARY_SHA:
                raise ValueError('Audited primary training summary differs')
            report = json.loads(summary_raw)
            if (report['protocol']['primary_configuration'] != CONFIGURATION or
                    report['selected_median_seeds'][CONFIGURATION] != SEED or
                    report['primary_research_eligible'] is not False or report['enhanced_control_enabled'] is not False):
                raise ValueError('Pinned failed primary status differs')
            row = next(r for r in report['models'] if r['configuration'] == CONFIGURATION and r['seed'] == SEED)
            raw = z.read(f'primary/{CONFIGURATION}_seed{SEED}.pt')
        self.checkpoint_sha256 = row['checkpoint_sha256']
        predict = checkpoint_predictor(raw, self.checkpoint_sha256)
        if self.fault == 'load':
            raise OSError('Controlled failure after actual pinned checkpoint reconstruction')
        if self.fault == 'predict':
            def fail(*args):
                predict(*args)  # Exercise real forward before the controlled error.
                raise RuntimeError('Controlled failure after actual model forward')
            return fail
        return predict
