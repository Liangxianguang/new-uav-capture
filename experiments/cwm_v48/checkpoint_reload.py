"""Strict cold model/optimizer/RNG reconstruction; never optimizer.step()."""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/'cwm_v48'), str(HERE.parent/'cwm_v33'), str(HERE.parent/'cwm_v28')]
from contract import load_contract
from prefix_response_core import PrefixQueryResponse
from train_ranking_response import LocalTwoHead, CalibratedOrigin, FrozenOriginResponse, state_digest
from qualification_gate import preserved_cpu_rng
from repeatability_identity import exact_tree


def finite_state(node):
    if torch.is_tensor(node):
        if (node.is_floating_point() or node.is_complex()) and not torch.isfinite(node).all():
            raise ValueError('Finite complete checkpoint tensor state required')
    elif isinstance(node, dict):
        for value in node.values(): finite_state(value)
    elif isinstance(node, (tuple, list)):
        for value in node: finite_state(value)
    elif isinstance(node, float) and not math.isfinite(node):
        raise ValueError('Finite checkpoint scalar state required')


def reconstruct(checkpoint, name, seed, train_indices, stage=False):
    contract, plan = load_contract()
    if (checkpoint['v34_plan'] != plan or checkpoint['v36_contract'] != contract or checkpoint['seed'] != seed or
        name not in plan['models'] or seed not in plan['training']['seeds'] or
        checkpoint['online_promoted'] is not False or checkpoint['baseline_weights_included'] is not False or
        (not stage and (checkpoint['configuration'] != name or checkpoint['common_motion_in_response_optimizer'] is not False)) or
        (stage and name != 'cv_motion_only')):
        raise ValueError('Fixed original/default-off final checkpoint identity required')
    finite_state(checkpoint)
    with preserved_cpu_rng():
        torch.manual_seed(seed)
        common = CalibratedOrigin(LocalTwoHead('motion_only', 2.5, 2.5), 'cv')
        if name == 'cv_motion_only':
            model = common; initial = state_digest(common); parameters = list(model.parameters())
        else:
            # Response initialization reseeds after the independent motion stage.
            torch.manual_seed(seed); core = PrefixQueryResponse(name == 'cv_prefix_rank_l2')
            initial = state_digest(core); model = FrozenOriginResponse(common, core)
            parameters = [p for p in model.parameters() if p.requires_grad]
        expected_torch_rng = torch.get_rng_state()
        if initial != checkpoint['initial_state_digest']:
            raise ValueError('Fixed copied response/motion initialization provenance differs')
        model.load_state_dict(checkpoint['model_state'], strict=True)
        if not exact_tree(model.state_dict(), checkpoint['model_state']):
            raise ValueError('Cold model state changed dtype/shape/bytes')
        # Loading into a real Adam restores all moments/options, without an update.
        optimizer = torch.optim.Adam(parameters, lr=.001)
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_state']))
        if not exact_tree(optimizer.state_dict(), checkpoint['optimizer_state']):
            raise ValueError('ALL Adam state/options must survive actual optimizer reload')
        expected_steps = 80*math.ceil(len(train_indices)/16)
        options = checkpoint['optimizer_state']['param_groups']
        if (len(options) != 1 or options[0]['lr'] != .001 or options[0]['betas'] != (.9, .999) or
            options[0]['eps'] != 1e-8 or options[0]['weight_decay'] != 0 or options[0]['amsgrad'] is not False or
            len(options[0]['params']) != len(parameters)):
            raise ValueError('Fixed complete Adam parameter/options contract differs')
        parameter_map = dict(zip(options[0]['params'], parameters))
        if not checkpoint['optimizer_state']['state']:
            raise ValueError('Empty optimizer state cannot prove fixed80epoch training')
        names = {id(p): name for name, p in model.named_parameters()}
        expected_moments = {key for key, p in parameter_map.items()
            if name != 'cv_motion_only' or not names[id(p)].startswith('core.response_head.')}
        if set(checkpoint['optimizer_state']['state']) != expected_moments:
            raise ValueError('ALL actually trained parameters require complete Adam moments')
        for key, state in checkpoint['optimizer_state']['state'].items():
            if key not in parameter_map or set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Unexpected/incomplete Adam moment state')
            if (float(state['step']) != expected_steps or state['exp_avg'].shape != parameter_map[key].shape or
                state['exp_avg_sq'].shape != parameter_map[key].shape or
                state['exp_avg'].dtype != parameter_map[key].dtype or state['exp_avg_sq'].dtype != parameter_map[key].dtype):
                raise ValueError('Final Adam step/moment budget differs from ALL TRAIN80epoch batches')
        sampler = np.random.default_rng(seed)
        for _ in range(80): sampler.permutation(train_indices)
        if not exact_tree(sampler.bit_generator.state, checkpoint['sampler_rng_state']):
            raise ValueError('Final fixed ALL-TRAIN80epoch sampler RNG differs')
        sampler.bit_generator.state = checkpoint['sampler_rng_state']
        if not exact_tree(sampler.bit_generator.state, checkpoint['sampler_rng_state']):
            raise ValueError('Actual sampler RNG restoration differs')
        if not exact_tree(expected_torch_rng, checkpoint['torch_rng_state']):
            raise ValueError('No stochastic training layer is allowed to change fixed initialization Torch RNG')
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if not exact_tree(torch.get_rng_state(), checkpoint['torch_rng_state']):
            raise ValueError('Actual Torch RNG restoration differs')
    model.eval().requires_grad_(False)
    common = model.common if isinstance(model, FrozenOriginResponse) else model
    if not stage and state_digest(common) != checkpoint['common_motion_state_digest']:
        raise ValueError('Cold frozen common-motion digest differs')
    return model, {'model_state_exact': True, 'actual_optimizer_state_exact': True,
        'actual_sampler_torch_rng_exact': True, 'expected_optimizer_steps': expected_steps,
        'optimizer_updated_by_reload': False}
