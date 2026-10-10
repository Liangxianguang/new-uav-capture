"""Pin unchanged V34 plan plus explicit V36 evaluation/transport details."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'cwm_v35'))
from audited_stream import sha


def load_contract():
    contract = json.loads((HERE/'protocol.json').read_bytes())
    for stage, key in (('data_protocol.json', 'v34_data_protocol_sha256'),
                       ('training_protocol.json', 'v34_training_protocol_sha256')):
        if sha(HERE.parent/'cwm_v34'/stage) != contract[key]:
            raise ValueError('Unchanged V34 scientific plan identity required')
    plan = json.loads((HERE.parent/'cwm_v34/training_protocol.json').read_bytes())
    validate_contract(contract, plan)
    return contract, plan


def validate_contract(contract, plan):
    if (contract != json.loads((HERE/'protocol.json').read_bytes()) or
        plan != json.loads((HERE.parent/'cwm_v34/training_protocol.json').read_bytes()) or
        contract['primary_configuration'] != plan['primary_configuration'] or
        plan['models'] != ['cv_motion_only', 'cv_full_query_rank_l2', 'cv_prefix_rank_l2'] or
        plan['training']['seeds'] != [1034101, 1034102, 1034103] or
        (plan['training']['motion_epochs'], plan['training']['response_epochs'], plan['training']['batch_calls']) != (80, 80, 16) or
        any(plan[k] is not False for k in ('enhanced_control_enabled', 'original_gru_in_optimizer',
            'common_motion_in_response_optimizer', 'private_labels_model_inputs', 'holdout_used', 'prior_gate_override_allowed')) or
        any(contract[k] is not False for k in ('enhanced_control_enabled', 'holdout_used', 'old_failed_gates_overridden')) or
        any(contract[k] is not True for k in ('two_complete_runs_required', 'independent_full_reload_required',
            'original8Levels_new_scene_capture_safety_latency_fallback_holdout_required'))):
        raise ValueError('Fixed original/default-off/full-comparison training contract required')
