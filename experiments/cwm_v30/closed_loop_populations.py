"""Complete pinned original development population; no new/holdout generation."""
import copy
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def validate_protocol(protocol):
    if protocol != json.loads((HERE/'closed_loop_protocol.json').read_bytes()):
        raise ValueError('Published fixed full-population protocol differs')
    if (protocol['original_population']['total_episodes'] != 1280 or
        protocol['original_population']['primary_levels'] != list(range(1, 9)) or
        protocol['reserved_holdout_not_collected']['seed_start'] != 966010 or
        protocol['reserved_holdout_not_collected']['layout_seed_start'] != 1966010 or
        protocol['max_steps'] != 250 or protocol['dt_seconds'] != .1 or protocol['horizon_steps'] != 8 or
        protocol['maximum_command_speed_mps'] != 5 or protocol['cbf_margin_m'] != .35 or
        any(protocol[k] is not False for k in ('new_training_or_tuning_allowed', 'deployment_control_enabled',
            'active_research_experiment_started', 'holdout_used', 'prior_gate_override_allowed'))):
        raise ValueError('Frozen full-Level/untouched-holdout/original contract differs')


def decode_original_population(scene_raw, curriculum_raw, specification):
    for raw, key in ((scene_raw, 'scenes_sha256'), (curriculum_raw, 'curriculum_sha256')):
        if hashlib.sha256(raw).hexdigest() != specification[key]:
            raise ValueError('Pinned original scene/curriculum source differs')
    curriculum = yaml.safe_load(curriculum_raw)
    actual = {str(c['level']): c['variants'] for c in curriculum['curriculum']['variants_by_level']}
    expected = specification['level_variants']
    if actual != expected or set(expected) != {str(i) for i in range(9)}:
        raise ValueError('All original8Levels plus L0/two L7 variants required')
    mapping = {v: int(level) for level, variants in expected.items() for v in variants}
    if len(mapping) != sum(len(v) for v in expected.values()):
        raise ValueError('Variant mapped to multiple Levels')
    records = [json.loads(line) for line in scene_raw.decode('utf8').splitlines() if line.strip()]
    if (len(records) != specification['total_episodes'] or
        len({r['episode_index'] for r in records}) != len(records) or
        Counter(r['variant'] for r in records) != Counter({v: specification['episodes_each_variant'] for v in mapping})):
        raise ValueError('Complete preassigned original development corpus required')
    result = []
    for record in records:
        if ('development' not in record['scene_block'] or record.get('level', mapping[record['variant']]) != mapping[record['variant']] or
            966010 <= record['episode_index'] <= 966025 or 966010 <= record['episode_seed'] <= 966025 or
            record['layout_seed'] is not None and 1966010 <= record['layout_seed'] <= 1966017):
            raise ValueError('Wrong Level/development split or reserved identity')
        result.append({**copy.deepcopy(record), 'level': mapping[record['variant']]})
    order = {variant: index for level in range(9) for index, variant in enumerate(expected[str(level)])}
    result.sort(key=lambda r: (r['level'], order[r['variant']], r['mirror_group_id'], r['mirror_pair_member'], r['episode_index']))
    return result


def original_population(capsule, protocol):
    validate_protocol(protocol)
    if hashlib.sha256(Path(capsule).read_bytes()).hexdigest() != protocol['baseline_capsule_sha256']:
        raise ValueError('Pinned original model capsule differs')
    spec = protocol['original_population']
    with zipfile.ZipFile(capsule) as archive:
        return decode_original_population(archive.read(spec['scenes_member']), archive.read(spec['curriculum_member']), spec)


def restricted_safe_capture_time(row, seconds=25.):
    """Do not drop failures or report success-only biased capture timing."""
    if row['safe_capture_success'] is True:
        import math
        value = row['capture_time_seconds']
        if (any(row[k] is not False for k in ('collision', 'boundary_violation', 'target_invalid_episode')) or
            not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= seconds+1e-8):
            raise ValueError('Safe-capture physical/time contract inconsistent')
        return min(float(value), seconds)
    if row['safe_capture_success'] is not False:
        raise ValueError('Exact safe-capture outcome required')
    return float(seconds)
