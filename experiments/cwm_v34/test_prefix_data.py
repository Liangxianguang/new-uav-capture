"""SYNTHETIC public-prefix and pipeline fixtures, never native data evidence."""
import copy
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_prefix_data as collect
import audit_prefix_data as audit
from prefix_library import augment, check_native_prefix
from stream_normalization import stream_history_normalization


def public_library():
    anchor = np.zeros((8, 4, 3), dtype=np.float64)
    joint = np.repeat(anchor[None], 3, 0)
    joint[1, :, 0, 0] = .5; joint[2, :, 0, 1] = 1.
    return joint, anchor


def labels(commands):
    return {'target': commands[:, 0].cumsum(0), 'defenders': commands.cumsum(0),
        'commanded': commands.copy(), 'executed': commands.copy(), 'valid': np.ones(8, dtype=bool),
        'branch_sign_label_only': np.zeros(8, dtype=np.int8), 'termination': np.full(8, 'running', dtype='U128')}


def toy_values():
    joint, anchor = public_library(); augmented, pairs = augment(joint, anchor, 0)
    values = {k: np.stack([labels(c)[k] for c in augmented]) for k in labels(anchor)}
    values.update({'anchor_'+k: v for k, v in labels(anchor).items()})
    values.update(proposed=augmented, anchor=anchor, prefix_pairs=pairs,
        base_candidate_count=np.asarray(len(joint), dtype=np.int64),
        bounded_shared_indices=np.arange(len(joint), dtype=np.int64),
        public_geometry_indices=np.arange(len(joint), dtype=np.int64), actual_candidates=joint[:, :, 0])
    return values


def test_all_base_commands_peers_prefixes_and_original_arrays_preserved():
    joint, anchor = public_library(); before = joint.copy(); old_anchor = anchor.copy()
    extended, pairs = augment(joint, anchor, 0)
    assert extended[:len(joint)].tobytes() == joint.tobytes()
    assert joint.tobytes() == before.tobytes() and anchor.tobytes() == old_anchor.tobytes()
    assert pairs.tolist() == [[1, 2, 3], [2, 2, 4], [1, 5, 5], [2, 5, 6]]
    for original, k, probe in pairs:
        assert extended[probe, :k].tobytes() == joint[original, :k].tobytes()
        assert np.array_equal(extended[probe, k:, 0], anchor[k:, 0])
        assert np.array_equal(extended[probe, :, 1:], joint[original, :, 1:])
    assert np.linalg.norm(extended, axis=-1).max() <= 5.
    independent, mapping = audit.independent_library(joint, anchor, 0)
    assert independent.tobytes() == extended.tobytes() and mapping.tobytes() == pairs.tobytes()


def test_alias_probes_retained_in_mapping_without_duplicate_candidates():
    joint, anchor = public_library(); joint[1, 2:] = anchor[2:]
    extended, pairs = augment(joint, anchor, 0)
    assert pairs[0].tolist() == [1, 2, 1]
    assert pairs[2].tolist() == [1, 5, 1]
    assert len(extended) == 5


@pytest.mark.parametrize('kind', ['dtype', 'shape', 'nonfinite', 'speed', 'peers', 'anchor_missing', 'duplicate', 'too_many', 'agent_bool', 'prefixes'])
def test_public_library_tampering_refused(kind):
    joint, anchor = public_library(); agent = 0; prefixes = (2, 5)
    if kind == 'dtype': joint = joint.astype(np.float32)
    elif kind == 'shape': joint = joint[:, :7]
    elif kind == 'nonfinite': joint[1, 0, 0, 0] = np.nan
    elif kind == 'speed': joint[1, 0, 0, 0] = 6.
    elif kind == 'peers': joint[1, 0, 1, 0] = 1.
    elif kind == 'anchor_missing': anchor[:, 0, 2] = 1.
    elif kind == 'duplicate': joint[2] = joint[1]
    elif kind == 'too_many': joint = np.repeat(joint[:1], 25, 0)
    elif kind == 'agent_bool': agent = False
    else: prefixes = (1, 7)
    with pytest.raises(ValueError): augment(joint, anchor, agent, prefixes)


@pytest.mark.parametrize('field', ['target', 'defenders', 'commanded', 'executed', 'valid', 'branch_sign_label_only', 'termination'])
def test_every_native_prefix_field_must_match(field):
    values = toy_values(); assert check_native_prefix(values) == 4
    changed = copy.deepcopy(values)
    if field == 'valid': changed[field][3, 0] = False
    elif field == 'termination': changed[field][3, 0] = 'after_terminal'
    else: changed[field][3, 0] += 1
    with pytest.raises(ValueError): check_native_prefix(changed)


def test_terminal_padding_preserved_not_imputed_or_removed():
    values = toy_values()
    for field in ('target', 'defenders', 'commanded', 'executed', 'branch_sign_label_only'):
        values[field][:, 3:] = 0
    values['valid'][:, 3:] = False; values['termination'][:, 3:] = 'after_terminal'
    assert check_native_prefix(values) == 4


@pytest.mark.parametrize('count', [True, 3., np.array([3], dtype=np.int64)])
def test_native_base_count_type_and_shape_refused(count):
    values = toy_values(); values['base_candidate_count'] = count
    with pytest.raises(ValueError): check_native_prefix(values)


@pytest.fixture
def inspect_fixture(monkeypatch):
    joint, anchor = public_library()
    captured = {'scenarios': types.SimpleNamespace(trajectories=np.zeros((1, 8, 3)))}
    config = types.SimpleNamespace(risk_mode='mean', cvar_alpha=.2)
    call = {'agent': 0, 'ordinal': 2, 'local_candidates': joint[:, :, 0], 'selected': anchor[:, 0],
        'observation': {}, 'planner': types.SimpleNamespace(config=config),
        'scenarios': types.SimpleNamespace(normalized_weights=np.array([1.])), 'selected_costs': np.array([0.])}
    base = types.SimpleNamespace(planner_config=None, distributed_config=None,
        evaluator=types.SimpleNamespace(belief_reference=lambda *a: (np.zeros(3), np.zeros(3))))
    result = {'inputs': (np.zeros((8, 252)), np.zeros((4, 6)), joint, anchor, np.zeros((8, 3)), np.zeros((8, 3))),
        'info': {'seed_count': 3, 'status': 'toy'}, 'actions': joint[:, :, 0],
        'forecast': {'prediction': np.zeros((3, 8, 3)), 'reference': np.zeros((3, 8, 3)), 'response': np.zeros((3, 8, 3))}}
    marker = {'frozen': False, 'rolls': 0}
    monkeypatch.setattr(collect.prior, 'public_call_snapshot', lambda _: {'scope': 'SYNTHETIC PUBLIC'})
    monkeypatch.setattr(collect.prior, 'restore_public_call', lambda *a: call)
    monkeypatch.setattr(collect.prior, 'expand_candidates', lambda *a, **k: result)
    monkeypatch.setattr(collect.prior, 'local_cost', lambda *a: np.arange(3, dtype=float))
    package = types.ModuleType('encirclement3d'); module = types.ModuleType('encirclement3d.minimax_mpc')
    module.aggregate_scenario_costs = lambda *a: 0.
    monkeypatch.setitem(sys.modules, 'encirclement3d', package)
    monkeypatch.setitem(sys.modules, 'encirclement3d.minimax_mpc', module)
    original_augment = collect.augment
    def freeze(*args):
        value = original_augment(*args); marker['frozen'] = True; return value
    def private_rollout(base, parent, commands, key):
        assert marker['frozen']; marker['rolls'] += 1; return labels(commands)
    monkeypatch.setattr(collect, 'augment', freeze); monkeypatch.setattr(collect.prior, 's4_rollout', private_rollout)
    return base, captured, result, marker


def test_inspector_freezes_all_public_probes_before_any_private_roll(inspect_fixture):
    base, captured, result, marker = inspect_fixture
    row, values, snapshot = collect.inspect(base, object(), captured, np.zeros((8, 252)), object(), 17)
    assert row['native_prefix_bytes_equal'] is True and row['prefix_pair_count'] == 4
    assert marker['rolls'] == len(values['proposed'])+2
    assert snapshot['scope'].startswith('SYNTHETIC')
    assert np.array_equal(values['proposed'][:3], result['inputs'][2])


def test_missing_peer_never_imputes_or_reads_private_labels(inspect_fixture):
    base, captured, result, marker = inspect_fixture
    result['inputs'] = None; result['info'] = {'status': 'missing_received_peer_context'}
    row, values, _ = collect.inspect(base, object(), captured, np.zeros((8, 252)), object(), 17)
    assert values is None and row['status'] == 'missing_received_peer_context'
    assert marker['rolls'] == 0


def test_support_statistics_never_retain_branch_arrays():
    values = toy_values(); protocol = json.loads((collect.HERE/'data_protocol.json').read_bytes())
    row = {'episode_index': 1, 'step': 2, 'agent': 0, 'group': 'toy', 'split': 'train', 'variant': 'toy'}
    support = collect.Support(); support.add(row, values, protocol)
    assert all(not isinstance(v, np.ndarray) for entry in support.rows for v in entry.values())
    assert [r['candidate_count'] for r in support.rows] == [3, 3, 3, 7]
    assert support.pairs == 4
    before = copy.deepcopy(support.rows); values['target'][:] = 1000.
    assert support.rows == before


def test_actual_protocol_has_all_fresh_groups_and_no_old_holdout_overlap():
    protocol = json.loads((collect.HERE/'data_protocol.json').read_bytes()); collect.validate_protocol(protocol)
    assert protocol['groups'] == 64 and protocol['train_groups'] == 48 and protocol['development_groups'] == 16
    assert len({v['seed_start']+i for v in protocol['variants'] for i in range(128)}) == 512


@pytest.mark.parametrize('field,value', [('groups', 32), ('target_branch_rule_overrides', {'x': 1}),
    ('holdout_used', True), ('maximum_command_speed_mps', 6), ('prefix_lengths', [1, 7])])
def test_protocol_changes_refused(field, value):
    protocol = json.loads((collect.HERE/'data_protocol.json').read_bytes()); protocol[field] = value
    with pytest.raises(ValueError): collect.validate_protocol(protocol)


def test_existing_collection_output_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path/'existing'; output.mkdir(); (output/'marker').write_bytes(b'user-owned')
    monkeypatch.setattr(collect, 'published', lambda: pytest.fail('Must reject before remote read'))
    with pytest.raises(ValueError): collect.main(output)
    assert (output/'marker').read_bytes() == b'user-owned'


def test_unpublished_collection_refuses_before_output_creation(tmp_path, monkeypatch):
    def refuse(): raise ValueError('Unpublished')
    monkeypatch.setattr(collect, 'published', refuse)
    output = tmp_path/'not-created'
    with pytest.raises(ValueError): collect.main(output)
    assert not output.exists()


def test_stream_normalizer_matches_uniform_history_math_and_repeats_exactly():
    generator = np.random.default_rng(3401); blocks = [generator.normal(size=(8, 252)) for _ in range(7)]
    a = stream_history_normalization(iter(blocks)); b = stream_history_normalization(iter(blocks))
    combined = np.stack(blocks)
    assert np.allclose(a[0], combined.mean((0, 1), keepdims=True), atol=1e-15, rtol=1e-15)
    assert np.allclose(a[1], combined.std((0, 1), keepdims=True), atol=1e-15, rtol=1e-15)
    assert a[2] == 56 and a[0].dtype == np.float64 and a[0].shape == (1, 1, 252)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()


@pytest.mark.parametrize('blocks', [[], [np.zeros((7, 252))], [np.full((8, 252), np.nan)]])
def test_invalid_or_empty_normalizer_support_refused(blocks):
    with pytest.raises(ValueError): stream_history_normalization(iter(blocks))


def test_constant_history_std_floor_is_original_point01():
    mean, scale, count = stream_history_normalization([np.ones((8, 252))])
    assert np.array_equal(mean, np.ones((1, 1, 252))) and np.array_equal(scale, np.full((1, 1, 252), .01))
    assert count == 8


@pytest.fixture
def geometry_fixture(monkeypatch):
    @dataclass
    class Wall:
        center_xy: np.ndarray
        half_extents_xy: np.ndarray
    @dataclass
    class Scene:
        obstacles: tuple
        defender_positions: np.ndarray
        obstacle_zone_x: tuple
        name: str
        target_escape_direction: tuple = (1., 0., 0.)
    module = types.ModuleType('encirclement3d.showcase')
    module.s4_adaptive_branching_scenario = lambda env, layout, bias, variation: Scene(
        (Wall(np.zeros(2), np.array([.5, 4.])),), np.array([[-7., 2., 5.], [-7., -2., 5.], [-6., 2., 5.], [-6., -2., 5.]]),
        (-.5, .5), f'SYNTHETIC_{layout}_{bias}')
    module.scenario_metadata = lambda s: {'name': s.name, 'wall': s.obstacles[0].center_xy.tolist(),
        'extents': s.obstacles[0].half_extents_xy.tolist(), 'defenders': s.defender_positions.tolist(),
        'target_escape_direction': list(s.target_escape_direction)}
    module.s4_branch_route_metrics = lambda *a: {'scope': 'SYNTHETIC ROUTE'}
    module.validate_s4_branching_scenario = lambda *a: None
    monkeypatch.setitem(sys.modules, 'encirclement3d', types.ModuleType('encirclement3d'))
    monkeypatch.setitem(sys.modules, 'encirclement3d.showcase', module)
    base = types.SimpleNamespace(configuration=lambda r: {}, env_class=lambda *a, **k: object())
    return base, module


def test_all_geometry_axes_mirrors_and_paired_rng_share_one_split(geometry_fixture):
    base, _ = geometry_fixture; protocol = json.loads((collect.HERE/'data_protocol.json').read_bytes())
    population = collect.scenes(base, protocol)
    assert len(population) == 512 and len({r['episode_index'] for r in population}) == 512
    assert len({r['mirror_group_id'] for r in population}) == 64
    for group in range(64):
        selected = [r for r in population if r['layout_seed'] == protocol['layout_seed_start']+group]
        assert len(selected) == 8
        assert {r['model_split'] for r in selected} == {'train' if group < 48 else 'development_validation'}
        assert len({r['variant'] for r in selected}) == 4
        for member in ('upper', 'lower'):
            mirrored = [r for r in selected if r['mirror_pair_member'] == member]
            assert len({r['episode_seed'] for r in mirrored}) == 1
            assert len({r['controller_sampling_identity'] for r in mirrored}) == 1
        assert all(r['target_motion_mode'] == 'adaptive_branching' for r in selected)
        assert all(r['scenario']['target_escape_direction'] == [1., 0., 0.] for r in selected)
    reference, shifted, wider, shorter = population[:4]
    assert reference['scenario']['wall'] == [1.5, 0.]
    assert shifted['scenario']['wall'] == [2.25, 0.]
    assert wider['scenario']['defenders'][0][1] == pytest.approx(2.8)
    assert shorter['scenario']['extents'] == [.5, 3.2]


def test_route_errors_are_retained_for_all_variants_never_replaced(geometry_fixture, monkeypatch):
    base, module = geometry_fixture; protocol = json.loads((collect.HERE/'data_protocol.json').read_bytes())
    original = module.s4_adaptive_branching_scenario
    def fail_first(env, layout, bias, variation):
        if layout == protocol['layout_seed_start']: raise ValueError('SYNTHETIC invalid route')
        return original(env, layout, bias, variation)
    monkeypatch.setattr(module, 's4_adaptive_branching_scenario', fail_first)
    population = collect.scenes(base, protocol)
    assert len(population) == 512 and sum(not r['route_valid'] for r in population) == 8
    assert all(r['route_error'] == 'SYNTHETIC invalid route' for r in population[:8])


def test_paired_run_copies_only_sampling_identity_without_mutating_record(monkeypatch):
    instance = collect.PairedTap.__new__(collect.PairedTap)
    record = {'episode_index': 123, 'controller_sampling_identity': 17, 'nested': [42]}
    def original_run(self, supplied, trajectory, observer):
        assert supplied['episode_index'] == 17
        supplied['nested'][0] = 99
        return {'episode_index': supplied['episode_index']}, []
    monkeypatch.setattr(collect.prior.SequentialTap, 'run', original_run)
    row, _ = instance.run(record)
    assert row['episode_index'] == 123 and record['nested'] == [42]


def test_plain_unhooks_capture_class_and_restores_even_on_failure():
    instance = collect.PairedTap.__new__(collect.PairedTap); current, parent = object(), object()
    instance.evaluator = types.SimpleNamespace(DistributedMinimaxDNMPC=current); instance.collection_parent = parent
    def fail(*args):
        assert instance.evaluator.DistributedMinimaxDNMPC is parent
        raise ValueError('SYNTHETIC native failure')
    instance.run = fail
    with pytest.raises(ValueError): instance.plain({}, None, None)
    assert instance.evaluator.DistributedMinimaxDNMPC is current


def test_existing_audit_output_refuses_before_data_reads(tmp_path, monkeypatch):
    output = tmp_path/'existing'; output.mkdir()
    monkeypatch.setattr(audit, 'preflight', lambda _: pytest.fail('Must reject before evidence read'))
    with pytest.raises(ValueError): audit.audit(tmp_path/'missing-data', output)


def test_audit_refuses_uncompleted_or_failed_collection_before_native_restore(tmp_path):
    report = {'status': 'partial', 'protocol': json.loads((collect.HERE/'data_protocol.json').read_bytes())}
    (tmp_path/'summary.json').write_text(json.dumps(report), encoding='utf8')
    with pytest.raises(ValueError): audit.audit(tmp_path, tmp_path/'never-created')
    assert not (tmp_path/'never-created').exists()


def test_manifest_requires_all_trajectories_commands_sources_and_rejects_mutation(tmp_path):
    report = {'source_hashes': {'experiments/toy.py': 'unused'}}
    population = [{'episode_index': 17}]; records = [{'context_path': 'calls/a.json', 'arrays_path': 'calls/a.npz'}]
    expected = {'summary.json', 'preregistration.json', 'scenes.jsonl', 'geometry_preflight.json', 'records.json', 'episodes.json',
        'source/experiments/toy.py', 'calls/a.json', 'calls/a.npz', 'observed/17.npz', 'observed/17.commands.npz',
        'plain/17.npz', 'plain/17.commands.npz'}
    for name in expected:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'SYNTHETIC EVIDENCE')
    manifest = audit.evidence_manifest(tmp_path, report, population, records)
    assert set(manifest) == expected
    (tmp_path/'plain/17.commands.npz').write_bytes(b'changed')
    assert audit.evidence_manifest(tmp_path, report, population, records) != manifest


def test_manifest_unsafe_relative_paths_refused(tmp_path):
    with pytest.raises(ValueError): audit.evidence_manifest(tmp_path, {'source_hashes': {}}, [], [{'context_path': '../outside'}])
