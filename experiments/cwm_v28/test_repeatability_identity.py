"""Serialization/strict-rejection fixtures; not scientific release evidence."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repeatability_identity as identity
import package_ranking_release_v2 as release
import test_package_release as old_fixtures


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf8')


def checkpoint(reverse=False):
    sources = [('a.py', 'a'*64), ('b.py', 'b'*64)]
    return {'model_state': {'weight': torch.tensor([0., 1.], dtype=torch.float32)},
        'optimizer_state': {'state': {0: {'exp_avg': torch.tensor([.2, .3])}}, 'lr': .001},
        'torch_rng_state': torch.tensor([1, 2, 3], dtype=torch.uint8),
        'sampler_rng_state': {'position': 12, 'state': [7, 8]},
        'normalizer_mean': torch.tensor([0., 1.]), 'source_hashes': dict(reversed(sources) if reverse else sources),
        'data_summary_sha256': 'c'*64, 'protocol': {'enabled': False}}


@pytest.fixture
def pair(tmp_path):
    folders = [tmp_path/'primary', tmp_path/'repeated']
    protocol = {'models': ['cv_motion_only', 'cv_l2', 'cv_rank_l2'],
        'training': {'seeds': [994101, 994102, 994103]}}
    reports = []
    for i, folder in enumerate(folders):
        folder.mkdir()
        rows = []
        for model in protocol['models']:
            for seed in protocol['training']['seeds']:
                path = folder/f'{model}_seed{seed}.pt'
                torch.save(checkpoint(bool(i)), path)
                rows.append({'configuration': model, 'seed': seed, 'checkpoint_sha256': identity.digest(path),
                    'development': {'ade': .3}, 'parameters': 2})
                (folder/f'{model}_seed{seed}_history.json').write_bytes(b'[{"epoch":80}]')
        for seed in protocol['training']['seeds']:
            torch.save(checkpoint(bool(i)), folder/f'cv_seed{seed}_motion_stage.pt')
            (folder/f'cv_seed{seed}_motion_stage_history.json').write_bytes(b'[{"epoch":80}]')
        report = {'protocol': protocol, 'models': rows, 'holdout_used': False,
            'data_summary_sha256': 'd'*64, 'qualification': {'research_eligible': False},
            'selected_median_seeds': {'cv_rank_l2': 994101}, 'split_groups': ['g0', 'g1']}
        save_json(folder/'summary.json', report)
        reports.append(report)
    return folders, reports


def test_distinct_serialization_keeps_all_individual_identities_and_no_rewrite(pair):
    folders, reports = pair
    before = {str(p): identity.digest(p) for f in folders for p in f.iterdir()}
    result = identity.verify_pair(*folders, *reports)
    assert result['checkpoints_each_run'] == 12
    assert result['different_serialized_checkpoint_pairs'] == 12
    assert result['individual_final_file_hashes_verified'] is True
    assert result['checkpoint_files_rewritten'] is False
    assert before == {str(p): identity.digest(p) for f in folders for p in f.iterdir()}
    assert reports[0] != reports[1]


@pytest.mark.parametrize('field', ['model_state', 'optimizer_state', 'torch_rng_state',
    'sampler_rng_state', 'normalizer_mean', 'source_hashes', 'data_summary_sha256', 'protocol'])
@pytest.mark.parametrize('stage', ['final', 'motion_stage'])
def test_coherently_rehashed_content_change_is_rejected(pair, field, stage):
    folders, reports = pair
    name = 'cv_motion_only_seed994101' if stage == 'final' else 'cv_seed994101_motion_stage'
    path = folders[1]/(name+'.pt')
    value = torch.load(path, weights_only=True)
    if field == 'model_state': value[field]['weight'][1] += .01
    elif field == 'optimizer_state': value[field]['state'][0]['exp_avg'][0] += .01
    elif field == 'torch_rng_state': value[field][0] += 1
    elif field == 'sampler_rng_state': value[field]['position'] += 1
    elif field == 'normalizer_mean': value[field][1] += .01
    elif field == 'source_hashes': value[field]['a.py'] = '0'*64
    elif field == 'data_summary_sha256': value[field] = '0'*64
    else: value[field]['enabled'] = True
    torch.save(value, path)
    if stage == 'final':
        reports[1]['models'][0]['checkpoint_sha256'] = identity.digest(path)
        save_json(folders[1]/'summary.json', reports[1])
    with pytest.raises(ValueError, match='contents differ'):
        identity.verify_pair(*folders, *reports)


@pytest.mark.parametrize('change', ['metric', 'gate', 'seed_selection', 'support', 'data', 'holdout_type', 'missing_model'])
def test_summary_excludes_no_scientific_fields(pair, change):
    folders, reports = pair
    report = reports[1]
    if change == 'metric': report['models'][0]['development']['ade'] += .01
    elif change == 'gate': report['qualification']['research_eligible'] = True
    elif change == 'seed_selection': report['selected_median_seeds']['cv_rank_l2'] = 994102
    elif change == 'support': report['split_groups'].pop()
    elif change == 'data': report['data_summary_sha256'] = '0'*64
    elif change == 'holdout_type': report['holdout_used'] = 0
    else: report['models'].pop()
    save_json(folders[1]/'summary.json', report)
    with pytest.raises(ValueError): identity.verify_pair(*folders, *reports)


def test_own_digest_cannot_be_ignored(pair):
    folders, reports = pair
    reports[1]['models'][0]['checkpoint_sha256'] = '0'*64
    save_json(folders[1]/'summary.json', reports[1])
    with pytest.raises(ValueError, match='Individual checkpoint file digest'):
        identity.verify_pair(*folders, *reports)


@pytest.mark.parametrize('name', ['cv_rank_l2_seed994103', 'cv_seed994103_motion_stage'])
def test_history_byte_change_rejected(pair, name):
    folders, reports = pair
    (folders[1]/(name+'_history.json')).write_bytes(b'[{"epoch":79}]')
    with pytest.raises(ValueError, match='history bytes'):
        identity.verify_pair(*folders, *reports)


@pytest.mark.parametrize('left,right', [
    (False, 0), ([1], (1,)), (0., -0.),
    (torch.tensor([0.]), torch.tensor([-0.])),
    (torch.tensor([1.], dtype=torch.float32), torch.tensor([1.], dtype=torch.float64)),
    ({1: 'a'}, {True: 'a'})])
def test_types_signed_zero_and_tensor_dtype_are_not_relaxed(left, right):
    assert not identity.exact_tree(left, right)


@pytest.fixture
def full_inventory(tmp_path, monkeypatch):
    roots = old_fixtures.complete_inventory_fixture.__wrapped__(tmp_path, monkeypatch)
    here = tmp_path/'repo/experiments/cwm_v28'
    monkeypatch.setattr(release, 'ROOT', tmp_path/'repo')
    monkeypatch.setattr(release, 'HERE', here)
    for name in ('package_ranking_release_v2.py', 'release_ranking_training_v2.py', 'repeatability_identity.py'):
        (here/name).write_bytes(b'fixture-v2-source')
    reports = []
    for i, stage in enumerate(('primary', 'repeated')):
        report = release.read_json(roots[stage]/'summary.json')
        for row in report['models']:
            path = roots[stage]/f"{row['configuration']}_seed{row['seed']}.pt"
            torch.save(checkpoint(bool(i)), path)
            row['checkpoint_sha256'] = identity.digest(path)
        for seed in report['protocol']['training']['seeds']:
            torch.save(checkpoint(bool(i)), roots[stage]/f'cv_seed{seed}_motion_stage.pt')
        save_json(roots[stage]/'summary.json', report)
        reports.append(report)
    verified = release.read_json(roots['reload']/'summary.json')
    verified['repeatability_identity'] = identity.verify_pair(roots['primary'], roots['repeated'], *reports)
    verified['reload_source_hashes'] = {n: identity.digest(here/n) for n in ('release_ranking_training_v2.py', 'repeatability_identity.py')}
    for stage in ('primary', 'repeated'):
        verified[stage+'_summary_sha256'] = identity.digest(roots[stage]/'summary.json')
    save_json(roots['reload']/'summary.json', verified)
    return roots


def test_v2_curated_inventory_preserves_both_distinct_file_hashes(full_inventory):
    roots = full_inventory
    members, previous = release.inventory(**roots)
    assert previous['primary_research_eligible'] is False
    assert len([n for n in members if n.startswith(('primary/', 'repeated/')) and n.endswith('.pt')]) == 24
    assert 'verification_source/experiments/cwm_v28/repeatability_identity.py' in members
    assert identity.digest(members['primary/cv_rank_l2_seed994101.pt']) != identity.digest(members['repeated/cv_rank_l2_seed994101.pt'])


@pytest.mark.parametrize('changed', ['identity', 'reload_source'])
def test_v2_old_or_forged_reload_refused(full_inventory, changed):
    roots = full_inventory
    report = release.read_json(roots['reload']/'summary.json')
    if changed == 'identity': report.pop('repeatability_identity')
    else: report['reload_source_hashes']['repeatability_identity.py'] = '0'*64
    save_json(roots['reload']/'summary.json', report)
    with pytest.raises(ValueError, match='contents/new reload source'):
        release.inventory(**roots)


@pytest.mark.parametrize('fail', [None, 'data', 'reload', 'reload_compare'])
def test_v2_archive_uses_new_full_reload_and_fails_closed(full_inventory, tmp_path, monkeypatch, fail):
    """Real complete ZIP/contents checks, mocked scientific subprocesses."""
    roots = full_inventory
    previous = release.read_json(roots['reload']/'summary.json')
    entries = []
    def child(script, arguments, output):
        entries.append(script)
        if (script == 'audit_ranking_data.py' and fail == 'data') or (
            script == 'release_ranking_training_v2.py' and fail == 'reload'):
            raise subprocess.CalledProcessError(1, script)
        if script == 'release_ranking_training_v2.py':
            path = Path(arguments[-1]); path.mkdir()
            release.write_json(path/'summary.json', {} if fail == 'reload_compare' else previous)
    monkeypatch.setattr(release, 'run_child', child)
    monkeypatch.setattr(release, 'compare_replayed_data', lambda *a: None)
    archive, output = tmp_path/'v2.zip', tmp_path/'v2-replay'
    if fail:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            release.package(**roots, archive_path=archive, replay_output=output)
        assert not (output/'summary.json').exists()
    else:
        result = release.package(**roots, archive_path=archive, replay_output=output)
        assert entries == ['audit_ranking_data.py', 'release_ranking_training_v2.py']
        assert result['primary_research_eligible'] is False
        assert result['enhanced_control_enabled'] is False
        assert result['holdout_used'] is False
