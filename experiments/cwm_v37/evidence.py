"""Explicit complete run inventory; never trust a partial summary or rglob."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent/'cwm_v35'), str(HERE.parent/'cwm_v36'), str(HERE.parent/'cwm_v28')]
from audited_stream import safe_file, checked_json, sha
from contract import load_contract
from repeatability_identity import exact_tree, summary_semantics


def summary(directory, corpus):
    contract, plan = load_contract(); report = json.loads(safe_file(directory, 'summary.json').read_bytes())
    if (report['status'] != 'complete_fixed_streamed_nine_model_training_pending_repeat_reload' or
        report['v34_plan'] != plan or report['v36_contract'] != contract or
        report['data_summary_sha256'] != sha(corpus.data/'summary.json') or
        report['data_audit_summary_sha256'] != sha(corpus.audit/'summary.json') or
        any(report[k] is not False for k in ('enhanced_control_enabled', 'holdout_used', 'prior_gate_overridden',
            'common_motion_in_response_optimizer', 'two_complete_runs_verified', 'independent_full_reload_verified')) or
        len(report['models']) != 9 or {(m['configuration'], m['seed']) for m in report['models']} !=
            {(m, s) for m in plan['models'] for s in plan['training']['seeds']}):
        raise ValueError('Complete exact nine-model offline scientific run required')
    prereg = json.loads(safe_file(directory, 'preregistration.json').read_bytes())
    if prereg['exact_published_bytes_verified'] is not True or prereg['scientific_sources'] != report['source_hashes']:
        raise ValueError('Actual scientific source preregistration identity differs')
    for name, digest in report['source_hashes'].items():
        if sha(safe_file(ROOT, name)) != digest or sha(safe_file(directory, 'source/'+name)) != digest:
            raise ValueError('Actual used scientific sources changed')
        raw = subprocess.check_output(['git', 'show', prereg['commit']+':'+name], cwd=ROOT)
        if raw != (ROOT/name).read_bytes(): raise ValueError('Actual source differs from exact published commit history')
    return report


def inventory(directory, report, corpus):
    files = {'summary.json', 'preregistration.json', 'normalization.npz', 'population_weights.json',
        'public_input_precision_audit.json', 'initial_gru_score_audit.json', 'initial_cv_score_audit.json',
        'initial_optimizer_prerequisites.json'}
    files.update('source/'+name for name in report['source_hashes'])
    plan = report['v34_plan']; total = {str(i) for i in range(len(corpus))}
    for seed in plan['training']['seeds']:
        files.update((f'cv_seed{seed}_motion_stage.pt', f'cv_seed{seed}_motion_stage_history.json'))
        base = f'cache{seed}/'; path = base+'manifest.json'
        manifest = json.loads(safe_file(directory, path).read_bytes())
        if (manifest['status'] != 'all_canonical_references_and_complete_BASE_cost_derivatives_checked' or
            manifest['calls'] != len(corpus) or set(manifest['files']) != total):
            raise ValueError('ALL per-seed canonical references and cost arrays required')
        files.update((path, base+'score_audit.json'))
        if sha(safe_file(directory, base+'score_audit.json')) != manifest['score_audit_sha256']:
            raise ValueError('Whole deployed native score-audit identity differs')
        for index, row in manifest['files'].items():
            if row['path'] != index+'.npz' or type(row['complete_base']) is not bool:
                raise ValueError('Exact per-call canonical identities required')
            name = base+row['path']; files.add(name)
            if sha(safe_file(directory, name)) != row['sha256']: raise ValueError('Canonical per-call array identity differs')
    for model in report['models']:
        identifier = f"{model['configuration']}_seed{model['seed']}"
        files.update((identifier+'.pt', identifier+'_history.json', identifier+'_library_contributions.json'))
        if sha(safe_file(directory, identifier+'.pt')) != model['checkpoint_sha256']:
            raise ValueError('Individual final model serialization identity differs')
        base = 'predictions/'+identifier+'/'; path = base+'manifest.json'
        manifest = json.loads(safe_file(directory, path).read_bytes())
        if set(manifest) != total: raise ValueError('ALL supported-call final predictions required')
        files.update((path, base+'decision_manifest.json'))
        for index, row in manifest.items():
            if row['path'] != index+'.npz': raise ValueError('Exact native prediction call path required')
            name = base+row['path']; files.add(name)
            if sha(safe_file(directory, name)) != row['sha256']: raise ValueError('Individual final prediction identity differs')
        decisions = json.loads(safe_file(directory, base+'decision_manifest.json').read_bytes())
        if set(decisions) != set(corpus.indices): raise ValueError('Both complete native prediction splits required')
        for split, populations in decisions.items():
            if set(populations) != {'actual_unique', 'public_geometry', 'bounded_shared'}:
                raise ValueError('All original candidate populations required')
            for population, row in populations.items():
                if row['path'] != f'{split}_{population}.jsonl' or type(row['calls']) is not int or row['calls'] <= 0:
                    raise ValueError('Declared complete native decision evidence required')
                name = base+row['path']; files.add(name)
                if sha(safe_file(directory, name)) != row['sha256']: raise ValueError('Final original-cost JSONL identity differs')
    return {name: sha(safe_file(directory, name)) for name in sorted(files)}


def require_same_summary(primary, repeated):
    # Individual final serialization hashes must have been verified FIRST.
    if not exact_tree(summary_semantics(primary), summary_semantics(repeated)):
        raise ValueError('Two complete nine-model scientific summary semantics differ')
