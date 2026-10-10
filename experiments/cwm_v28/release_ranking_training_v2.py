"""V2 full reload: exact contents plus individual serialized file identities.

This script produces validation reports only, never online promotion. Complete
data/branch replay audit must already have passed; it is not replaced by hashing.
Full data/training artifact packaging and independent release replay are later
publication requirements and are explicitly NOT satisfied by this report alone.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'cwm_v10')]
from train_ranking_response import (verified_calls,normalization,cost_population_weights,complete_contexts,FrozenLocalScore,
    frozen_configs,control_metrics,CalibratedOrigin,LocalTwoHead,FrozenOriginResponse,evaluate_cost_model,
    ranking_metrics,ranking_cache,decision_records,library_contributions,qualify,validate_training_protocol,
    save_json,sha,BASELINE_SHA,state_digest)
from qualification_gate import preserved_cpu_rng
from geometry_release import arrays,compare_arrays
from repeatability_identity import verify_pair


def model_from_checkpoint(checkpoint):
    protocol=json.loads((HERE/'training_protocol.json').read_text())
    if (checkpoint['protocol']!=protocol or checkpoint['configuration'] not in protocol['models'] or
        checkpoint['seed'] not in protocol['training']['seeds'] or
        any(checkpoint[k] is not False for k in ('online_promoted','baseline_weights_included','common_motion_in_response_optimizer'))):
        raise ValueError('Fixed offline final checkpoint contract differs')
    config=protocol['training']
    with preserved_cpu_rng():
        common=CalibratedOrigin(LocalTwoHead('motion_only',config['motion_scale_m'],config['response_scale_m']),'cv')
        model=common if checkpoint['configuration']=='cv_motion_only' else FrozenOriginResponse(common,LocalTwoHead('plain',config['motion_scale_m'],config['response_scale_m']))
        model.load_state_dict(checkpoint['model_state'],strict=True)
    model.eval().requires_grad_(False)
    common=model.common if isinstance(model,FrozenOriginResponse) else model
    if state_digest(common)!=checkpoint['common_motion_state_digest']:
        raise ValueError('Frozen common component checkpoint differs')
    return model


def audit_runs(data,audit_output,primary,repeated,output):
    protocol=json.loads((HERE/'training_protocol.json').read_text());validate_training_protocol(protocol)
    calls,data_report=verified_calls(data,audit_output)
    output.mkdir(parents=True,exist_ok=False)
    capsule=HERE.parent/'cwm_v1/baseline/capsule.zip'
    if sha(capsule)!=BASELINE_SHA: raise ValueError('Original capsule changed')
    configs=frozen_configs(capsule,output/'restored')
    indices={split:[i for i,c in enumerate(calls) if c['record']['split']==split] for split in ('train','development_validation')}
    mean,scale=normalization(calls,indices['train']);weights,population=cost_population_weights(calls,indices['train'])
    contexts=complete_contexts(calls,configs,data)
    scores={i:FrozenLocalScore(context,calls[i]['values']['proposed'][:,:,calls[i]['record']['agent']]) for i,context in contexts.items()}
    expected_models={(m,s) for m in protocol['models'] for s in protocol['training']['seeds']}
    checkpoints=[];summaries=[]
    for directory in (primary,repeated):
        report=json.loads((directory/'summary.json').read_text())
        if (report['status']!='offline_fixed_fresh_ranking_training_finished_pending_two_run_release' or report['protocol']!=protocol or
            report['data_summary_sha256']!=sha(data/'summary.json') or report['data_audit_summary_sha256']!=sha(audit_output/'summary.json') or
            any(report[k] is not False for k in ('enhanced_control_enabled','holdout_used','prior_gate_overridden','common_motion_in_response_optimizer','two_complete_runs_verified')) or
            len(report['models'])!=9 or {(r['configuration'],r['seed']) for r in report['models']}!=expected_models):
            raise ValueError('Fixed complete nine-model run required')
        for name,digest in report['source_hashes'].items():
            if sha(ROOT/name)!=digest or sha(directory/'source'/name)!=digest:
                raise ValueError('Run-used training source differs')
        saved=arrays((directory/'normalization.npz').read_bytes())
        if not compare_arrays(saved,{'mean':mean,'scale':scale}): raise ValueError('Train-only normalization differs')
        expected_weights={'population':population,'weights':weights}
        # JSON integer mapping keys become strings, normalize expected likewise.
        expected_weights=json.loads(json.dumps(expected_weights))
        if json.loads((directory/'population_weights.json').read_text())!=expected_weights:
            raise ValueError('Fixed full-train population weights differ')
        controls=control_metrics(calls,indices['development_validation'])
        if report['controls']!=controls: raise ValueError('Public GRU/CV controls differ')
        decisions={};recomputed_models=[];saved_checkpoints={};contributions={}
        for row in report['models']:
            identifier=f"{row['configuration']}_seed{row['seed']}"
            path=directory/(identifier+'.pt')
            if sha(path)!=row['checkpoint_sha256']: raise ValueError('Final checkpoint digest differs')
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            if (checkpoint['configuration']!=row['configuration'] or checkpoint['seed']!=row['seed'] or
                checkpoint['source_hashes']!=report['source_hashes'] or checkpoint['data_summary_sha256']!=report['data_summary_sha256'] or
                checkpoint['data_audit_summary_sha256']!=report['data_audit_summary_sha256'] or
                checkpoint['population_weights_sha256']!=sha(directory/'population_weights.json') or
                checkpoint['ranking_cache_sha256']!=sha(directory/f"cv_seed{row['seed']}_ranking_cache.npz") or
                not np.array_equal(checkpoint['normalizer_mean'].numpy(),mean) or not np.array_equal(checkpoint['normalizer_scale'].numpy(),scale)):
                raise ValueError('Final checkpoint provenance differs')
            model=model_from_checkpoint(checkpoint)
            expected_initial=checkpoint['initial_state_digest']
            if expected_initial!=row['initial_state_digest'] or row['parameters']!=sum(p.numel() for p in model.parameters()):
                raise ValueError('Fixed model budget/initial provenance differs')
            history=json.loads((directory/f'{identifier}_history.json').read_text())
            if [r['epoch'] for r in history]!=list(range(1,81)) or not np.isfinite([r['mean_batch_loss_terms'] for r in history]).all():
                raise ValueError('Complete fixed80epoch history required')
            measured_splits={};computed_decisions={};computed_contributions={}
            for split,chosen in indices.items():
                measured,predictions=evaluate_cost_model(model,calls,chosen,mean,scale,protocol['training']['batch_calls'])
                cache={i:ranking_cache(scores[i],o['reference'],calls[i]['values']) for i,o in zip(chosen,predictions) if i in contexts}
                cache_saved=arrays((directory/f"cv_seed{row['seed']}_ranking_cache.npz").read_bytes())
                for i,values in cache.items():
                    for key,value in values.items():
                        if torch.is_tensor(value) and not np.array_equal(value.numpy(),cache_saved[f'call{i}_{key}']):
                            raise ValueError('Reloaded canonical ranking supervision differs')
                measured.update(ranking_metrics(calls,chosen,predictions,cache))
                stored=arrays((directory/f'{identifier}_{split}_predictions.npz').read_bytes())
                expected={f'call{j}_{key}':value for j,o in enumerate(predictions) for key,value in o.items()}
                if not compare_arrays(stored,expected): raise ValueError('All-call reloaded prediction bytes differ')
                measured_splits[split]=measured
                computed_decisions[split]=decision_records(calls,chosen,predictions,contexts)
                computed_contributions[split]=library_contributions(computed_decisions[split],protocol)
                if split=='development_validation': decisions[identifier]=computed_decisions[split]['bounded_shared']
            if (measured_splits['train']!=row['train'] or measured_splits['development_validation']!=row['development'] or
                computed_decisions!=json.loads((directory/f'{identifier}_decisions.json').read_text()) or
                computed_contributions!=json.loads((directory/f'{identifier}_library_contributions.json').read_text())):
                raise ValueError('Recomputed original-cost metrics/contributions differ')
            recomputed_models.append(row);saved_checkpoints[identifier]=checkpoint;contributions[identifier]=computed_contributions
            print(json.dumps({'status':'all_final_predictions_original_costs_recomputed','run':directory.name,'model':identifier}),flush=True)
        selected,gates=qualify(recomputed_models,controls,decisions,protocol)
        if (report['selected_median_seeds']!=selected or report['qualification']!=gates or report['primary_research_eligible']!=gates['cv_rank_l2']['research_eligible'] or
            report['library_contributions']!=contributions or report['split_calls']!={k:len(v) for k,v in indices.items()} or
            report['split_groups']!={k:sorted({calls[i]['record']['group'] for i in v}) for k,v in indices.items()} or
            report['full_cost_calls']!={k:sum(i in contexts for i in v) for k,v in indices.items()}):
            raise ValueError('Recomputed fixed median/decision gates or support differ')
        checkpoints.append(saved_checkpoints);summaries.append(report)
    identity=verify_pair(primary,repeated,*summaries)
    verified_calls(data,audit_output)
    reload_sources={name:sha(HERE/name) for name in ('release_ranking_training_v2.py','repeatability_identity.py')}
    result={'reload_source_hashes':reload_sources,'status':'two_complete_fresh_ranking_runs_reloaded_original_costs_verified','models_per_run':9,
        'complete_runs':2,'repeatability_identity':identity,'all_final_weights_optimizer_rng_histories_equal':True,'all_prediction_bytes_reloaded':True,
        'train_only_normalization_verified':True,'actual_geometry_shared_cost_contributions_recomputed':True,
        'data_summary_sha256':sha(data/'summary.json'),'data_audit_summary_sha256':sha(audit_output/'summary.json'),
        'primary_summary_sha256':sha(primary/'summary.json'),'repeated_summary_sha256':sha(repeated/'summary.json'),
        'primary_research_eligible':summaries[0]['primary_research_eligible'],'qualification':summaries[0]['qualification'],
        'enhanced_control_enabled':False,'holdout_used':False,'artifact_packaged_and_replayed':False,
        'scope':'Complete independent data replay audit plus two-run model re-inference/cost/repeatability. Artifact transport/publication replay still pending. No active selector, holdout, full-Level/new-scene closed-loop capture/safety/latency qualification.'}
    save_json(output/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','audit','primary','repeated','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    report=audit_runs(args.data,args.audit,args.primary,args.repeated,args.output)
    print(json.dumps({'status':report['status'],'primary_research_eligible':report['primary_research_eligible']}),flush=True)
