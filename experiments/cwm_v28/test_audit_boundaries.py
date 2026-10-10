import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from audit_ranking_data import audit,_equal,check_record_paths
from release_ranking_training import model_from_checkpoint
from train_ranking_response import CalibratedOrigin,LocalTwoHead,FrozenOriginResponse,state_digest


@pytest.mark.parametrize('fault',['partial','control','holdout','training_allowed','protocol'])
def test_incomplete_or_altered_data_stops_before_replay(tmp_path,fault):
    data=tmp_path/'data';data.mkdir()
    p=json.loads((HERE/'data_protocol.json').read_text())
    report={'protocol':p,'status':'fresh_bounded_sequential_data_collected_pending_independent_audit',
        'enhanced_control_enabled':False,'new_model_trained':False,'holdout_used':False,'training_authorized_by_this_report':False}
    if fault=='partial':report['status']='collecting'
    elif fault=='control':report['enhanced_control_enabled']=True
    elif fault=='holdout':report['holdout_used']=True
    elif fault=='training_allowed':report['training_authorized_by_this_report']=True
    else:report['protocol']['groups']=31
    (data/'summary.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):audit(data,tmp_path/'audit')
    assert not (tmp_path/'audit').exists()


def test_dtype_is_part_of_original_history_equivalence():
    a=np.zeros((8,252),dtype=np.float32)
    assert _equal(a,a.copy()) and not _equal(a,a.astype(np.float64))


@pytest.mark.parametrize('path',['../outside.json','C:/outside.json','calls/994010_2_0_0.json'])
def test_record_paths_cannot_escape_or_overwrite_another_round(path):
    row={'episode_index':994010,'step':2,'ordinal':4,'agent':0,'context_path':path}
    with pytest.raises(ValueError):check_record_paths(row)
    row['context_path']='calls/994010_2_4_0.json';check_record_paths(row)


@pytest.mark.parametrize('configuration',['cv_motion_only','cv_l2','cv_rank_l2'])
def test_actual_v28_module_reload_preserves_frozen_common(configuration):
    p=json.loads((HERE/'training_protocol.json').read_text())
    common=CalibratedOrigin(LocalTwoHead('motion_only',2.5,2.5),'cv')
    model=common if configuration=='cv_motion_only' else FrozenOriginResponse(common,LocalTwoHead('plain',2.5,2.5))
    cp={'protocol':p,'configuration':configuration,'seed':994102,'model_state':model.state_dict(),
        'common_motion_state_digest':state_digest(common),'online_promoted':False,
        'baseline_weights_included':False,'common_motion_in_response_optimizer':False}
    loaded=model_from_checkpoint(cp)
    assert state_digest(loaded)==state_digest(model)
    assert all(not v.requires_grad for v in loaded.parameters())
    forged=copy.deepcopy(cp);forged['online_promoted']=True
    with pytest.raises(ValueError):model_from_checkpoint(forged)
    forged=copy.deepcopy(cp);forged['common_motion_state_digest']='0'*64
    with pytest.raises(ValueError):model_from_checkpoint(forged)
