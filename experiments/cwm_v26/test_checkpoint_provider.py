import hashlib
import io
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),str(HERE.parent/'cwm_v25')]
from checkpoint_provider import checkpoint_predictor, PinnedV23Loader, CONFIGURATION, SEED
from qualification_gate import OptionalResponseAdapter, QualificationGate


@pytest.fixture(scope='session')
def actual_raw():
    root=HERE.parents[1]
    cp=root/f'results/cwm_v23/primary_fixed_20261010/{CONFIGURATION}_seed{SEED}.pt'
    if not cp.exists():
        artifact=HERE/'artifacts/real_sequential_shadow_20261010.zip'
        if not artifact.exists():
            pytest.skip('Actual V23 checkpoint fixture not locally present')
        with zipfile.ZipFile(artifact) as z:
            raw=z.read('model/primary.pt')
        assert hashlib.sha256(raw).hexdigest()=='907f6c8c7df68ee53c626d0b8ade973e6053be53f2973b30366df1d444ecfa14'
        return raw
    return cp.read_bytes()


def public_inputs():
    p=np.zeros((2,8,4,3),dtype=np.float64)
    p[1,:,0,0]=1.
    return (np.zeros((8,252)),np.zeros((4,6)),p,p[0].copy(),np.zeros((8,3)),np.zeros((8,3)))


@pytest.mark.parametrize('mode',['off','guarded'])
def test_disabled_modes_never_resolve_artifact(mode,tmp_path):
    loader=PinnedV23Loader(tmp_path/'nonexistent.parts.json',tmp_path/'cache')
    adapter=OptionalResponseAdapter(QualificationGate(mode=mode,qualification={'research_eligible':False}),loader)
    assert adapter.forecast(*([object()]*6))[0] is None
    assert adapter.load_calls == adapter.prediction_calls == 0
    assert not (tmp_path/'cache').exists() and loader.checkpoint_sha256 is None


def test_real_checkpoint_public_forward_and_rng_preserved(actual_raw):
    py,npstate,ts=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    predict=checkpoint_predictor(actual_raw,hashlib.sha256(actual_raw).hexdigest())
    assert random.getstate()==py and torch.equal(torch.get_rng_state(),ts)
    now=np.random.get_state()
    assert now[0]==npstate[0] and np.array_equal(now[1],npstate[1]) and now[2:]==npstate[2:]
    adapter=OptionalResponseAdapter(QualificationGate(mode='shadow',qualification={'research_eligible':False}),lambda:predict)
    values=public_inputs()
    result,info=adapter.forecast(*values)
    assert result is not None and not info['control_eligible']
    assert (result['response'][0]==0).all()
    assert np.array_equal(result['reference'][0],result['reference'][1])
    assert np.array_equal(result['prediction'][0],result['reference'][0])
    repeated,_=adapter.forecast(*values)
    assert all(np.array_equal(result[k],repeated[k]) for k in result)


def test_changed_checkpoint_hash_rejected(actual_raw):
    with pytest.raises(ValueError,match='checkpoint bytes differ'):
        checkpoint_predictor(actual_raw,'0'*64)


@pytest.mark.parametrize('change,reason',[
    ('seed','primary checkpoint contract'),('enabled','primary checkpoint contract'),
    ('protocol','training protocol differs'),('normalizer','normalizer contract differs')])
def test_hash_updated_semantic_checkpoint_forgery_rejected(actual_raw,change,reason):
    cp=torch.load(io.BytesIO(actual_raw),map_location='cpu',weights_only=True)
    if change=='seed':
        cp['seed']+=1
    elif change=='enabled':
        cp['online_promoted']=True
    elif change=='protocol':
        cp['protocol']['training']['response_epochs']+=1
    else:
        cp['normalizer_scale'].fill_(float('nan'))
    raw=io.BytesIO();torch.save(cp,raw)
    forged=raw.getvalue()
    with pytest.raises(ValueError,match=reason):
        checkpoint_predictor(forged,hashlib.sha256(forged).hexdigest())
