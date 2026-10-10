import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from loss_training_release import tensor_tree_equal,verify_manifest


def test_repeat_comparison_includes_rng_and_optimizer_not_only_weights():
    original = {'weights':torch.zeros(2),'optimizer':{'state':{0:{'step':torch.tensor(80.),'exp_avg':torch.zeros(2)}}},
                'rng':torch.tensor([1,2,3],dtype=torch.uint8),'sampler':{'state':{'value':123}}}
    copy = {'weights':torch.zeros(2),'optimizer':{'state':{0:{'step':torch.tensor(80.),'exp_avg':torch.zeros(2)}}},
            'rng':torch.tensor([1,2,3],dtype=torch.uint8),'sampler':{'state':{'value':123}}}
    assert tensor_tree_equal(original,copy)
    copy['rng'][0] ^= 1
    assert not tensor_tree_equal(original,copy)


@pytest.mark.parametrize('kind',['omitted','duplicate','wrong_digest'])
def test_manifest_cannot_hide_changed_members(tmp_path,kind):
    path = tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('data.bin',b'original')
        if kind == 'duplicate':
            with pytest.warns(UserWarning):
                archive.writestr('data.bin',b'changed')
        archive.writestr('ARTIFACT_MANIFEST.json',json.dumps({} if kind == 'omitted' else {'data.bin':'wrong'}))
    with pytest.raises(ValueError):
        verify_manifest(path)
