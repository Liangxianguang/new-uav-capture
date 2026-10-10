import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from frozen_training_release import tensor_tree_equal,verify_manifest


def test_optimizer_rng_or_weight_difference_is_rejected():
    left = {'weights':torch.tensor([1.,2.]),'optimizer':{'step':torch.tensor(80)},
            'rng':np.arange(3,dtype=np.int64)}
    right = {'weights':left['weights'].clone(),'optimizer':{'step':torch.tensor(80)},
             'rng':left['rng'].copy()}
    assert tensor_tree_equal(left,right)
    right['optimizer']['step'] += 1
    assert not tensor_tree_equal(left,right)


def test_manifest_rejects_omitted_and_duplicate_members(tmp_path):
    path = tmp_path/'test.zip'
    raw = b'public pilot'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('data.bin',raw)
        archive.writestr('ARTIFACT_MANIFEST.json',json.dumps({'data.bin':hashlib.sha256(raw).hexdigest()}))
    verify_manifest(path)
    with zipfile.ZipFile(path,'a') as archive:
        archive.writestr('unexpected.bin',b'not covered')
    with pytest.raises(ValueError,match='coverage'):
        verify_manifest(path)
    with zipfile.ZipFile(path,'a') as archive:
        archive.writestr('data.bin',b'changed')
    with pytest.raises(ValueError,match='Duplicate'):
        verify_manifest(path)


def test_array_dtype_change_not_equal():
    assert not tensor_tree_equal(np.ones(2,dtype=np.float32),np.ones(2,dtype=np.float64))
