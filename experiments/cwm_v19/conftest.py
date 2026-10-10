"""Isolate V19's data_audit import from the historical V12 test namespace.

Standalone experiments run in fresh Python processes. Combined pytest suites
cache historical basename imports. Preload the V19 trainer with its own audit,
then restore the older audit cache; no experiment source or runtime is changed.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
previous = sys.modules.get('data_audit')
spec = importlib.util.spec_from_file_location('_cwm_v19_data_audit',ROOT/'data_audit.py')
local_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_audit)
sys.modules['data_audit'] = local_audit
try:
    import train_frozen_response
finally:
    if previous is not None:
        sys.modules['data_audit'] = previous
    else:
        sys.modules.pop('data_audit',None)
