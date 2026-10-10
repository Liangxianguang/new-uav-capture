"""Unmodified V10 full original audit mapped to the preassigned V15 protocol."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'cwm_v10'))
from two_head_release import audit_data as original_audit


def audit_data(read,candidate_read,configs,restored):
    protocol = json.loads((ROOT/'data_protocol.json').read_text())
    if json.loads(read('data/summary.json'))['protocol'] != protocol:
        raise ValueError('Preregistered V15 data protocol differs')
    scenes = [json.loads(line) for line in read('data/scenes.jsonl').decode().splitlines()]
    if {r['episode_seed'] for r in scenes} != set(range(978010,978074)) or {r['layout_seed'] for r in scenes} != set(range(1978010,1978042)):
        raise ValueError('Fresh V15 scene identity mismatch')
    def mapped(name):
        return read('data/source/cwm_v15/data_protocol.json') if name == 'data/source/cwm_v10/data_protocol.json' else read(name)
    return original_audit(mapped,candidate_read,configs,restored)
