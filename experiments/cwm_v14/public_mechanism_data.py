"""Reuse original complete geometry/public/local audit with a new protocol path."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent / 'cwm_v10'), str(ROOT.parent / 'cwm_v13')]
from two_head_release import audit_data as original_audit
from geometry_release import arrays, sha


def audit_fresh(read, source_read, configs, restored):
    protocol = json.loads((ROOT / 'data_protocol.json').read_text())
    report = json.loads(read('data/summary.json'))
    if report['protocol'] != protocol:
        raise ValueError('Preregistered V14 collection protocol mismatch')
    scenes = [json.loads(line) for line in read('data/scenes.jsonl').decode().splitlines()]
    if {r['episode_seed'] for r in scenes} != set(range(975010, 975074)) or {r['layout_seed'] for r in scenes} != set(range(1975010, 1975042)):
        raise ValueError('V14 fresh scene seeds differ')
    def mapped(name):
        if name == 'data/source/cwm_v10/data_protocol.json':
            return read('data/source/cwm_v14/data_protocol.json')
        return read(name)
    return original_audit(mapped, source_read, configs, restored)
