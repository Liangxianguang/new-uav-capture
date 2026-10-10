import json
import subprocess
import sys
from pathlib import Path


def test_actual_completed_shadow_semantic_rejections_in_fresh_process(tmp_path):
    script=Path(__file__).with_name('check_shadow_evidence.py')
    subprocess.run([sys.executable,str(script),'--output',str(tmp_path/'semantic')],
        capture_output=True,text=True,check=True,timeout=180)
    result=json.loads((tmp_path/'semantic/summary.json').read_text())
    assert result['status']=='passed_full_actual_shadow_semantic_checks'
    assert len(result['checks'])==9
    assert not result['enhanced_control_enabled']
