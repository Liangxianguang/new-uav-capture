import sys
import json
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_real_entry_off_refusal_and_load_failure_preserve_baseline(tmp_path):
    capsule = Path(__file__).resolve().parents[1] / "cwm_v1" / "baseline" / "capsule.zip"
    if not capsule.is_file():
        pytest.skip("baseline capsule is not present")
    script = Path(__file__).with_name('original_entry_boundary.py')
    subprocess.run([sys.executable, str(script), '--capsule', str(capsule), '--output', str(tmp_path / 'audit'),
                    '--episodes', '1'], check=True, capture_output=True, text=True, timeout=180)
    summary = json.loads((tmp_path / 'audit/summary.json').read_text())
    assert summary["status"] == "passed_original_entry_optional_boundary_fallback_audit"
    assert summary["all_outcomes_equal_to_off"]
    assert summary["all_trajectories_byte_equal_to_off"]
    assert summary["modes"]["off"]["load_calls"] == 0
    assert summary["modes"]["off"]["prediction_calls"] == 0
    assert summary["modes"]["refusal"]["load_calls"] == 0
    assert summary["modes"]["load_failure"]["status"] == "load_failed"
    assert summary["modes"]["load_failure"]["prediction_calls"] == 0
    assert summary['modes']['prediction_failure']['status'] == 'prediction_failed'
    assert summary['historical_capsule_trajectories_equal']
    assert summary['commands_and_plans_equal_to_plain']
    assert all(not e['control_eligible'] for mode in summary['modes'].values() for e in mode['events'])
