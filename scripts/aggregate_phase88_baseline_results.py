"""Aggregate Phase88 baseline evaluation JSON files without mixing denominators."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

METRICS = (
    "safe_capture_rate", "capture_event_rate", "collision_rate",
    "boundary_violation_rate", "target_invalid_episode_rate",
    "target_boundary_violation_rate", "target_obstacle_violation_rate",
    "timeout_rate", "mean_min_clearance_m", "mean_policy_latency_ms",
)

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()

def main() -> None:
    a=parse_args(); rows=[]
    for path in a.inputs:
        data=json.loads(path.resolve().read_text(encoding="utf-8"))
        if bool(data.get("locked_test_used", False)):
            raise ValueError(f"locked-test result is not allowed: {path}")
        row={"source":str(path.resolve()),"algorithm":data.get("algorithm"),"use_cbf":bool(data.get("use_cbf",False)),"episodes":int(data.get("episodes",0)),"episode_start_index":data.get("episode_start_index"),"episode_end_index_exclusive":data.get("episode_end_index_exclusive"),"diagnostic_only":int(data.get("episodes",0))<100}
        row.update({key:data.get(key) for key in METRICS})
        rows.append(row)
    payload={"experiment":"phase88_baseline_result_aggregation","locked_test_used":False,"rows":rows,"denominator_warning":"Rows with different episode counts or ranges are not pooled automatically; diagnostic_only must be reviewed before final reporting."}
    a.output.resolve().parent.mkdir(parents=True,exist_ok=True)
    a.output.resolve().write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))
if __name__ == "__main__": main()
