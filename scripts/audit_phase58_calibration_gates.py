"""Apply the pre-registered Phase 58 calibration gates without retuning.

This script consumes the frozen block aggregate and the process-isolated
runtime aggregate.  It reports Go/No-Go per hypothesis; a single failed gate
keeps confirmation and locked-diagnostic closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runs-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def interval(item: dict[str, Any]) -> tuple[float, float]:
    low, high = item["paired_bootstrap_95_ci"]
    return float(low), float(high)


def outcome_mean(aggregate: dict[str, Any], block: str, method: str, metric: str) -> float:
    return float(aggregate["blocks"][block]["methods"][method]["episode_metrics"][metric]["mean"])


def max_exhaustion(runs_glob: str, block: str, method: str, scenes_path: Path) -> float:
    import glob

    scenes = {
        int(json.loads(line)["episode_index"]): json.loads(line)["scene_block"]
        for line in scenes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    values: list[float] = []
    for root in glob.glob(runs_glob):
        episode_path = Path(root) / method / "episodes.jsonl"
        if not episode_path.is_file():
            continue
        for line in episode_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if scenes.get(int(row["episode_index"])) != block:
                continue
            value = row.get("qdr_suffix_gate_max_exhaustion_streak_steps")
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
    return max(values) if values else float("nan")


def main() -> None:
    args = parse_args()
    aggregate = read_json(args.aggregate)
    runtime = read_json(args.runtime)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    policy = protocol["gate_policy"]
    blocks = aggregate["blocks"]
    m6 = "M6_qdr_synchronous_mpc"
    m0 = "M0_current_state_delayed_mpc"
    m7 = "M7_qdr_asynchronous_mpc"
    m5 = "M5_asynchronous_distributed_mpc"
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, criterion: str, evidence: str) -> None:
        checks.append(
            {
                "name": name,
                "status": "GO" if passed else "NO-GO",
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "evidence": evidence,
            }
        )

    id_delta = blocks["id_reference"]["paired_comparisons"]["M6_vs_M0"]["safe_capture_success"]
    id_low, id_high = interval(id_delta)
    add(
        "M6_ID_safe_capture_noninferiority",
        id_low >= float(policy["id_safe_capture_noninferiority_lower_bound"]),
        {"paired_delta": id_delta["mean_delta_candidate_minus_reference"], "ci_low": id_low, "ci_high": id_high},
        f"paired CI lower >= {policy['id_safe_capture_noninferiority_lower_bound']}",
        "blocks.id_reference.paired_comparisons.M6_vs_M0.safe_capture_success",
    )

    stress_delta = blocks["joint_stress_transfer"]["paired_comparisons"]["M6_vs_M0"]["collision"]
    stress_low, stress_high = interval(stress_delta)
    add(
        "M6_joint_stress_collision",
        stress_high < float(policy["stress_collision_delta_upper_bound"]),
        {"paired_delta": stress_delta["mean_delta_candidate_minus_reference"], "ci_low": stress_low, "ci_high": stress_high},
        f"paired CI upper < {policy['stress_collision_delta_upper_bound']}",
        "blocks.joint_stress_transfer.paired_comparisons.M6_vs_M0.collision",
    )

    joint_timeout = outcome_mean(aggregate, "joint_stress_transfer", m6, "timeout")
    joint_timeout_delta = blocks["joint_stress_transfer"]["paired_comparisons"]["M6_vs_M0"]["timeout"]
    _, joint_timeout_delta_high = interval(joint_timeout_delta)
    add(
        "M6_joint_stress_timeout",
        joint_timeout <= float(policy["timeout_upper_bound"]),
        joint_timeout,
        f"timeout <= {policy['timeout_upper_bound']}",
        "blocks.joint_stress_transfer.methods.M6_qdr_synchronous_mpc.episode_metrics.timeout.mean",
    )
    add(
        "M6_joint_stress_timeout_delta",
        joint_timeout_delta_high <= float(policy["timeout_delta_upper_bound"]),
        {"ci_high": joint_timeout_delta_high},
        f"paired timeout-delta CI upper <= {policy['timeout_delta_upper_bound']}",
        "blocks.joint_stress_transfer.paired_comparisons.M6_vs_M0.timeout",
    )

    scenes_path = args.scenes.resolve()
    exhaustion_values = {
        block: max_exhaustion(args.runs_glob, block, m6, scenes_path) for block in blocks
    }
    max_streak = max(value for value in exhaustion_values.values() if np.isfinite(value))
    add(
        "M6_max_exhaustion_streak",
        max_streak <= float(policy["maximum_exhaustion_streak_steps"]),
        {"by_block": exhaustion_values, "maximum": max_streak},
        f"maximum exhaustion streak <= {policy['maximum_exhaustion_streak_steps']} steps",
        "raw calibration episode qdr_suffix_gate_max_exhaustion_streak_steps",
    )

    runtime_methods = runtime["methods"]
    m6_p95 = float(runtime_methods[m6]["matched_prefix"]["total_control_latency_ms"]["p95"])
    m7_p95 = float(runtime_methods[m7]["matched_prefix"]["total_control_latency_ms"]["p95"])
    reduction = (m6_p95 - m7_p95) / m6_p95
    async_delta = blocks["id_reference"]["paired_comparisons"]["M7_vs_M6"]["safe_capture_success"]
    async_delta_low, async_delta_high = interval(async_delta)
    add(
        "M7_async_qdr_efficiency",
        reduction >= float(policy["asynchronous_total_p95_reduction"]),
        {"m6_p95_ms": m6_p95, "m7_p95_ms": m7_p95, "reduction": reduction},
        f"matched-prefix total p95 reduction >= {policy['asynchronous_total_p95_reduction']}",
        "runtime.methods.M6/M7.matched_prefix.total_control_latency_ms.p95",
    )
    add(
        "M7_async_qdr_ID_safe_capture_noninferiority",
        async_delta_low >= float(policy["id_safe_capture_noninferiority_lower_bound"]),
        {"paired_delta": async_delta["mean_delta_candidate_minus_reference"], "ci_low": async_delta_low, "ci_high": async_delta_high},
        f"paired CI lower >= {policy['id_safe_capture_noninferiority_lower_bound']}",
        "blocks.id_reference.paired_comparisons.M7_vs_M6.safe_capture_success",
    )

    payload = {
        "schema_version": "phase58-calibration-gate-audit-v1",
        "policy": policy,
        "checks": checks,
        "overall_status": "GO" if all(check["passed"] for check in checks) else "NO-GO",
        "confirmation_allowed": all(check["passed"] for check in checks),
        "claim_boundary": "local_cbf_empirical_filter_only; robust_clbf_qp_proof_not_claimed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
