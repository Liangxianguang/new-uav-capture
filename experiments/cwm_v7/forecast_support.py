"""Post-training support-matched oracle diagnostic; never changes gates or control."""
import argparse
import json
from pathlib import Path

import numpy as np

from geometry import ROOT, sha


def support_metric(error, mask, groups):
    by_group = {str(g): float(error[groups == g][mask[groups == g]].mean()) for g in sorted(set(groups.tolist())) if mask[groups == g].any()}
    return {"valid_points": int(mask.sum()), "states_with_support": int(mask.any((1, 2)).sum()), "groups_with_support": len(by_group),
            "group_equal_ade_m": float(np.mean(list(by_group.values()))) if by_group else None, "by_group": by_group}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.training / "summary.json").read_text())
    with np.load(args.data / "pairs.npz", allow_pickle=False) as raw:
        data = {k: raw[k] for k in raw.files}
    if sha(args.data / "pairs.npz") != report["dataset_sha256"] or report["holdout_used"] or report["enhanced_control_enabled"]:
        raise ValueError("Frozen diagnostic dataset / disabled contract mismatch")
    chosen = data["split"] == "development_validation"
    target, valid = data["target"][chosen, :, :8], data["valid"][chosen, :, :8]
    backbone, groups = data["backbone"][chosen], data["group"][chosen]
    common = valid & valid[:, :1]
    joint = valid & valid.all((1, 2))[:, None, None]
    paths = {"original_gru_zero_response": np.repeat(backbone[:, None], 8, 1),
             "gru_plus_exact_response": backbone[:, None] + target - target[:, :1],
             "fixed_reference_truth_zero_response": np.repeat(target[:, :1], 8, 1), "action_specific_truth": target}
    for kind, seed in report["chosen_median_seed_per_architecture"].items():
        file = args.training / f"{kind}_seed{seed}_development_predictions.npz"
        with np.load(file, allow_pickle=False) as raw:
            paths[kind + "_learned"] = raw["prediction"]
            paths[kind + "_response_with_reference_motion_truth"] = target[:, :1] + raw["response"]
    results = {}
    for support, mask in (("common_reference_valid", common), ("all_probes_joint_full8", joint)):
        results[support] = {method: support_metric(np.linalg.norm(path - target, axis=-1), mask, groups) for method, path in paths.items()}
    summary = {"status": "post_training_support_diagnostic_only", "results": results,
               "chosen_seed_per_architecture": report["chosen_median_seed_per_architecture"],
               "development_gate_unchanged": report["development_gate_passed"], "enhanced_control_enabled": False, "holdout_used": False,
               "dataset_sha256": report["dataset_sha256"], "training_summary_sha256": sha(args.training / "summary.json"), "source_sha256": sha(Path(__file__)),
               "limitations": ["Post-training development diagnostic; not new selection, held-out validation, causal proof or control utility.",
                               "Exact truth is labels only, never public model inputs or online commands.",
                               "Published training ADE uses all branch-valid points, including frames without reference validity; this diagnostic compares shared support.",
                               "Joint full8 support is necessary for later original eight-step diagnostic MPC scoring; it does not prove physical safety or realized returns."]}
    with args.output.open("x") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps({k: {m: v["group_equal_ade_m"] for m, v in a.items()} for k, a in results.items()}))


if __name__ == "__main__":
    main()
