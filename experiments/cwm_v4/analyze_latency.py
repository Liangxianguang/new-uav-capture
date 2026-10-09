"""Post-hoc response latency/support diagnosis; never changes the fixed gate."""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnose import summarized


def analyze(data, protocol):
    mask = data["valid"][:, 1:] & data["valid"][:, :1]
    effect = np.linalg.norm(data["target"][:, 1:] - data["target"][:, :1], axis=-1)
    crossing = (effect > .05) & mask
    any_crossing = crossing.any(axis=-1)
    first_step = np.argmax(crossing, axis=-1) + 1
    result = {}
    for variant in protocol["variants"]:
        chosen = data["variant"] == variant
        signals = any_crossing[chosen]
        first = first_step[chosen][signals]
        common = mask[chosen]
        prefix_length = common.sum(axis=-1)
        bucket = {"windows": int(chosen.sum()), "pairs": int(signals.size),
                  "pairs_with_observed_effect_over_0_05m": int(signals.sum()),
                  "first_observed_response_step": summarized(first),
                  "first_observed_response_time_seconds": summarized(first * .1),
                  "response_first_after_original_8_step_horizon": int(np.sum(first > 8)),
                  "response_first_within_original_8_step_horizon": int(np.sum(first <= 8)),
                  "common_prefix_steps": summarized(prefix_length.reshape(-1)),
                  "full24_pair_count": int(np.sum(prefix_length == 24)), "branches": {}}
        for i, name in enumerate(protocol["branches"][1:]):
            branch_signal = signals[:, i]
            branch_first = first_step[chosen, i][branch_signal]
            signal_groups = sorted({str(g) for g, s in zip(data["group"][chosen], branch_signal) if s})
            bucket["branches"][name] = {"signal_pairs": int(branch_signal.sum()),
                                        "signal_groups": signal_groups,
                                        "first_response_step": summarized(branch_first)}
        result[variant] = bucket
    return {"status": "posthoc_diagnostic_not_a_new_gate", "effect_threshold_m": .05, "dt_seconds": .1,
            "variants": result,
            "limitations": ["Latency is first observed displacement threshold crossing, not a separately identified reaction delay.",
                            "A censored pair with no crossing is unknown after censoring, not proof of no response.",
                            "Post-hoc explanatory analysis only; cannot override the predeclared data adequacy gate.",
                            "Each window/action pair is not an independent scene; group support reported separately."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "summary.json").read_text())
    with np.load(args.run / "pairs.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    result = analyze(data, report["protocol"])
    with (args.run / "latency_analysis.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status": result["status"], "variants": {k: {x: v[x] for x in
          ("pairs_with_observed_effect_over_0_05m", "response_first_after_original_8_step_horizon", "first_observed_response_step")} for k, v in result["variants"].items()}}))


if __name__ == "__main__":
    main()
