from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from select_phase56_evaluation_split import select_split  # noqa: E402


def test_select_phase56_split_keeps_mirror_pair() -> None:
    records = [
        {"mirror_group_id": 1, "defender_bias": "upper", "evaluation_split": "development_calibration"},
        {"mirror_group_id": 1, "defender_bias": "lower", "evaluation_split": "development_calibration"},
        {"mirror_group_id": 2, "defender_bias": "upper", "evaluation_split": "locked_diagnostic"},
        {"mirror_group_id": 2, "defender_bias": "lower", "evaluation_split": "locked_diagnostic"},
    ]
    selected, metadata = select_split(records, "development_calibration")
    assert len(selected) == 2
    assert metadata["selected_mirror_groups"] == 1
    assert metadata["locked_tuning"] is False

