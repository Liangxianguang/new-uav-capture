from __future__ import annotations

import numpy as np

from scripts.check_phase58_qdr_time_index import main
from src.encirclement3d.qdr_formalization import audit_qdr_time_index


def test_phase58_checker_contract_covers_queue_length_ten() -> None:
    positions = np.zeros((2, 3), dtype=np.float64)
    velocities = np.ones((2, 3), dtype=np.float64)
    queue = np.arange(60, dtype=np.float64).reshape(10, 2, 3)
    suffix = np.arange(24, dtype=np.float64).reshape(4, 2, 3)

    result = audit_qdr_time_index(positions, velocities, queue, suffix)

    assert result.passed
    assert result.first_controllable_step == 10
    assert result.candidate_action_application_steps == (10, 11, 12, 13)
    assert result.expected_terminal_index_steps == 14


def test_phase58_checker_cli_writes_tensorboard_artifact(tmp_path, monkeypatch) -> None:
    output = tmp_path / "qdr"
    monkeypatch.setattr(
        "sys.argv",
        ["check_phase58_qdr_time_index.py", "--output", str(output), "--queue-lengths", "0", "10"],
    )
    main()

    summary = (output / "summary.json").read_text(encoding="utf-8")
    assert '"overall_pass": true' in summary
    assert (output / "tensorboard").exists()
