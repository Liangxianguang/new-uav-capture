import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from scout import commands, select_scenes


def test_scout_commands_bounded_and_public_only():
    source = np.ones((8, 4, 3))
    positions = np.arange(12, dtype=float).reshape(4, 3)
    original = positions.copy()
    output = commands(source, {"defender_positions": positions}, np.zeros(3))
    assert output.shape == (4, 8, 4, 3)
    assert np.array_equal(output[0], source)
    assert np.linalg.norm(output, axis=-1).max() <= 5.0000001
    assert np.array_equal(positions, original)
    assert np.all(output[-1] == 0)


def test_selection_disjoint_deterministic(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    path = root / "scenes.jsonl"
    rows = [{"variant": "fast_target", "mirror_group_id": f"g{i}", "episode_index": i * 2 + j}
            for i in range(5) for j in range(2)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    protocol = {"variants": {"fast_target": 7}, "groups_per_variant": 2, "selection_seed": 920910}
    first = select_scenes(path, protocol, {"g0"})
    assert first == select_scenes(path, protocol, {"g0"})
    assert len(first) == 2
    assert len({r["mirror_group_id"] for r in first}) == 2
    assert "g0" not in {r["mirror_group_id"] for r in first}
    with pytest.raises(ValueError, match="partition"):
        select_scenes(tmp_path / "not_train.jsonl", protocol, set())
