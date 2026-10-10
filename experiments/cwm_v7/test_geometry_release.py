import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from geometry_release import audit, target_bad


@pytest.fixture
def evidence():
    root = Path(__file__).parent
    with zipfile.ZipFile(root / "artifacts/geometry_qualification_20261010.zip") as archive:
        primary = {n: archive.read(n) for n in archive.namelist()}
    with zipfile.ZipFile(root.parent / "cwm_v6/artifacts/s4_mechanism_20261010.zip") as archive:
        old = {n: archive.read(n) for n in archive.namelist()}
    return primary, old


def test_geometry_release_recomputes_physical_validity(evidence):
    primary, old = evidence
    result = audit(primary.__getitem__, old.__getitem__)
    assert result["geometry_qualified_for_finite_training_distribution"]
    assert result["stages"]["qualification"]["candidates"][0]["stress_full_horizon_branches"] == 185
    assert not result["enhanced_control_enabled"]


def test_geometry_release_rejects_concealed_invalid_original_episode(evidence):
    primary, old = evidence
    key = "development/wallx0/episodes.json"
    episodes = json.loads(primary[key])
    for row in episodes:
        row["target_invalid_episode"] = False
    primary[key] = json.dumps(episodes).encode()
    with pytest.raises(ValueError, match="target-validity flag"):
        audit(primary.__getitem__, old.__getitem__)


def test_geometry_release_rejects_unjustified_better_capture_selection(evidence):
    primary, old = evidence
    key = "development/summary.json"
    summary = json.loads(primary[key])
    summary["selected_wall_center_x_m"] = 3.
    primary[key] = json.dumps(summary).encode()
    with pytest.raises(ValueError, match="Selection"):
        audit(primary.__getitem__, old.__getitem__)


def test_physical_validity_does_not_count_padded_zero_target_positions():
    scene = {"obstacles": [{"shape": "wall", "center_xy": [1.5, 0.], "height": 9.4, "half_extents_xy": [.6, 4.2]}]}
    assert target_bad([[1.5, 0., 5.]], scene)[0]
    assert not target_bad([[-3., 0., 5.], [1.5, 5.3, 5.]], scene).any()
    assert target_bad([[0., 0., 0.]], scene)[0]
