import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from diagnose_interactions import diagnostic_commands, summarized
from verify_release import check_archive


def test_diagnostic_uses_only_public_geometry_and_does_not_mutate():
    sequence = np.ones((8, 4, 3))
    observation = {"defender_positions": np.arange(12, dtype=float).reshape(4, 3)}
    reference = np.ones(3)
    saved = [sequence.copy(), observation["defender_positions"].copy(), reference.copy()]
    commands = diagnostic_commands(sequence, observation, reference)
    assert commands.shape == (8, 16, 4, 3)
    assert np.isfinite(commands).all()
    assert np.linalg.norm(commands, axis=-1).max() <= 5.0000001
    assert np.array_equal(commands[0, :8], sequence)
    assert np.array_equal(commands[-1], np.zeros((16, 4, 3)))
    assert all(np.array_equal(a, b) for a, b in zip(saved, [sequence, observation["defender_positions"], reference]))


def test_boolean_diagnostic_rates_have_valid_quantiles():
    value = summarized([False, True, False, True])
    assert value["count"] == 4
    assert value["mean"] == .5
    assert value["p95"] == 1.


def test_release_verifies_member_hashes(tmp_path):
    path = tmp_path / "release.zip"
    data = b"frozen data"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.bin", data)
        archive.writestr("MANIFEST.json", json.dumps({"data.bin": hashlib.sha256(data).hexdigest()}))
    assert check_archive(path, "MANIFEST.json", False)["members_verified"] == 1


@pytest.mark.parametrize("name,expected", [("../data.bin", "Unsafe"), ("data.bin", "hash mismatch")])
def test_release_rejects_unsafe_or_corrupted(tmp_path, name, expected):
    path = tmp_path / "invalid.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, b"bad")
        archive.writestr("MANIFEST.json", json.dumps({name: "0" * 64}))
    with pytest.raises(ValueError, match=expected):
        check_archive(path, "MANIFEST.json", False)
