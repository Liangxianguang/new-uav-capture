import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "backfill_closed_loop_tensorboard.py"
SPEC = importlib.util.spec_from_file_location("backfill_closed_loop_tensorboard", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_finite_rejects_nan_and_infinity():
    assert MODULE._finite(0.0)
    assert MODULE._finite(3)
    assert not MODULE._finite(float("nan"))
    assert not MODULE._finite(float("inf"))


def test_iter_method_summaries_only_reads_method_children(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "distributed_delayed").mkdir(parents=True)
    (run_dir / "worst_case").mkdir()
    (run_dir / "summary.json").write_text(json.dumps({}), encoding="utf-8")
    for name in ("distributed_delayed", "worst_case"):
        (run_dir / name / "summary.json").write_text(json.dumps({}), encoding="utf-8")

    found = [path.parent.name for path in MODULE._iter_method_summaries(run_dir)]

    assert found == ["distributed_delayed", "worst_case"]
