"""Independent audit and archive builder for the V18 public geometry pilot."""
import argparse
import hashlib
import io
import json
import tempfile
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO/'cwm_v17'), str(REPO/'cwm_v9'), str(REPO/'cwm_v7')]
BASELINE_SHA = "c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arrays(path_or_bytes):
    raw = path_or_bytes if isinstance(path_or_bytes, bytes) else path_or_bytes.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def arrays_equal(left, right):
    return set(left) == set(right) and all(
        left[key].dtype == right[key].dtype
        and left[key].shape == right[key].shape
        and left[key].tobytes() == right[key].tobytes()
        for key in left
    )


def load_protocol():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if protocol["baseline_capsule_sha256"] != BASELINE_SHA:
        raise ValueError("Baseline capsule pin changed")
    if any(protocol[key] for key in ("enhanced_control_enabled", "new_model_trained",
                                     "private_labels_model_inputs", "holdout_used")):
        raise ValueError("V18 is no longer a public shadow-only pilot")
    if protocol["target_branch_rule_overrides"] or protocol["horizon_steps"] != 8:
        raise ValueError("Frozen target or horizon contract changed")
    return protocol


def _json(path):
    return json.loads(path.read_text())


def _check_source_snapshots(run):
    summary = _json(run / "summary.json")
    for name, expected in summary["source_hashes"].items():
        snapshot = run / "source" / name
        if not snapshot.is_file() or digest(snapshot) != expected:
            raise ValueError(f"Source snapshot mismatch: {name}")
        current = REPO / name
        if not current.is_file() or digest(current) != expected:
            raise ValueError(f"Current source mismatch: {name}")


def _check_branch(values, prefix, scene=None):
    valid = values[prefix + "valid"]
    termination = values[prefix + "termination"]
    if valid.shape != (8,) or termination.shape != (8,):
        raise ValueError("Invalid branch mask shape")
    after = termination == "after_terminal"
    count = int((~after).sum())
    if not count:
        raise ValueError("Empty branch support")
    if after.any() and not np.all(after[np.argmax(after):]):
        raise ValueError("Terminal padding is not a suffix")
    if np.any(valid[after]) or np.any(~np.isfinite(values[prefix + "target"])):
        raise ValueError("Invalid or imputed branch values")
    for key, shape in (("target", (8, 3)), ("defenders", (8, 4, 3)),
                       ("commanded", (8, 4, 3)), ("executed", (8, 4, 3))):
        if values[prefix + key].shape != shape or not np.isfinite(values[prefix + key]).all():
            raise ValueError("Invalid branch field shape/finiteness")
    commanded, executed = values[prefix + "commanded"], values[prefix + "executed"]
    if np.linalg.norm(commanded, axis=-1).max() > 5. + 1e-8:
        raise ValueError("CBF command speed violated")
    # Execution dynamics and noise are disabled by this frozen scene protocol.
    if not np.allclose(commanded, executed, rtol=0, atol=1e-12):
        raise ValueError("Frozen commanded/executed label contract changed")
    if scene is not None:
        from geometry_release import target_bad
        bad = target_bad(values[prefix + "target"][:count], scene)
        expected = np.zeros(8, bool)
        expected[:count] = ~np.maximum.accumulate(bad)
        if not np.array_equal(valid, expected):
            raise ValueError("Branch target validity differs from actual geometry")
    if after.any():
        start = int(np.argmax(after))
        for key in ("target", "defenders", "commanded", "executed"):
            if np.any(values[prefix + key][start:] != 0):
                raise ValueError("Future branch values were imputed")


def audit_run(run, protocol, base, expected_scenes):
    summary = _json(run / "summary.json")
    if summary["protocol"] != protocol:
        raise ValueError("Run protocol differs from preregistration")
    if summary["status"] != "fresh_local_geometry_probe_pilot_finished_not_control_qualification":
        raise ValueError("Unexpected pilot status")
    if not summary["eligible_for_separate_fresh_training_protocol"] or not all(summary["data_checks"].values()):
        raise ValueError("Pilot data gate is not complete")
    if (len(summary["episodes"]) != 16 or summary["calls"] != 384 or summary["records"] != 384
            or summary["skipped_calls"] != 0 or summary["enhanced_control_enabled"]
            or summary["new_model_trained"] or summary["holdout_used"]):
        raise ValueError("Incomplete or activated pilot population")
    _check_source_snapshots(run)
    scenes = [json.loads(line) for line in (run / "scenes.jsonl").read_text().splitlines()]
    if scenes != expected_scenes or digest(run / "scenes.jsonl") != summary["scene_sha256"]:
        raise ValueError("Fresh scene identity changed")
    by_id = {r["episode_index"]: r for r in scenes}
    records = _json(run / "records.json")
    episodes = _json(run / "episodes.json")
    decisions = _json(run / "decisions.json")
    if len(records) != 384 or len(episodes) != 16:
        raise ValueError("Saved population count changed")
    if len(decisions) == 0:
        raise ValueError("No full-support decision records")
    if episodes != summary["episodes"] or {r["episode_index"] for r in episodes} != set(by_id):
        raise ValueError("Original episode population differs")
    from geometry_release import target_bad
    for episode in episodes:
        if not episode["trajectory_byte_equal"] or episode["observed_sha256"] != episode["plain_sha256"]:
            raise ValueError("Original trajectory equivalence failed")
        observed = run / "observed" / f"{episode['episode_index']}.npz"
        plain = run / "plain" / observed.name
        if digest(observed) != episode["observed_sha256"] or digest(plain) != episode["plain_sha256"]:
            raise ValueError("Original trajectory digest differs")
        if not arrays_equal(arrays(observed), arrays(plain)):
            raise ValueError("Original observed/plain trajectory mismatch")
        if bool(target_bad(arrays(observed)["target_positions"], by_id[episode['episode_index']]["scenario"]).any()) != episode["target_invalid_episode"]:
            raise ValueError("Original target geometry flag differs")
    from replay_probe_public import decode_public, restore_public_call
    from geometry_probes import public_geometry_pool
    from local_shadow import local_cost
    from geometry_release import compare_arrays
    from freeze_baseline import sha as source_sha
    from local_shadow import delayed_joint_context
    from support_audit import map_actual, scored_record
    from collect_probe_pilot import summarize

    # Reconstruct the frozen original planner once. Every saved local score is
    # independently recomputed from the public call snapshot below.
    calls, recomputed_decisions = [], []
    expected_ids = {(scene['episode_index'], step, agent) for scene in scenes
                    for step in protocol['snapshot_steps'] for agent in range(4)}
    if len(records) != len(expected_ids) or {(r['episode_index'], r['step'], r['agent']) for r in records} != expected_ids:
        raise ValueError("Actual local call population/duplicates differ")
    for record in records:
        scene = by_id[record['episode_index']]
        if record['group'] != scene['mirror_group_id'] or record['split'] != 'development_pilot':
            raise ValueError("Fresh group or split differs")
        expected_key = int.from_bytes(hashlib.sha256(
            f"{protocol['noise_seed']}/{record['episode_index']}/{record['step']}/{record['agent']}".encode()).digest()[:4], 'little')
        if record['noise_key'] != expected_key:
            raise ValueError("Fixed branch noise identity differs")
        context_path = run / record["context_path"]
        arrays_path = run / record["arrays_path"]
        if digest(context_path) != record["context_sha256"] or digest(arrays_path) != record["arrays_sha256"]:
            raise ValueError("Saved call digest mismatch")
        values = arrays(arrays_path)
        if not all(np.isfinite(value).all() for name, value in values.items()
                   if value.dtype.kind in "fc"):
            raise ValueError("Nonfinite saved public/branch array")
        for key in ("history", "backbone", "proposed", "anchor", "actual_candidates", "original_local_costs"):
            if key not in values:
                raise ValueError("Incomplete public call array")
        if values["history"].shape != (8, 252) or values["backbone"].shape != (8, 3):
            raise ValueError("Public history/backbone shape mismatch")
        for branch_index in range(values["valid"].shape[0]):
            _check_branch({key: values[key][branch_index] for key in
                           ("valid", "termination", "target", "defenders", "commanded", "executed")}, "", scene['scenario'])
        _check_branch(values, "anchor_", scene['scenario'])
        snapshot = _json(context_path)
        call = restore_public_call(snapshot, base.planner_config, base.distributed_config, values["backbone"])
        if call['agent'] != record['agent'] or not np.array_equal(call['local_candidates'], values['actual_candidates']):
            raise ValueError("Original actual candidates differ from public context")
        reference, velocity = base.evaluator.belief_reference(call["observation"], base.planner_config)
        pool = public_geometry_pool(call, reference, velocity, protocol["probe_magnitude_mps"])
        if not np.array_equal(pool, values["proposed"][:, :, record["agent"]]):
            raise ValueError("Public geometry pool reconstruction differs")
        context = delayed_joint_context(call['observation'], call['known'], call['peer_sequences'],
                                        call['agent'], pool, call['selected'], reference, velocity)
        if context is None or any(not np.array_equal(value, values[key]) for value, key in
                                  zip(context, ('proposed', 'anchor', 'relative'))):
            raise ValueError("Public delayed joint/anchor/relative reconstruction differs")
        if not np.array_equal(reference, values['reference']) or not np.array_equal(velocity, values['velocity']):
            raise ValueError("Public reference/velocity differs")
        recomputed = local_cost(call, values["actual_candidates"],
                                np.repeat(values["backbone"][None], len(values["actual_candidates"]), 0))
        if not np.allclose(recomputed, values["original_local_costs"], rtol=1e-10, atol=1e-8):
            raise ValueError("Original local cost reproduction failed")
        selected = call["selected"]
        selected_index = next(i for i, candidate in enumerate(values["actual_candidates"])
                              if np.array_equal(candidate, selected))
        actual_indices = [next(i for i, candidate in enumerate(pool)
                               if np.array_equal(candidate, actual))
                          for actual in values["actual_candidates"]]
        if int(np.argmin(recomputed)) != selected_index or int(np.argmin(values["original_local_costs"])) != selected_index:
            raise ValueError("Original selected action changed")
        if not np.allclose(recomputed, values["original_local_costs"], rtol=1e-10, atol=1e-8):
            raise ValueError("Copied original scores differ")
        if record["actual_choice_pool_index"] not in actual_indices:
            raise ValueError("Actual choice is outside reconstructed public pool")
        if not np.array_equal(pool[record['actual_choice_pool_index']], selected):
            raise ValueError("Actual chosen pool index differs")
        anchor_index = record['actual_choice_pool_index']
        for key in ('target', 'defenders', 'commanded', 'executed', 'valid', 'termination', 'branch_sign_label_only'):
            if not np.array_equal(values[key][anchor_index], values['anchor_'+key]):
                raise ValueError("Equal-probe anchor labels differ")
        current = {'record':record, 'values':values}
        calls.append(current)
        actual, _ = map_actual(values, record['agent'])
        for name, indices in (('actual_unique', actual), ('geometry_union', np.arange(len(pool)))):
            if values['valid'][indices].all() and values['anchor_valid'].all():
                recomputed_decisions.append(scored_record(current, indices, name, call, {}, protocol))
    if recomputed_decisions != decisions:
        raise ValueError("Independent full original cost/ranking recomputation differs")
    recomputed_summary = summarize(calls, recomputed_decisions, episodes, protocol)
    if any(summary[key] != value for key, value in recomputed_summary.items()):
        raise ValueError("Independent support/decision/data gate summaries differ")
    if source_sha(REPO / "cwm_v1/baseline/capsule.zip") != BASELINE_SHA:
        raise ValueError("Baseline capsule changed")
    return {"summary_sha256": digest(run / "summary.json"), "records": len(records),
            "decisions": len(decisions), "episodes": len(episodes),
            "eligible_for_separate_fresh_training_protocol": True,
            "independent_full_cost_rankings_and_summaries_recomputed": True}


def _compare_runs(first, second):
    if (first / "scenes.jsonl").read_bytes() != (second / "scenes.jsonl").read_bytes():
        raise ValueError("Repeated pilot differs: scenes.jsonl")
    for name in ("records.json", "decisions.json", "episodes.json", "summary.json"):
        if _json(first / name) != _json(second / name):
            raise ValueError(f"Repeated pilot differs: {name}")
    for subdir in ("observed", "plain", "calls"):
        left = {p.relative_to(first / subdir).as_posix(): p for p in (first / subdir).rglob("*") if p.is_file()}
        right = {p.relative_to(second / subdir).as_posix(): p for p in (second / subdir).rglob("*") if p.is_file()}
        if set(left) != set(right) or any(left[name].read_bytes() != right[name].read_bytes() for name in left):
            raise ValueError(f"Repeated pilot arrays differ: {subdir}")


def audit_replays(data, replay, protocol):
    summary = _json(replay / "summary.json")
    _check_source_snapshots(replay)
    if summary["status"] != "independent_original_public252_and_local_geometry_candidates_equal":
        raise ValueError("Unexpected public replay status")
    if summary["protocol"] != protocol or len(summary["checked_calls"]) != 384:
        raise ValueError("Incomplete public replay")
    rows = len(summary["checked_calls"])
    if len(summary["episodes"]) != 16 or rows != len(_json(data / "records.json")):
        raise ValueError("Public replay population mismatch")
    if summary['data_summary_sha256'] != digest(data / 'summary.json'):
        raise ValueError('Public replay data provenance differs')
    expected_rows = [{'episode_index':r['episode_index'], 'step':r['step'], 'agent':r['agent'],
                      'public_context_equal':True, 'eligible_arrays':'arrays_path' in r}
                     for r in _json(data / 'records.json')]
    if summary['checked_calls'] != expected_rows or any(summary[k] for k in
        ('enhanced_control_enabled', 'new_model_trained', 'holdout_used')):
        raise ValueError('Public replay call coverage or original-only status differs')
    for episode in summary["episodes"]:
        if not episode["trajectory_byte_equal"]:
            raise ValueError("Public replay trajectory was not byte equal")
        original = data / "observed" / f"{episode['episode_index']}.npz"
        replayed = replay / "trajectories" / original.name
        if digest(original) != digest(replayed) or not arrays_equal(arrays(original), arrays(replayed)):
            raise ValueError("Public replay trajectory differs")
    return {"summary_sha256": digest(replay / "summary.json"), "checked_calls": rows, "episodes": 16}


def _files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and not {'restored', 'audit_restored', '__pycache__'}.intersection(p.relative_to(root).parts)}


def verify_archive(path, recompute=False):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if len(names) != len(archive.namelist()):
            raise ValueError('Duplicate archive names')
        if "ARTIFACT_MANIFEST.json" not in names or "release_summary.json" not in names:
            raise ValueError("Incomplete V18 archive")
        manifest = json.loads(archive.read("ARTIFACT_MANIFEST.json"))
        if set(manifest) != names - {'ARTIFACT_MANIFEST.json'}:
            raise ValueError('Incomplete archive manifest coverage')
        for name, expected in manifest.items():
            if name not in names or hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Archive member mismatch: {name}")
        report = json.loads(archive.read("release_summary.json"))
        if report["status"] != "v18_public_geometry_pilot_release_audit_passed":
            raise ValueError("Archive report is not an audited release")
        if recompute:
            with tempfile.TemporaryDirectory(prefix='cwm_v18_verify_') as temp_name:
                temp = Path(temp_name)
                for name in manifest:
                    target = temp / name
                    if not target.resolve().is_relative_to(temp.resolve()):
                        raise ValueError('Unsafe archive path')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
                expected = audit_all(temp/'pilot_first', temp/'pilot_repeat', temp/'public_first', temp/'public_repeat')
                if report != expected:
                    raise ValueError('Independent archived report recomputation differs')
        return {"members": len(manifest), "sha256": digest(path)}


def audit_all(first, second, replay_first, replay_second):
    protocol = load_protocol()
    from replay_probe_public import BackboneTap
    from geometry import translated_records
    import torch
    if digest(REPO / 'cwm_v1/baseline/capsule.zip') != BASELINE_SHA:
        raise ValueError('Frozen capsule differs')
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='cwm_v18_baseline_') as temp_name:
        base = BackboneTap(REPO / 'cwm_v1/baseline/capsule.zip', Path(temp_name)/'restored')
        scenes = translated_records(base, protocol, protocol['wall_center_x_m'])
        first_result = audit_run(first, protocol, base, scenes)
        second_result = audit_run(second, protocol, base, scenes)
    _compare_runs(first, second)
    replay_result = audit_replays(first, replay_first, protocol)
    replay_result_repeat = audit_replays(second, replay_second, protocol)
    report = {"status": "v18_public_geometry_pilot_release_audit_passed", "protocol": protocol,
              "pilot_runs": [first_result, second_result], "public_replays": [replay_result, replay_result_repeat],
              "repeated_arrays_contexts_byte_equal": True, "repeated_json_values_equal": True,
              "enhanced_control_enabled": False,
              "new_model_trained": False, "holdout_used": False,
              "conclusion": "Eligible only for a separate fresh training protocol; no controller qualification or promotion."}
    return report


def build(first, second, replay_first, replay_second, output):
    report = audit_all(first, second, replay_first, replay_second)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cwm_v18_archive_") as temp_name:
        staging = Path(temp_name)
        (staging / "release_summary.json").write_text(json.dumps(report, indent=2))
        for label, root in (("pilot_first", first), ("pilot_repeat", second),
                            ("public_first", replay_first), ("public_repeat", replay_second)):
            for name, raw in _files(root).items():
                destination = staging / label / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
        for file in ROOT.glob('*.py'):
            destination = staging / 'verification_source' / file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file.read_bytes())
        members = {name: hashlib.sha256(raw).hexdigest() for name, raw in _files(staging).items()}
        (staging / "ARTIFACT_MANIFEST.json").write_text(json.dumps(members, indent=2, sort_keys=True))
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in _files(staging).items():
                if name == "ARTIFACT_MANIFEST.json":
                    continue
                archive.writestr(name, raw)
            archive.writestr("ARTIFACT_MANIFEST.json", (staging / "ARTIFACT_MANIFEST.json").read_bytes())
    return report, verify_archive(output, recompute=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--public-first", type=Path)
    parser.add_argument("--public-second", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_archive(args.verify, recompute=True)), flush=True)
        return
    if any(v is None for v in (args.first, args.second, args.public_first, args.public_second, args.output)):
        parser.error('Four completed runs and an exclusive output required')
    report, archive = build(args.first, args.second, args.public_first, args.public_second, args.output)
    print(json.dumps({"status": report["status"], "archive": archive}), flush=True)


if __name__ == "__main__":
    main()
