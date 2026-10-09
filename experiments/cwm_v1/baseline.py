"""Isolated adapter around the unmodified historical evaluator in a capsule."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from freeze_baseline import REFERENCE, restore


class Baseline:
    def __init__(self, capsule: Path, restored: Path):
        self.root = restored.resolve()
        self.capsule_manifest = restore(capsule, self.root)
        sys.path[:0] = [str(self.root / "src"), str(self.root / "scripts")]
        import evaluate_minimax_mpc as evaluator
        from encirclement3d.observation_encoding import policy_observations
        from encirclement3d.distributed_dn_mpc import DistributedDNMPCConfig
        from encirclement3d.minimax_mpc import MinimaxMPCConfig
        from encirclement3d.showcase import scenario_from_metadata

        self.evaluator = evaluator
        self.provenance = json.loads((self.root / REFERENCE / "manifest.json").read_text())
        self.scenario_from_metadata = scenario_from_metadata
        self.planner_config = MinimaxMPCConfig.from_mapping(self.provenance["planner"])
        self.distributed_config = DistributedDNMPCConfig.from_mapping(self.provenance["distributed"])
        self.checkpoint = evaluator.model_from_checkpoint(
            self.root / "models/formal_phase2_v3/gru_baseline_seed745101.pt", torch.device("cpu"))
        original = yaml.safe_load((self.root / "configs/capture_radius_pursuit_central_v4_flee.yaml").read_text())

        def encoder(env, observation=None):
            keys = ["max_observation_obstacles", "include_uncertainty_features", "include_prediction_features"]
            saved = {key: env.pursuit[key] for key in keys}
            saved_task = dict(env.task)
            try:
                env.pursuit.update({key: original["task"]["pursuit"][key] for key in keys})
                env.task.update(policy_obstacle_geometry=original["task"]["policy_obstacle_geometry"],
                                policy_clearance_features=False, policy_role_slot_features=False,
                                policy_route_intent_features=False)
                features = policy_observations(env, observation)
                if features.size != 252:
                    raise ValueError("Frozen encoder must have exactly 252 features")
                return features
            finally:
                env.pursuit.update(saved)
                env.task.clear()
                env.task.update(saved_task)

        evaluator.policy_observations = encoder
        original_filter = evaluator.PursuitCBFSafetyFilter

        class FrozenMarginFilter(original_filter):
            def __init__(self, env, *args, **kwargs):
                super().__init__(env, *args, **kwargs)
                self.margin = 0.35

        self.filter_class = FrozenMarginFilter
        evaluator.PursuitCBFSafetyFilter = FrozenMarginFilter
        original_env = evaluator.CaptureRadiusPursuit3DEnv
        original_planner = evaluator.DistributedMinimaxDNMPC
        adapter = self
        self.observer = None
        self.last_sequence = None
        self.last_info = {}

        class ObservedPlanner(original_planner):
            def plan(self, *args, **kwargs):
                plan = super().plan(*args, **kwargs)
                adapter.last_sequence = plan.action_sequence.copy()
                return plan

        class ObservedEnv(original_env):
            def step(self, actions, *args, **kwargs):
                if adapter.observer is not None:
                    adapter.observer(self, self.observe(), np.asarray(actions).copy(),
                                     None if adapter.last_sequence is None else adapter.last_sequence.copy())
                result = super().step(actions, *args, **kwargs)
                adapter.last_info = copy.deepcopy(result[-1])
                return result

        evaluator.DistributedMinimaxDNMPC = ObservedPlanner
        evaluator.CaptureRadiusPursuit3DEnv = ObservedEnv
        self.env_class = original_env  # Branches must not invoke the observer recursively.

    def configuration(self, record: dict) -> dict:
        config = copy.deepcopy(self.provenance["environment"])
        config["task"]["pursuit"].update(copy.deepcopy(record["pursuit_overrides"]))
        config["task"]["pursuit"]["target_motion_mode"] = record["target_motion_mode"]
        execution = copy.deepcopy(record["execution"])
        if "command_noise_std_mps" in execution:
            execution["command_noise_std"] = execution.pop("command_noise_std_mps")
        config.setdefault("dynamics", {}).setdefault("execution", {}).update(execution)
        config["world"]["max_steps"] = 250
        config["experiments"] = [{"obstacle_count": record["obstacle_count"], "target_speed_scale": record["target_speed_scale"]}]
        return config

    def run(self, record: dict, trajectory: Path | None = None, observer=None):
        self.observer, self.last_sequence, self.last_info = observer, None, {}
        try:
            row, steps = self.evaluator.run_episode(
                self.configuration(record), seed=record["episode_seed"], method="distributed_delayed",
                planner_config=self.planner_config, candidate_source="checkpoint", checkpoint_data=self.checkpoint,
                device=torch.device("cpu"), num_samples=1, sampling_steps=8,
                sampling_seed=745102 + record["episode_index"] * 1000, projection_iterations=4,
                use_local_cbf=True, safety_layer="local_cbf", prediction_refresh_interval_steps=1,
                distributed_config=self.distributed_config,
                scenario=self.scenario_from_metadata(record["scenario"]), validate_scenario=False,
                record_history=trajectory is not None, trajectory_path=trajectory)
            row.update(level=record.get("level"), episode_index=record["episode_index"],
                       boundary_violation=bool(self.last_info.get("defender_boundary_violation", False)))
            return row, steps
        finally:
            self.observer = None

    def records(self) -> list[dict]:
        return [json.loads(line) for line in (self.root / REFERENCE / "scenes.jsonl").read_text().splitlines() if line]
