"""Train reproducible MAPPO/IPPO baselines on the four-defender pursuit task.

The actor is parameter-shared across defenders and uses decentralized local
observations.  ``mappo`` uses ``env.centralized_state()`` for the critic;
``ippo`` uses the corresponding defender local observation for the critic.
Both variants share the same PPO implementation, action bounds, environment,
seed protocol, and optional execution CBF filter at evaluation time.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.learning import CentralizedSharedActorCritic, RecurrentCentralizedSharedActorCritic, SharedActorCritic  # noqa: E402
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_controllers import PursuitCBFSafetyFilter  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from evaluate_s4_closed_loop import read_scenes  # noqa: E402
from train_phase79_dagger_residual import build_environment  # noqa: E402


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--algorithm", choices=("mappo", "ippo"), required=True)
    p.add_argument("--recurrent", action="store_true", help="Use recurrent centralized-critic MAPPO.")
    p.add_argument("--seed", type=int, default=791601)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--use-cbf-eval", action="store_true")
    p.add_argument("--training-scenes", type=Path, help="Development scene JSONL used for training rollouts.")
    p.add_argument("--training-episodes", type=int, help="Maximum records loaded from --training-scenes.")
    p.add_argument("--expert-dataset", type=Path, help="Optional audited local-observation/action dataset for actor warm-start.")
    p.add_argument("--bc-epochs", type=int, default=0, help="Behavior-cloning epochs before PPO; zero disables warm-start.")
    p.add_argument("--bc-batch-size", type=int, default=1024)
    return p.parse_args()


def device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "environment_config" not in document:
        raise ValueError("Baseline config must contain environment_config.")
    env_path = Path(str(document["environment_config"]))
    if not env_path.is_absolute():
        env_path = path.parent / env_path
    environment = yaml.safe_load(env_path.resolve().read_text(encoding="utf-8"))
    if not isinstance(environment, dict):
        raise ValueError("environment_config must contain a mapping.")

    def merge(base: dict[str, Any], update: dict[str, Any]) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    overrides = document.get("environment_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("environment_overrides must be a mapping.")
    merge(environment, overrides)
    return document, environment


def make_env(config: dict[str, Any], seed: int, *, max_steps: int | None) -> CaptureRadiusPursuit3DEnv:
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=4, target_speed_scale=0.65)
    if max_steps is not None:
        env.max_steps = int(max_steps)
    env.reset(seed=seed)
    return env


def discounted_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        advantages[t] = last
    return advantages, advantages + values


def collect_episode(
    policy: torch.nn.Module,
    env: CaptureRadiusPursuit3DEnv,
    algorithm: str,
    dev: torch.device,
    action_scale: float,
    seed: int,
    max_steps: int | None,
    gamma: float,
    gae_lambda: float,
    record: dict[str, Any] | None = None,
    recurrent: bool = False,
) -> dict[str, Any]:
    if record is None:
        observation = env.reset(seed=seed)
    else:
        env, observation, _scenario = build_environment(env.config, record, max_steps=max_steps)
    locals_: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    old_log_probs: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []
    values: list[float] = []
    hidden = policy.initial_actor_hidden(4, device=dev) if recurrent else None  # type: ignore[union-attr]
    limit = int(max_steps or env.max_steps)
    for _ in range(limit):
        local = policy_observations(env, observation).astype(np.float32)
        state = env.centralized_state().astype(np.float32)
        with torch.no_grad():
            local_t = torch.as_tensor(local, device=dev)
            if recurrent:
                distribution, hidden = policy.distribution_step(local_t, hidden)  # type: ignore[union-attr]
                value = policy.value(torch.as_tensor(state, device=dev)[None])[0]  # type: ignore[union-attr]
            elif algorithm == "mappo":
                distribution = policy.distribution(local_t)  # type: ignore[union-attr]
                value = policy.value(torch.as_tensor(state, device=dev)[None])[0]  # type: ignore[union-attr]
            else:
                distribution, value_per_agent = policy.distribution_and_value(local_t)  # type: ignore[union-attr]
                value = value_per_agent.mean()
            raw = distribution.sample()
            action = torch.tanh(raw) * action_scale
            log_prob = policy._squashed_log_probability(distribution, raw).sum(dim=-1).sum()  # type: ignore[union-attr]
        next_observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        locals_.append(local)
        states.append(state)
        actions.append(action.cpu().numpy().astype(np.float32))
        old_log_probs.append(float(log_prob.cpu()))
        rewards.append(float(reward))
        dones.append(bool(terminated or truncated))
        values.append(float(value.cpu()))
        observation = next_observation
        if terminated or truncated:
            break
    advantages, returns = discounted_gae(
        np.asarray(rewards, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        np.asarray(dones, dtype=bool),
        gamma,
        gae_lambda,
    )
    return {"local": np.asarray(locals_), "state": np.asarray(states), "action": np.asarray(actions), "old_log_prob": np.asarray(old_log_probs), "advantage": advantages, "return": returns, "info": info}


def update(policy: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: dict[str, torch.Tensor], algorithm: str, action_scale: float, epochs: int, minibatch: int, clip_range: float, recurrent: bool = False) -> float:
    n = int(batch["advantage"].shape[0])
    total_loss = 0.0
    for _ in range(epochs):
        order = torch.randperm(n, device=batch["advantage"].device)
        for start in range(0, n, minibatch):
            idx = order[start : start + minibatch]
            local = batch["local"][idx]
            actions = batch["action"][idx]
            if recurrent:
                # Each item is one complete episode.  Keeping the sequence intact
                # makes the old log-probability and GRU state contract explicit.
                local_seq = batch["local"][idx]
                action_seq = batch["action"][idx]
                reset = torch.zeros(local_seq.shape[:2], device=local_seq.device)
                initial = torch.zeros((local_seq.shape[0], 4, policy.hidden_dim), device=local_seq.device)
                logp_agents, entropy_agents, _ = policy.evaluate_actions_sequence(local_seq, initial, reset, action_seq, action_scale)  # type: ignore[union-attr]
                logp = logp_agents.sum(-1)
                entropy = entropy_agents.sum(-1).mean()
                value = policy.value(batch["state"][idx].reshape(-1, batch["state"].shape[-1])).reshape(local_seq.shape[0], -1)  # type: ignore[union-attr]
                valid = batch["valid"][idx]
                old_logp = batch["old_log_prob"][idx]
                adv = batch["advantage"][idx]
                ratio = torch.exp(logp - old_logp)
                clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
                policy_loss = -torch.minimum(ratio * adv, clipped)
                value_target = batch["return"][idx]
                value_loss = 0.5 * (value - value_target).pow(2)
                loss = (policy_loss * valid).sum() / valid.sum().clamp_min(1.0) + (value_loss * valid).sum() / valid.sum().clamp_min(1.0) - 0.01 * entropy
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); optimizer.step()
                total_loss += float(loss.detach())
                continue
            if algorithm == "mappo":
                flat_local = local.reshape(-1, local.shape[-1])
                flat_actions = actions.reshape(-1, actions.shape[-1])
                dist = policy.distribution(flat_local)  # type: ignore[union-attr]
                raw = torch.atanh(torch.clamp(flat_actions / action_scale, -0.999999, 0.999999))
                logp = policy._squashed_log_probability(dist, raw).sum(-1).reshape(-1, 4).sum(-1)  # type: ignore[union-attr]
                value = policy.value(batch["state"][idx]).reshape(-1)  # type: ignore[union-attr]
            else:
                flat = local.reshape(-1, local.shape[-1])
                dist, value_agents = policy.distribution_and_value(flat)  # type: ignore[union-attr]
                raw = torch.atanh(torch.clamp(actions.reshape(-1, actions.shape[-1]) / action_scale, -0.999999, 0.999999))
                logp = policy._squashed_log_probability(dist, raw).sum(-1).reshape(-1, 4).sum(-1)  # type: ignore[union-attr]
                value = value_agents.reshape(-1, 4).mean(-1)
            ratio = torch.exp(logp - batch["old_log_prob"][idx])
            adv = batch["advantage"][idx]
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
            policy_loss = -torch.minimum(ratio * adv, clipped).mean()
            value_loss = 0.5 * (value - batch["return"][idx]).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + value_loss * 0.5 - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            total_loss += float(loss.detach())
    return total_loss / max(1, epochs * ((n + minibatch - 1) // minibatch))


def behavior_clone(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset_path: Path,
    algorithm: str,
    action_scale: float,
    epochs: int,
    batch_size: int,
    device_: torch.device,
) -> list[float]:
    archive = np.load(dataset_path.resolve())
    if "local_observations" not in archive or "actions" not in archive:
        raise ValueError("Expert dataset must contain local_observations and actions.")
    local = np.asarray(archive["local_observations"], dtype=np.float32).reshape(-1, archive["local_observations"].shape[-1])
    actions = np.asarray(archive["actions"], dtype=np.float32).reshape(-1, archive["actions"].shape[-1])
    if local.shape[-1] <= 0 or actions.shape[-1] != 3:
        raise ValueError("Expert dataset has incompatible dimensions.")
    expected_dim = int(
        policy.actor_body[0].in_features
        if algorithm == "mappo"
        else policy.body[0].in_features  # type: ignore[union-attr]
    )
    if local.shape[-1] != expected_dim:
        raise ValueError(
            "Expert dataset local-observation dimension does not match the baseline actor: "
            f"dataset={local.shape[-1]}, actor={expected_dim}. "
            "Use a dataset collected with the same route/prediction feature contract."
        )
    local_tensor = torch.as_tensor(local, device=device_)
    action_tensor = torch.as_tensor(actions, device=device_)
    if algorithm == "mappo":
        actor = policy.distribution  # type: ignore[union-attr]
    else:
        actor = policy.distribution_and_value  # type: ignore[union-attr]
    history: list[float] = []
    generator = torch.Generator(device=device_).manual_seed(17)
    for _ in range(epochs):
        order = torch.randperm(local_tensor.shape[0], generator=generator, device=device_)
        losses: list[float] = []
        for start in range(0, local_tensor.shape[0], batch_size):
            index = order[start : start + batch_size]
            distribution = actor(local_tensor[index])[0] if algorithm == "ippo" else actor(local_tensor[index])
            predicted = torch.tanh(distribution.loc) * action_scale
            loss = torch.nn.functional.mse_loss(predicted, action_tensor[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history


def main() -> None:
    a = args()
    if a.updates <= 0 or a.episodes_per_update <= 0:
        raise ValueError("updates and episodes-per-update must be positive")
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)
    dev = device(a.device)
    torch.set_num_threads(a.torch_threads)
    torch.set_num_interop_threads(a.torch_threads)
    document, config = load_config(a.config)
    training_records = None
    if a.training_scenes is not None:
        training_records = read_scenes(a.training_scenes.resolve(), a.training_episodes)
        if not training_records:
            raise ValueError("--training-scenes contains no records")
        if any("locked" in str(item.get("scene_block", "")).lower() for item in training_records):
            raise ValueError("Training scenes must not contain locked-test records")
    probe = make_env(config, a.seed, max_steps=a.max_steps)
    observation = probe.observe()
    local_dim = int(policy_observations(probe, observation).shape[-1])
    state_dim = int(probe.centralized_state().shape[-1])
    action_scale = float(config["agents"]["defender_max_speed"])
    if a.recurrent and a.algorithm != "mappo":
        raise ValueError("--recurrent is only supported with --algorithm mappo")
    if a.recurrent:
        policy: torch.nn.Module = RecurrentCentralizedSharedActorCritic(local_dim, state_dim, hidden_dim=a.hidden_dim).to(dev)
    elif a.algorithm == "mappo":
        policy: torch.nn.Module = CentralizedSharedActorCritic(local_dim, state_dim, hidden_dim=a.hidden_dim).to(dev)
    else:
        policy = SharedActorCritic(local_dim, hidden_dim=a.hidden_dim).to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=a.learning_rate)
    history: list[dict[str, Any]] = []
    bc_history: list[float] = []
    if a.expert_dataset is not None and a.bc_epochs > 0:
        bc_history = behavior_clone(
            policy,
            optimizer,
            a.expert_dataset,
            a.algorithm,
            action_scale,
            a.bc_epochs,
            a.bc_batch_size,
            dev,
        )
        (a.output / "behavior_cloning.json").write_text(
            json.dumps({"dataset": str(a.expert_dataset.resolve()), "dataset_sha256": hashlib.sha256(a.expert_dataset.resolve().read_bytes()).hexdigest(), "epochs": a.bc_epochs, "loss": bc_history}, indent=2),
            encoding="utf-8",
        )
    def save_progress() -> None:
        payload = {
            "state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "local_observation_dim": local_dim,
            "centralized_state_dim": state_dim,
            "action_dim": 3,
            "action_scale": action_scale,
            "hidden_dim": a.hidden_dim,
            "algorithm": a.algorithm,
            "actor_recurrent": bool(a.recurrent),
            "seed": a.seed,
            "training_scene_file": str(a.training_scenes.resolve()) if a.training_scenes else None,
            "training_scene_sha256": hashlib.sha256(a.training_scenes.resolve().read_bytes()).hexdigest() if a.training_scenes else None,
            "config_sha256": hashlib.sha256(a.config.resolve().read_bytes()).hexdigest(),
        }
        torch.save(payload, a.output / "checkpoint_latest.pt")
        (a.output / "progress.json").write_text(
            json.dumps({"algorithm": a.algorithm, "updates_completed": len(history), "updates_target": a.updates, "history": history}, indent=2),
            encoding="utf-8",
        )
    for update_index in range(a.updates):
        episodes = [
            collect_episode(
                policy,
                make_env(config, a.seed + update_index * a.episodes_per_update + j, max_steps=a.max_steps),
                a.algorithm,
                dev,
                action_scale,
                a.seed + update_index * a.episodes_per_update + j,
                a.max_steps,
                a.gamma,
                a.gae_lambda,
                record=training_records[(update_index * a.episodes_per_update + j) % len(training_records)] if training_records else None,
                recurrent=a.recurrent,
            )
            for j in range(a.episodes_per_update)
        ]
        if a.recurrent:
            length = max(len(e["reward"] if "reward" in e else e["old_log_prob"]) for e in episodes)
            def pad(key: str, fill: float = 0.0) -> np.ndarray:
                values = []
                for e in episodes:
                    value = np.asarray(e[key]); out = np.full((length,) + value.shape[1:], fill, dtype=value.dtype); out[:len(value)] = value; values.append(out)
                return np.asarray(values)
            batch_np = {key: pad(key) for key in ("local", "state", "action", "old_log_prob", "advantage", "return")}
            batch_np["valid"] = np.asarray([[1.0] * len(e["old_log_prob"]) + [0.0] * (length - len(e["old_log_prob"])) for e in episodes], dtype=np.float32)
        else:
            batch_np = {key: np.concatenate([episode[key] for episode in episodes], axis=0) for key in ("local", "state", "action", "old_log_prob", "advantage", "return")}
        batch = {key: torch.as_tensor(value, device=dev, dtype=torch.float32) for key, value in batch_np.items()}
        if a.recurrent:
            valid = batch["valid"]
            flat_adv = batch["advantage"][valid > 0]
            batch["advantage"] = (batch["advantage"] - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        else:
            batch["advantage"] = (batch["advantage"] - batch["advantage"].mean()) / (batch["advantage"].std() + 1e-8)
        loss = update(policy, optimizer, batch, a.algorithm, action_scale, a.ppo_epochs, a.minibatch_size, a.clip_range, recurrent=a.recurrent)
        row = {"update": update_index + 1, "loss": loss, "safe_capture_rate": float(np.mean([bool(e["info"]["safe_capture_success"]) for e in episodes]))}
        history.append(row)
        save_progress()
        if (update_index + 1) % max(1, a.updates // 10) == 0 or update_index == 0:
            print(json.dumps(row), flush=True)
    payload = {"state_dict": policy.state_dict(), "local_observation_dim": local_dim, "centralized_state_dim": state_dim, "action_dim": 3, "action_scale": action_scale, "hidden_dim": a.hidden_dim, "algorithm": a.algorithm, "actor_recurrent": bool(a.recurrent), "seed": a.seed, "use_cbf_eval": bool(a.use_cbf_eval), "training_scene_file": str(a.training_scenes.resolve()) if a.training_scenes else None, "training_scene_sha256": hashlib.sha256(a.training_scenes.resolve().read_bytes()).hexdigest() if a.training_scenes else None, "config_sha256": hashlib.sha256(a.config.resolve().read_bytes()).hexdigest()}
    torch.save(payload, a.output / "checkpoint.pt")
    (a.output / "training.json").write_text(json.dumps({"algorithm": a.algorithm, "history": history, "behavior_cloning_loss": bc_history, "document": document}, indent=2), encoding="utf-8")
    (a.output / "config.yaml").write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output": str(a.output.resolve()), "checkpoint": str((a.output / "checkpoint.pt").resolve()), "algorithm": a.algorithm, "updates": a.updates}, indent=2))


if __name__ == "__main__":
    main()
