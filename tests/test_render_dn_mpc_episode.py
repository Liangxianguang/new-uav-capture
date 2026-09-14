from scripts.render_dn_mpc_episode import spec_from_scene_record


def test_s4_flat_scene_record_is_normalized_for_replay() -> None:
    record = {
        "episode_index": 17,
        "episode_seed": 726118,
        "layout_seed": 1726118,
        "mirror_group_id": 8,
        "observation_condition": "delayed_noisy",
        "defender_bias": "lower",
        "target_speed_scale": 0.75,
        "pursuit_overrides": {"message_delay_steps": 4},
        "scenario": {"obstacles": [{"shape": "wall"}]},
    }

    spec = spec_from_scene_record(record)

    assert spec["episode_seed"] == 726118
    assert spec["target_motion_mode"] == "adaptive_branching"
    assert spec["obstacle_count"] == 1
    assert spec["target_speed_scale"] == 0.75


def test_nested_scene_record_keeps_explicit_spec_fields() -> None:
    record = {
        "scenario": {"obstacles": []},
        "spec": {
            "episode_seed": 726101,
            "layout_seed": 1726101,
            "target_motion_mode": "adaptive_branching",
            "obstacle_count": 1,
        },
    }

    spec = spec_from_scene_record(record)

    assert spec["target_motion_mode"] == "adaptive_branching"
    assert spec["obstacle_count"] == 1
