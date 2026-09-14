# Representative Capture Media

These files show a successful **V5 development** replay, not a formal locked
test result. The checkpoint is `models/v5_development_exact_reactive_seed661606.pt`.
It was evaluated on S3 validation episode `0` (`episode_seed=646101`,
`layout_seed=1646101`) with CBF enabled and a recurrent-state reset interval of
one control step.

The scene contains one cylinder, one wall, and one box. It terminates in a
safe capture: a defender reaches the `0.80 m` capture radius with no collision
and no boundary violation. The capture snapshot/GIF/MP4 are visual evidence of
one replay, whereas rates must be read from the development status and formal
reports.

The release verification reran the complete V5 validation block in this
repository on 2026-08-23: `57/60` safe captures (`95.0%`), collision `0%`,
boundary violation `0%`, and Transit `100%`. This remains development-only
evidence because the checkpoint is a single training seed. In the displayed
episode, capture occurs at `4.8 s` with final nearest distance `0.78284 m`.

- `v5_development_s3_episode0_capture_3d.png`: final 3-D capture frame.
- `v5_development_s3_episode0_capture_3d.gif`: rendered 3-D animation.
- `v5_development_s3_episode0_capture_3d.mp4`: H.264 version of the same
  animation.

## Phase 68：Official S4 interception showcase

These four replay-only videos use the frozen locked-test Official S4 checkpoint
with distributed delayed DN-MPC and local CBF. They are visual demonstrations,
not new statistics or additional tuning. Each replay ends in a captured target
and uses the v3 light publication renderer with `+z` altitude upward.

The source, metrics, and replay command are documented in
[`docs/PHASE68_INTERCEPTION_VIDEO_SHOWCASE.md`](../PHASE68_INTERCEPTION_VIDEO_SHOWCASE.md).

| Condition | MP4 | GIF | Final frame |
| --- | --- | --- | --- |
| Nominal, episode 0 | [`official_s4_episode_000.mp4`](phase68_interception_showcase/official_s4_episode_000.mp4) | [`GIF`](phase68_interception_showcase/official_s4_episode_000.gif) | [`PNG`](phase68_interception_showcase/official_s4_episode_000_final.png) |
| Partial dropout, episode 2 | [`official_s4_episode_002.mp4`](phase68_interception_showcase/official_s4_episode_002.mp4) | [`GIF`](phase68_interception_showcase/official_s4_episode_002.gif) | [`PNG`](phase68_interception_showcase/official_s4_episode_002_final.png) |
| Delayed noisy, episode 4 | [`official_s4_episode_004.mp4`](phase68_interception_showcase/official_s4_episode_004.mp4) | [`GIF`](phase68_interception_showcase/official_s4_episode_004.gif) | [`PNG`](phase68_interception_showcase/official_s4_episode_004_final.png) |
| Nominal, episode 6 | [`official_s4_episode_006.mp4`](phase68_interception_showcase/official_s4_episode_006.mp4) | [`GIF`](phase68_interception_showcase/official_s4_episode_006.gif) | [`PNG`](phase68_interception_showcase/official_s4_episode_006_final.png) |

The replay manifest and raw trajectory files remain in the local ignored
`results/phase68_interception_showcase_official_s4_seed727201/` directory.

Regenerate the media with the two commands in the repository README. The
checkpoint SHA-256 is
`535098773be05687e147043435649378532362d479bdc0375842970370ba40ba`.

## Phase 69：Opposite-side obstacle-bypass showcase

These two controlled development replays explicitly initialize the target
and the four-defender cluster on opposite sides of the central mixed obstacle
field. All four defenders cross the obstacle zone before the safe capture in
both directions. They are visual evidence rather than additional benchmark
statistics; see [`docs/PHASE69_OPPOSITE_SIDE_OBSTACLE_BYPASS_SHOWCASE.md`](../PHASE69_OPPOSITE_SIDE_OBSTACLE_BYPASS_SHOWCASE.md)
for metrics and reproduction commands.

| Condition | MP4 | GIF | Final frame |
| --- | --- | --- | --- |
| Defenders left → target right | [`MP4`](phase69_opposite_side_showcase/v5_opposite_side_mixed_episode.mp4) | [`GIF`](phase69_opposite_side_showcase/v5_opposite_side_mixed_episode.gif) | [`PNG`](phase69_opposite_side_showcase/v5_opposite_side_mixed_episode_final.png) |
| Defenders right → target left | [`MP4`](phase69_opposite_side_showcase/v5_opposite_side_mirror_episode.mp4) | [`GIF`](phase69_opposite_side_showcase/v5_opposite_side_mirror_episode.gif) | [`PNG`](phase69_opposite_side_showcase/v5_opposite_side_mirror_episode_final.png) |
