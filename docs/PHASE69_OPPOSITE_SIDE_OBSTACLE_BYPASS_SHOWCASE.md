# Phase 69: Opposite-Side Obstacle-Bypass Showcase

The previous replay placed the target and the defender cluster on the same
side of the obstacle field. That was useful for checking capture rendering,
but it did not visibly test obstacle bypass. This phase adds two explicit
opposite-side replays: the target and the four defenders start on opposite
sides of the central mixed obstacle layout, and the defenders must enter and
cross the obstacle zone before capture.

These are controlled development replays for visual inspection. They are not
new locked-test statistics, do not tune the locked test, and do not constitute
a formal safety proof. The CBF used here is the empirical local CBF filter;
the rejected R-CLBF-QP/robust-CBF composition is not claimed to be feasible or
formally certified.

## Replays

| Direction | MP4 | GIF | Final 3-D frame |
| --- | --- | --- | --- |
| Defenders left → target right | [MP4](media/phase69_opposite_side_showcase/v5_opposite_side_mixed_episode.mp4) | [GIF](media/phase69_opposite_side_showcase/v5_opposite_side_mixed_episode.gif) | [PNG](media/phase69_opposite_side_showcase/v5_opposite_side_mixed_episode_final.png) |
| Defenders right → target left | [MP4](media/phase69_opposite_side_showcase/v5_opposite_side_mirror_episode.mp4) | [GIF](media/phase69_opposite_side_showcase/v5_opposite_side_mirror_episode.gif) | [PNG](media/phase69_opposite_side_showcase/v5_opposite_side_mirror_episode_final.png) |

The renderer uses a fixed, lightly elevated camera, positive upward altitude,
solid semi-transparent obstacles, directional UAV icons, red target motion,
and a frozen capture frame. The central obstacle-zone crossing is therefore
visible in both the animation and the final trajectory frame.

## Replay evidence

| Replay | Safe capture | Capture time | Minimum clearance | Defender crossing | Collision / boundary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Left → right (`s1`, seed `642121`) | yes | 4.2 s | 0.554 m | 4/4 | 0 / 0 |
| Right → left (`s2`, seed `642122`) | yes | 3.6 s | 0.505 m | 4/4 | 0 / 0 |

Both runs report a central encounter, all four defenders entering and crossing
the obstacle zone, and cooperative safe capture. The first run also completes
the independent transit-route diagnostic for both defender and target routes.
The second run's target transit diagnostic is conservative and terminates on
an exact-clearance boundary condition; this does not change the actual replay
outcome, which is a collision-free capture. It should not be presented as a
target route-completion result.

## Reproduce

From the repository root, run the two deterministic showcase commands below.
The checkpoint is a development checkpoint, so these commands are intended to
reproduce the visual demonstration rather than a formal benchmark.

```powershell
python scripts\run_mixed_obstacle_showcase.py --method capture `
  --checkpoint models\v5_development_exact_reactive_seed661606.pt `
  --seed 642121 `
  --output-dir results\phase69_opposite_side_mixed_candidate `
  --initial-side-distance 7.0 `
  --scenario s1 --layout mixed `
  --detection-range 14.0 --target-speed-scale 0.45 `
  --use-cbf --device cpu --fps 12 --frame-stride 1

python scripts\run_mixed_obstacle_showcase.py --method capture `
  --checkpoint models\v5_development_exact_reactive_seed661606.pt `
  --seed 642122 `
  --output-dir results\phase69_opposite_side_mirror_candidate `
  --initial-side-distance 7.0 `
  --scenario s2 --layout mixed `
  --detection-range 14.0 --target-speed-scale 0.45 `
  --use-cbf --device cpu --fps 12 --frame-stride 1
```

The raw trajectories and episode JSON files remain in the ignored
`results/phase69_opposite_side_*_candidate/` directories. Their trajectory
hashes are `183feade75b35c64cbcaf3adefd494e9e41dc39fecc22e64b6951bd124d65007`
and `1f9c91c7873b6384b925611a9268679d941cb27effa130850b05f0d3efd7a407`.

