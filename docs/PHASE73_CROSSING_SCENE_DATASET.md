# Phase 73: 600-scene target-crossing obstacle-bypass dataset

## Scope and preservation

Phase 73 creates a new validation/development scene pool for the failure mode
identified in Phase 72: the adversary must genuinely traverse the central
obstacle region instead of starting and escaping on the same side as the
defenders. It is a frozen scene resource, not a closed-loop model result.

The existing locked-test manifest, checkpoints, and result directories are not
read or modified. The new dataset is explicitly marked `not_a_locked_test` and
must not be used to tune or replace the existing locked test.

## Dataset contents

| Item | Value |
| --- | ---: |
| Total scenes | 600 |
| Mirror groups | 300 |
| Scenes per block | 200 |
| Blocks | development_calibration, development_validation, external_holdout |
| Target crossing required | 600/600 |
| Direct target path blocked | 600/600 |
| Minimum certified bypass routes | 2 per scene |
| Geometry | mixed cylinder, box, and wall obstacles |
| Observation conditions | 300 nominal, 300 delayed_noisy |
| Target behavior | adaptive_maneuvering |

Each mirror group contains one left-defender/right-target scene and one
right-defender/left-target scene. The obstacle layout is reflected about the
`x=0` plane, while obstacle shape, extent, height, speed condition, and route
contract are preserved. This makes paired controller comparisons less
sensitive to one-sided geometry.

The central obstacle zone is `x ∈ [-3.0, 3.0]`. A generated scene contains a
tall central gate that blocks the direct `z=4.2` target path. The generator
then certifies both positive-`y` and negative-`y` lateral routes using bounded
polyline clearance and finite-horizon transit checks. Independent defender
transit is also certified through a lateral channel. These are generation-time
reachability checks; they are not a closed-loop safety proof and do not turn
the local CBF into an R-CLBF-QP certificate.

## Reproducible artifacts

The generated artifacts are intentionally kept under the ignored `results/`
directory:

```text
results/phase73_crossing_scene_dataset/
├── scenes.jsonl
├── blocks/development_calibration.jsonl
├── blocks/development_validation.jsonl
├── blocks/external_holdout.jsonl
├── manifest.json
└── README.md
```

The main manifest SHA-256 is:

```text
10c7ac7f67f55bf077d7b1218229e26afacf28e855762e7efdd6f513a3160f02
```

The generator is deterministic from the versioned protocol and environment
configuration:

```powershell
python scripts\generate_phase73_crossing_scene_dataset.py `
  --protocol configs\phase73_crossing_scene_dataset.yaml `
  --environment-config configs\phase70_maneuvering_adversary_v2.yaml `
  --output-dir results\phase73_crossing_scene_dataset
```

All model arms should consume the same `scenes.jsonl` (or the same named block)
and report the same scene manifest hash. The `external_holdout` block is
reserved for confirmation after the model, planner, and safety-layer choices
are frozen.

## Decision and next use

The dataset-generation gate passed: 600 records, 300 complete mirror groups,
three 200-scene blocks, valid symmetry, and route certificates were verified.
No capture-rate conclusion can be drawn yet. The next experiment should be a
same-scene, multi-seed comparison on `development_validation`, using the
released V5/local-CBF arm and a rule or planner baseline. Only after that
comparison is complete should the frozen `external_holdout` be opened. Report
safe capture, collision, boundary/timeout, target crossing, minimum clearance,
and predictor/planner/QDR-or-tube/safety/total p50/p95/p99 when those modules
are evaluated.
