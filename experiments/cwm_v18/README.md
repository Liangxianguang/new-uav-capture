# V18 public local geometry probe pilot

Freeze before collection.16fresh original episodes/8mirror groups under qualified
wallx1.5, unchanged target, DN-MPC and CBF. Six fixed actual local invocation
snapshots, eight-step horizon only. Shadow actual/CV/radial/tangential/stop own
commands use available delayed peer plans, never current privileged peers.

This is a new local-action scope, not relaxation of V16's failed80step new-scene
gate. Candidate and legality gains are separate from any learned-model gains.
No model trained/loaded and no enhanced action executed. Failed/skipped/prefix
support retained. A successful data pilot requires separate fresh training and
validation groups; original holdout remains closed.

## Reproduction

Run each collector and public replay in a new Python process with exclusive
output directories. The collector is frozen to the preregistered 8 mirror groups.

```powershell
python experiments/cwm_v18/collect_probe_pilot.py --output results/cwm_v18/pilot_first
python experiments/cwm_v18/collect_probe_pilot.py --output results/cwm_v18/pilot_repeat
python experiments/cwm_v18/replay_probe_public.py --data results/cwm_v18/pilot_first --output results/cwm_v18/public_first
python experiments/cwm_v18/replay_probe_public.py --data results/cwm_v18/pilot_repeat --output results/cwm_v18/public_repeat
python experiments/cwm_v18/release_verify.py --first results/cwm_v18/pilot_first --second results/cwm_v18/pilot_repeat --public-first results/cwm_v18/public_first --public-second results/cwm_v18/public_repeat --output results/cwm_v18/reproduced_pilot.zip
python experiments/cwm_v18/release_verify.py --verify experiments/cwm_v18/artifacts/public_local_geometry_pilot_audited_20261010.zip
python -m pytest -q experiments/cwm_v18
```

The released archive excludes restored baseline copies, but includes both full
collections, both independent public replays, raw prefix/terminal/command labels,
public contexts, source snapshots and verification code. Its verifier restores
the SHA-pinned capsule separately and recomputes original full costs, rankings,
support, geometry validity and data gates. Public252/GRU history equality is
checked during the independent replay; full per-frame observations are not
archived and are not claimed to be re-encoded by the release verifier.

Pilot response: actual unique 0.001507m, geometry union 0.091091m;
group-equal >0.05m fractions 0% and 29.309%. All16 original safe captures and
all six data checks pass. This is data signal, not learned-world-model gain.
V15's failed response/decision gates remain unchanged.
