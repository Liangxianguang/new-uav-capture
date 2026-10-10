# V28: fresh fixed-library candidate ranking experiment

This is a new falsifiable ranking hypothesis, not a revision of V23's failed
cost-contrast result. Data/training protocols must be committed and pushed BEFORE
fresh collection. `collect_ranking_data.py` checks exact pushed HEAD and current
source/protocol bytes before it starts. V23 and reserved holdout are not reused.

Fresh 32 mirror groups (64 episodes): first24 train, last8 development. Every
sequential local invocation at steps2/3/4/5/6/8 is recorded, including later
best-response rounds. Candidate library uses frozen V27 shared public proposals
from the fixed failed V23 teacher, identically for all scoring models. Proposal
inputs/forecasts are frozen before cloned paired target labels are generated.
Original selected anchor, received peer plans,252 encoding,GRU,DN-MPC,CBF and
target/safety rules are unchanged. Original trajectory, plan and CBF-command
equality are checked against an unexpanded baseline replay for every episode.

New primary `cv_rank_l2` trains an anchored target response, with matched
`cv_l2` and `cv_motion_only` controls, three seeds and two complete independent
runs. The auxiliary KL loss fits the candidate FULL COST distribution, including
original target-independent offsets. Physical vector-L2 response supervision
remains. Motion is independently pretrained/frozen. Private costs/temperature
are detached supervision only, never forward or deployment inputs.

Same-score different-library comparisons isolate candidate expansion. Same-
library different-scoring comparisons isolate response/ranking learning. Use
shared complete call support and realized true cost or common-library regret;
different library minima cannot be subtracted as an expansion gain. Original
basic V23 accuracy/decision thresholds remain, with additional fixed primary
ranking-vs-L2 criteria. No best-seed/epoch/gate switching or holdout tuning.

```powershell
python -m pytest experiments/cwm_v27 experiments/cwm_v28 -q
python experiments/cwm_v28/collect_ranking_data.py --output results/cwm_v28/NEW_DATA
```

Collector completion and data-gate pass alone DO NOT authorize training. A
separate independent public252/received-context/candidate/branch/full-cost audit
is required. The trainer/release audit are still pending implementation; the
training protocol and loss are frozen first, not claimed as trained results.
Later active sequential selector, untouched holdout, original completeLevels
and new scenes closed-loop capture/safety/latency remain mandatory. Everything
stays default off; empirical local CBF is not a new formal safety proof.
