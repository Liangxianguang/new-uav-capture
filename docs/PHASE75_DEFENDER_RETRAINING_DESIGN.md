# Phase 75: public-belief route-intent defender retraining design

## Why retraining is necessary

Phase 74 showed that the failure is not repaired by changing the prediction
backbone. On the same 200-scene target-crossing validation block, GRU,
Diagonal SSM, and Official S4 all reached only about 10.7--10.8% safe capture
without physical execution delay. With a 4-step command delay and 0.08 m/s
bounded command noise, all three fell to 0% safe capture and 100% collision.
The current recurrent defender was warm-started from same-side pursuit data;
its teacher, `DynamicEncirclementController`, tracks a belief and distributes
perimeter points but does not explicitly select a bypass side or assign a
blocking role before the target reaches the obstacle.

Continuing to behavior-clone that teacher would reproduce the wrong behavior.
The redesign therefore changes the supervision contract and the defender
policy interface before changing the neural backbone.

## Proposed method: Public-Belief Route-Intent Cooperative Defender

The teacher and the learned policy receive only the public information
available to the defender: delayed/noisy target beliefs, shape-aware obstacle
geometry, teammate states/messages, and the execution queue. Target ground
truth is prohibited online. A conservative grid route planner enumerates
left, right, and upper bypass candidates, then scores them using predicted
interception time, target escape margin, defender arrival time, clearance, and
role-switch cost. Every 6--10 steps the teacher may update the route and role
assignment, while the low-level controller tracks the selected waypoint under
speed, acceleration, turn-rate, and jerk limits.

The four roles are one interceptor, two gate blockers, and one recovery agent.
The exact assignment is not fixed by defender index. It is recomputed from
public-belief reachability and held for a short interval to prevent the
oscillation seen when a noisy observation changes the nearest defender.

The learned defender is hierarchical:

1. A small route-intent head selects `left/right/upper` and updates every
   eight steps.
2. A role head assigns interceptor/gate/recovery responsibilities.
3. A recurrent low-level actor produces the velocity command from the current
   local observation, route embedding, waypoint vector, and a compact pending
   queue summary.
4. The existing local CBF remains the final empirical filter. It is not
   upgraded rhetorically into an R-CLBF-QP proof.

GRU is the first defender backbone because Phase 74 found no stable closed-loop
gain from SSM/S4 and GRU is cheaper for DAgger iterations. The target predictor
can remain the selected `both` checkpoint; predictor-backbone comparison is a
separate experiment.

## Training data and curriculum

Generate a fresh, mirror-balanced Phase 75 training archive with 1,200 scenes,
plus disjoint 300-scene calibration and 300-scene validation blocks. All
training/validation scenes require target crossing and have a blocked direct
route, but the exact layouts and mirror groups must not overlap between
blocks. The current Phase 73 validation block is not converted into training
labels; it remains the frozen comparison block for Phase 74.

Each accepted demonstration must contain a safe capture, at least two
defenders entering the obstacle zone, no collision/boundary termination, and
the route/role/queue labels listed in
`configs/phase75_defender_retraining.yaml`. Rejected attempts stay in a
diagnostic manifest and are never silently used as positive action labels.

Use four stages: easy crossing warm-up, route-side learning, delayed/noisy
execution, and adversarial transfer. The difficulty is increased by one main
factor at a time. Mirror augmentation must preserve the left/right label
transformation, and route labels should be approximately balanced so the
policy cannot win by always choosing one side.

## Optimization and DAgger

First run hierarchical behavior cloning with action Huber loss plus route,
role, waypoint, and action-smoothness auxiliary losses. Then run three
development-only DAgger rounds. Roll out the current policy under the actual
delay/noise contract, query the public-belief teacher at every step, and add
corrective labels. The expert/policy mixture decreases from 75/25 to 25/75.
This directly targets compounding errors: an early wrong bypass choice should
be corrected before the target reaches the central obstacle.

Do not accept a checkpoint solely because action MSE falls. Select by the
calibration gates and then evaluate the untouched validation block with three
training seeds. Preserve the route-intent confusion matrix, role-switch rate,
queue length, first unsafe command, target-crossing rate, and failure contact
type in addition to safe capture.

## Acceptance gates and execution order

1. Implement and unit-test the public-belief route/role teacher; include a
   no-target-truth audit and route-feasibility certificate for every scene.
2. Generate and audit the disjoint Phase 75 scene blocks.
3. Collect quality-gated demonstrations. Stop if rejection exceeds 40% or if
   any route class is under-represented; repair the teacher/curriculum before
   training.
4. Train BC, then run the three DAgger rounds on training scenes only.
5. Select on calibration, confirm on the fresh validation block, and compare
   against the released GRU, best planner baseline, local CBF-off, and the
   old teacher under identical delay/noise.
6. Only if the validation gates pass, freeze the model and open the reserved
   external holdout once. Locked-test remains closed until the full protocol is
   signed off.

The target confirmation is at least 70% safe capture, at most 5% collision and
boundary violation, at most 10% timeout, and at least 95% target crossing on
the strict validation condition. These are promotion gates, not current
results. Latency must be reported as predictor/planner/QDR-or-tube/safety/total
p50, p95, and p99; 100 ms is not a hard gate.

The executable design source is
`configs/phase75_defender_retraining.yaml`. No Phase 75 checkpoint is claimed
until the teacher implementation and quality-gated data collection succeed.
