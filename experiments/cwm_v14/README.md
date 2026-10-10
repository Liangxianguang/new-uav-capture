# V14 preregistered public mechanism pilot

Fresh32 mirror groups,48 train/16 development episodes,975010–975073;
layouts1975010–1975041. Preserve holdout966010–966025. Fixed snapshots2,3,4,5,6,8.
No future-label window selection, target-rule changes, or control activation.
The existing qualified wallx1.5 finite distribution and frozen V9 library are
retained. New sample seeds and windows do not establish geometry or8Level transfer.

Train3 configurations x3seeds x40epochs: snapshot-only branch, public-history
branch, public-history CBF-command estimator. True3-class branch state and
filtered commands are supervision ONLY. Matched per-snapshot/offset training
prior, zero paired-branch response and raw proposed commands are controls.
History/geometry features remain original public252 and delayed4x6 context.
No simulator-private labels, target state, future execution or terminal masks
are network inputs. Neither model is a new executable controller.

Protocols are fixed before fresh collection/labels/training. All gates are
offline mechanism gates, not performance promotion. Two independent training
runs must reproduce checkpoint/optimizer/RNG/history/prediction evidence.
If unsuccessful, keep original GRU + delayed DN-MPC + empirical local CBF.

Collection reuses the unmodified V10 collector with this data protocol; all
outputs must be exclusive fresh directories. Runtime commands and terminal
results will be recorded once implemented and executed.
