# V19 frozen-common-motion causal geometry response

Preregister before collection or training. Fresh32 mirror groups, first24 train
and last8 development, separate from V18 pilot and reserved holdout. Original
controller and CBF are never modified; all new candidates are shadow only.

V18 showed actionable paired-response labels but also that exact response added
to biased motion can worsen ranking. Stage1 calibrates common reference motion;
stage2 trains raw/mediated response using a separate core, with the shared
stage1 reference prediction fully frozen. Within-seed response comparisons
therefore cannot attribute a changed common-motion model to response effects.

All three seeds, final fixed epochs and two deterministic runs are required.
Candidate library gain and learned response gain remain separate. Offline
gates do not activate a controller; previous failures remain recorded.
