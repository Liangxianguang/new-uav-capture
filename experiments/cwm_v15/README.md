# V15 preregistered frozen public CBF-mediated target response

Fresh32 mirror groups,24train/8development; episodes978010–978073,
layouts1978010–1978041. Preserve holdout966010–966025. Same qualified
wallx1.5 scenes and fixed steps2,3,4,5,6,8; no future-label sampling.

V14 cbf_history seed976102 is fixed from its error-median selection. It is a
public-input estimator, never the safety layer. Cached estimated commands
are inputs only to the optional two-head core. Raw proposed commands remain
unchanged in the original controller, label masks and full local cost engine.

Three configs x3seeds x80epochs: independent motion-only, raw plain two-head,
mediated plain two-head. Match raw/mediated init, batch RNG and trainable
budget. No private command oracle, branch head, task loss or candidate changes.
Response error must improve over zero and raw; original full-cost gains must
have positive group lower bounds versus own motion/CV/motion-only/raw.
Offline gates never activate control. Two independent runs and full audit
required; failed checkpoints remain unpromoted and original model remains.

Collection uses original V10 collector with this data protocol. Public replay
uses V10 replay_public. Protocol committed before new data/training; all output
paths must be exclusive fresh directories. Results/commands will follow.
