# Experimental Evidence

This directory retains the two reports required to interpret the active
release. They are evidence files, not executable inputs.

| File | Meaning |
| --- | --- |
| [V4_LOCKED_TEST_REPORT.md](V4_LOCKED_TEST_REPORT.md) | Formal random/fixed benchmark over three independently trained retained-BC checkpoints. The S3 `Policy + CBF` result is `75.3% +/- 6.5%`. |
| [V4_LOCKED_TEST_SUMMARY.json](V4_LOCKED_TEST_SUMMARY.json) | Machine-readable aggregation for the formal report. |
| [V5_EXACT_REACTIVE_DEVELOPMENT_STATUS.md](V5_EXACT_REACTIVE_DEVELOPMENT_STATUS.md) | Development-only status of the released V5 checkpoint. Its `57/60` S3 result is a single training seed and is not a locked test. |

The original V4 checkpoints and the audited expert archives are not in this
repository. The V4 reports are consequently retained as historical evidence;
the released V5 checkpoint and its validation/replay workflow are the runnable
reproduction target.
