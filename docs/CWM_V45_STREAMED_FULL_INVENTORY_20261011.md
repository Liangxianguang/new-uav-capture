# V45: disk-backed full paired-development inventory

This is an engineering milestone, not a training result or efficacy certificate.

`experiments/cwm_v45/paired_prefix_streamed.py` delegates all five input arguments to the unchanged V42 native runner. Only its final evidence-inventory builder and JSON writer aliases are replaced in a scoped context and restored on success or failure. Ordinary diagnostic JSON still uses the original writer. Qualification, models, original costs, scene populations, controller deadlines, timing, safety gates and the full 10,752 method-episode workload are unchanged. No holdout or deployment entry is added.

The retained SQLite ledger avoids accumulating the complete filename set and manifest dictionary in Python. Its cache is configured to 4 MiB and temporary storage to disk; this is not a bound on total process memory. Episode rows, individual event lists, native rollouts and other allocations are not streamed by this change. Actual full-workload memory savings have not been measured.

All original inventory membership is retained: protocol/certificates/scenes/preload, imported-source snapshots, cohort episode/step files, every trajectory/command/event file and every event array/context file. Canonical paths, regular files, event hashes, case aliases and later duplicate mutations are checked. The operational SQLite ledger is not a member of the scientific manifest. Manifest records are serialized one at a time in lexical order using the original JSON indentation, escaping, field order and platform newline behavior.

Verification on 2026-10-11:

- V45: 26 tests passed in 5.87 seconds; actual process exit 0.
- V42/V43/V44/V45 combined: 120 tests passed in 5.99 seconds; actual exit 0.
- CLI `--help`: actual exit 0.
- Exact byte comparisons use the actual original writer, including Windows CRLF translation. The initial two LF-only expected-byte assertions failed and were corrected in tests; production serialization was not changed to force LF.

These tests use synthetic fixtures, including a 512-member inventory, toy callbacks and failed-primary refusal. They do not establish million-file scaling, native enhanced closed-loop results, active-model fault behavior, independent scientific replay, causal identification or policy improvement. No qualified V38 primary was loaded and no native enhanced evaluation was launched.

After complete training/reload/archive qualification, the future full paired evaluation can use this entry with the same five arguments as V42. It must still pass independent replay, genuine active-model fault checks, capture/safety/latency gates and the single sealed holdout before promotion. Until then the enhancement stays disabled and the original GRU + DN-MPC + CBF remains the supported fallback.
