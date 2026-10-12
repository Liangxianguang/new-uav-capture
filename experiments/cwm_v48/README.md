# Coordinate-local derivative audit revision (2026-10-12)

User explicitly authorized correction and a new from-scratch run. V36 failed
before Adam creation and stays failed. No existing output/source is overwritten.

V47 diagnosis identified supported call15256, complete BASE call10781,
episode1036089/step3/ordinal0/agent0, TRAIN wider_encirclement. The role gap
3.2430380124459646e-5 triggered the old 4e-5 global veto. All nine sampled
finite differences matched autograd; roles stayed2/2/2. Roles depend only on
first-time position, so later-time coordinates cannot cause a role switch.

This prospective fork preserves first-time4*step, all original full-score,
label, kink and gradient thresholds and sampled candidates/coordinates. It
verifies first-time bytes and roles before auditing later-time coordinates.
No sample is removed. A call with no verified smooth coordinate still fails.
Physical/controller/safety/capture/latency/efficacy contracts are unchanged.

All production V36/V37/V38 files are mechanically forked into this directory;
only local import routing, derivative audit, and protocol version differ.
Both training and cold reload use the corrected coordinate audit. Archive
replay executes this version's reload entry in the independent extraction.
Exact published source snapshots bind all local Python files and protocol.
Original V23/V34/V35/V36/V37/V38 remain unchanged.

140 implementation/structural tests passed in14.88s. This is not training,
full native audit qualification, efficacy, latency or active control evidence.
V47 full GRU/CV native audit remains running at publication preparation.

New outputs: D:/uav-capture/cwm48-p-20261012 and cwm48-r-20261012;
two fixed9-model runs, same3 configurations/seeds and80epoch budgets.
Then cwm48-reload-20261012 through reload_streamed.py, and complete
package_streamed_release.py archive with a new archive/replay/parts path.
Never feed these versioned contracts to an old active-policy gate as if equal.
New enhanced control remains disabled pending full scientific and closed-loop
qualification; no holdout is accessed.

scripts/run_cwm48_after_coordinate_audit.ps1 holds the actual V47 process
handle, requires real exit0 and complete bound diagnostic evidence, resources
and exact published HEAD, then serially calls both new native training entries.
Waiting and native input/derivative audits are NOT optimizer evidence. Any
native failure preserves output and stops the queue; resource deferral is not
a scientific efficacy failure. V36 failed directory is never resumed.
