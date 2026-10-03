# Exact native AdamW continuation preparation

The user set a maximum total of 300,000 training updates per run after asking
whether later recovery and decreasing failure frequency should be measured.
The complete 150,000-update lower-rate calibration is preserved separately.
The scientific continuation changes only total steps from 150,000 to 300,000.

`prepare_budget_extension.py` tests the existing
core resume path on CPU with CUDA hidden and the full width-128/two-layer model.
Its smaller mod-7 corpus and batch four keep this an implementation oracle.
Twenty updates followed by exact checkpoint restoration and twenty more match
an uninterrupted forty-update run in model tensors, every native AdamW moment
and step, shuffled permutation/cursor/generator state and example exposure.
Every native parameter step is forty. Canonical metrics agree exactly; dense
gradient and full diagnostic logs agree byte for byte. All four original log
prefixes remain byte-identical. Warmup is not restarted and cumulative costs
increase. Changed rate, decay, initialization, batch policy and decreased budget
are rejected without modifying the original resources. The completed portable
archive verifies without PyTorch. No CUDA context is created.

The [validation](budget-extension-preparation/validation.json), exact histories,
source snapshot and hashes retain that oracle. It verifies continuation mechanics
rather than future scientific convergence or GPU bitwise equivalence.

[stability_recovery_series.py](../../stability_recovery_series.py) adds fixed
10,000-update windows over the whole post-long-onset history, including the
150,000–250,000 portion outside the new persistence tail. A separate leading
interval preserves observations between onset and the first grid boundary.
Its partial width cannot contribute a normalized rate or a window-to-window
change. All future/partial windows retain their actual/expected support; empty
fractions and incomplete rates stay unavailable. A boundary-crossing episode
has one onset and contributes every failed observation to its actual window.
Both joint and held-out series remain distinct. The
[series audit](recovery-series-validation.json) records three passing meaningful
tests and exact support/failure accounting on five complete historical cases.
The original metric source and all original archives remain unchanged.

The strict extension guard pins the original full result, native checkpoint,
seven original resources, six copied resources, criterion, corpus, environment,
31 core/manuscript files, five extension sources and three recovery sources.
The new launcher calls the existing core `train(..., resume=True)` directly
after verification; the ordinary stage wrapper correctly rejects an old-budget
child config. The child records `posthoc_budget_extension`, while the new
root manifest freezes all additional updates before training. No checkpoint
metadata is rewritten. Final persistence uses the unchanged last-50,000 rule
at 250,000–300,000; six fresh full-budget confirmations remain required later.

CUDA peaks are the maximum over the preserved parent and all observed extension
segments. Canonical peak snapshots add host recording cost to wall time; they
do not change optimizer updates or exhaustive evaluation cadence. Cumulative
training/diagnostic/execution-wall costs restore checkpoint offsets and exclude
the intervening pause. The NoTorch worker waits for complete stage state before
reading finalized whole-budget peaks, archiving all histories, and separately
retaining both recovery reports. An early crossing never stops training.
