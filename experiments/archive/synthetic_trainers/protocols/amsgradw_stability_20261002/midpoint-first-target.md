# Midpoint first target without the required memorization plateau

The raw AMSGradW / GPTMini midpoint calibration is an explicit adaptation of
*Convexifying Transformers*, Section 4. At canonical update 93,250 it first
reaches the frozen held-out 99% complete RHS target: train accuracy is 100%
and held-out accuracy is 99.0120% (46 incorrect equations out of 4,656).
Recorded costs at this single observation are 2,666.34 training / 2,806.54
wall seconds. These are first-crossing costs, not long-confirmation timing
or evidence of a speedup.

The [recorded prefix](midpoint-first-target-prefix.json) retains every one of
the 374 canonical observations from update zero through 93,250. The required
train≥99% / held-out≤10% memorization plateau is absent before this first
crossing. The unchanged criterion requires five consecutive observations
spanning at least 1,000 updates before first held-out 99%. Later observations
cannot add a plateau before an already recorded crossing. This recipe is
therefore ineligible for stable-grokking confirmation under the frozen rule.
This conclusion concerns the phase gate, not its still-unmeasured final
persistence. A single target crossing is distinct from the required
20-observation joint confirmation.

The prefix is incomplete; long-confirmation timing has support zero at this
point, and the final 50,000-update window has not begun. The driver continues
the identical 150,000-update budget, with all failures and later outcomes
retained. Full-budget archives, plots, and comparison remain required before
choosing another adaptation.

The prefix schedule, exhaustive split sizes, metric bounds, integer-count
accuracies, and component-loss identities were checked. The live and committed
plans match byte for byte; all 19 source and two local manuscript fingerprints
are unchanged. The JSON retains the manifest and raw-prefix SHA256 values,
config, corpus, unchanged criterion, and explicitly incomplete assessment.
System Python produced the assessment without importing PyTorch.

The [midpoint protocol](midpoint-protocol.md) remains unchanged. Independent
confirmation, architecture improvement, and complementary-mechanics studies
remain pending; no scientific independent confirmation has started.
