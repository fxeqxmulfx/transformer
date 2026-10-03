# First AdamW failure inside the frozen final window

Update 104,000 is the first canonical joint-target failure in the final
100,000–150,000 window of `adamw-short-lr001`. Exhaustive train/held-out
accuracy is 90.0129%/87.9940%, below the frozen 99% target. The preceding
16 canonical observations in that window pass. One failed observation
already rules out the prospective requirement that every final-window
observation passes, irrespective of any later recovery. This remains a
partial run; the full 150,000-update budget must still finish and be archived.

The [portable source bundle](first-tail-failure/summary.json) retains the
unchanged frozen plan, every canonical observation from zero through 104,000
(417 total), five complete exhaustive observations and full tensor diagnostics,
and every gradient norm from 103,750 through 104,001 (252 total). The
[offline verifier](verify_first_tail_failure.py) checks artifact hashes,
criterion, exact coverage, exhaustive scoring, batch exposure, native AdamW
moment norms, dense/sampled agreement and the derived partial assessment.
It needs neither PyTorch nor the original run. Source-byte equivalence and
negative checks are recorded in the [validation](first-tail-failure-validation.json).

At 103,750 and 103,751 both scores are 100%. At 103,999 they are
89.5619%/87.6289%; at 104,001 they are 90.8290%/89.0679%. Both failing
neighbors follow ordinary 512-example batches, whereas canonical 104,000
follows the 48-example epoch tail. EOS remains 100% at all five samples;
the failures concern numeric answers. This observation is not confined to
the canonical short batch, and does not identify its trigger.

The largest gradient norm in the preserved interval is 68.1002 at 103,977,
following a 512-example batch. Accuracy was not measured at that update,
so the peak's ordering before the first failed canonical evaluation does
not establish an ordering before the underlying accuracy regression or
a causal explanation. Native second-moment norm changes from about
2.9e-9 at 103,751 to 7.0244 at 103,999; these are sampled stored buffer
norms, not effective bias-corrected denominators.

The earlier low-held-out memorization phase is also absent. Thus this recipe
cannot pass either the frozen phase criterion or final persistence, and cannot
open independent confirmation. Preserve both original optimizer budgets and
all subsequent failures, then execute the already frozen mod-193 task
adaptation requested by the user. This is a GPTMini/optimizer adaptation of
*Convexifying Transformers*, Section 4, not evidence that AdamW is universally
unstable or that a larger modulus will fix it. No configuration, criterion,
or source used by the live or queued scientific runs was changed here.
