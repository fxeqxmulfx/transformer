# Early regression in the AdamW optimizer calibration

The first canonical joint-target failure after long confirmation is update
18,750: train/held-out accuracy falls to 81.1211%/79.7895%. The immediate
neighbors at 18,749 and 18,751 also fail, with held-out 78.8875%/82.8179%.
EOS remains 100%; numeric answers fail. At 18,500 and 18,501 both complete
RHS scores are still 100%. This regression is not confined to the canonical
48-example batch; the pre-failure neighbor follows an ordinary batch of 512.
Its trigger remains unresolved.

The [complete bracketing gradient trace](first-regression-gradients.jsonl)
contains all 252 consecutive records from 18,500 through 18,751. Its largest
norm is 47.2065 at 18,728, after a full 512-example batch; at the three
failed sampled observations norms are 4.4290/7.4554/7.2089. Dense tracing
therefore captures a larger gradient before the first failed evaluation.
Accuracy was not measured at 18,728, so this ordering does not establish whether
the gradient preceded the underlying regression or identify a causal trigger.

The [source samples](first-regression.json) retain all 76 canonical observations
through 18,750, five complete exhaustive observations and full tensor diagnostic
records, actual native AdamW moment norms, gradient-interval SHA256 and frozen
source/paper/configuration fingerprints. AdamW uses betas (0.9, 0.98), bias
correction and no maximum buffer, as specified prospectively. No recipe or
criterion was changed after this inspection.

This is an early regression, before the frozen final-tail window begins at
100,000. It prevents treating early 100% accuracy or the optimizer name as a
stability guarantee, but does not by itself decide the final-tail criterion.
Complete both 150,000-update budgets and preserve all later observations before
selecting a further intervention. The task remains the explicit GPTMini
adaptation of *Convexifying Transformers*, Section 4; observed associations are
not a proof of a cause or an internal learning algorithm.
