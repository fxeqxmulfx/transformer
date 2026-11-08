# Joint update control of difficult-query margins and loss

Frozen 2026-11-08 for task #187 before measuring these bounds. This extends
the exact-source single-update audit in PROTOCOL.md. Preserve its complete
24-case cohort, all input and PTX checks, replay checks, and geometry checks.
Never overwrite its report or original checkpoints. Step-10000 updates
remain post-budget probes. The original architecture ranking is unchanged.

## Conditions being tested

Use normalized ambient features, so arbitrary changes of fitted PCA bases
cannot themselves change a class score. For each competitor-minus-target
readout w, write the actual update as w'=w+p+rho. Here p is the ideal buffered
Adam increment with each class's own diagonal scale, and rho is the measured
FP32 rounding remainder already retained by task #186. Let dx=x'-x.

Keep the signed first-order contribution together:

```
L = <p,x> + <w,dx>
gap' - gap = L + <p,dx> + <rho,x'> + execution_change
R = ||p|| ||dx|| + ||rho|| ||x'|| + |execution_change|
gap + L - R <= gap' <= gap + L + R.
```

Compare the joint interval with the separated norm allowance
`S=||p||||x||+||w||||dx||+R`, yielding `[gap-S,gap+S]`. Neither interval
assumes an unchanged feature map or a common preconditioner across classes.
Record the omitted mixed term and actual rounding contribution; no observed
loss or new margin is substituted for R. The feature displacement and
execution remainder are measured after the update: this is conditional local
control, not yet a forecast that can be made before running the network.

Take the maximum over **all** wrong-class bounds to enclose each query's
worst-competitor margin. A positive lower margin certifies a strict correct
decision. A lower margin above the old actual margin certifies improvement;
an upper margin below it certifies regression. Retain inconclusive queries,
old errors, and changes of the worst competitor.

For CE, include the target's zero gap and apply log-sum-exp to the lower and
upper competitor gaps. Equivalently, relative CE change is the log of an
old-softmax-weighted exponential moment of the relative gap changes. This
keeps the signed contributions and emphasizes difficult competitor classes.
Certify loss decrease only when the upper new loss is below the old loss,
and loss increase only when the lower new loss is above it. Average the
independently valid ceilings over the declared query sets. No hypothesis
assumes the desired new accuracy or loss.

## Query sets and reporting

- Both complete splits (4704 train, 4705 validation) for all 24 probes.
- For each old checkpoint and split, freeze the lowest-margin 10% before
  evaluating the new state: ceil(n/10)=471 rows, ties by canonical pair index.
  Report that same set after the update; never reselect it using new margins.
- Report old-error rows separately, with explicit null means for empty sets.
- Record actual old/new CE and minimum/mean margins, lost/gained decisions,
  interval widths, joint/separate classification coverage, certified margin
  improvement/regression and certified loss increase/decrease, with all
  inconclusive cases retained. Compare new state hashes with task #186.
- Analyze all arms and both seeds. This is a retrospective diagnostic using
  the learned readout and observed feature motion, not a new statistical
  generalization guarantee or architecture selection.

## Verification and limits

Prove the finite-step identity, norm remainder, all-competitor margin
condition and actual-CE bound in Lean; connect the increment to the actual
buffered Adam update. Include a nonzero cancellation witness and an actual
two-query CE counterexample to inference from increasing mean margins.
No new sorry, extra axiom or theorem claiming optimizer discovery.

Retain task #186 tolerances. New FP64 identity/enclosure checks use
`1e-8*(1+largest relevant magnitude)`. Classification/improvement/descent
certificates require a positive numerical slack of that tolerance, and no
false positive is accepted. Check the FP64 CE of executed logits against the
source GPU metric (`2e-6*(1+loss)`). Audit all queries; errors fail the case.
Use native synthetic tests where parameter and feature effects cancel,
the mixed term is nonzero, the worst competitor switches, and mean margin
and loss move in opposite directions. Runtime cap remains 600 seconds,
excluding compilation. Reports and binaries stay ignored and local.
