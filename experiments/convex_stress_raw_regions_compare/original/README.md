# Original observed-region comparison

The reviewed CM48 component is fixed at `0fee1a9d`. Task305 compares it across
all 35 original seed1 tasks, both 105-choice policies and all three modes. The
complete task296 source, task301 guarded CE result, nested task298 tradeoff and
every earlier baseline/failure remain retained.

The training-only component inspection covers 3,878 unseen inputs in the prior
and 3,932 in the supplement, across the full finite input domain. The respective
93 and 94 ambiguous source blocks have disjoint observed class hulls. This does
not establish unseen-label correctness or the convex-target assumption. Fixed
confidence is one; no evaluation label or metric changed the component.

No original held-out score has been measured for this adapter. Its source,
protocol, binary and request must be frozen before the complete run. Every
original case and independent reconstruction replay will be preserved, including
refusals and regressions. Authored findings will replace this pending status
only after the complete result exists.
