# Future forcing from checkpoint state

Task #205 addresses the user's remaining discovery/future-gradient gap.
Freeze this protocol before new measurements. Do not train, alter the copied
optimizer, use Python, fit a bound to future updates, or claim discovery from
an optimizer magnitude bound.

## Mathematics

For the actual exact-real .9/.95 moment recurrences, derive the one-step
ratio bound `r_next <= (81/95)*r + 1/5`, where `m^2<=r*v`, `v>=0`, `r>=0`.
Prove the warm-start cone
`K_h = 19/14 + (r-19/14)*(81/95)^h` for arbitrary future gradients, including
zero variance. Connect warm starts to the existing moment recurrence. Retain
epsilon and both absolute-time bias corrections in the direction bound.
Prove the sharp one-step remainder and an attaining gradient when defined;
this sharpness concerns real gradients, not a full clipped CE trajectory.
Show sign symmetry of the actual moment recurrence to explain why magnitude
control alone does not create task alignment. Do not refute typical CE
discovery using an arbitrary-gradient counterexample.

## Fixed native study

Use all three existing main-model cohorts and the eight fixed seeds
`[7,11,1,3,5,13,17,19]`. Use exactly six anchors per trajectory:
`[0,1000,4000,6000,8000,9500]`. Each window contains every subsequent update
through `min(anchor+4000,10000)`: 144 windows and 444000 future prefixes.
The initial and early anchors remain in the results even if they certify no
useful geometry. No new endpoint query predictions are needed.

The constructor accepts only one anchor's parameters/moments, its absolute
time, the declared schedule, and the class partition from training labels.
It cannot access future weights, gradients, updates, losses or query outcomes.
Freeze all bounds to disk before the audit opens any embedding/update trace.
Other anchors are separate inputs to separate constructor calls, never inputs
to an earlier anchor's bound. Hash every checkpoint and input report.

Compare two tracks with identical initial geometry and numerical guards:
the new per-coordinate warm-state cone and AD2's uniform cone `19/14`.
Sum coordinate cones to bound the whole readout direction norm and its
active-row block norm. Orthogonal projection cannot increase these norms.
Accumulate damped forcing radii with the known future WSD learning rates.
Use the active-mode radius for the lower active norm and the whole-readout
radius for the upper competing (unused plus between) norm. Keep the common
decay exactly. Report active-energy-share lower bounds and the longest
consecutive certified prefix above 0%, 50%, 60%, and 2/3 for every case.
Also report the first violation of each threshold; do not skip an early dip.

The exact-real theorem is separate from FP32 execution. The numerical track
uses the existing fixed relative direction allowance `1e-6` and forecasts
the existing per-coordinate rounding allowance as
`8*EPS32*(1+M_h+eta*Bmax_h*(1+1e-6))`, where M is an a priori coordinate
parameter bound propagated with the same direction and rounding bounds.
No measured future rounding value enters construction. Audit these guards
against every recorded update, and report their assumptions explicitly;
this is not a formal IEEE/CUDA rounding proof. Keep scaled `1e-8` enclosure
tolerance. Never enlarge a tolerance after observing outcomes.

The separate auditor recomputes class projections from the preserved full
embedding traces, checks each recorded forcing against the frozen direction
budget, checks future rounding and coordinate magnitudes against their
predicted caps, checks tube distances and energy-share lower bounds at all
444000 prefixes, and verifies hashes/cadence/rates. All failed and
inconclusive cases are retained. Numerical validity is not certificate
coverage. Existing traces are a replay check, not fresh-seed confirmation.

## Budget and preservation

Construction plus audit has an 1800-second cap, excluding compilation,
unit checks and Lean development. Preserve all declared slots if it expires.
Use native Rust, jq and Lean. Run the full Lean build, theorem/global axiom
checks, native INDEX and forbidden scan; run meaningful Rust tests, clippy
and fmt. Keep generated evidence ignored at the root, and commit source,
protocol, commands, authored findings and generated INDEX for source changes.

The study may prove useful finite future control in exact arithmetic. It
does not assume or prove that the CE gradients create useful geometry from
random initialization. Report that distinction even if late checkpoints
have long certified horizons.
