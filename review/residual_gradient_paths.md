# Task 151: same-assistant review

The user requested a new synthetic benchmark inspired by this Lean project,
with no delegated reviewer. This review follows task145's closed GPU comparison.
All new learning used CPU Python 3.15.2/Torch 2.15.0 and Schedule-Free AdamW only.

The new witness families have analytically known solutions in both compared
parameterizations and matched initial input/output functions. The sign-vector
features are supplied, so recovery does not stand in for discovering modular
arithmetic or attention routes. Pre-RMS gate optima are finite for all declared
scales; none of the failed cases is excluded by the bounded gate's range.
The transport families have different parameter distances and Jacobians despite
matching initial outputs. They do not prove any universal optimizer failure.

Read `Residual/GradientPaths.lean` beside benchmark README Eqs. 1–4. Nine theorems
connect the scalar calculation to the existing vector RMSNorm and residualRun.
Positive epsilon, nonzero branch amplitude in the ratio, unit-RMS direction and
positive depth are explicit; concrete examples satisfy all hypotheses. The
radial theorem differentiates gain, not sigmoid logit; both placements multiply
it by the same sigmoid derivative in the independent numerical test. The zero-
derivative example is a counterexample to an unconditional identity-skip claim,
not to the existing norm upper bounds. The depth-scaled lower bound applies only
to the displayed fixed linear branches. All nine `#print axioms` checks contain
only propext, Classical.choice and Quot.sound. Full build, audit, generated index
and forbidden checks pass; no new sorry, rests, vacuous or placeholder claim.

Six focused tests check Decimal derivatives, finite differences where resolved,
finite exact optima, literal closed-form transport values/derivatives, independently
computed labels, pairwise disjoint splits, full checkpoint/sampler equality,
eager/compiled max-autotune continuation and sealed final test. Self-review
strengthened resume from a scalar batch-insensitive example to eight branches,
compared every non-clock checkpoint field, and deliberately perturbed model and
optimizer during preparation to verify restoration. All six pass (7.68s).
The full CPU suite passed 468 tests/13 skips and 104 optimizer tests/14 skips;
the stronger sixth focused test was added and rerun afterward. No engine changes
followed the source freeze.

All 36 frozen cases reached 10,000 updates: 360,000 finite recorded gradient norms,
171.11 scored CPU wall seconds including compilation. Post-RMS gates recover
12/12, pre-RMS gates 4/12, depth-scaled transport 6/6, ordinary transport 2/6.
Every success crossed validation threshold by update 1000. Every failed case
remains in results.json. All final test evaluations occurred after cohort closure.
Replaying eager averages differs from compiled validation relative MSE by at most
1.69e-8; no architecture or checkpoint was selected using the final test.

At scale one, pre-RMS gradients are exactly zero throughout the four runs.
At scale 1000, nonzero FP32 gradients can be rounding artifacts; Decimal and
FP64 record the small positive real derivative independently. In depth-32
ordinary transport the derivative is nonzero but the first ideal parameter
step (0.00033746) is below the spacing toward zero (256) at w=2^32; the actual
update is unchanged. Depth-eight ordinary transport still moves. Successful
runs also reach zero-gradient numerical floors, so gradient size alone is not
used as the success criterion. The coefficient norm bound is a real-arithmetic
diagnostic from estimated coefficients, not a certified FP32 interval bound.

Profiling retained the interrupted literal depth recurrence's Inductor operation-
counting traceback. The algebraically equivalent per-layer factor (1+c)*h
avoids repeated traversal and compiles depth32 in about 2.3–3.0s. The independent
real-value/gradient oracle still checks every D; FP32 rounding order is disclosed.
Tachyon attachment was denied by ptrace policy; no OS policy was changed. This
was CPU preparation, not renewed GPU training. Throughput estimates were taken
alongside CPU verification and are not a controlled transformer performance claim.

GPT `379a9f2` adds optional post-RMS branch gates; no default selection changes.
Actual-model tests use independent dense equations, all parameter gradients,
unequal gate values, causal and cached inference, checkpointed gradients, initial
ordinary-model forward equality and FP32-tolerant common-gradient parity. The
new held-out amplitude regression learns with Schedule-Free AdamW and retains
an intentionally pre-RMS control. The same assistant also injected a real
misplaced-normalization mutation: the test rejects it. The landed affected GPT
suite passes 179 tests with 15 CUDA skips; focused review passes 26 tests.
An initial exact FP32 gradient assertion was corrected for a measured 3.73e-9
accumulation-order difference; forward values remain bit-identical at unit gates.

The benchmark confirms a gain-placement repair and a conditioned linear witness.
It does not show learned gates outperform the task145 ordinary/fixed-scale
transformer controls on grokking. Keep QK norms and source RoPE/XSA/Peri behavior.
A further long full-transformer comparison needs a separately authorized budget.
Raw evidence is in `.kanban/artifacts/151/`; committed curves, source hashes,
Decimal/FP32 oracles, profiles and every endpoint are beside the experiment.
