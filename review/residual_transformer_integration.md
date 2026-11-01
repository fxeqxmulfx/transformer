# Full-transformer residual integration (task145)

The GPT implementation is committed at `ad76812`, based on `0fab228`.
The lab port matches the source's initial parameters, forward values and every
parameter gradient for all five residual modes, both normal and clipped XSA
inputs. The source fixture is generated only by the explicit pinned-GPT checker;
tests do not regenerate expected values from the lab implementation.

Attention Residuals, arXiv:2603.15031v1 §3.1 Eqs2–4 and §5, supplies the full depth
recurrence, key-only scoring, learned key gain and zero-query initialization.
The host retains its Peri output norms; this is an explicit paper deviation.
Q/K variants instead use parameter-free RMS on both operands, unit-RMS initial
queries and a sqrt(width) score divisor. The corrected variant transforms all
query/base/moment/coordinate-scale state before training forwards. Existing
Lean `QueryCoordinates` and `QueryConditioning` prove the current-state
coordinate identities; they do not prove subsequent optimizer equivalence,
full-transformer convergence or held-out generalization.

The first depth input is the embedding. Each completed attention or FFN branch
is appended once, and the final depth mixture is read by the original final
norm and tied head. There is no redundant singleton query. Common source
matrices are initialized before any additional query draw, preserving paired
initialization. The repeated-prefix guard remains valid only while all earlier
attention heads are above XSA clipping; its mask cannot reactivate.

Validation so far: 170 affected GPT CPU tests pass, 15 CUDA cases skip; the
17 newly added tests are included. All10 new lab/protocol tests pass, including
independent optimizer equations, full-state resume, stable validation stopping,
reserved-test isolation and a durable wall-time ceiling. The full CPU lab suite passed459 tests (13 skipped) in850.12s and
the optimizer suite passed104 tests (14 skipped) in12.14s. The final
10 targeted residual tests also passed after the compilation-state guard
was added. CUDA compilation and gradient fidelity will be checked
in the separately budgeted profile before the learning cohort is frozen.

Two initial test failures were retained in local task145 evidence: cache output
tolerance incorrectly assumed all-FP64 source arithmetic (the source explicitly
uses FP32 single-query softmax); and a stopping test assumed a random model would
already classify at least one validation example correctly. The cache tolerance
now reflects source arithmetic. The stopping test uses declared fixed metrics
so it tests control flow independently of random learning success. No learned
cohort has been retuned or discarded.
