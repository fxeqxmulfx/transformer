# Finite-epsilon RMS backward and recovery

Task155 asks whether cancellation in the normalization backward removes a
mathematically nonzero gradient and interferes with learning. This extends the
task151 gate-placement witnesses, with unchanged forward functions and optima.
All comparisons remain in the laboratory; the production GPT will have one
normalization implementation, without a new architecture flag.

For x with mean-square energy m and positive epsilon, the usual RMS VJP is

```text
(1) inv * g - inv^3 * x * mean(x*g),   inv = 1/sqrt(m + eps).
```

For m>0 write p=x*mean(x*g)/m. The same exact derivative is

```text
(2) inv * (g-p) + eps * inv^3 * p.
```

At x=0 the derivative is g/sqrt(eps). The candidate computes the projection
and this decomposition in FP64, then returns the gradient in the input dtype.
Its forward is the native RMSNorm. This does not remove subsequent upstream
rounding or turn a tiny mathematical gradient into a well-conditioned update.

On a unit-RMS direction with amplitude a, the local derivative is

```text
(3) eps / (a*a + eps)^(3/2).
```

`Transformer.Residual.radialRMS_hasDerivAt` proves Eq. (3). This is a finite-
epsilon derivative, not ideal zero-epsilon homogeneity. The previously proved
gain-placement ratio explains why a radial gradient can be very small.

| Axis | Values | Purpose |
| --- | --- | --- |
| RMS backward | Native / separated radial-tangent, FP64 arithmetic | Same forward, compare numerical gradients |
| Gate placement | Before / after RMS | Affected radial path / unaffected control |
| Branch scale | .001 / 1 / 1000 | Visible radial derivative / cancellation / tiny true derivative |
| Branch count | 1 / 8 | Isolated radial path / other branch contributions |
| Data and batch seed | 11 / 17 | Paired disjoint sign vectors and order |

All 48 cases use Schedule-Free AdamW, lr .01, betas(.9,.98), epsilon1e-8,
zero decay, batch64 and 10,000 updates under CPU fullgraph max-autotune.
Evaluate every250 updates and save full state. Final test opens only after
all cases have ended. Recovery requires test relative MSE<=1e-4 and maximum
teacher-coefficient error<=.01, matching task151. Preserve every failure.

Independent checks compare forward values, scalar real derivatives, FP64
autograd VJPs, zero inputs, higher derivatives and compiled continuation.
Runtime comparisons report compilation separately from warmed updates.
Synthetic amplitude recovery does not establish faster transformer grokking.

The unscored pilot found native zero/wrong-sign gradients at scales1/1000.
The candidate matched the scalar derivatives; at scale1 it recovered both
one/eight-branch teachers in5,000 updates while native gradients stayed zero.
Scale1000 remained poorly conditioned even with a corrected gradient. These
pilot cases motivated the frozen cohort and are not substituted for its results.
A separate rejected hypothesis was dtype-dependent half-precision epsilon:
local Torch2.15 already used FP32 epsilon for FP16/BF16 RMSNorm in the probe.

Run `./run.py rms-backward freeze`, `./run.py rms-backward run`, and
`./run.py rms-backward report`. `protocol.json` freezes source/recipe hashes.

## Frozen results

All 48 runs completed at 10,000 updates; total recorded run wall time was
261.43 seconds. Every run used the frozen Python 3.15 / Torch 2.15 CPU
max-autotune recipe. Final test was opened only after the cohort closed.

| Placement | Input scale | Native recovered | Accurate recovered |
| --- | ---: | ---: | ---: |
| Before RMS | .001 | 4/4 | 4/4 |
| Before RMS | 1 | 0/4 | 4/4 |
| Before RMS | 1000 | 0/4 | 0/4 |
| After RMS | each of .001, 1, 1000 | 12/12 | 12/12 |

At scale 1, all native runs had exactly zero gradient at all 10,000 updates
and maximum coefficient error .75. Corrected runs had no zero-gradient update
and final coefficient errors 0.0000033–0.0000054. Native and accurate after-RMS
runs have identical recovery curves and final metrics. The improvement is
numerical derivative accuracy, not a changed target or forward function.

The scale-1000 failure is retained. Eight-branch corrected runs still have
some zero upstream gate gradients: the local accurate VJP is returned to
FP32, and later sums can lose its tiny radial component. Even the one-branch
case with a positive gradient throughout does not recover within this budget.
The optimizer's epsilon and tiny true derivative remain limiting factors.

`results.json` retains all histories, full train/validation/test metrics,
coefficient errors, gradient ranges/zero counts and checkpoint/split hashes.
The regression tests also check compiled full-state continuation and independent
FP64 derivatives. Lean proves the exact decomposition, coefficient ratio,
zero-input derivative and scalar squared-error derivative; it makes no claim
about floating-point roundoff or universal optimizer convergence.

An exploratory RTX 3050 profile of the actual GPT (FP32, width 64, four
layers/heads, batch 256, context 2, compiled forward/backward without optimizer
updates) measured median warmed passes of 5.828 ms native and 6.334 ms accurate,
about 8.7% overhead. Peak allocated memory was 1,832,448 / 1,901,568 bytes.
Preparation including compilation was 49.57 / 41.28 seconds. The lab suite was
running concurrently, so this is an indicative cost, not an isolated throughput
claim. `profile/exploratory.json` records every sample and source hashes;
`profile/gpt.py` records the measurement. With the GPT source on PYTHONPATH,
run it through `uv run --locked python ../experiments/rms_backward/profile/gpt.py`
from `python/`. The script compares normalization implementations in disposable
model copies and does not expose a production architecture flag.

Validation: the complete lab suite passed 475 tests (three skips), and the
mandatory Lean build, audit, generated index and forbidden checks passed.
The audit reports 36 existing unfinished statements, zero results resting on
one and zero extra axioms. All six new statements are proved. The companion
GPT repair is `7314dec`: 194 model/provenance tests passed before landing.

The full `./run.py test` command was **not entirely green**: two existing
ScheduleFreeMuon CUDA allocation subcases failed (FP32/FP64; repeated
8,519,680-byte allocations). They reproduce using unchanged root optimizer
sources at `3978d528`; no optimizer source is changed here. Task157 tracks them.
The paired learning experiments and the model recovery regressions use
Schedule-Free AdamW, whose tested paths passed.
