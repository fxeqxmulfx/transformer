# Conditional amplitude gradients after branch normalization

Task158 continues the requested research loop after the RMS backward repair.
The question is whether normalization still obstructs learning an input-dependent
branch strength even when its local derivative is numerically correct.

For a sign z in {-1,1}, supply features f(z)=(1,z) and learn a linear gate:

```text
(1) g(z) = 2 sigmoid(w0 + w1*z), initially w0=w1=0.
(2) before: c(z) = a*g(z)/sqrt(a*a*g(z)^2 + eps)
    after:  c(z) = g(z)*a/sqrt(a*a + eps).
```

Both variants initially have exactly the same forward values and two parameters
per branch. Both use the corrected task155 RMS backward. Independent unit-RMS
sign vectors carry each branch; their coefficients depend on the first sign of
the identity input. No label or evaluation result enters the feature generation.
Negative/positive signs select the cyclic teacher rows (1/4,1/2,3/4) and
(3/4,1/4,1/2), respectively.

For either desired coefficient c, the finite optimal gate is

```text
(3) before: g* = sqrt(eps)*c / (a*sqrt(1-c*c))
    after:  g* = c*sqrt(a*a+eps)/a.
```

All declared cases have 0<g*<2. Set l_z=log(g*/(2-g*)), then
w0=(l_1+l_-1)/2 and w1=(l_1-l_-1)/2. Thus both parameterizations can represent
both conditional targets exactly. The fixed-input chain rule and
Transformer.Residual.gain_derivative_ratio explain the radial conditioning;
this is not a theorem that a nonlinear transformer will generalize faster.

| Axis | Values |
| --- | --- |
| Placement | Before / after output RMS |
| Scale | .001 / 1 / 1000 |
| Branch count | 1 / 8 |
| Data and batch seed | 11 / 17 |

Freeze all 24 cases before scored learning. Use Schedule-Free AdamW with lr .01,
betas(.9,.98), epsilon1e-8, zero decay, batch64, 10,000 updates, evaluation and
checkpoint every250, CPU fullgraph max-autotune. Test opens after all cases end.
Recovery requires test relative MSE<=1e-4 and worst conditional coefficient
error<=.01. Report both conditional test scores and every failed case.

The unscored 5,000-step pilot found a scale-1 before-RMS case fitting one
condition while the other stalled (held-out MSE .0336914); the after-RMS case
reached 5.16e-13. At scale1000 the before-RMS MSE was .292969 and the after-RMS
case again recovered. These pilot observations motivated the frozen cohort;
they are not substituted for the complete results.

A full-model candidate would apply 2*sigmoid(W_gate*RMS(branch_input)) after the
existing output RMS and retain the fixed 1/sqrt(2L) residual coefficient. Zero
gate weights and no bias preserve initial forward values and common gradients.
Only the two-feature synthetic witness is studied here; the transformer uses
its full normalized input. A synthetic improvement alone does not establish
faster grokking. Any production integration is unconditional, without a flag,
and requires the separate matched transformer comparison to first99% validation.

Run ./run.py context-gains freeze, run, and report. The protocol pins source,
framework and recipes, and results retain all histories and state hashes.

All24 scored cases completed10,000 updates. After-RMS recovered12/12; before-RMS
recovered4/12, all at the smallest scale. Both depths and seeds are retained.

| Branch scale | Before RMS recovered | After RMS recovered | Before test relative MSE | After test relative MSE |
| --- | ---: | ---: | ---: | ---: |
| .001 | 4/4 | 4/4 | 1.27e-13–1.66e-12 | 9.62e-14–4.07e-12 |
| 1 | 0/4 | 4/4 | .0216–.1775 | 8.05e-14–6.40e-12 |
| 1000 | 0/4 | 4/4 | .2549–.7672 | 8.05e-14–6.40e-12 |

The largest after-RMS conditional coefficient error was3.52e-6. At scale1,
before-RMS retained errors up to.5; at scale1000, up to.75. Its gradients were
not identically zero, so this is failure within10,000 updates, not mathematical
impossibility. Multi-branch upstream FP32 reductions can still incur cancellation
after the corrected local RMS backward. Some successful one-branch after-RMS
runs reached zero training gradient; zero alone is not evidence of failure.

Summed case wall time was146.44s, including compilation/preparation. The general
regression suite shared the CPU during this cohort, so these times are not an
isolated throughput comparison. `results.json` retains full curves, per-condition
test scores, coefficient errors, gradient summaries and checkpoint hashes.
The four-case transformer comparison remains a separate test of generalization.

Replay this scored cohort at commit `63c2d22a`: later lab additions deliberately
invalidate its all-source fingerprint. The retained JSON is readable at any revision.
