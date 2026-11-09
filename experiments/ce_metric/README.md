# Feature-weighted CE momentum control

Feature-dependent scaling can be absorbed in the partial-CE momentum proof,
but this extension does not produce useful full-model future control in the
fixed comparison. Twenty-one new Lean theorems prove the weighted result.
All 144 metric constructions and saved-update audits pass. Full raw potential
increases change from 31 with the isotropic metric to 30 with feature weights;
the conservative condition that avoids observing future probabilities fails
on all 120 moving cases. Reliable discovery and full-model control remain open.

The [protocol](PROTOCOL.md) was frozen at `82f979da`, the native implementation
at `cdb5e66d`, and the proof at `bc90d03f`. No training, optimizer, prediction
or coverage computation changed. Different metrics measure different
potentials, so the changed increase count is not an optimizer improvement.

## Formal extension and its limits

Let r_k>0 be fixed feature weights shared across all unused classes.
For the actual partial CE gradient, after common clipping a>=0,

```
sum_(u,k) r_k*g_U(u,k)^2 <= L_r*<E_U,g_U>,
L_r = a*q*mean_i sum_k r_k*T(i,k)^2.
```

The unweighted radial product is nonnegative. Class-shared feature weights
commute with the computed unused-class mean. For arbitrary saved momentum,
`m_R=.9*m+.1*g`, `w_R=w-s*r*m_R` and

```
V_(r,s)(w,m)=sum ((w-9*s*r*m)^2/r+90*s²*r*m²),
delta V=-2*s*<w,g>+19*s²*sum r*g²-(171/10)*s²*sum r*(m-g)².
```

Thus s>=0 and `19*s*L_r<=2` give nonincrease. The fixed-r, fixed-s theorem
allows changing features and minibatches, with the coefficient condition
explicit at each step. A universal-probability corollary uses q=1, a<=1,
an ordinary mean feature-energy cap M and r_k<=R; its condition is
`19*s*R*M<=2`. It does not infer a future feature cap from one measurement.

Conversion to ordinary unused-class energy costs the largest weight:
`||w||² <= 1.9*R*V_(r,s)(w,m)`. Omitting R would overstate the certificate.
The exact full-step formula retains decay, differences between class scales,
changes since the checkpoint, tied encoder input and execution residuals.
The signed linear credits add; the quadratic term uses their total and
preserves cross terms. These identities do not bound unknown remainders.

See the [FM1–FM4 derivation](../../review/flat_spectrum.md#checkpoint-fixed-feature-metrics)
and [`FeatureMetricControl.lean`](../../src/Transformer/Grokking/GradientEvidence/FeatureMetricControl.lean).
This extends the dynamical partial reference, without assuming or proving
that training discovers flat spectra or useful feature alignment.

## Frozen comparison

Use all three original cohorts and every #206 case: eight seeds
`[7,11,1,3,5,13,17,19]`, anchors `[0,1000,4000,6000,8000,9500]`, 144 slots.
The standalone constructor opens only original reports, training-label
support and old checkpoint arrays. It freezes every metric and its hash
before a separate verifier opens any #206 next-update evidence. The verifier
reconstructs those metrics first, then checks the original next readout
bitwise against its preserved trace and recomputes partial CE independently.
These are reused observations, not fresh training or independent trials.

| Cohort | Original schedule | Original source |
| --- | --- | --- |
| lr001 | Main WSD, peak .001 | `class_geometry_wsd/runs/audit-02` |
| lr01 | Main WSD, peak .01 | `class_geometry_lr/runs/lr01-01` |
| settle | .01 to .0001 over 1000–4000; hold through 8000; cooldown | `geometry_gap/runs/settle-01` |

From old variance and its absolute-time bias correction, form D_old.
Take alpha=min D_old over the full readout and
`r_k=min_(unused c) D_old(c,k)/alpha`. The isotropic control uses r=1;
both use checkpoint scalar `s=eta(t)*alpha/(1-.9^(t+1))`. Unlike #206's raw
reference, this scalar also uses the old second moment. The comparisons
within this study hold that choice fixed. At initialization, D_old=1e8,
r=1 and eta=0; all 24 such cases have zero potential change.

Each metric has a raw reference and a separate stable diagnostic reference
that caps s at `1/(19*L_r)` using current q. This cap changes neither the
executed update nor its saved arrays. Current-q successes do not establish
the universal-probability condition or full-model stability.

## Findings

Across the 120 moving cases, all four partial reference tracks decrease in
the recorded arithmetic. The sufficient raw condition passes 58 isotropic
cases and 44 feature-metric cases; all 120 pass for each stable reference.
The feature raw coefficient is larger because r_k>=1: absorbing more of the
full diagonal also makes this sufficient condition harder to satisfy.
Full potential increases remain in 31/30 raw and 36/34 stable cases
(isotropic/feature). Of the raw increases, 13/11 also meet their respective
reference conditions. No failing condition or increasing update is omitted.

Every row contains eight seeds. Entries are **isotropic / feature** counts.
Both stable references satisfy their current-q condition in all rows.

| Cohort | Anchor | Raw condition passes | Raw full increases | Stable full increases |
| --- | ---: | ---: | ---: | ---: |
| lr001 | 0 | 8 / 8 | 0 / 0 | 0 / 0 |
| lr001 | 1000 | 7 / 5 | 6 / 6 | 6 / 6 |
| lr001 | 4000 | 8 / 8 | 4 / 4 | 4 / 4 |
| lr001 | 6000 | 7 / 7 | 1 / 1 | 1 / 1 |
| lr001 | 8000 | 7 / 7 | 3 / 3 | 3 / 3 |
| lr001 | 9500 | 6 / 4 | 0 / 0 | 0 / 0 |
| lr01 | 0 | 8 / 8 | 0 / 0 | 0 / 0 |
| lr01 | 1000 | 0 / 0 | 3 / 2 | 5 / 4 |
| lr01 | 4000 | 0 / 0 | 4 / 4 | 5 / 5 |
| lr01 | 6000 | 0 / 0 | 3 / 3 | 3 / 3 |
| lr01 | 8000 | 0 / 0 | 4 / 5 | 4 / 4 |
| lr01 | 9500 | 2 / 1 | 0 / 0 | 0 / 0 |
| settle | 0 | 8 / 8 | 0 / 0 | 0 / 0 |
| settle | 1000 | 0 / 0 | 3 / 2 | 5 / 4 |
| settle | 4000 | 4 / 2 | 0 / 0 | 0 / 0 |
| settle | 6000 | 4 / 2 | 0 / 0 | 0 / 0 |
| settle | 8000 | 5 / 0 | 0 / 0 | 0 / 0 |
| settle | 9500 | 8 / 8 | 0 / 0 | 0 / 0 |

All 32 settling cases from anchor 4000 onward decrease in all four
potentials. Raw current-q stability passes 21 isotropic and 12 feature
cases there. This remains a set of isolated next-step diagnoses with
different metrics and steps, not a fixed-metric certificate for the hold.
The duplicated lr01/settle anchor-1000 changes are the same underlying
updates and must not be counted as independent confirmation.

Mean adaptive linear credit divided by the old potential falls from
7.1898e-4 to 6.4629e-4 on the moving raw tracks, and from 7.3611e-4 to
7.2576e-4 on the stable tracks. It remains large and positive on the long
plateaus. After settling it is favorable on average in both metrics; there
the feature metric also removes some favorable credit. The average raw
encoder credit rises from 1.5773e-5 to 5.0531e-5, and total quadratic credit
from 2.4905e-5 to 4.8308e-5. These are different potential accountings of
unchanged updates, not causal effects of altering adaptive scaling.

The largest feature weight ranges from 1.228 to 22.546 over moving cases.
The ordinary-energy bound divided by actual old unused energy is
2.053–13.412 for the feature raw metric, versus 1.899–1.906 for the isotropic
control. In later settling it is 2.053–2.502 versus approximately 1.900.
The weighted extension therefore makes this ordinary-energy bound looser
in every moving case; a smaller weighted potential is not a tighter bridge
to the original geometry.

With q=1, clipping<=1 and the declared normalization cap, **neither raw
metric passes the prospective reference condition on any moving case**.
The smallest stability numbers are 121.12 isotropic and 991.00 feature,
against a required maximum of 2. In later settling the feature number is
2.71e6–2.05e7. Only the 24 zero-LR initialization slots pass. Even passing
this condition would control only the reference; the full perturbations
would still need bounds. The sampled small probabilities cannot simply be
assumed for future updates.

The next useful mathematical target is to propagate a small unused-class
probability bound through actual parameter and feature motion, alongside
control of the class-dependent diagonal and tied encoder terms. Changing
the feature metric alone has not supplied a useful future certificate.
Discovery, the full 4000-step hold and prediction-certificate coverage are
unchanged by this study.

## Checks and evidence

All 144 slots and 576 metric/step tracks pass the declared audit; no cap,
missing slot or failed numerical check occurred. All original source and
case hashes are checked. Maximum partial-gradient scaled error is 3.480e-8
(allowance 1e-5), maximum feature energy 128.000032
(cap `128*(1+5e-6)`), and maximum full-identity scaled error 1.135e-14
(allowance 1e-8). No allowance changed. Construction takes 1.40 seconds,
verification 27.70 seconds, combined 29.10 against the fixed 600-second cap.

Three native tests (including independent finite differences and a nonzero
weighted CE/memory example), clippy with warnings denied, fmt and release
build pass. The full Lean build passes 13348 jobs. All 21 new theorems use
only standard axioms; the global audit remains 36 `sorry`, zero rests and
zero extra axioms. Native INDEX reports 4251 modules, 21878 theorems and zero
vacuous/placeholder declarations; the forbidden scan passes. No Python,
subagents, new GPU execution or modifications to #206 sources were used.

Ignored evidence is preserved at the root in `runs/metric-01/`, containing
all frozen metrics, `construction.json`, `verification.json` and rendered
`summary.json`. Logs and the measured executable are in
`.kanban/artifacts/207/`. Native source fingerprint:
`848027911e51dda8604a6a5eee5794c76de2a6d84b649677721ce242ad139fe9`.
Executable SHA-256:
`1b2212051bbe3ca9aa2859ba86fba6d8bf0bf6586f9b7fe0bf7223af9310e4d2`.
The committed generator validates the exact inventory and retains all
condition failures and actual increases.

## Reproduction

Run from the repository root with the original cohorts and the complete
#206 evidence available. Their READMEs document regeneration. This audit
requires no GPU access. Use a new output path; construction refuses overwrite.

```sh
export CARGO_TARGET_DIR="$PWD/.lake/ce-metric-target"
cargo build --offline --locked --release --manifest-path experiments/ce_metric/Cargo.toml
cm_run="$PWD/experiments/ce_metric/runs/metric-reproduction"
"$CARGO_TARGET_DIR/release/ce-metric" "$cm_run" experiments/class_geometry_wsd/runs/audit-02 experiments/class_geometry_lr/runs/lr01-01 experiments/geometry_gap/runs/settle-01
"$CARGO_TARGET_DIR/release/ce-metric" verify "$cm_run" experiments/ce_lyapunov/runs/lr001-01 experiments/ce_lyapunov/runs/lr01-01 experiments/ce_lyapunov/runs/settle-01 > "$cm_run/verification.json"
jq -f experiments/ce_metric/summary.jq "$cm_run/verification.json" > "$cm_run/summary.json"

cargo test --offline --locked --manifest-path experiments/ce_metric/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/ce_metric/Cargo.toml -- -D warnings
cargo fmt --manifest-path experiments/ce_metric/Cargo.toml --check
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/FeatureMetricAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml -- --check
rg -n 'native_decide|^axiom |linter\..* false|autoImplicit' src
```

The final scan should find nothing (rg exit 1). To regenerate the measured
summary only, use `runs/metric-01/verification.json` as the jq input.
