# CE damping with momentum

The actual partial CE gradient supplies enough damping to absorb arbitrary
saved momentum in a suitable potential. Twenty-four new Lean theorems prove
this under an explicit scalar-step condition. All 144 declared full-model
replays pass the numerical checks, but 31 actual updates increase the raw
potential, including 13 whose partial reference meets that condition.
Full AdamW therefore needs additional control of its signed perturbations.
Reliable discovery and useful control over the entire hold remain unproved.

The [protocol](PROTOCOL.md) was frozen at `1b0d6b31` before measurement;
the replay implementation is `686aadbe` and the proofs are `cb33b97a`.
This study changes no training recipe or prediction certificate.

## What the proof closes

For the computed within-unused class block, partial cross-entropy gives
`||G_U||² <= L <E_U,G_U>` with nonnegative radial product and
`L=q*mean_i ||T_i||²`. Here q bounds the unused probabilities in the batch.
Common clipping multiplies L by the same nonnegative scalar. This is derived
from the actual derivative, not supplied as a favorable-gradient assumption.

For the main beta1=.9, define `m_R=.9*m+.1*g`, `w_R=w-s*m_R` and
`V_s(w,m)=||w-9*s*m||²+90*s²||m||²`. The exact identity is

```
V_s(w_R,m_R)-V_s(w,m) =
  -2*s*<w,g>+19*s²*||g||²-(171/10)*s²*||m-g||².
```

Thus `s>=0` and `19*s*L<=2` imply nonincrease for every saved moment m.
Also `||w||² <= 1.9*V_s(w,m)`. Fixed-s partial-CE trajectories inherit the
bound even with changing features and minibatches, subject to the declared
per-step coefficient bound. The uniform q=1 is always available; a tighter
observed q is not a future guarantee. The theorem removes the memory
alignment premise for this reference dynamics. It does not prove growth of
active contrasts, full tied AdamW stability, or discovery from initialization.

The [source derivation](../../review/flat_spectrum.md#loss-specific-momentum-control)
gives CL1–CL4. The proof entry point is
[`CEMomentumControl.lean`](../../src/Transformer/Grokking/GradientEvidence/CEMomentumControl.lean).
Full-update identities retain decay, coordinate-dependent adaptive scaling,
the tied encoder, and execution residuals. Projection is applied after the
diagonal-times-moment product. Signed linear source credits add, while the
quadratic term is the energy of their total, including cross terms.

## Fixed replay and findings

Each case restores a saved state and executes exactly its original next
minibatch. There are eight seeds `[7,11,1,3,5,13,17,19]`, six anchors
`[0,1000,4000,6000,8000,9500]`, and three original complete cohorts:

| Cohort | Schedule | Preserved source |
| --- | --- | --- |
| lr001 | Main WSD, peak .001 | `class_geometry_wsd/runs/audit-02` |
| lr01 | Main WSD, peak .01 | `class_geometry_lr/runs/lr01-01` |
| settle | .01 to .0001 over 1000–4000; hold through 8000, then cool | `geometry_gap/runs/settle-01` |

All use the unchanged main-model .9/.95 AdamW kernels, epsilon 1e-8, decay
.1 and the original global clipping. The model solves `(a+b)^3 mod 97`;
all 33 possible targets occur in training, leaving 64 unused classes. The
same data, order, initialization and checkpoint hashes are checked. These
144 indexed cases are not independent trials: schedules share initial
states, and lr01 and settle have identical updates through anchor 1000.

The raw scalar reference is `s=eta*min(D)/(1-.9^(t+1))`, with D from the
actual updated second moment over the whole readout. A separately declared
stable reference uses `min(s,1/(19*L))` for L>0. Neither changes the actual
training step. Both the reference trajectory and the potential depend on s,
so different counts between references are not effects of an LR intervention.

All 144 cases pass replay, independent partial CE, damping and identity
checks. The 24 initialization cases have eta=0 and zero potential change.
Across the 120 moving cases, both references strictly decrease in the
recorded arithmetic. The raw sufficient condition passes only 58 of those
120 cases; its 62 failures remain inconclusive by that condition even though
the measured reference changes are negative. The stable reference passes
all 120. Actual potential increases occur in 31 raw and 36 stable cases;
13 of the raw increases also satisfy the raw sufficient condition.

Every row below contains all eight seeds. The stable reference passes its
condition in all rows. Increases use the frozen scaled 1e-8 allowance.

| Cohort | Anchor | Raw condition passes | Actual increases, raw | Actual increases, stable |
| --- | ---: | ---: | ---: | ---: |
| lr001 | 0 | 8 | 0 | 0 |
| lr001 | 1000 | 7 | 6 | 6 |
| lr001 | 4000 | 8 | 4 | 4 |
| lr001 | 6000 | 7 | 1 | 1 |
| lr001 | 8000 | 7 | 3 | 3 |
| lr001 | 9500 | 6 | 0 | 0 |
| lr01 | 0 | 8 | 0 | 0 |
| lr01 | 1000 | 0 | 3 | 5 |
| lr01 | 4000 | 0 | 4 | 5 |
| lr01 | 6000 | 0 | 3 | 3 |
| lr01 | 8000 | 0 | 4 | 4 |
| lr01 | 9500 | 2 | 0 | 0 |
| settle | 0 | 8 | 0 | 0 |
| settle | 1000 | 0 | 3 | 5 |
| settle | 4000 | 4 | 0 | 0 |
| settle | 6000 | 4 | 0 | 0 |
| settle | 8000 | 5 | 0 | 0 |
| settle | 9500 | 8 | 0 | 0 |

At anchor 4000 the following mean signed credits show why damping alone
does not decide the full step. Each row averages eight single-step raw
diagnoses in potential units; these are not cumulative trajectory credits.
The baseline plus four linear credits plus total quadratic equals the
actual change, up to the declared numerical tolerance.

| Cohort | Partial baseline | Decay | Adaptive | Encoder | Execution | Total quadratic | Actual change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lr001 | -2.23884e-5 | -1.88674e-2 | +3.27088e-2 | +1.23026e-3 | -3.13176e-6 | +1.47588e-4 | +1.51937e-2 |
| lr01 | -1.41936e-2 | -6.50714e-1 | +6.31030e-1 | +2.08064e-2 | +8.36074e-6 | +3.43714e-2 | +2.13081e-2 |
| settle | -6.54550e-4 | -2.61115e-3 | -2.53237e-4 | +9.77945e-5 | -3.59710e-6 | +6.91602e-7 | -3.42405e-3 |

Adaptive scaling supplies the largest positive mean linear term on the
two long-plateau cohorts at anchors 1000–8000. It is favorable on average
at every settling anchor from 4000 onward. This is a diagnostic attribution,
not a causal claim that removing adaptivity improves training.

All 32 settling cases at anchors 4000, 6000, 8000 and 9500 decrease the
actual potential under both references. The raw condition passes 21/32;
the stable condition passes 32/32. Their raw actual changes range from
-0.004349 to -0.0003724. This supports the measured settling behavior,
but these are only four next updates per seed, with different s at different
states. They cannot be chained into a fixed-potential or full-hold guarantee.

## Numerical and formal checks

All replayed readouts equal the original next embedding trace bit for bit.
The complete optimizer check has zero ulp discrepancy and maximum relative
norm discrepancy 7.075e-8. The independent partial CE derivative differs by
at most 3.480e-8 in the declared scaled norm (allowance 1e-5). Maximum feature
energy is 128.000032, below `128*(1+5e-6)`. Maximum scaled full potential
closure error is 1.059e-14 (allowance 1e-8). No tolerance was changed.
All evidence hashes and diagnoses pass independent saved-array verification.
Replay plus verification takes 27.52, 23.88 and 22.55 seconds by cohort,
each below the predeclared 600-second cap. There are no missing or failed slots.

Eleven native tests, clippy with warnings denied, fmt, all three release
builds, full Lean build and axiom audits pass. The 24 new theorems use only
the standard axioms. The global audit remains 36 existing `sorry`, zero
results resting on `sorry`, and zero extra axioms. Native INDEX reports
4248 modules and 21857 theorems, with zero vacuous or placeholder declarations;
the forbidden scan passes. No Python or subagents were used.

Measured source fingerprint:
`0c657c116402b8faf2d5b65ef0c940a5fc3c571a8bd0371e0d5b721fb8000ba8`.
The pinned trainer fingerprint is
`14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da`.
The native build verifies the pinned engine and Kernite revision. Reports
also preserve the generated optimizer PTX hash, batch rows, checkpoint and
trace hashes, executed features and gradients, and the next full state.

Generated evidence is ignored and kept at the repository root in
`experiments/ce_lyapunov/runs/{lr001-01,lr01-01,settle-01}/`, with the complete
rendered `runs/summary.json`. Logs, measured executables and their SHA-256
ledger are under `.kanban/artifacts/206/`. The committed `summary.jq` checks
the exact cohort/seed/anchor inventory and retains every condition failure
and actual increase; no failed slot is removed from the denominator.

## Reproduction

Run from the repository root with the pinned native engine/Kernite checkouts
and the three preserved source cohorts available. Their respective READMEs
document regeneration. GPU replay needs access to the NVIDIA driver.
Use a fresh output directory; the harness refuses to overwrite a replay.

```sh
export CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target"
ce_run="$PWD/experiments/ce_lyapunov/runs/reproduce-01"
mkdir "$ce_run"
mkdir "$ce_run/bin"

cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin ce-lyapunov
cp "$CARGO_TARGET_DIR/release/ce-lyapunov" "$ce_run/bin/lr001"
"$ce_run/bin/lr001" "$ce_run/lr001" "$PWD/experiments/class_geometry_wsd/runs/audit-02"
"$ce_run/bin/lr001" verify "$ce_run/lr001" > "$ce_run/lr001/verification.json"

cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin ce-lyapunov --features lr01
cp "$CARGO_TARGET_DIR/release/ce-lyapunov" "$ce_run/bin/lr01"
"$ce_run/bin/lr01" "$ce_run/lr01" "$PWD/experiments/class_geometry_lr/runs/lr01-01"
"$ce_run/bin/lr01" verify "$ce_run/lr01" > "$ce_run/lr01/verification.json"

cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin ce-lyapunov --features settle
cp "$CARGO_TARGET_DIR/release/ce-lyapunov" "$ce_run/bin/settle"
"$ce_run/bin/settle" "$ce_run/settle" "$PWD/experiments/geometry_gap/runs/settle-01"
"$ce_run/bin/settle" verify "$ce_run/settle" > "$ce_run/settle/verification.json"

jq -s -f experiments/ce_lyapunov/summary.jq "$ce_run/lr001/report.json" "$ce_run/lr01/report.json" "$ce_run/settle/report.json" > "$ce_run/summary.json"
```

The verifier rehashes source and evidence files and reconstructs every
diagnosis from saved arrays without further GPU updates. For the measured
summary, pass `runs/{lr001-01,lr01-01,settle-01}/report.json` instead.
Mandatory source checks, using no Python:

```sh
cargo test --offline --locked --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin ce-lyapunov
cargo clippy --offline --locked --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin ce-lyapunov -- -D warnings
cargo fmt --manifest-path experiments/class_geometry_wsd/Cargo.toml --check
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/CEMomentumAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml -- --check
rg -n 'native_decide|^axiom |linter\..* false|autoImplicit' src
```

The final scan should find nothing (rg exit 1). Full unknown-future control
still requires bounding the signed adaptive/encoder perturbations, and any
change in s, ahead of time. The previous checkpoint-only final-500-step
active-share guarantee is unchanged; this study does not extend it over
the 4000-step hold or increase prediction-certificate coverage.
