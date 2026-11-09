# Prospective probability control

Can a checkpoint's small unused-class probabilities remain small through
actual training motion, long enough to make the CE momentum condition useful?
The [frozen protocol](PROTOCOL.md) fixes the comparison before measurements.
The overall goal still includes full optimizer/encoder control, useful geometry
and prediction; the probability bound alone does not establish those results.

## Formal result

For a particular class u, let p be its actual checkpoint softmax probability.
If every relative logit `z_u-z_c` increases by at most D, then

```
p_future <= p*exp(D) / (1-p+p*exp(D)).
```

This ceiling is sharp, stays in [0,1], and increases with the initial probability
ceiling. It permits signed D. Separate budgets for each rival give a stronger
bound with all rivals retained. Contrast increments telescope over any finite
prefix. A nonzero Lean witness attains the bound: p moves from 1/5 to 1/3
when its contrast increases by log 2.

For an old readout contrast of norm at most A, old feature norm at most X,
readout-motion radius R, feature-motion radius S and execution-contrast change
at most e, the deterioration budget is `R*X+A*S+R*S+e`. The mixed term cannot
be dropped. These are explicit bounds on parameter/feature motion, not an
assumption that future probabilities are small. Feature motion must cover the
full tied encoder and final normalization.

Seventeen new Lean theorems compose the transported ceiling with the existing
weighted actual-partial-CE momentum theorem. This proves the reference step's
nonincrease when `19*s*q_future*max_weight*feature_energy_cap <= 2`.
Full AdamW/encoder perturbations remain separate, and no useful numerical
horizon or full-hold guarantee has yet been measured.

The derivation is [PM1–PM4](../../review/flat_spectrum.md#prospective-probability-control).
The native library evaluates the same probability and motion bounds with
stable log odds. It is preparation for the frozen constructor/verifier, not
a completed training audit.

## Recreating the evidence

The historical ignored run directories and the task #207 artifacts are absent
in this checkout. The user authorized recreation on 2026-11-09. This machine
reports a GTX 1050 with 2 GiB, rather than the previously documented RTX 3050;
new runs must therefore carry separate execution provenance.

The pinned Kernite revision `a1b6bc2c956e1a5f35960e26b209e1b6c213096c`
has been recovered into the root's ignored `.lake/reproduction/kgen` checkout.
The public `grokking` remote ends at `a3eb4686` and contains no
`experiments/rotate90_complex`, no other advertised branch or tag, and no
resolvable `b70811e` commit. The required trainer archive
`source-b70811e.tar.gz` is also absent locally. Its required combined source
digest is `14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da`.
The user-provided [diff-gpt source](https://github.com/fxeqxmulfx/diff-gpt/tree/6bb9d03a7184cb0a43bfaf240977639213e2cbeb)
has been recovered into `.lake/reproduction/diff-gpt`. All four GPT manifest,
optimizer, optimizer-kernel and schedule digests required by the old runner
match exactly. This recovers the optimizer's provenance; its full attention
model is a different architecture from the missing Rotate90 trainer.
The user explicitly retained the original Rotate90 testbed and exact-source
recovery as a prerequisite. Exact recreation therefore still needs that
trainer source, including its initialization and minibatch ordering. The
committed independent forward equations and copied optimizer kernels do not
establish those missing bytes.
No replacement baseline report or historical result has been manufactured.

## Checks

The full Lean build passes 13351 jobs. All 17 new theorems use only
`propext`, `Classical.choice` and `Quot.sound`; the global audit remains
36 existing `sorry`, zero results resting on them and zero extra axioms.
The generated INDEX contains 4254 modules and 21895 theorems, with zero
vacuous or placeholder declarations. The four native bound tests, strict
Clippy and formatting pass. These checks do not supply the missing runs.

```sh
cargo test --offline --locked --manifest-path experiments/probability_tube/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/probability_tube/Cargo.toml -- -D warnings
cargo fmt --manifest-path experiments/probability_tube/Cargo.toml --check
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ProbabilityMotionAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --locked --release --manifest-path scripts/native_index/Cargo.toml -- --check
rg -n 'native_decide|^axiom |linter\..* false|autoImplicit' src
```

The final scan should find nothing (rg exit 1). Generated logs are preserved
at the root in `.kanban/artifacts/79/`. Native tests compare the formulas
against direct softmax, an attained ceiling, signed cumulative motion and
nonzero mixed terms. They are numerical checks, not a floating-point proof.

The read-only native source preflight checks the trainer's concatenated source
digest, Kernite's exact revision and clean sources, and the four GPT digests.
Supply the recovered trainer's directory as the first argument:

```sh
cargo run --offline --locked --manifest-path experiments/probability_tube/Cargo.toml \
  --bin check-rotate90-sources -- /path/to/rotate90_complex \
  "$PWD/.lake/reproduction/kgen" \
  "$PWD/.lake/reproduction/diff-gpt/research/main_model"
```

The current preflight verifies Kernite and all four GPT inputs, then exits
unsuccessfully because the trainer is absent. It never launches training or
substitutes a different model. A successful source check would still need
the original baseline evidence regenerated before the old runner can run.
