# Fixed support fallback for bounded complex heads

Can a complex candidate keep the saved predecessor's answers outside its
observed support without changing its ordinary training cross-entropy? CM38,
proved and reviewed at `16e34b2f`, gives that identity on an affine domain with
the parent frozen. This native component implements the actual route and its
resource accounting. It has eight numerical checks; no original held-out
comparison has run through it, so the existing benchmark counts are unchanged.

The complete causal comparison motivating the change gains forward branch
amplitudes but regresses on forward associative divisor. The successful saved
predecessor of the latter is the earlier count model. A fallback therefore
needs its actual state, not just the newer Fourier component. Dispatch uses
the saved representation (`Causal` or `Counts`); other families receive an
explicit refusal. Neither a task identifier nor an arithmetic answer formula
is a model input. The same route applies to every valid categorical prefix.

| Component | Parameters and forward computation | Bound |
| --- | --- | --- |
| Candidate | Saved real/imaginary Fourier coefficients and actual complex code products | Existing column radius; code coordinates at most one |
| Causal parent | Saved complex coefficients, code dictionaries and fixed forward pass | Existing radius, with all coefficients and code storage charged |
| Count parent | Saved shared/query integer joint coefficients, embedded in complex coordinates with zero imaginary part | Nonnegative joint mass, indicator features bounded by one, explicitly checked column sums |

Support is reconstructed from every original input: both the observed queries
and the selected feature values must equal the saved dictionary. Input hashes
retain order, repeated records, raw labels, vocabulary and context. The route
uses the candidate inside this support and the predecessor elsewhere. It never
checks the target to decide which head predicts. Both heads remain stored;
only one contributes on an individual prediction.

The count embedding ports `AdaptiveModel::raw_logits` from
`convex_stress_counts` at `9496a606`. Each shared key/value row and each local
query/key/value row is a fixed indicator feature. Missing query mass supplies
the class-zero row; the zero-total model supplies its original uniform row.
Joint counts are complex coefficients with imaginary part zero. Normalization
and scale are fixed. Complex addition retains the original integer merging,
division and final scaling order, including shared/local overlap. The declared
`2^52` integer limit makes every merged integer exactly representable before
division. Numerical tests compare the resulting bits with the original saved
model implementation on every prefix in the fixture domain.

For vocabulary size `V`, value-alphabet size `A`, nonnegative scale `s`, shared
mass at most `T` and each local mass at most `T`, the charged parent column
radius is `s * (V + A * (1 + V))`. This bounds the sum of all nonnegative joint
coefficients and therefore every single column. The implementation also sums
the actual columns, including implicit class-zero coefficients, and checks
them against that radius. All logical complex rows, including zero coordinates,
are charged in `real_slots`; integer weights and code storage are counted
separately in `stored_scalars`. Buffer forecasts cover sparse storage and its
audit/serialization copies, plus original record/token storage and batch
temporaries. The inherited 200,000-record and 20-million-token limits are
checked before encoding. The combined forecast must stay within 56 GiB;
native commands have a 64 GiB address-space cap.

This is CM38's generic bounded-feature case for the frozen count parent, using
the CFt1-CFt3 joint channel representation. Count indicators and the routed
input matrix are not claimed to be unitary. The candidate's flat code supplies
its feature bound, while ordinary raw-vocabulary CE remains convex in its
real/imaginary coefficients. Discovery and selecting a fixed feature family
remain separate from that convex optimization claim.

Full, mini and forward endpoints can be wrapped without retraining. Every
covered original record has exactly the candidate logits. `batch_gradient`
retains the actual record IDs and denominator; repeated inputs under different
IDs remain distinct. Finite differences use the actual wrapped complex forward
pass for full batches and two/two/one minibatches. Additional checks replay
saved full and cyclic-mini histories unchanged, and an explicit shorter-tail
epoch. The old cyclic causal schedule is not relabeled as a one-epoch run.
Forward construction and prediction compute no CE derivatives.

Saved inference models load only through external candidate, parent and input
bindings. The loader reconstructs support and charged bounds. Tests change
coefficients, parent scale, inputs, budgets and bindings; even a self-rehashed
parent fails the external binding. Tests also retain an actual bounded Fourier
witness in which a supported unseen alias gets a worse answer. Exact parent
preservation outside support is not a claim of universal nonregression.

`protocol.json` declares the future comparison before original metrics. It
retains the complete parent's existing observed-CE head selection, then tests
a separately declared transformation of selected causal heads with importable
predecessors. This is necessary because the wrapped model has identical training
CE: the old strict-decrease selector would not select it on a tie. Missing or
unsupported predecessors retain the old selected endpoint with a recorded
refusal. Both initializer policies, all 35 tasks/modes, original primary success,
auxiliary diagnostics, all depth/transformer results and costs must remain in
that complete comparison. Existing frozen comparisons are unchanged. A reviewed
adapter and fresh original evidence are still required for any improved score.

From the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/fallback
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path experiments/convex_stress_fallback/Cargo.toml --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_fallback/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_fallback/Cargo.toml --all-targets -- -D warnings
LAKE_ARTIFACT_CACHE=false prlimit --as=68719476736:68719476736 -- lake build
LAKE_ARTIFACT_CACHE=false prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/Axioms.lean
```

Freeze the committed component, protocol and complete local Cargo dependency
source inventory before its review with `bash
experiments/convex_stress_fallback/freeze_source.sh REPOSITORY PRODUCER_COMMIT
NEW_DIRECTORY`. The native Cargo graph supplies each local package; all tracked
files in those packages and the borrowed `convex_stress` modules are hashed and
archived, with the locked metadata and compiler versions retained. The later adapter must also freeze its
exact executable and each imported checkpoint. Generated logs and evidence
remain outside the worktree; no synthetic fixture is an original benchmark pass.
The numerical checks are separate from the exact-real CM38 proof; they do not
turn floating-point outputs into Lean kernel certificates.
