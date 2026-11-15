# Joint coordinate programs on the remaining synthetic cases

Task343 asks whether CM57's joint complex head can learn raw output decoders
for the thirteen reviewed340 synthetic cases. The proof is commit258dd055;
the native constrained head is commit aee826aa. This crate supplies input-only
programs and training-only discovery. No new catalog accuracy is measured yet.
The reviewed scope remains58/165 cases meeting all-heldout99 in all three modes.

| Component | Construction | Limitation |
| --- | --- | --- |
| Numeric family | Fixed302 range, cursor and radix configurations | Explicit decimal/input-format prior; not arithmetic learned from arbitrary raw symbols |
| Prefix family | Inclusive counts, comparisons, negation and conjunction; count depth at most3 | Bounded discrete beam search; the target need not survive |
| Coordinate head | Joint program masses and five scaled coefficients, complete16-coordinate complex products per program/class | Injective ordinal class geometry and identity copy are supplied priors |
| Categorical control | Same retained programs; frozen observed atom-to-label dictionary, then the same joint CE core | Unseen entries use class0 and are marked unsupported |

Numeric programs return latent quantities, structural roles or tokens copied
from input. Neither learned parameters nor output labels enter program
execution. The declared format uses BOS1, separator18, operand separator20,
decimal digits26..35 and index symbols100..228. The terminal output role has
no supplied raw answer label. Coordinates are
`e(c)=(c-(V-1)/2)/((V-1)/2)`; a relabeling must transport this prior. Quantity
features use scale128. The entire output formatter must be defined before a
query is accepted, including terminal-role queries.

The prefix grammar is independently implemented from the source family
described in340. It never imports the data generator's selected target AST,
formula seed, evaluation routine or random state. Search receives original
training record IDs, causal prefixes and observed raw labels only. At each
depth a bounded beam combines fit, simplicity and a deterministic hash of
observed truth signatures. Child deduplication by observed signatures is a
heuristic, not a semantic-equivalence proof. The final fitting bank retains
distinct syntaxes even when their training signatures coincide. Search stops
at its first fitting depth; policy, truncation and failure must be reported.

Training anchors test whether each retained program admits a positive-mass
affine decoder with the explicit copy prior. The witness sets a finite local
budget; it does not freeze the decoder during CE. All six scaled coordinates
are optimized by the342 core. Discrete search and the categorical dictionary
are outside that convex stage. Hard-anchor incompatibility is an explicit fit
refusal, including for inconsistent stochastic labels.

Observable span coverage is reported separately from unknown target
realizability. Neither training agreement nor flat Fourier blocks certify the
answers of unseen inputs. A separate postorder prefix interpreter and full
numeric renderer audit every operator. Sum rendering uses digit-wise carries,
independently of the main bounded-integer implementation.

Seven native checks cover all302 programs on valid/malformed prompts and
causal positions, nested inclusive counts, independent complex products,
unseen raw-label completion versus the categorical control, unsupported output
roles, label transport, resource refusal, contradictory labels, complete
mode replay, effective CE updates and the actual minibatch tail denominator.
These fixtures import no benchmark target generators.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/synthetic-programs
export CARGO_BUILD_JOBS=2 LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
manifest=experiments/convex_stress_synthetic_programs/Cargo.toml
cargo fmt --manifest-path "$manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path "$manifest" -- --test-threads=2
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path "$manifest" --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path "$manifest"
```

Before any heldout benchmark measurement, finish and commit the complete
driver, freeze its protocol and transitive source/input closure, and seal all
training endpoints. Preserve original short/long/novel/carry splits, own-output
EOS rollouts or original all-prefix scoring, raw/exact99 and original95% exact
success, six prior/mode endpoints and all refusals. Fresh reconstruction and
an independent rebuild/repeat must audit complete products, projections and
budgets under64GiB. Keep the whole339 report, the original35 comparison and
transformer histories unchanged. Generated evidence belongs on storage.
