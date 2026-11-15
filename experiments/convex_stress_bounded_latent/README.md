# Bounded joint latent decoder core

Can learned confidence reduce ordinary cross-entropy while keeping the full
many-to-one latent decoder convex and explicitly bounded? This Rust library
implements CM64 after Lean commits `9048ce88` and `aa1d0371`. Candidate discovery
and the original task348 dataset adapter remain separate work. There are no
new stress-test scores or catalog passes from this component.

For a fixed finite family of input-only programs, the learned variables are
every shared mass `w[p]` and every decoder coefficient `gamma[p,state,class]`.
They are nonnegative, each decoder row sums to its program mass, observed rows
equal that mass times the observed one-hot answer, and total mass lies in a
declared positive bounded interval. The native domain rejects inconsistent
programs: an adapter must first eliminate exactly those zero blocks and retain
every compatible candidate, as justified by CM64. It must refuse a resource
overflow rather than truncate compatible candidates. Training rejects any
domain constraints absent from its supplied original records.

The complete source is a direct sum of four-point Fourier matrices. Lean proves
its unitarity and a complete column bound of `2 * R * states`, independently of
program count. Both real and imaginary head tensors are stored as `U* gamma`.
Predictions contract the saved coefficients; skipping a product is justified
only by an exactly zero source entry. A second contraction uses a hard-coded
integer phase table and reversed summation order. The column audit reads every
coefficient, and an inverse audit compares against the full learned tensor.

| Policy | Computation |
| --- | --- |
| Forward | Feasible uniform-mass initialization, uniform unknown decoder rows, zero gradient updates |
| Full | Actual full-record mean CE gradients in all joint coordinates, followed by joint projection |
| Mini | The same gradient and projection on consecutive original record IDs, including duplicates and the actual short final batch |

Unknown rows remain free variables. The initial uniform choice is not an unseen
target rule. Projection eliminates observed-row equalities, projects each
unknown row onto a simplex at the shared mass, and solves the one global mass
multiplier. A uniform-row shortcut gives the same projection when the proposed
rows happen to be uniform; the generic solver accepts nonuniform proposals.
Independent feasibility and KKT residuals must stay below `1e-8` and `1e-7`.
These are numerical audits, not a Lean proof of floating-point projection or
optimizer convergence. Large or ill-conditioned proposals can be refused.

Observed CE decreases with confidence but does not identify unseen decoder
rows. The tests include two feasible heads with identical observed CE and
opposite answers on an unseen state. Target realizability remains an explicit
unproved assumption in CM64's transfer result, never a native certificate.
Flat spectrum supplies the capacity bound; the future program family will be
a separate structural prior.

Predictions certify a strict winner with outward binary64 intervals. Ambiguous
comparisons use independently accumulated exact signed-half products of the
actual saved coefficients, with common denominator `2^1075`. True ties retain
all maximizers and choose the first class. Exact maximum shifting preserves
ordinary CE; if conversion loses a strictly negative score gap, the query is
refused. These numerical certificates concern the saved head only. An exact
query cap and an 8192-bit integer cap are explicit; limb-operation charges are
separate from coefficient products and do not imply a CPU-time bound.

Before allocation, shapes and records are admitted against a 48GiB buffer
forecast under a 64GiB process cap. With `N = programs + programs*states*classes`,
the forecast reserves sixteen full binary64 arrays, the encoded original
records, and 1GiB of slack. Projection rows use exact preallocated capacities;
the reserve includes row metadata, prefix/sorted buffers, labels, current and
proposed parameters, gradients, and both complete complex heads. At most128
epochs on65536 records bound the preallocated step trace below1GiB. This is an
implementation buffer forecast, not a whole-system memory theorem. Runner
data, candidate discovery and serialization will need their own admission.
Every operation path has a finite counter capped at40T and refuses on exhaustion.

Eleven native controls pass: every gradient coordinate against finite
differences of actual complex CE; analytic and nonuniform projection cases;
variational inequalities and idempotence; upper/lower mass bounds and zero
blocks; duplicate weighting and minibatch tails; both update modes reducing
observed CE; exact integer Gram identities and dense complete products;
coefficient corruption; near unequal scores, genuine ties and subnormal
refusals; independent dyadic conversion; unseen-state ambiguity; and resource
and data-boundary refusals. These are component checks, not dataset accuracy.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/bounded-latent-core-v1
prlimit --as=8589934592:8589934592 -- cargo test --offline --locked --manifest-path experiments/convex_stress_bounded_latent/Cargo.toml -j2
prlimit --as=8589934592:8589934592 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_bounded_latent/Cargo.toml --all-targets -j2 -- -D warnings
prlimit --as=8589934592:8589934592 -- cargo build --offline --locked --release --manifest-path experiments/convex_stress_bounded_latent/Cargo.toml -j2
```

No Python is executed. Generated evidence stays in persistent ignored storage.
The48 original datasets must be trained and sealed before any heldout scoring,
then independently repeated, while preserving the previous77 verified cases
and historical transformer references.
