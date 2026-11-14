# Convex mixtures for raw-label refinement

Can a training-only mixture of context maps improve raw-label choices after
the reviewed CM43 confidence calibration? CM45 at `2f0dc9ba` proves the actual
mixed complex products, the unchanged feature/head bound, convex ordinary CE
in the mixture, coefficient slopes and a quadratic upper model. It also proves
that exact observed-row preservation makes the original observed CE constant
in the mixture. A separate diagnostic is therefore declared here.

| Part | Fixed policy |
| --- | --- |
| Bank | Reviewed parent first, then 90 combinations of 5 neighborhood sizes, 3 distance powers, 3 axis weights and 2 confidence scales |
| Neighborhoods | 1, 3, 5, 9, 17 nearest observed states within the same latent block |
| Distance | Squared ordinal coordinate distance with axis weights (1,1), (4,1), (1,4); ties by cell then state |
| Neighbor weights | Inverse squared-distance powers 0, 1, 2, normalized before the fixed scale 0.25 or 0.5 |
| Diagnostic | Original raw training records and multiplicities, excluding the queried source state, using the fixed canonical margin24 conjugate head |
| Full | Best vertex followed by 64 conditional-gradient steps on the full diagnostic |
| Mini | Best vertex followed by 8 epochs in original record order, at the declared batch size including each short tail |
| Forward | Best vertex by forward CE; zero new derivatives and optimizer updates |
| Line search | Steps 0, 1/16, 1/8, 1/4, 1/2, 1 toward the vertex minimizing the coefficient slope |
| Retention | Best full-diagnostic epoch endpoint, including the initial best vertex; strict improvement greater than 1e-10 |
| Inference | Actual mixed complex coordinates times the unchanged complete source coefficients |

All raw classes remain in the ordinary softmax denominator. The diagnostic
stores the nonzero class logits and includes the remaining zero-logit classes
analytically. The canonical initializer identity is checked against actual
complex products; sparse loss and slopes are checked against dense CE and
finite differences. The canonical tensor is fixed during mixture training.
Its selected mixture is then transferred to the unchanged trained source
tensor. This transfer supplies no guarantee about held-out accuracy or CE.

The bank changes only rows already interpolated by CM42/43. Observed inputs,
unique-label blocks, unmatched inputs and unsupported axes keep the reviewed
parent. Exact observed logits, raw losses and complete coefficient derivatives
remain unchanged, including repeated records and one-record tails. The original
source training metadata and new mixture gradient/score/update counts are
recorded separately. The algorithm has no claimed convergence rate.

The locality assumption is explicit and uses input coordinates without an
answer formula or a distance between output labels. Discovery still uses all
training records: source exclusion is not a leave-one-out refit of discovery.
Flatness bounds the features and complete coefficient budget; it does not
identify missing answers. Bank and mixture selection never read held-out labels.

External digests bind the parent and endpoint metadata. Loading reconstructs
the complete bank, diagnostic optimization, mixture rows and costs before
accepting a saved model. Allocation and work forecasts are checked before
building the diagnostic cache. Limits are 56 GiB forecast buffers, 32 billion
scalar work units and 4096 gradient batches, with a 64 GiB process address cap.
They are native guards rather than formally verified floating-point bounds.

The eight native fixtures pass, covering actual nonzero complex products,
ordinary CE and slopes, observed gradients in all three actual source training
modes, repeated records and tails, source-coefficient-independent diagnostics,
deterministic reconstruction, mutation rejection and prediction/limit errors.
This component has no original-data accuracy result. Its separate comparison
must retain all 35 tasks, both policies, all modes, every baseline and every
regression, with selection frozen before held-out evaluation.

On the noisy 47-record synthetic fixture, the parent's diagnostic CE is
0.37842 and the best vertex reaches 0.35847 with 42 correct raw labels. Full
mixture optimization reaches 0.29668 with three active weights and 41 correct
labels. Mini and forward retain the vertex. This exposes the CE/argmax tradeoff:
lower diagnostic CE does not by itself prove higher raw accuracy. These are
fixture observations, not the original Fibonacci result.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_context_mixture/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_context_mixture/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_context_mixture/Cargo.toml --all-targets -- -D warnings
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false scripts/native_index/target/release/lean-index --check
```
