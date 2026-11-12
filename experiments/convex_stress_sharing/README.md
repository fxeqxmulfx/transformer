# Fixed input sharing before spectral completion

Can a generic training-only sharing proposal repair the missing-answer failure
that remains after fitting ordinary cross-entropy exactly under wider spectral
bounds? Lean CM10, landed at `78a6743d`, comes first: it proves that an uncertain
completion can be a feasible global CE minimum, and that a correct fixed sharing
map repairs that witness. It also proves convexity and the CE fit certificate
for linear lookup from a table on fixed groups, retaining original multiplicities.

The map proposal is discrete and data dependent. Convexity applies after the map
is fixed, or to the direct prediction hull of fixed families. It does not claim
joint convexity of map search and table training, or prove that agreement on
observed labels implies agreement on missing labels. The predictor uses token
identities, saved maps and learned probabilities. It evaluates no arithmetic rule.

Rows or columns can be proposed for sharing when at least four observed
counterparts agree and none disagree. Proposals are ordered by decreasing
overlap, then axis and token IDs. Each union must preserve label consistency
under both current maps on every original training observation. Groups receive
compact indices in representative order. Rejected unions are saved too.
The threshold and ordering are frozen before any candidate training.

Learn a probability table on these fixed groups. Inactive padding is uniform,
so its nonconstant Fourier moments vanish. Bound each nonconstant Fourier
class moment of the grouped table in operator and nuclear norms. Group sizes
and vocabulary size determine the bounds; unknown target spectra do not enter them.

| Candidate, in training-only selection order | Operator bound | Nuclear bound |
| --- | --- | --- |
| `rank_one` | sqrt(r c) | sqrt(r c) |
| `full_rank` | sqrt(V) | min(r,c) sqrt(V) |

Here r and c are the numbers of row and column groups, and V is the unchanged
output vocabulary. The second candidate retains the original ambient operator
bound; reducing both input dimensions alone does not justify a smaller one.
These are upper bounds;
the model does not impose a rank equality. No claim puts every task's target
inside either family.

Every original observation remains in ordinary CE, even when several map to
one grouped cell. Full-batch sums their gradients. Minibatches refresh a
gradient for each original example and update the summed cell gradient, using
the original shuffle, batches and tails. This is incremental aggregated
gradient, not unbiased SGD. Both modes use step size `1/max_cell_multiplicity`.
Forward-only fixes consistent observed rows and uses no training derivatives.
CE, FFT, simplex and spectral projection code is inherited unchanged from
`deef98cb`; the splitting solver is ported to weighted grouped observations.
No finite-step convergence theorem is claimed.

The frozen pilot has six tasks, two budgets and three modes: 36 records and
18 choices. Power-division mod 97 with exponents two and three and square-times
are failure diagnostics; addition, division and multiplication are controls.
The older unshared cohorts remain unchanged as baselines. This follow-up was
designed after seeing their failures and is not a prospective assessment on
previously unseen task families.

Caps remain 128 full-batch, 512 minibatch and 128 forward-only updates. Stopping
uses scheduled training CE excess at most `1e-8` or the cap. Held-out accuracy
never stops training or chooses the budget. Selection takes the first ideal
candidate in the table's order, otherwise the lowest training CE without a
global certificate. Keep every failure, all original split metrics, group-search
time, both candidates' costs and exposures. A floating-point ideal certificate
is not an interval proof or a guarantee of correct unseen answers.

## Reproduce

Use Rust, FFTW3 and OpenBLAS/LAPACK, with one BLAS thread. The launcher defaults
to two independent numerical processes, each with eight frequency workers.
Generated outputs stay ignored; worktree run directories link to the root.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
LAKE_ARTIFACT_CACHE=false lake env lean scripts/SchattenMomentsAxioms.lean
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_sharing/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_sharing/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_sharing/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_sharing/Cargo.toml --bins
experiments/convex_stress_sharing/target/release/convex-stress-sharing prepare
bash experiments/convex_stress_sharing/run_pilot.sh experiments/convex_stress_sharing/runs/pilot-v1 2
OPENBLAS_NUM_THREADS=1 experiments/convex_stress_sharing/target/release/audit_saved CANDIDATE_DIRECTORY PRODUCER_FINGERPRINT
experiments/convex_stress_sharing/target/release/audit_saved coverage CANDIDATE_DIRECTORY PRODUCER_FINGERPRINT
experiments/convex_stress_sharing/target/release/audit_saved compare OLD_SCHATTEN_DIRECTORY NEW_DIRECTORY PRODUCER_FINGERPRINT
```

Replay reconstructs the groups from training data, the endpoint from saved
splitting state, all metrics, bounds, exposure counts and fresh full-data
residuals. The independent auditor reads raw maps and probabilities, checks
group consistency and original multiplicities, and computes plain-exponential
CE, predictions and direct Fourier sums. Data construction and LAPACK SVD are
shared; the norm checks are numerical rather than interval certificates.
The auditor's comparison requires exact coordinates against each of the 18
original control endpoints, including failed geometries, with identity maps.

The coverage command separates predictions on training-covered groups from
wholly unobserved groups. It checks whether each held-out label agrees with an
observed label in its group and counts correct predictions in both subsets.
This is post-training attribution; no held-out label feeds back into the map
or fit. It does not repeat the spectral audit. Lean `quotient_observed_group`
proves that exact observed CE determines the prediction on a covered group;
agreement with held-out ground truth still requires a correct group map.

## Findings

The Lean ambiguity and conditional sharing repair are proved. Native tests
cover overlap and label-renaming behavior, a transitive-agreement conflict,
finite-difference weighted gradients, minibatch aggregate memory, exact identity
controls, forward batch invariance, and a withheld toy answer recovered in all
three modes with saved-state replay. Stress-task outcomes are pending. This
pilot does not establish an all-35 pass or a fresh matched transformer comparison.
