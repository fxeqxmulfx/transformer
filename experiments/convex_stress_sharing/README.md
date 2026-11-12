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

All 36 candidates completed and replayed. All 36 independent raw-table audits,
36 coverage analyses, 18 exact old-control comparisons and the replay of all
18 training-only choices pass. All six native tests pass, covering overlap and
label renaming, transitive conflicts, finite-difference weighted gradients,
minibatch aggregate memory, exact identity controls, forward batch invariance
and withheld toy recovery in all three modes with saved-state replay.
The frozen producer is `ddf9aa4a`, fingerprint
`3475ea397f514d1ab0932ebdbf2d671848ce556ec97a5014e5ba1e221007dd7b`.
Evidence remains local in `runs/pilot-v1` and root `.kanban/artifacts/169`.
The later coverage auditor leaves the producer binary byte-identical.

All 18 selected task/mode outcomes pass the original 99% criterion, improving
from three to six of these six tasks in each mode. These are percentages on
the original validation split, selected using training CE alone.

| Task | Full batch | Minibatch | Forward only |
| --- | ---: | ---: | ---: |
| Power division, p=97, k=2 | 100 | 99.9570 | 100 |
| Power division, p=97, k=3 | 99.7423 | 99.6778 | 99.8067 |
| Square-times | 100 | 100 | 100 |
| Addition | 100 | 100 | 100 |
| Division | 100 | 99.9785 | 100 |
| Multiplication | 100 | 99.9575 | 100 |

Addition selects `rank_one`; every other task selects `full_rank`. The three
square-times selections hit their respective caps with training-CE excess
`9.36e-4`, `1.43e-4` and `1.75e-3`. They pass accuracy but have no ideal-loss
certificate. The other 15 choices meet the frozen `1e-8` ideal-CE tolerance.
No checkpoint or budget is substituted using held-out accuracy. All alternate
candidates remain available, including their failed fits; the alternate
forward rank-one k=3 model also passes accuracy while missing the CE bound.

The grouping proposal finds 49-by-49 coordinates for k=2, 33-by-33 for k=3,
and 97-by-49 for square-times. All three controls retain identity maps, and
all 18 of their old endpoint probability tables match exactly, including the
opposite geometries. Original observations retain their multiplicities.

The coverage audit finds no observed-group label conflict on any held-out
example. For k=2, 550 of 4,656 validation examples belong to wholly unobserved
groups: full-batch and forward recover all 550, minibatch 548. For k=3, only
18 belong to unobserved groups; the corresponding correct counts are 6, 3
and 9. Its high total accuracy mostly comes from correct sharing, with every
remaining error confined to those unobserved groups. Square-times has 2,387
of 4,705 examples in unobserved groups, all recovered by all three modes.
Thus the audit distinguishes copying a learned shared answer from additional
completion, rather than attributing every correct answer to the spectral bound.

The Lean ambiguity, conditional sharing repair and observed-group certificate
are proved without new unfinished statements. They do not prove correct
group discovery, unique missing-answer recovery or finite-step convergence.
This adaptive six-task pilot does not establish an all-35 pass or a fresh
matched transformer comparison; the original full-suite cohort is separate.
