# Compare declared-readout complex heads on the original stress suite

Does CM35's use of the declared output partition repair unseen answers in the
original stress comparison? This Rust adapter imports the reviewed training
cohort, checks the genuine grouped complete bases and their separate raw-class
expansions, and evaluates the saved complex heads. It adds no training or
arithmetic function of input names. Original execution is a separate task after
reviewed complete parent255, training259 and this adapter260.

The latest completed comparison retains primary counts of 34/35 full-batch,
34/35 minibatch and 33/35 forward-only. The conventional transformer is 33/35.
DiffGPT separately records 35/35 on each of seeds 0, 1 and 2 for the selected
`phase-score-lerope`: 75 short plus 30 long successes in its
`research/main_model/model.json`. Its original run/audit files are absent in the
available checkout. That recorded result is not a new replay or an endpoint in
this original seed-1 comparison.

## Inputs and scope

| Input | Frozen producer | Retained evidence |
| --- | --- | --- |
| Complete CM34 comparison | `d51aa422` | Both 105-choice arms, all earlier parents, 336 depth outcomes and 35 transformer endpoints |
| CM35 training | `21e86e7c` | 23 original pairs, all 55 primes, seven available heads in each mode including Fibonacci |
| CM34 training | `166bebc3` | Complete training evidence within the parent |
| Causal and refined training | `b516af90`, `4f210282` | Exact task237/task243 summaries and inventories within the parent |

`protocol.json` pins source fingerprints, executable hashes and producers. It
also pins the completed259 summary/inventory and the earlier237/243 evidence.
A request binds the complete parent and training summaries, inventories,
independent review summary, exit and reviewed-commit files. Preparation and
final summarization run both frozen foreign summary executables and require
byte-identical results. Every refusal, process failure, partial artifact and
construction cost remains in the complete imported evidence.

The adapter rebuilds the original native data and checks its fingerprints,
raw vocabulary, labels, observation order, minibatches and policies. It binds
the entire output partition, including unobserved vocabulary labels, to both
the discovery request and its saved `readout.json`. The original field metric
maps each output name into a modulo-97 answer group; ordinary tasks use the
identity map. CM35 explicitly uses this metadata in its observed equations.
The generic constructor receives no input arithmetic rule. The partition does
not replace the original full-vocabulary cross-entropy or its derivatives.

Before offering a head, the checker validates its genuine grouped complete
basis, separate raw vector expansion, initialization, saved real/imaginary
coefficients, CE update trajectory, work and refusal records. An expansion is
never represented as a complete basis of the original ungrouped equations.
Rehashed mutations of the partition, grouped observations/basis, raw vectors,
coefficients, charged work and conditional scope must still fail replay.

## Selection and success

Each task/mode and initializer arm considers its own parent first, then the
eligible new heads in increasing prime order. Replacement requires an observed
raw-label training CE improvement greater than `1e-12`. There is no held-out
ranking or certificate tie-breaker. Offers and selections are sealed before
metrics are evaluated. A failed evaluation retains all six parent choices.

The original primary split's sequence accuracy of at least 99% determines
success. Each per-mode count covers all 35 tasks; `full35_pass` requires all
105 task/mode choices in that initializer arm. Auxiliary split failures remain
diagnostics. `standalone_candidates` reports every evaluated new endpoint,
even when unselected. `selected_changes` reports every replacement, new primary
pass, primary regression and available accuracy/loss delta against its parent.
Neither diagnostic feeds selection. All discovery and training costs are
retained and charged once across both arms.

Masks distinguish encoded axes, complete vector matches to observed classes,
and exact observed inputs. An unseen raw alias can remain ambiguous despite a
correct pooled answer. Unknown axes and unmatched active vectors have separate
counts. Low training CE does not prove unknown-target compatibility.

CM35 in `review/flat_spectrum.md` and
`Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel` derives the
factorization under a surjective declared readout, equal code/answer cardinality,
unknown-code compatibility and additive realizability. The executable checks
the observed construction and preserves the unproved target premises. It does
not establish general convergence or discovery convexity. This adapter adds no
Lean claim and leaves frozen252/253 sources unchanged.

## Checks and reproduction

Ten native tests cover real full/mini/forward heads, two-dimensional grouped
codes, pooled-correct/raw-wrong predictions, complete synthetic35-task scope,
primary success with an auxiliary failure, CE ties, independent initializer
arms, refusal/process failures, review and input binding, actual metric/mask
replay, rehashed certificate/state corruption, standalone/selected differences
and an explicit regression. The shell fixture records a low-disk case as
exit125 and continues remaining cases and summary. Initial disk refusal starts
no run. Synthetic scores are not original benchmark results.

```sh
cd experiments/convex_stress_readout_compare
prlimit --as=68719476736:68719476736 -- cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- \
  cargo test --offline --locked -- --test-threads=1
prlimit --as=68719476736:68719476736 -- \
  cargo clippy --offline --locked --all-targets -- -D warnings
bash -n run_original.sh freeze_original.sh
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release
```

The protocol and scripts require the full reviewed predecessor chain, including
255, 259 and260, before original evaluation. Freeze the reviewed executable,
protocol, scripts and source inventory outside the worktree. Run
`bash freeze_original.sh REPO PRODUCER PARENT_ROOT PARENT_BINARY PARENT_REVIEW
STAGE_ROOT STAGE_BINARY STAGE_REVIEW NEW_REQUEST`, then set
`READOUT_COMPARE_BINARY` to that executable and run
`bash run_original.sh REQUEST FRESH_ROOT`. Every review directory supplies
`process.exit`, `reviewed.commit` and `summary.json`. Independently replay all
complete cases with `replay-case ROOT TASK`, then reproduce `summarize ROOT`
byte-for-byte before approving original results.

All numerical processes have a 64 GiB address-space limit. Metrics retain the
32 billion scalar-work forecast cap per endpoint across all splits. A 16 GiB
disk reserve is checked before preparation and each case; all refusals remain
visible. These are resource caps, not a matched-compute comparison.

Generated outputs stay ignored and outside temporary worktrees. The available
filesystem labelled `storage` is mounted at `/run/media/misha/storage`; this
session uses `transformer-runs/convex_stress_readout_compare` there for new runs
and `transformer-runs/build/readout-compare` for build files. Other destinations
can be passed explicitly. Existing evidence keeps its recorded paths.
