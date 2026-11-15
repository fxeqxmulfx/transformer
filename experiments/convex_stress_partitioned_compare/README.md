# Partitioned boundaries on the original 35 cases

Task336 asks whether the reviewed335 partitioned complex head improves original
heldout raw accuracy and ordinary cross-entropy. The completed, independently
reproduced comparison reaches raw99 and exact99 on all35 tasks in every mode
and on every reported heldout split. Both selection policies retain all105
choices and all original success flags. The previous best Fibonacci result,
4612/4704 (98.0442%), becomes4695/4704 (99.8087%) in all three modes.

Lean first at `4e25eef2`: `scripts/PartitionedBoundaryFallbackAxioms.lean`
instantiates CM38/CM55 with the52-coordinate maximum, repeated original records,
the true short batch, and nonzero complete fallback logits. The fixed partition
expansion and its coefficient costs were audited at `47ba037f`. Ordinary CE
and its derivatives agree on covered records; the fixed affine complete head
has a convex budget domain. Partition and categorical decoder discovery happen
before this convex stage. No unknown-label certificate or float refinement
theorem is claimed. Flat codes bound the normalized input features; raw labels
are categorical and no task-name or arithmetic prediction rule is added.

| Mode | Reviewed335 training | New original scoring |
| --- | --- | --- |
| Forward | Same6144-config constructor, zero CE gradients | Final sealed state |
| Full | Same constructor,128 steps at0.1 | Final iterate |
| Mini | Same constructor,8 epochs of127 at0.1; final tail5 | Final iterate |

## Reviewed findings

Producer `3186b9c3aa8bbd9a62384bcf36e9de3595fc1e41` was frozen before scoring.
Original and independently rebuilt/repeated runs completed with zero failures;
all165 JSON/JSONL artifacts match exactly, including saved models, complete
raw logits, replay results and the final summary. The independently rebuilt
binary also matches the frozen binary. All eight native checks, formatting,
Clippy, release build, applicable Lean audits, full Lean build, global axioms,
native INDEX and forbidden scans passed at the producer commit.

| Mode | Fibonacci raw correct | Fibonacci heldout CE | Aggregate primary raw correct | Aggregate raw accuracy |
| --- | --- | --- | --- | --- |
| Forward only | 4695/4704 | 0.075084803 | 279943/279952 | 99.996785% |
| Full batch | 4695/4704 | 0.075016804 | 279943/279952 | 99.996785% |
| Minibatch | 4695/4704 | 0.074949462 | 279941/279952 | 99.996071% |

These counts are identical under `minimum_ce` and `first_small_ce`. Each policy
has105/105 primary raw99, primary exact99, all-heldout raw99 and all-heldout
exact99 outcomes. Original success is35/35 in each mode. The lowest primary
raw accuracy is the Fibonacci row above. Only its eligible endpoints were
retrained: the other34 cases explicitly inherit reviewed319 metrics. All
35 conventional transformer endpoints and336 depth outcomes remain retained,
alongside the separate historical DiffGPT success on seeds0/1/2. This experiment
does not provide a fresh matched-budget comparison or a new three-seed run.

Against319, the new Fibonacci head fixes84/87/84 heldout errors in
forward/full/mini, with no regressions. Against330 it fixes87/88/88; against332
it fixes83/85/90, again with no regressions. Full/minibatch ordinary CE is lower
than forward-only CE, and all three are lower than319 and332. The score winner
is unchanged across the three new modes. The constructor's partition search
is essential: the preserved global affine335 attempt remained below99% even
on training. This evidence supports the shared partitioned representation on
these inputs; it does not certify arbitrary unseen labels or prove discovery
by convex optimization.

The new component stores68 shared parameters across both roles, while the
complete combined head has7,180,800 real coefficient slots and radius bound
2,730,368. Full and minibatch apply256 and608 new CE gradient batches across
the roles; forward-only applies none. Every endpoint accounts for65,856
prediction/product calls, and the complete invocation reserves
65,580,802,229,664 scalar work units. The largest forecast buffer is
58,600,283,680 bytes, below the fixed56GiB ceiling. Original/repeat drivers
took40:00.49 and39:58.00, with peak RSS7,314,320 and7,306,884KiB, under the
64GiB address-space cap. Forecasts include sequential replay and resident
models; they are conservative implementation allowances, not allocator proofs.

Evidence is local under `/run/media/misha/storage/transformer-runs/partitioned-compare/`
(`original-v1`, `review-v1`) and `freezes/partitioned-compare-v1/findings-review`.
The165-file relative manifest SHA256 is
`b68b45de39acd9a27b94fd8c680b54ab105cf0efa06589e62172b500a7d9384c`;
the complete summary SHA256 is
`c9b3c559069fa1e9a4cd68a5cf3ee93123008bcd54753cf3a2485e2c6ad9cef2`.
Generated evidence remains untracked. The16 separately reviewed additional
stress cases remain100% raw/exact;114 additional cases remain unmeasured,
including the seven data-only frequency cases from337.

## Protocol and reproduction

The adapter retains all35 cases, both105-choice policies,35 conventional
transformer endpoints,336 depth outcomes and separate historical DiffGPT
seeds0/1/2. It verifies all331 artifacts from319,156 from330,163 from332 and56
from the partitioned335 component, including the failed affine335 evidence.
The complete prior summaries and manifests remain available. This is not a
fresh matched-budget transformer comparison. Eligibility uses the same frozen
representation prefix as330/332; it does not use task names or new scores.

Every fresh fit must exactly reproduce the corresponding reviewed335 outcome,
including all candidates, split attempts, CE updates and work counters. All
three endpoints are saved and sealed before any new original scoring. The
original prior/supplement route is preserved. Covered blocks use the new head;
unsupported blocks retain the actual complete319 head. Every scored record
stores full319/new logits plus retained330/332 logits, predictions and CE.
IDs, tokens, targets, fields, roles and vocabulary must match. Fixes and
regressions against every retained model are counted; changed-state scores
are freshly computed. Independent expanded Fourier products check all raw
classes. Failures stay in the35-case denominator, and original success,
per-task raw99, native exact99 and CE remain separate.

Complete coefficients and fixed offsets are charged at1048576 per new role,
plus the complete old head and radius. Report the compact stored parameters
separately from the expanded real coefficient count. Different model budgets
prevent a matched-capacity claim. Only eligible endpoints are retrained;
unchanged cases retain reviewed319 metrics with explicit provenance.

The prospective work cap is80T scalar units per complete case invocation,
including three new construction/training passes and two old constructions
per endpoint, all6144 configurations, every split trial, gradients, projections,
metric/field calls, independent products and both retained raw audits. The
larger partition search is charged even though CE alone is inexpensive.
Serialization, hashing and disk I/O are additionally measured.

Memory forecasts separate preparation, fitting/roundtrip, reconstruction/scoring
and summary phases. New roles are fitted and replayed sequentially. The largest
one-role constructor buffer is charged alongside resident old models and
saved states during reconstruction. During staging, no old runtime exists;
two complete typed saved copies and parent parsing are reserved. JSON expansion
uses32 times immutable file sizes, with2GiB plus8MiB additional reserve. The
saved-file allowance comes from pinned complete335 outcomes and is checked on
every sealed model. Summaries are read separately from model execution. The
explicit56GiB buffer ceiling and64GiB address-space limit remain unchanged.
These are prospective implementation allowances, not a formal allocator proof.

Eight native checks pass, including both actual roles in all modes, complete
expanded products, nonzero unsupported fallbacks, original record/field-call
counts, exact raw replay against330/332, retained regressions and failure
denominators, and memory overflow/refusal. The synthetic source uses disjoint
categorical cycles to obtain binary aliases in different latent fields. The
earlier acyclic fixture produced only singleton blocks and was refused; those
failed test logs remain retained. The production component is unchanged.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/PartitionedBoundaryFallbackAxioms.lean
cargo fmt --manifest-path experiments/convex_stress_partitioned_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_partitioned_compare/Cargo.toml -- --test-threads=2
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_partitioned_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_partitioned_compare/Cargo.toml
bash experiments/convex_stress_partitioned_compare/freeze.sh "$FREEZE" "$CARGO_TARGET_DIR/release/convex-stress-partitioned-compare"
bash "$FREEZE/source/experiments/convex_stress_partitioned_compare/run_original.sh" "$FREEZE/convex-stress-partitioned-compare" "$OUTPUT" "$REQUEST"
```

The request fields are `producer_commit`, `parent_root`, `parent_manifest`,
`component_root`, `component_manifest`, `comparison_root`, `comparison_manifest`
and `diff_gpt_history`. Use319 `signed-compare/original-v2`,335
`partitioned-boundary/inspection-v1`,332 `shared-geometry-compare/original-v1`,
their complete reviewed manifests, and the frozen task318 historical snapshot.
Exact hashes and component sizes are in `protocol.json`. Freeze the committed
source closure, driver and binary before scoring. Run the full driver, rebuild
the same source, repeat in a fresh directory and compare all JSON/JSONL hashes.
Keep generated evidence on storage, not in Git. Full Lean, applicable audits,
global axioms, native INDEX and forbidden-construct checks also apply.
