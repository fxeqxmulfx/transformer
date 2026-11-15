# Partitioned boundaries on the original35 cases

Task336 asks whether the reviewed335 partitioned complex head improves original
heldout raw accuracy and ordinary cross-entropy. Its conditional training folds
exceed99%, but the original heldout result is unknown before this frozen run.
The existing best original Fibonacci result is4612/4704 (98.0442%); all35
original success flags and34/35 raw99 are retained as explicit baselines.

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
