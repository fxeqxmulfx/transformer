# Balanced original-task density follow-up

This separately frozen CM24 follow-up asks whether observable feature balancing
and explicit missing-axis completion improve the original nine categorical
pair tasks. Its harness is ported from `../cohort` at `3311e13c`. The original
dataset generator, split fingerprints, metrics and full-vocabulary CE are
unchanged. The constructor receives training cell/label pairs only.

| Candidate family | Policy |
| --- | --- |
| Complete CM21–23 parent | Rebuild every earlier CM15 trial, exact and interval attempt, and certified refinement with the original policies. |
| Balanced uniform density | Normalize each nonzero supplied feature row to squared norm equal to rank; audit the explicit uniform density. Its kernel column is an empirical proposal with no objective-optimality claim. |
| Balanced interval densities | Tolerances `0,0.1,0.3`, 4,096 iterations each, 73,728 total; first two observed rows are the objective anchors. Keep all accepted, refused and capped outcomes. |
| Missing phase entries | Sort only observed nonzero distinct phases; assign unused ranks to missing axis entries in index order. This arbitrary completion supplies no unseen labels. |
| Certified refinement | At most 24 rounds per seed, 16 seeds, 256 assignments; preserve all integer duals, decoder counts and states. At most 24 new seed audits. |

Lean `75ca9276` proves actual normalization, uniform-density feasibility,
unchanged observed CE under unseen-axis completion, and unchanged latent
spectral budgets. It also proves a capacity limit: a surjective phase axis
cannot represent a constant opposite row alongside nonconstant rows. This
follow-up keeps that narrower native family and tests the numerical repair;
it does not claim to solve the capacity limitation.

Near-zero means measured row energy at most `1e-24` in the native policy.
Such a row is refused when observed, and explicitly replaced with ones when
unobserved. This numerical convention is recorded separately from the exact
Lean zero-row statement. No re-centering or re-orthogonalization follows
balancing. Every accepted matrix and candidate has raw numerical checks;
none is described as an exact real-arithmetic certificate.

All parent candidates precede new candidates on CE ties. Selection is sealed
before split metrics. Full original-input replay reconstructs every feature,
missing-axis mask, matrix, assignment certificate, decoder and cost ledger;
raw all-frequency DFT/SVD/CE and original split metrics are recomputed.
The harness retains every process failure and all 27 references to the nine
forward constructions. These references are not three optimizer runs.
Full35/modes/transformer integration waits for complete verified parents.

Checks and reproduction use native tooling only:

```sh
cd experiments/convex_stress_density/balanced_cohort
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo build --locked --release
cd ../../..
BALANCED_PRODUCER_COMMIT=<committed-producer> \
BALANCED_BINARY=<frozen-binary> \
  bash experiments/convex_stress_density/balanced_cohort/run_cohort.sh \
  experiments/convex_stress_density/runs/balanced-original-v1
```

Freeze the tested committed source, full source fingerprint, binary hash,
protocol and launcher before preparing original data. The launcher is serial,
uses one BLAS thread, enforces the 64 GiB process limit, reserves 16 GiB disk
space and refuses existing destinations. Partial artifacts remain local.
Generated evidence stays ignored in the root experiment's runs directory.

## Original follow-up result

The frozen producer `4510ee7d6dc722fb9724833995cabc38386e0223` completed all
nine tasks and their independent full replays, with zero process failures.
Both primary and every-split passes remain **3/9**. All nine original inputs,
earlier proposals and complete CM21–23 reports are preserved unchanged.

| Task | Observed errors | Validation sequence accuracy | Selected candidate |
| --- | ---: | ---: | --- |
| Permuted addition | 0/4704 | 100% | Earlier `prior-0` |
| Addition | 0/4704 | 100% | Earlier `prior-0` |
| Subtraction | 0/4704 | 100% | Earlier `prior-0` |
| Division | 2609/4656 | 38.810% | `balanced-refined-0` |
| Multiplication | 1198/4704 | 72.604% | Earlier `prior-0` |
| Sum cube | 2523/4704 | 42.912% | Earlier `prior-0` |
| Sum fourth power | 3758/4704 | 7.396% | Earlier `prior-0` |
| Absolute difference | 4061/4704 | 3.783% | Earlier `prior-0` |
| Blind absolute difference | 2218/4704 | 50.223% | `balanced-refined-12` |

Division previously had no admissible candidate in this family. Blind absolute
difference improves from 41.339% validation accuracy and 2500 observed errors.
These two improvements produce no additional task passes. Earlier full-rank
models already solve division and multiplication, and the separate general-bank
constructor solves both absolute-difference cases; these stronger candidates
remain part of the full comparison.

All 54 supplied subspaces pass the balanced feasibility audit. Of 162 new
interval attempts, 90 meet the numerical stopping rule and 72 reach the cap.
There are 144 audited seeds including the 54 uniform proposals. The bounded
refinement runs 124 seeds and 2180 certified assignments; 20 further slots hit
the assignment budget. Twelve refined slots fit their observations, 103 stop
at repeated states, and nine hit the round cap. These are proposal counts,
not extra successful tasks. The original 54 exact refusals and 162 interval
caps remain in the parent reports.

The new interval work totals 316542 iterations and uses zero CE gradients.
Construction, including reconstruction of all parents, takes 727.398 seconds;
this excludes raw endpoint evaluation and independent replay. The largest
recorded run/replay RSS is 87736 KiB (85.7 MiB), under the enforced 64 GiB limit.
All 27 mode references still refer to nine forward constructions.

Evidence is rooted at `runs/balanced-original-v1` and
`.kanban/artifacts/218/balanced-original-v1` in the main checkout. The full
source fingerprint is
`3b5db804ed686523499453a4e5c6a8681fb0ed0acdbb2d44c8d2af0f01ec9a16`,
the frozen binary SHA256 is
`2abc076ca8e03b98c973181b61a57108f050893b8dc5aacedf4f97c3ec0da698`,
and the complete summary SHA256 is
`5250f76d0d89882064b7f9a4eadc7e377850973cd49d6c7a1e35b001ddc09dd2`.
The result supports the feasibility repair but leaves both phase discovery
and native permutation capacity unresolved. Task 219 addresses the latter
with categorical tags and repeated phase assignments, starting with Lean.
