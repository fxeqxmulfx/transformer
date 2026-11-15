# Observed-prefix quotients for deeper count discovery

Task344 asks whether training-only discovery can recover the count programs
missed by task343's128-child beam. The ordinary cross-entropy stage remains the
joint CM57 complex head. Discrete grammar search is outside its convex claim.

CM58 (`df52cffd`) first proves finite prefix-closure storage and preservation of
observed nested counts under equal complete prefix signatures. CM59 (`8c458234`)
then proves necessary child bits at actual count-threshold transitions. Both
extensions pass full Lean builds and standard-axiom audits; the repository
retains36 existing sorry and zero extra axioms. Neither proves unseen answers.

The native search takes only original training prefixes and raw labels. It
builds a prefix trie, caches exact inclusive counts, and retains each distinct
first-layer signature within a declared cap. The fixed supplied grammar has
three symbol predicates, negations, Boolean constants, two count operands,
an optional symbol count, offsets-2..2, optional negation and a literal gate.
The program AST and both interpreters come unchanged from task343. Complete
observed signatures choose a smallest retained syntax, without claiming global
semantic equivalence. All final programs undergo the original raw-label fit
check and a separately implemented interpreter check.

For depth3, candidate root configurations yield necessary child constraints
from adjacent observed labels. Unknown parent labels and inactive gates reveal
no comparison truth. Inferred child bits constrain only that search branch;
they do not create additional CE observations. Capped latent banks refuse their
branch. The first uncapped fitting branch in declared order is retained, with
at most32 final syntaxes, ordered by node count and AST. Every search cap and
work counter remains explicit. Node counts are cached; this changes no program.

| Development run | Training-only finding | Limits and observed resources |
| --- | --- | --- |
| v1 | Depth1 finds16 programs; depth2 finds4 after retaining1926 first-layer classes; depth3 has no depth2 fit | Depth2:1.40s,7764KiB RSS; no heldout measurement |
| v2 | All13 possible depth3 root branches hit a latent-bank cap before final pair comparison |8192 classes;7.45s,32808KiB RSS |
| v3 | Three root configurations complete their pair comparisons without a fit; ten other configurations still hit a cap |32768 classes;3:06.41,199556KiB RSS;316946809 pairs in the largest complete branch |

Development artifacts are local under
`/run/media/misha/storage/transformer-runs/prefix-quotients/training-dev-v{1,2,3}`.
V2/v3 include exact development source snapshots and binary hashes. Later
source changes preserve failure/work-cap counters and cache AST node counts.
These runs do not change the reviewed68/165 raw/exact99 coverage. All thirteen
cases, categorical controls, original35 and entire343 history still need a
complete frozen independently repeated producer before a new score is claimed.

Reproduce native validation without Python:

```sh
cargo fmt --manifest-path experiments/convex_stress_prefix_quotients/Cargo.toml --check
CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/prefix-quotients-check \
  prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_prefix_quotients/Cargo.toml
CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/prefix-quotients-check \
  prlimit --as=68719476736:68719476736 -- cargo clippy --all-targets --offline --locked \
  --manifest-path experiments/convex_stress_prefix_quotients/Cargo.toml -- -D warnings
```

The `discover CASE FRESH_JSON` binary generates only the original training
split. It never reads a selected target AST, formula seed, generator RNG,
heldout target or task identity into the search. Tests compare interval pruning
against exhaustive grammar evaluation, check depth3 transition witnesses,
prefix closure, duplicate observations, independent count evaluation, raw-label
contradictions, signature caps and work-cap evidence. Scalar forecasts are
conservative admission bounds; actual pair/row/signature counters are separate.
