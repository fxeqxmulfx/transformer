# Joint coordinate programs on the remaining synthetic cases

Task343 asks whether CM57's joint complex head can learn raw output decoders
for the thirteen reviewed340 synthetic cases. The proof is commit258dd055;
the native constrained head is commit aee826aa. This crate supplies input-only
programs and training-only discovery. Frozen producer
`178771856b7bc93603fe310345e4ad3d3da28cbd` now completes the original run and a
fresh independently rebuilt repeat. Ten new seed1 cases reach100% raw-label
and exact-sequence accuracy on every split in forward/full/minibatch modes:
Count, all eight Addition variants and depth1 C-RASP. The verified catalog
scope rises from58 to68/165 cases meeting all-heldout99 in all three modes.
The original35 comparison and complete earlier23-case cohort remain inherited,
unchanged references; no new transformer training or seed replication is claimed.

| Mode | Coordinate test CE | Categorical test CE | Correct test tokens | Exact test sequences |
| --- | ---: | ---: | ---: | ---: |
| Forward | 0.3263473621 | 0.2753128544 | 63,700/63,700 | 3,456/3,456 |
| Full | 0.0030031147 | 0.0033692898 | 63,700/63,700 | 3,456/3,456 |
| Mini | 0.0024034406 | 0.0027562013 | 63,700/63,700 | 3,456/3,456 |

Both arms attain100% accuracy on these ten cases. The896 test occurrences of
unseen raw output labels are copied hints; both arms have the explicit identity
copy prior. CE is weighted by every supervised token across all54 original test
splits, including long and hard-carry probes. Each successful endpoint evaluates
all182 splits of the ten cases, including validation/novel and train diagnostics.
Forward mode performs training-only discovery and anchor projection, with zero
gradient steps. Full/minibatch perform640/680 gradient batches across the ten
fits and visit651,392/81,424 training records, including actual short tail batches.

The other three attempted cases are fit refusals, with null model measurements:
depth2/3 C-RASP search finds no observed-compatible program, and RandomLM is
undefined for this supplied program family. Its training pool also contains
six identical-prefix groups with contradictory labels, incompatible with a
positive-mass deterministic hard-anchor fit. All refusals remain in the13-case
denominator. Thus97 catalog cases still lack complete model measurements,
including these three; task344 addresses deeper training-only count discovery.

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
no supplied raw answer label. Prefix programs exclude optional leading BOS
from the counted symbol word; adding BOS leaves their semantics unchanged.
Coordinates are
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
at its first fitting depth. The frozen driver policy permits128 beam entries,
32 fitting syntaxes, depth3 and10,000,000 visited candidates. These are explicit
search limits, not completeness guarantees; truncation and failures are recorded.

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

Ten native checks cover all302 programs on valid/malformed prompts and
causal positions, nested inclusive counts, independent complex products,
unseen raw-label completion versus the categorical control, unsupported output
roles, label transport, resource refusal, contradictory labels, complete
mode replay, CE updates and the actual minibatch tail denominator. Driver tests
also cover independent raw-count/EOS scoring, prefix ignore/transition masks,
all-case seals, tamper rejection, streaming hashes and refusal denominators.
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

The complete driver binds the binary, committed protocol and pinned340 data
inventory. Its staged construction forecasts all candidate work, six fits,
fresh reconstruction, actual complex products and disk output before fitting.
It seals every case/mode before permitting heldout scores. It preserves
original short/long/novel/carry splits, own-output
EOS rollouts or original all-prefix scoring, raw/exact99 and original95% exact
success, six prior/mode endpoints and all refusals. Fresh reconstruction and
an independent rebuild/repeat must audit complete products, projections and
budgets under64GiB. Keep the whole339 report, the original35 comparison and
transformer histories unchanged. Generated evidence belongs on storage.

Training-only development finds a compatible range program for Count, the
appropriate radix formatter for Addition and12 fitting syntaxes for depth1
C-RASP. Count coordinate CE falls from0.562538 to0.00703683 full-batch and
0.00438676 minibatch; a plain Addition fixture falls from0.110811 to0.000275762
and0.000388806. No heldout score was used. Increasing the prefix beam from64
to128 still finds no compatible program on depth2/3 training pools. The final
driver retains those refusals and the independent-label negative control. A
successful CE fit is not a measured generalization result.

The core's effective-update counter means bitwise parameter change. Projection
roundoff can increment it even when CE is numerically zero; endpoint audits
also report the largest actual parameter step and initial/final CE. The
categorical binary control can start with numerically zero training CE.

```sh
store=/run/media/misha/storage/transformer-runs
binary="$CARGO_TARGET_DIR/release/convex-stress-synthetic-programs"
prlimit --as=68719476736:68719476736 -- "$binary" forecast FRESH_FORECAST_JSON
# Optional development diagnostics generate the training split only.
prlimit --as=68719476736:68719476736 -- "$CARGO_TARGET_DIR/release/train-only" suite-count FRESH_TRAIN_JSON
```

After committing, direct Lean/global/CM57 audits, native INDEX and forbidden
checks, freeze the complete producer before any heldout model evaluation:

```sh
freeze="$store/freezes/synthetic-programs-v1"
experiments/convex_stress_synthetic_programs/freeze.sh "$freeze" "$binary"
mkdir -p "$store/synthetic-programs"
/usr/bin/time -v -o "$store/synthetic-programs/original-v1.resources" \
  "$freeze/source/experiments/convex_stress_synthetic_programs/run.sh" \
  "$freeze/convex-stress-synthetic-programs" "$store/synthetic-programs/original-v1" \
  original "$(cat "$freeze/producer.commit")" \
  "$store/remaining-synthetic-data/original-v1/inventory.json" \
  "$store/remaining-synthetic-data/original-v1/original35.json"
# Independent compilation into an empty target directory, then byte comparison.
export CARGO_TARGET_DIR="$store/build/synthetic-programs-review"
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path "$manifest"
cmp "$CARGO_TARGET_DIR/release/convex-stress-synthetic-programs" "$freeze/convex-stress-synthetic-programs"
/usr/bin/time -v -o "$store/synthetic-programs/review-v1.resources" \
  "$freeze/source/experiments/convex_stress_synthetic_programs/run.sh" \
  "$CARGO_TARGET_DIR/release/convex-stress-synthetic-programs" "$store/synthetic-programs/review-v1" \
  repeat "$store/synthetic-programs/original-v1"
```

Compare every original/repeat JSON and JSONL file, including both complete
references, state/projection/CE traces, seals, raw vectors, metric replays and
the complete report. Preserve process exits/resource logs and all failures.
Do not alter this frozen producer after observing heldout results.

Both complete runs exit0, including all13 stages, score commands and fresh
replays. All189 JSON/JSONL artifacts are byte-identical; their manifest SHA256
is `56f68a3141418222f652a4796dc35933eda9713443d7603cead3875f962aeb6f`.
The complete report SHA256 is
`d7714f2d2d261b9680aa4818b9877501a6c2586b09f5762d4c6c402687b4139d`.
Independent compilation yields the same production binary SHA256,
`3c1079275f6c706d1e4323e2d031bef65269bbe733cf7ea890eb69f10cb28da2`.
The freeze preserves5,338 source inputs. Original/repeat wall times are
11:12.83/10:52.54; peak RSS is7,985,260/7,985,272KiB, about7.62GiB. The
largest preflight buffer bound is13,902,658,560 bytes and scalar-work bound is
204,699,389,116,416, below the56GiB/600-trillion admission caps. Each process
runs under64GiB address-space limits.

Every scored query checks the separate program interpreter and complete
complex multiplication. Maximum product residual is8.73115e-11; rechecking
every saved projection against original anchors and complete budgets gives
maximum normalized violation7.40770e-16. These numerical audits supplement the
real-valued CM57 theorem and leave target realizability explicitly uncertified.
All ten native checks, formatting, clippy with warnings denied, full Lean,
direct global/CM57 axiom checks, INDEX and forbidden scans pass. The global
audit retains36 existing sorry with0 rests and0 extra axioms.

The separate post-run `audit-retained` executable reads each report back,
authenticates the full nested340 inventory and339 report, rehashes the separate
original35 reference, checks every case/state/replay binding and all120 raw
files, and recomputes accuracy thresholds from their stored integer counts.
Both readbacks match. Adding this read-only auditor leaves the frozen production
binary unchanged. The complete transformer histories remain inside the exactly
retained references. Generated evidence stays in
`synthetic-programs/{original-v1,review-v1}` and `freezes/synthetic-programs-v1`
on storage. Reproduce the recorded producer at commit17877185; the post-run
auditor and analysis source are committed separately with these findings.

```sh
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path "$manifest" --bin audit-retained
for run in original-v1 review-v1; do
  prlimit --as=68719476736:68719476736 -- "$CARGO_TARGET_DIR/release/audit-retained" \
    "$store/synthetic-programs/$run" "$freeze/convex-stress-synthetic-programs" \
    "$(cat "$freeze/producer.commit")"
  jq -s -f experiments/convex_stress_synthetic_programs/analyze.jq \
    "$store/synthetic-programs/$run"/*/metrics.json
done
```
