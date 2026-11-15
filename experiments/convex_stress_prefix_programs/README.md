# Learned prefix counts and hint copying

Task334 adds four parity scratchpad cases to the12 reviewed additional cases
from328. It retains their full states/results and every prior refusal. The
original35 suite and its Fibonacci gap remain a separate comparison.

Lean first, at `5be03278`: `scripts/PrefixCountingGenerationAxioms.lean` checks
a generic count program, a training-learned categorical decoder, new covered
inputs, an unknown-state refusal, complete complex Fourier logits, coefficient
budget and convex ordinary CE. CM54 transfers accepted predictions through
actual generated prefixes and EOS, conditional on realizability and coverage.
No native realization theorem or flatness-only identification is claimed.

The unchanged328 binder runs first. Only lack of its usable initialization
enables the new finite bank. A sorted prefix antichain finds minimal observed
inputs; their last symbols supply possible delimiters. Symbols occurring at
least once per minimal prefix on average supply possible counters. This uses
input tokens alone. The bank enumerates first/last delimiter, skip0..3,
stride/offset(1,0),(2,0),(2,1), moduli2..8, final-only/every-position/selected-symbol
emissions and optional adjacent-token copies. It contains no task name or
raw answer oracle. A running mode starts at the zero count. Raw labels for
count states and exhaustion are learned from categorical training dictionaries.

Every candidate, conflict witness and compatible dictionary is retained. The
first compatible program in the fixed order initializes the model, following
CM53's vertex construction. Its256-state complete Fourier head uses every raw
class, margin24 and radius16384. The real head has2*256*vocabulary coordinates;
the candidate bank is not misreported as simultaneously active head blocks.
After discrete discovery, ordinary CE runs on the selected template's singleton
simplex slice. This does not claim optimization over the full joint domain.
Full16 iterations, mini4 epochs of32 original supervised prefixes and forward0
retain every final state and true tail size. Effective updates may be zero on
an already identified affine slice; report that honestly.

States below240 can copy raw input symbols;240..247 hold count residues.
Exhausted254 and missing255 have no assigned raw output labels. Unseen decoder
entries remain `None`. The numerical completion explicitly copies an unknown
raw symbol if it is inside the vocabulary, and otherwise uses raw0. This prior
does not identify an unseen label under CM53. Report missing-state and copied
labels separately. Hinted data use229 classes, beyond328's126-class bank.

All16 states must be sealed before any heldout scoring. Every new result uses
actual own-output generation through EOS/limit, separate teacher-forced ordinary
CE and independent complete complex products. A separate count interpreter
recounts the selected prefix, without using the producer's running-state table.
Both original95% exact success and requested99% raw/exact goals remain visible,
with all short and32/64-length splits and all48 mode endpoints.

Search is capped at16,384 candidates,256 compatible dictionaries,64B scalar
units and56GiB explicit buffers. New CE training has a separate96B scalar cap,
declared from all original training dimensions before production. Model discovery, numerical head construction,
all original gradient records and repeated checks are charged. The prospective
new scoring cap is512B scalar units per case/mode, declared before original
scores; prior controls retain their original limits. Processes have a64GiB
address-space cap. Hashing, JSON, data reconstruction and disk I/O are measured
separately. Every refusal stays in the report and denominator.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/PrefixCountingGenerationAxioms.lean
cargo fmt --manifest-path experiments/convex_stress_prefix_programs/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_prefix_programs/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_prefix_programs/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_prefix_programs/Cargo.toml
experiments/convex_stress_prefix_programs/freeze.sh "$FREEZE" "$CARGO_TARGET_DIR/release/convex-stress-prefix-programs"
bash "$FREEZE/source/experiments/convex_stress_prefix_programs/run.sh" "$FREEZE/convex-stress-prefix-programs" "$OUTPUT" original "$PRODUCER" "$DATA333" "$BASELINE328"
bash "$FREEZE/source/experiments/convex_stress_prefix_programs/run.sh" "$FREEZE/convex-stress-prefix-programs" "$REPEAT" repeat "$OUTPUT"
```

`DATA333` is `parity-data/original-v1.json`; `BASELINE328` is
`generative-programs/original-v2/report.json`. Commit and freeze clean sources
before production. Keep drivers immutable while running. Use fresh storage
outputs; preserve complete artifact manifests and compare every JSON result.
Direct full Lean, global axioms, native INDEX and forbidden checks remain
required. Generated reports and logs remain ignored and local. New stress-test
results are pending; the12 reviewed passes do not yet become16.

Eight native checks pass before production. Besides count-three generalization
and categorical decoder renaming, they cover both interpreters over every
candidate, full binding/evaluation, complete coefficients, real mini tails,
forward gradients, strict control preservation, nested costs and early bounds.
All original new-case dimensions fit the declared limits. The largest new
full-batch forecast is66,490,403,328 scalar units for3,280 supervised prefixes;
the separate96B training cap admits this229-class head before any original
fitting or scoring. The maximum search/evaluation forecasts are42,953,035,776
and285,400,670,208 respectively. These are dimension checks, not accuracy results.
Each public command verifies the immutable complete inventory once; every
loaded state retains its own seal, input and forecast checks. This avoids
rehashing both large baseline files once per already sealed case in that command.
