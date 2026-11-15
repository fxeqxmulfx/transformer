# Complete Basis pools with weighted joint CE

Task346 asks whether the CM57 complex head can fit all complete Basis pools
under64GiB, retaining ordinary cross-entropy and the original scoring rules.
CM60 exact multiplicity grouping and CM61 selected-anchor feasibility were
proved first in commits819d4547 and822bb5e8. Their25 Lean theorems justify
complete weighted objectives/derivatives and a convex domain after fixing the
anchors. They do not identify unknown clean targets or certify these programs.

The reviewed345 data producer67863307 supplies all eight cases,45 splits and
5806712 supervised training positions. No pool, original mask, multiplicity,
heldout probe or novel split is shortened. `train_clean` is diagnostic only.
The complete344 report, original35 comparison and original history are retained
byte for byte. Historical transformer successes remain successes; this
experiment neither retrains those transformers nor matches their compute.

| Component | Coordinate model | Categorical control |
| --- | --- | --- |
| Input programs | One selected program from the same bounded grammar | Same selected program |
| Decoder prior | Injective ordinal raw-class coordinates, affine quantities, identity copy | Observed atom-to-coordinate means, unknown entries default to class0, identity copy |
| Exact anchors | All original groups for exact fits; otherwise only consistently labelled structural-role groups | Same rule on the control's features |
| Initializer | Complete observed-label moments, projected onto the joint domain | Observed dictionary, projected onto the same form of domain |
| CE modes | Forward0 updates; full64 epochs; mini8 epochs with batch1021 and actual tails | Same three modes |
| Measurements | Every original split: raw, exact, transitions, ordinary teacher-forced CE, independent products and coverage | Same complete measurements |

The fixed grammar contains causal lags, earlier/latest pointer walks, run-count
equalities and small residue counters. When the observed input alphabet has
more than eight symbols, only the copy bank is enumerated. Generated quantities
use the explicit BOS1/separator18 syntax prior. The bank has no access to a case
name, target generator, clean training labels or heldout labels. Discovery
prefers exact affine fits and otherwise ranks candidates by complete observed
training CE at the declared moment initializer. It retains one program; all
attempts and visited positions are recorded. This discrete discovery and the
choice of anchors are outside the convex claim. Fixed-grammar success is not
evidence for unrestricted end-to-end convex learning.

For noisy parity, different observed labels for identical features remain
different groups. Making all of them exact anchors would force zero mass and
conflict with the positive-mass requirement. The new domain keeps structural
role anchors supported by consistent observations while retaining every noisy
label in CE. It also reports the old all-anchor projection's refusal. An
observed coordinate mean is an initializer, never a pseudo-clean equality.
Without content anchors, the old content-identification guarantee is absent;
coverage and unknown-target realizability are reported separately.

CM60 grouping preserves each original ID and row/position. Full gradients
count every observation; minibatches use their own exact counts and true tail
denominators. Training raw traces store complete vectors once per exact
feature/label group and one reference for every original supervised position.
Both input interpreters are checked on every original prefix. Every other
query stores all raw-class logits and independently multiplies the complete
complex tensor. Generation feeds the model's own outputs back until EOS or
the original limit. A separate streaming readback reconstructs all source
rows, decisions, CE, transitions, EOS outcomes and integer denominators.

All six endpoints of all eight cases are saved and sealed before the first
heldout model score. A clean committed producer is required for preparation;
the executable, protocol, full data fingerprints, dependencies and Lean
sources are frozen. A second run rebuilds every dataset and trained state,
then reproduces every raw artifact. New coverage is provisional until the
complete manifests match. Each case must reach99% raw and exact on every
original heldout split in all three coordinate modes to add catalog credit.
Original selected-split success is also reported separately.

Resource forecasts cover discovery, all six fits, scoring, readback and the
independent repeat. Original records are capped at2million, exact groups4096,
candidates1024, buffers56GiB and process address space64GiB. Storage is checked
before preparing a run. Only one case is loaded at a time. The larger1021
minibatch differs from earlier127-record experiments and is explicit in every
saved policy. Exact/relaxed masses and learning rates are frozen in
`protocol.json`; they were set using training-only probes.

Six native tests currently pass. They cover independent input interpreters,
source/mask binding, all1024 noisy observations and all six fitted endpoints,
full gradient accounting and the three-record minibatch tail, unknown
dictionary entries, grouped versus individual scoring, a checksum-renewed CE
tamper, wrong own-output generation and premature EOS. Tests never score the
actual heldout pools. The weighted core separately checks full complex
derivatives, projection properties and exact agreement with the retained
all-anchor trainer.

Training-only development probes fit all six endpoints for noisy parity,
languageL6 and hard recall. They retain1024,939244 and160000 targets,
respectively. Coordinate hard-recall training CE changes from0.4975801269
to0.0460903935/full and0.0026721830/mini. Noisy training has921/1024 correct
targets, including the103 retained corruptions; full CE changes from
0.4242296065 to0.4242294261. Mini CE increases by approximately7.7e-9;
global monotonicity is not claimed for SGD. L6 already classifies every
training target correctly at initialization and prints zero CE due to floating
precision saturation. Its coordinate gradients execute but produce zero
representable parameter updates; categorical update counts can include
projection roundoff. None of these fitting observations adds heldout credit.

Reproduce from the producer worktree without Python:

```sh
basis_store=/run/media/misha/storage/transformer-runs
basis_manifest=experiments/convex_stress_basis_programs/Cargo.toml
export CARGO_TARGET_DIR="$basis_store/build/basis-programs" CARGO_BUILD_JOBS=2
cargo fmt --manifest-path "$basis_manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path "$basis_manifest"
prlimit --as=68719476736:68719476736 -- cargo clippy --all-targets --offline --locked --manifest-path "$basis_manifest" -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path "$basis_manifest"
basis_binary="$CARGO_TARGET_DIR/release/convex-stress-basis-programs"
basis_commit=$(git rev-parse HEAD)
basis_freeze="$basis_store/freezes/basis-weighted-programs-v1"
basis_original="$basis_store/basis-programs/original-v1"
basis_review="$basis_store/basis-programs/review-v1"
experiments/convex_stress_basis_programs/freeze.sh "$basis_freeze" "$basis_binary"
prlimit --as=68719476736:68719476736 -- "$basis_binary" prepare "$basis_store/basis-data/original-v1" "$basis_original" "$basis_commit"
prlimit --as=68719476736:68719476736 -- "$basis_binary" train "$basis_original"
prlimit --as=68719476736:68719476736 -- "$basis_binary" score "$basis_original"
prlimit --as=68719476736:68719476736 -- "$basis_binary" summary "$basis_original"
export CARGO_TARGET_DIR="$basis_store/build/basis-programs-review"
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path "$basis_manifest"
prlimit --as=68719476736:68719476736 -- cargo clippy --all-targets --offline --locked --manifest-path "$basis_manifest" -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path "$basis_manifest"
cmp "$basis_binary" "$CARGO_TARGET_DIR/release/convex-stress-basis-programs"
basis_binary="$CARGO_TARGET_DIR/release/convex-stress-basis-programs"
prlimit --as=68719476736:68719476736 -- "$basis_binary" repeat "$basis_original" "$basis_review"
prlimit --as=68719476736:68719476736 -- "$basis_binary" verify-pair "$basis_original" "$basis_review" "$basis_store/basis-programs/verified-v1.json"
```

All destination directories and the verification file must be fresh. Source,
protocol and required inputs are committed; generated states/raw traces/logs
stay local on the storage disk. The first complete heldout comparison is pending.
