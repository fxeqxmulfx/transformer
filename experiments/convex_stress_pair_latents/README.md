# Fixed pair latent programs for the bounded joint decoder

Can a finite input-only representation family extend the original pair tasks
without supplying a target formula or discarding compatible convex variables?
This component follows CM64's exact zero-block elimination and the complete
complex decoder at `a3959a46`. It is a generic pair-input structural prior;
flat spectrum supplies the separate head-capacity bound.

The family is fixed before labels: all49 Laurent monomials with each exponent
in `[-3,3]`, all4704 two-term combinations of distinct monomials with first
coefficient1 and second coefficient in `[-2,-1,1,2]`, and14 ordinary integer
primitives. The primitives are min/max/distance, two ordered quotients and
remainders, bitwise and/or/xor, three comparisons and gcd. Every output is
reduced modulo the prime input vocabulary. The declared totalizations and
enumeration order are in `protocol.json`; no family generator consumes labels,
task names, target syntax, generator seeds or validation examples. An outer
nonlinear map can be learned by the many-to-one decoder instead of being
supplied as a program feature.

Every candidate without conflicting labels at the same observed state remains
in the full joint domain. Every rejected candidate has two original record IDs
with a shared latent state and different labels, checked by a second interpreter.
That interpreter uses direct powers, Fermat inversion, repeated subtraction for
division, and a divisor search for gcd; it does not read the primary evaluator's
power table. Every retained program is checked on every original row with both
interpreters before encoding. Empty support and resource overflows are explicit
refusals. There is no top-k or syntax-size selection. A reload rechecks the full
candidate partition, so removing a compatible candidate is rejected.

| Item | Fixed protocol |
| --- | --- |
| Family | 4,767 programs, independent of task identity and labels |
| Native domain | Every retained mass and every state/class coefficient, including unknown rows |
| Confidence | Total mass in[1,32], initialized at8 |
| Forward | Zero gradient updates |
| Full | Four complete-record CE updates, learning rate1,000,000 |
| Mini | One complete epoch, consecutive127-record batches and actual tail, same learning rate |
| Admission | 40T counted operations,48GiB complete-buffer forecast,64GiB process cap |
| Current binary | Training-only selection and encoding; zero model accuracy measurements |

Five native controls pass: both interpreters on every candidate and every pair
in a complete small domain; exact elimination against an independent pairwise
collision definition; invariance under a bijective raw-label rename; retaining
all4,767 compatible candidates and rejecting forged subsets; duplicate records
and conflicting repeats; all three actual joint training modes on a noninjective
synthetic decoder; and input, resource and training-fingerprint refusals.
These complement the eleven tests of the joint core. They do not establish
that the family realizes any unseen target.

The training-only probe is frozen before reading the48 original task348 pools.
Its shell adapter verifies the pinned independent data review and every input
hash, then passes only the vocabulary and ordered training rows to the binary.
All48 cases are attempted; the six unavailable-source cases remain in the
copied original inventory. This probe reports support sizes, complete memory
forecasts and interpreter checks, with no heldout pools or accuracy scores.
An independent source rebuild must reproduce every training/result artifact.
The previous77 verified cases and original transformer results remain unchanged.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/pair-latents-v1
prlimit --as=8589934592:8589934592 -- cargo test --offline --locked --manifest-path experiments/convex_stress_pair_latents/Cargo.toml -j2
prlimit --as=8589934592:8589934592 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_pair_latents/Cargo.toml --all-targets -j2 -- -D warnings
prlimit --as=8589934592:8589934592 -- cargo build --offline --locked --release --manifest-path experiments/convex_stress_pair_latents/Cargo.toml -j2
# Commit before freezing and before any original-data probe:
bash experiments/convex_stress_pair_latents/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-pair-latents"
bash experiments/convex_stress_pair_latents/probe.sh FROZEN_BINARY FRESH_OUTPUT DATA348_ROOT VERIFIED348_JSON PRODUCER_COMMIT
# Independently rebuild the frozen source, repeat into a new directory, then
# rehash all original and repeated artifacts and compare manifest.sha256.
```

The later complete comparison must freeze and seal all training states before
heldout scoring and independently repeat the full saved coefficients, raw
predictions and ordinary CE. A training fit or a successful probe adds no
catalog accuracy credit.
