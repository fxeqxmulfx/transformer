# Seven frequency-generation datasets

Task337 ports seven more cases from the frozen165-case GPT catalog. It adds
data and scoring contracts before another model extension. The complete
reviewed334 report, all16 earlier additional passes, their failures and
historical GPT references remain unchanged. This inventory adds zero model
evaluations;114 additional cases still await evaluation.

| Cases | Answer | Vocabulary |
| --- | --- | ---: |
| histogram-bos / histogram-no-bos | Frequency at each input position; optional initial BOS | 229 |
| histogram2 | Number of distinct symbols sharing each position's frequency, with BOS | 229 |
| mode-none | Unique most frequent symbol | 100 |
| mode-counts / mode-itemized | Count/symbol pairs, then the unique mode | 229 |
| most_freq | Distinct symbols by decreasing frequency, padded with BOS | 100 |

Every answer ends in EOS. Frequency ties retain first occurrence, including
the increasing-count scratchpad. Mode repairs a sampled tie by moving one
occurrence from a randomly selected tied donor to a selected winner. Draw order
matches the frozen source: one paired length,128 format bits per example even
without hints, pool size, sampled alphabet, word draws, any tie repair, and
finally pair reversal. Only declared formatting controls are excluded from
the split seed. Histogram BOS variants and all three mode formats therefore
share problems. The unused number limit is absent from mode-none identity.

Seed1 retains128 training,32 validation and64 test examples at lengths8–16,
clean training,32/64 probes and prompt-based novel validation:11 splits per
case. Original source constraints and native size ceilings apply before
allocation. All raw targets use the causal shift, including scratchpads and
EOS. Own-output generation stops at EOS or the original input-derived limit.
Mode's final-token diagnostic is separate from full exact-sequence accuracy:
a correct final mode does not repair an incorrect scratchpad. Ordinary
teacher-forced CE, original95% exact success and requested99% raw/exact goals
remain separate.

Six native tests compare all21 historical test token/transition denominators,
independently recount every generated word, verify vocabulary/masks/novelty,
format alignment and odd stream prefixes, stable ordering, single-occurrence
tie repair, genuine own-output rollout, final-versus-exact scoring and early
refusals. Historical per-example fingerprints are unavailable; matching source
rules and aggregate denominators is not proof of old byte identity. The new
inventory records complete records and original-style split/tensor hashes.

Longer frequency examples require number tokens absent from short training.
The inventory records every such target occurrence and raw class, without
feeding them to a model or asserting they are identified. A later learner must
state any structural decoder prior it uses; flat feature bounds alone cannot
supply these labels. Data answer constructors are private and are not model
features, programs or selection inputs.

The complete16-case report is embedded by immutable hash. Before parsing, the
inventory reserves64 times combined source bytes plus4GiB for parsing/copying
and new records, below56GiB. Every process remains capped at64GiB. Freeze the
committed native sources and reference hashes before generating either run;
repeat into a fresh path and compare the complete bytes. Generated artifacts
and review evidence stay on storage. No Lean theorem changes in this data port.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
cargo fmt --manifest-path experiments/convex_stress_frequency_data/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_frequency_data/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_frequency_data/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_frequency_data/Cargo.toml
bash experiments/convex_stress_frequency_data/freeze.sh "$FREEZE" "$CARGO_TARGET_DIR/release/convex-stress-frequency-data"
prlimit --as=68719476736:68719476736 -- "$FREEZE/convex-stress-frequency-data" prepare "$HISTORY" "$BASELINE334" "$FRESH_INVENTORY"
```

`HISTORY` is the frozen task318 progress snapshot; `BASELINE334` is
`prefix-programs/original-v1/report.json` under the storage runs root. Exact
pins are in `protocol.json`. Also run direct `lake build`, the global axiom
audit, native INDEX check, reference-source hashes and the forbidden scan.

Producer `952b37ca` is reviewed. Its freeze preserves427 source inputs. Original
and independent inventories match complete SHA256
`21d0c176efdfe08b8bc04bdf83b62c91987e542d2e799160905dbea05d32c5ae`.
The entire retained334 report compares equal. Both runs exit0, taking3m22.07s
and3m09.32s with peak RSS7887260/7887088KiB (7.52GiB). The prospective buffer
allowance is46361964544 bytes. Six native checks, formatting, Clippy, release,
source hashes, full Lean, global axioms, native INDEX and forbidden checks pass
at the producer. Existing36 unfinished Lean statements remain explicit.

| Case | Unseen target occurrences, short test | Length32 | Length64 |
| --- | ---: | ---: | ---: |
| histogram-bos | 11 | 657 | 2792 |
| histogram-no-bos | 11 | 657 | 2792 |
| histogram2 | 6 | 0 | 0 |
| mode-none | 9 | 5 | 6 |
| mode-counts | 0 | 53 | 179 |
| mode-itemized | 0 | 53 | 179 |
| most_freq | 0 | 0 | 0 |

These counts concern labels absent from training **outputs**, not necessarily
absent from inputs: mode-none can use a declared copy prior. Numeric count
labels need a separate identifying assumption. Task338 investigates a supplied
class-coordinate prior in Lean before any such model extension. Complete
inventories and review evidence are preserved under `frequency-data/` and
`freezes/frequency-data-v1/` on storage. No new model pass follows from this data
review; all seven evaluations remain outstanding.
