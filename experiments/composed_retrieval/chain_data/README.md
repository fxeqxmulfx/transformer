# Original modular-chain data

The GPT benchmark has twelve modular composition cases. Its input is an
original sequence of residue objects `[z₀, b₀, c₀, …]`. The only supervised
object is the final residue after applying `z ← (z³ + bz + c) mod p` at
each pair. The generator computes that label. The model sees the original
prefix, and every earlier position has the original ignore mask.
[Original benchmark](../../../stress_tests_for_gpt.md)

| Original data | Values |
| --- | --- |
| Modulus | 17, 97, 193 |
| Composition steps | 1, 2, 4, 8 |
| Training split | First 2,048 unique short inputs |
| Validation split | Next 1,024 unique short inputs |
| Doubled-length test | 1,024 unique inputs from a separately seeded stream |
| Primary criterion | At least 99% exact final answers on validation |

The source enumerates and shuffles the small domain in the one-step
modulo-17 case. It rejection-samples complete inputs in all other cases.
The longest test prefix has 33 objects. With data seed 1, two Rust
implementations agree on all 49,152 examples across all 36 splits,
including row order, final labels, zero change masks and the full dense
tensor hashes. With the separate historical data seed 11, the generator
reproduces all 72 archived input and labeled-row hashes. The seed-1
digests come from reconstruction of the original source. This data audit
scored zero models. [Frozen source references](sources.json)

From the repository root, with `DIFF_GPT` set to the DiffGPT checkout
and `INPUTS` set to a fresh directory for the archived source texts, run

```sh
experiment=experiments/composed_retrieval
bash "$experiment/chain_data/freeze-sources.sh" "$PWD" "$DIFF_GPT" "$INPUTS"
cargo test --release --locked --offline --manifest-path "$experiment/Cargo.toml" \
  --features chain-data --test chain_source
cargo run --release --locked --offline --manifest-path "$experiment/Cargo.toml" \
  --features chain-data --bin chain-inventory -- "$INPUTS"
```
