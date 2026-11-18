# Original synthetic generation with the trained object radii

Can the CM117 successor generate the complete answer from an original
prompt when its own output becomes the next input? This cohort adds
23 cases to the 64-case comparison. Repeated-symbol copy appears in both.
Its six fitted models, training receipts and all 264 initial and final
raw files across both builds must match before combining the studies.
[Parent training protocol](../product_radii/README.md)

| Original cases | Cases | What is scored |
| --- | ---: | --- |
| Histogram, second-order histogram, mode and most frequent | 7 | Exact generated sequence, including scratchpads and EOS |
| Parity | 4 | Plain, running parity, and both original hinted scratchpads |
| Boolean AND | 2 | Random or shifted zero placement, including late-region probes |
| Decimal addition | 8 | Both directions, hint formats and balancing settings, including hard carries |
| Count | 1 | Original symbolic counting sequence |
| Random language control | 1 | Original sampled continuation of a shared prompt |
| Repeated-symbol copy | 1 | Same original copy case and training protocol as the parent study |

The original 165-case plan sets the selection target to 95% complete
sequences on novel validation for all 24 cases. Its Boolean AND source
pools match the archived 35-case subset, which used a 99% primary target.
The two original Boolean selections have seven and eight examples.
Every nonempty novel validation probe retains its source order and
multiplicity. The original random language control has eight training
examples and uses the full sampled teacher sequence to define novelty.
All 64 novel-validation examples have identical visible input and generation
limit with distinct answers. One deterministic output can exactly match at
most one example. The independent decoder checks that ceiling and the
empirical answer multiplicity for every other control split.
[Original benchmark](../../../stress_tests_for_gpt.md)

The seven frequency, four parity, count, addition and control definitions
are checked against the complete 21 metadata records in the November 15
compact archive at SHA-256
`267d5dee6c378a7faf4133fbbfa967a1a75545fa588b5f3a0bcb1fbd6daf2f65`.
They include the regenerated native tensor and complete-record digests.
The Boolean adapter verifies the 14 and eight historical base split
fingerprints. The preflight reconstructs 345 splits and 27,700 supervised
training positions. The legal model context includes every prefix reachable
under the original generation limit. A hinted parity rollout can reach
259 objects. [Source protocol](protocol.json)

The scorer saves full-vocabulary logits for each teacher-forced and
model-written prefix in source order. It counts missing objects after
early EOS and surplus objects past the reference length. Mode and parity
also report the final object before EOS, independently of scratchpad
accuracy. The 99% raw-object and exact-sequence criteria use free
generation on every held-out split. The input-dependent maximum raw
storage is 26,239,425,408 bytes across two builds, each with 144 fits.
Both builds seal their complete fit lists before scoring begins.

```sh
base=/run/media/misha/storage/transformer-runs
out="$base/composed-retrieval/learned-radii-generation24-v1"
producer=$(git log --diff-filter=A --format=%H -- experiments/composed_retrieval/generation/protocol.json)
bash experiments/composed_retrieval/run.sh "$PWD" /run/media/misha/storage \
  "$producer" "$out" generation-cohort "$base/arithmetic-data/inputs-v1"
"$out/original/target/release/reconcile-generation" "$PWD" /run/media/misha/storage \
  "$base/composed-retrieval/learned-radii64-v1" "$out"
```
