# Original GPT mechanism probe data and scoring

This native port reconstructs the six original GPT mechanism probes and retains
their historical protocols. It prepares data and a separate evaluator; it trains
no model. The next comparisons require minibatch and forward-only minibatch.
Full-batch is optional, and the frozen comparisons351/355/360 are unchanged.

Five probes currently reproduce every original record and teacher fingerprint.
Routing mixture remains unverified: the native tensor operations reproduce its
tokenized records but fail the independent unquantized teacher hash. Preparation
retains its explicit refusal, writes a complete six-case inventory, and exits
nonzero. A repeat can establish an identical refusal; that does not make the
missing dataset verified or add model accuracy. Task361 remains open.

| Probe | Original splits | Vocabulary | Original primary target | Current port |
| --- | --- | --- | --- | --- |
| Routing mixture | 1024/256/256 | 293 | Unquantized MSE ≤0.001 | Teacher hash refusal |
| Branch amplitudes | 1024/256/256 | 69 | Unquantized MSE ≤0.001 | Exact source fingerprints |
| Conditional amplitudes | 1024/256/256 | 69 | Unquantized MSE ≤0.001 | Exact source fingerprints |
| Linear transport | 1024/256/256 | 69 | Unquantized MSE ≤0.001 | Exact source fingerprints |
| Order discrimination | 496/496 | 70 | Sequence accuracy ≥99% | Exact source fingerprints |
| Key/value binding | 2048/512/512, plus novel validation | 100 | Sequence accuracy ≥99% | Exact source fingerprints |

The reference is DiffGPT `5a37c515639f23037d191b0659a5a0853984328d`.
`references/sources.json` records the exact source hashes. `references/cases.json`
is a required reproducibility input: the six plan entries joined, in plan order,
with their complete original `probes_seed1.json` history. The three earlier
controls are branch amplitudes, linear transport and key/value binding. The
newly reproduced conditional and order datasets are not new learned successes.

`native/reference.cpp` ports the CPU tensor construction in `probes.py`.
The executable links native ATen and c10, with no Python runtime. Random draws,
two intra-op threads, masks, vocabulary, multiplicity, split sizes, EOS and
unquantized binary64 words are retained. Key/value binding reuses the previously
reviewed native original35 generator. The pinned ATen runtime is2.14.1+cu126;
operations explicitly use CPU, with the AVX2 kernel family. The runtime's CUDA
support-library dependencies do not perform GPU data construction.

Routing's expected test teacher SHA256 is
`818cdbf141a2ad4a288e49c336bcca00f702917202af26485bd2b6ea533da7aa`;
the direct native result is
`aef39e606285cce7d852a28e18ff4030c15f29f5e55535bc2e1f6e52235aa13d`.
Diagnostic changes to reduction order, scalar square root and native exponential
kernels did not resolve the mismatch. CPU kernel differences are a hypothesis,
not an established cause. None of those diagnostic variants enters this producer.
The relevant native algorithms are in the pinned PyTorch
[softmax kernel](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SoftMaxKernel.cpp)
and [sum kernel](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SumKernel.cpp).

The model-facing training view contains only vocabulary, original row index,
tokens and masked training labels. The predictor interface receives only the
visible causal prefix and returns full-vocabulary logits. Source names, formulas,
numeric teachers and heldout answers stay in the data producer and scorer.
Rows remain in their original order, including repeated records and minibatch
tails. The data port itself has zero updates in every command.

Scoring preserves free generation from the model's own prefix, original EOS
limits, the full raw-answer denominator, exact sequence correctness and class
counts. Numeric error compares decoded outputs with the unquantized teacher;
each invalid or missing component costs64. Training scores use teacher forcing;
heldout generative scores use free generation. Dense order scores use only the
original supervised positions. Ordinary CE, raw99, exact99 and original success
are separate fields. CE is accumulated in native binary64, so no bitwise match
with the historical CUDA loss reduction is claimed. Nonfinite or malformed model
logits cause an explicit evaluation failure.

Nine native controls cover scoring, independent denominator checks, unquantized
teachers, malformed transport, retained source history, missing inventory entries,
both disks and the RAM reserve. The scorer's synthetic controls are not evidence
of a trained model. Readback checks every accepted dataset against all historical
fingerprints, its complete summary and serialized artifact hash. Refused cases
cannot contain an accepted dataset or claim the inventory is complete.

Both disks need at least2GiB free during preparation; freezing needs4GiB.
The wrapper caps process address space at8GiB, while admission reserves4GiB
above the summed live process RSS under the user's110,000,000,000-byte RAM limit.
Every data read/write is bounded at32MiB. Failure stops without deleting prior
artifacts. Generated datasets, binaries, manifests, logs and diagnostic files
belong outside Git and outside this worktree.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/probe-data-v1
cargo fmt --manifest-path experiments/convex_stress_probe_data/Cargo.toml --check
cargo test --offline --locked --manifest-path experiments/convex_stress_probe_data/Cargo.toml -j2
cargo clippy --offline --locked --all-targets --manifest-path experiments/convex_stress_probe_data/Cargo.toml -j2 -- -D warnings
cargo test --offline --locked --release --manifest-path experiments/convex_stress_probe_data/Cargo.toml -j2
cargo build --offline --locked --release --manifest-path experiments/convex_stress_probe_data/Cargo.toml -j2
bash experiments/convex_stress_probe_data/native/build.sh PINNED_TORCH_ROOT NEW_NATIVE_BINARY
bash -n experiments/convex_stress_probe_data/{freeze,run}.sh experiments/convex_stress_probe_data/native/build.sh
```

After committing, freeze the clean source and actual linked native libraries:

```sh
bash experiments/convex_stress_probe_data/freeze.sh NEW_FREEZE \
  "$CARGO_TARGET_DIR/release/convex-stress-probe-data" NEW_NATIVE_BINARY PINNED_TORCH_ROOT
bash NEW_FREEZE/source/experiments/convex_stress_probe_data/run.sh NEW_FREEZE \
  NEW_FREEZE/convex-stress-probe-data WORKSPACE NEW_ORIGINAL NEW_FREEZE/probe-reference
```

The current original command is expected to exit1 after retaining routing's
refusal and the other five complete datasets. Rebuild Rust from the frozen
source in a different Cargo target directory, rebuild the native helper against
`NEW_FREEZE/native-runtime`, and run the same wrapper at `NEW_REPEAT`. Compare
using the independently rebuilt executable:

```sh
REVIEW_BINARY verify WORKSPACE NEW_ORIGINAL NEW_REPEAT NEW_VERIFICATION.json
```

An identical inventory with `all_datasets_verified=false` preserves the remaining
gap. Do not submit361 as complete until every original numeric teacher matches.
Required repository checks use direct `LAKE_ARTIFACT_CACHE=false lake build`,
`lake env lean scripts/Axioms.lean`, native INDEX checking and forbidden-source
checking. This data/scoring change makes no new Lean theorem or optimizer claim.

Producer `a707943659d964e43c0673e316274389eb7cce1c` was frozen with65 source files
and9543 native runtime/header files. A separate Cargo build and C++ rebuild pass
all nine controls in default and release profiles, with strict Clippy. Original
and independent repeat preserve all15 splits of the five accepted probes and
the routing refusal. Their complete inventory bytes agree at SHA256
`12fd36f6076b922aa959ab0b9c0a9ace11d34a0df3f0579e5445ac107e3f9033`.
The verifier explicitly records `all_datasets_verified=false`; both preparation
commands retain exit1. No checkpoint, gradient update or model measurement exists.

Original preparation takes13.17 seconds and repeat12.60 seconds, peaking at
163080/163084KiB RSS. Both runs recheck the frozen runtime before and after use.
Evidence is retained under `transformer-runs/probe-data/{original-v1,review-v1}`
and `.kanban/artifacts/361`; the source/runtime freeze is
`transformer-runs/freezes/probe-data-v1`. The full Lean build passes13995 jobs;
the audit remains36 `sorry`, zero dependent rests and zero extra axioms. The
native index check reports4898 modules,25678 theorems and no vacuous/placeholder
declarations. These checks validate the infrastructure while routing's original
numeric teacher and future learned accuracy remain open.
