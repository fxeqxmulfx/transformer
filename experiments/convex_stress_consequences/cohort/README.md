# Original decoded-consequence comparison

This separately frozen harness tests whether bounded nonunique code discovery
and CM29 observed-row certificates improve the completed task223 comparison.
It retains the same 23 eligible original pair tasks, all 35 tasks in the parent,
all three modes and all 105 choices. The original splits, category names,
vocabularies and 0.99 success rule stay fixed. The complete task223 summary is
copied byte for byte and embedded in the new summary; its SHA256, source and
producer are pinned in `protocol.json`.

| Component | Fixed rule |
| --- | --- |
| Proposal inputs | Training categorical observations only; all original observations rechecked |
| Discovery | Tested default component, CRT, model and query budgets from the native library |
| Views | Original observed-strip proposals, support threshold 8, at most four attempted views |
| Bounds and loss | Actual full latent Fourier/SVD audit and ordinary CE over the original vocabulary |
| Certificates | Exact observed-only row programs; unknown injective-code realizability and fixed view remain hypotheses |
| Selection | Observed CE, conditional certificate within a 1e-12 tie, then earlier order |
| Prior certificates | Preserve the immutable parent's certificate; recognize an exact matching certified model hash |
| Accounting | One forward construction per task, shared by three mode references; zero new CE gradients |
| Memory | One dense view at a time, at most 60 GiB forecast plus 4 GiB wrapper reserve; 64 GiB process cap |
| Evidence | Every view, branch, cap, refusal, duplicate, model and query program retained; large reports streamed to files |
| Failure | Preserve all three prior choices and the partial files/logs |

The model bank and artifact hashes are sealed before fresh held-out metrics are
evaluated. The original data generator necessarily builds its splits; only
training observations enter discovery and selection. Full replay independently
checks actual code/row coefficients, reconstructs models, repeats raw audits,
reruns every constructor and verifies original split metrics. Summary generation
checks each retained artifact's size and SHA256 against the sealed replay.
The source fingerprint includes the native Rust generator, its inventory of
sources, protocol, manifests/locks, launcher and original data/metric stack.

Eight native tests pass: original protocol/policy preservation, streamed full
replay, prior-certificate and CE priority, rehashed certificate mutations,
all-105 failure retention, complete-summary artifact binding, parent provenance
and streaming hash/size checks. Together with the 41 core tests these provide
49 native checks before original evaluation. No new benchmark result is claimed
by these fixtures. Discrete discovery is separate from convex prediction-space
CE; neither spectral flatness nor observed strip agreement alone identifies
unknown answers.

Run from the repository root, after committing and archiving the tested
producer, binary, protocol, source fingerprint and binary SHA256:

```sh
cargo fmt --manifest-path experiments/convex_stress_consequences/cohort/Cargo.toml --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- \
  cargo test --locked --manifest-path experiments/convex_stress_consequences/cohort/Cargo.toml -- --test-threads=1
cargo clippy --locked --manifest-path experiments/convex_stress_consequences/cohort/Cargo.toml --all-targets -- -D warnings
cargo build --locked --release --manifest-path experiments/convex_stress_consequences/cohort/Cargo.toml
export CONSEQUENCES_PRODUCER_COMMIT="$(git rev-parse HEAD)"
bash experiments/convex_stress_consequences/cohort/run_cohort.sh \
  experiments/convex_stress_consequences/runs/original-v1 \
  experiments/convex_stress_identified/runs/original-v1
```

Set `CONSEQUENCES_BINARY` to an archived binary to rerun the exact producer.
The launcher uses a fresh destination, serial construction/replay, one BLAS
thread and a 16 GiB disk reserve. Generated evidence stays ignored in the root
experiment's `runs/`; a worktree must link this directory before running.
Retained-depth integration and the conventional transformer remain separate
required comparisons, with their original budgets and all outcomes preserved.
