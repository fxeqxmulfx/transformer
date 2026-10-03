# TinyShakespeare benchmark migration

- [x] Preserve every measured source, report, protocol, dataset, and checkpoint.
- [x] Put all six benchmark implementations under `src/infrastructure/benchmark/` and their
      tests under `tests/benchmark/`.
- [x] Keep the original measured sources and reports in a packaged archive;
      record their hashes before migration and retain compatibility paths.
- [x] Replace repository imports and working-directory-dependent paths in the
      active implementation. Keep numerical algorithms and training rules.
- [x] Separate domain rules, application ports/use cases, infrastructure, and CLI.
- [x] Make uv install the benchmark, its test suites, dataset, and evidence.
- [x] Run the numerical and migration contracts before any training smoke test.
- [x] Run the GPU contracts for both attentions, including all 24 full compiled
      updates and all three AMSGrad extensions.
- [x] Verify the installed wheel outside the repository and document commands.

Scope: optimizer, Magma, patience, compiled-forward, complete compiled-step,
and AMSGradW/MD comparisons. The synthetic-task and MQAR studies are separate.
Historical results keep their original protocol fingerprints. New runs record
the migrated source hashes and write to separate output directories.

Validation completed on 2026-10-02:

- CPU checkout suite: 125 tests, 111 passed and 14 explicit CUDA skips.
- NVIDIA GPU suite: all 125 tests passed, including complete compiled steps,
  CUDA Graph recording/replay, checkpoint reloads, and four short real-data
  runs covering both attention modes and both execution backends.
- Installed wheel tested from `/tmp`, using tests extracted from the source
  distribution: 125 tests, 111 passed and 14 explicit CUDA skips.
- The wheel's active source fingerprints match the checkout. All 307 distinct
  archived evidence/source files and the dataset match their recorded hashes.
- All 426 transferred checkpoint/curve files match their original hashes;
  legacy source paths still resolve. The loss-extraction check verifies 108
  exact recorded encodings.

The GPU run emitted PyTorch notices about limited SMs for GEMM autotuning and
ignored profiler regions, plus a Triton/Python `_POSIX_C_SOURCE` build warning.
The CUDA Graph capture/replay and steady compiler-counter checks passed.

Complete AMSGradW training reproduction was also verified on 2026-10-02.
The new CLI passed 98 CPU/CUDA preflight contracts and trained all six runs
from initialization using both attentions and seeds 0, 1, 2. Validation/training
curves, stopping/checkpoint steps, initial weights, paired batch plans, and
selected model hashes match the historical runs exactly. The mean test CE is
1.625374713347326 for Softmax and 1.6389354029880645 for Sparsemax, identical
to the archived measurements. Evidence is recorded in
`reports/amsgradw_reproduction_20261002/`; large checkpoints remain ignored.
