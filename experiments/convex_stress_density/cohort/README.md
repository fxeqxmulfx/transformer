# Original-task density and certified-refinement cohort

This freezes the policy measured on the complete/masked fixtures at `d97e1221`
for the nine original pair tasks used by the decoder-bank follow-up, including
division and the blind long absolute-difference case. The constructors receive
categorical training cells and labels; task names only identify frozen data.
The theory precedes the implementation: CM21 `1885957b`, CM22 `a4388412`,
and the CM23 assignment/decoder descent proof `0bb5496f`.

| Component | Declared policy |
| --- | --- |
| Earlier decoded proposal | 16 codes, 64 power steps, 24 refinements, seed 73 |
| Exact density proposals | Two codes, ranks 1/2/3, 256 iterations per stage, global 2,048 iterations, four model audits |
| Interval proposals | Tolerances 0.03/0.1/0.3, 4,096 iterations each, global 73,728, eight model audits |
| Certified refinement | 24 rounds per seed, 16 seeds, 256 assignments globally |
| Selection | Ordinary training CE; earlier-first ties within 1e-12 |
| Resource bounds | Serial cases, one BLAS thread, 64 GiB address space, 16 GiB disk reserve |

The full earlier proposal and every exact/interval/refinement attempt stay in
the artifact. The harness writes a digest-bound training choice before computing
held-out metrics. Every admitted candidate then receives the unchanged original
split evaluation and a raw latent DFT/SVD/CE audit. Replay regenerates the inputs,
checks their original fingerprints, recomputes every proposal and certificate,
and compares full states, selection and independently recomputed metrics.
The launcher retains partial files and process failures and continues the
remaining planned tasks. A summary requires a run outcome for every task and
replay seals for scored outcomes. Failed tasks retain all three mode references.

This is one forward construction per task. Its three mode references are
explicitly not three gradient training runs. Integration with the original 105
choices, all 35 short/long tasks and the conventional transformer waits for
complete verified parent evidence. No original result has been read at this
protocol freeze. CM23 proves local descent, not successful global discovery or
general unseen-label identification.

Four native harness tests use synthetic inputs, including success, total
proposal refusal, altered states with recomputed digests, invalid inventory,
oversized files, overwrite refusal and all-failed summary coverage. They do not
run original-data discovery. The parent library has 26 passing native checks.

From the repository root:

```sh
cargo fmt --manifest-path experiments/convex_stress_density/cohort/Cargo.toml -- --check
prlimit --as=68719476736:68719476736 -- cargo test --locked --offline --manifest-path experiments/convex_stress_density/cohort/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --locked --offline --manifest-path experiments/convex_stress_density/cohort/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --locked --offline --manifest-path experiments/convex_stress_density/cohort/Cargo.toml
# After committing, preserve this binary and its digest outside the worktree,
# set DENSITY_BINARY to that immutable copy and DENSITY_PRODUCER_COMMIT to HEAD.
bash experiments/convex_stress_density/cohort/run_cohort.sh experiments/convex_stress_density/runs/original-v1
```

Each destination must be fresh. `fingerprint`, `prepare ROOT`, `run-cell ROOT
TASK`, `replay-cell ROOT TASK` and `summarize ROOT` are native CLI subcommands.
The policy and all production source dependencies enter the compiled source
fingerprint. Generated reports remain ignored under the root experiment runs.
The inventory also binds the running binary's SHA-256; replay uses the preserved
frozen binary and rejects an inventory naming a different executable.
