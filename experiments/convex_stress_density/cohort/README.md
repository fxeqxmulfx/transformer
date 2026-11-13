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
complete verified parent evidence. The protocol was committed before original
discovery. CM23 proves local descent, not successful global discovery or general
unseen-label identification.

Four native harness tests use synthetic inputs, including success, total
proposal refusal, altered states with recomputed digests, invalid inventory,
oversized files, overwrite refusal and all-failed summary coverage. They do not
run original-data discovery. The parent library has 26 passing native checks.

## Frozen original-v1 findings

Producer `3311e13ce3d38123fc231f3d111f3cc7ee3fddc0`, source fingerprint
`b624c37ce670f33a2c65566ffd7c039f8c147b044bced4ae9684a9f4eb9e39b4`, and binary
`b804e357ec6553861de0a2e754494fe8e1db92124ef33607a853c3ceb0115257` produced
all nine terminal outcomes. Every task replays, including division's explicit
ineligible outcome; no process or replay failure occurred. The summary digest is
`b19e294991bb6cfe7f1fb5120dc48d10ac66d0cb16197cee3acfa699dcb07739`.

| Original task | Selected observed errors | Primary sequence accuracy | Outcome |
| --- | ---: | ---: | --- |
| Permuted addition | 0 / 4,704 | 100% | All splits pass |
| Addition | 0 / 4,704 | 100% | All splits pass |
| Subtraction | 0 / 4,704 | 100% | All splits pass |
| Division | No eligible model | — | Refusal retained |
| Multiplication | 1,198 / 4,704 | 72.60% | Below threshold |
| Sum-cube | 2,523 / 4,704 | 42.91% | Below threshold |
| Sum-power-four | 3,758 / 4,704 | 7.40% | Below threshold |
| Absolute difference | 4,061 / 4,704 | 3.78% | Below threshold |
| Blind absolute difference | 2,500 / 4,704 | 41.34% | Below threshold |

The result is **3/9 primary and all-split passes, with no improvement over the
earlier decoded proposal**. All eight selected models are `prior-0`; their
refined copies tie. The complete reports retain 54 exact slots, 162 interval
slots and 224 refinement slots. Every exact attempt refuses before an update;
every interval attempt reaches its 4,096-iteration cap, totaling 663,552
iterations. Eight seeds reach the refiner; twelve additional assignment solves
produce no improvement. All sixteen admitted vertices receive raw latent/CE
audits and original metrics. Capped iterates do not certify mathematical
infeasibility of the interval domain.

Division's sixteen earlier trials fail at the power/phase-clock step. Its
training observations cover all 97 columns but only 96 rows, exposing a missing
axis that the current full-phase clock refuses. Across the other tasks the
estimated feature rows also have large interval violations; fixing a small
masked fixture did not establish a usable feature model at this observation
scale. The next model work must address feature balancing and uncovered axes
while retaining these negative outcomes. Successful numerical admission would
still need observed-label refinement and separate held-out evaluation.

Peak RSS over construction and replay is 47,000 KiB (about 46 MiB), under the
64 GiB cap. Each construction takes about 1.6–7.9 seconds with one BLAS thread.
All 27 mode references remain, explicitly as references to the nine forward
constructions. The all35 comparison and conventional transformer result remain
pending; the better existing decoder-bank results are not replaced by this
standalone cohort.

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
