# CUDA compile for the validation-stopped optimizer comparison

The user requested CUDA compile during the 24-method patience experiment.
The eager process was stopped without deleting its completed result, curves
or checkpoints. Its sources/results remain separate and unchanged.

- [x] Write CUDA compile correctness, routing and hook tests before training.
- [x] Try `torch.compile` with Inductor, `reduce-overhead`, CUDA Graphs.
- [x] Verify that CUDA graphs are actually recorded, not merely configured.
- [x] Compare eager/compiled losses, gradients, optimizer steps and Fisher factors.
- [x] Measure warmed GPU throughput and peak memory on the actual model.
- [x] Check recompilation over 1220 updates and five full validations per case.
- [x] Test stopping and exact best-checkpoint restore/reload on both attentions.
- [x] Choose the tested backend before restarting all 24 methods.
- [x] Launch all 24 methods after all 80 CPU/CUDA tests passed.
- [ ] Finish softmax/Sparsemax, seeds0,1,2, with the existing patience settings.
- [ ] Audit best-checkpoint selection, source/data hashes and reported results.

Keep float32 and TF32 disabled. Use the original parameter objects and names:
an `_orig_mod.` prefix would incorrectly route Muon's tied embedding and
exclude Magma's hidden matrices. Compile the module in place. Optimizer
steps, CPU mask sampling, guards and validation stopping remain outside
captured model forward/backward graphs.

Sparsemax needs a capture-compatible adapter: retain the same centered
projection and support Jacobian on finite nonempty causal rows. Signal
invalid rows as NaN instead of a Python exception inside a CUDA graph;
the training loop detects the nonfinite loss and restores the previous best.
Test this extension, including simplex constraints, gradients, causality and
large score translations. Original source files are not modified.

AdaFisher's hooks must collect fresh current-input/output-derivative factors
on every step. Do not silently drop hooks or reuse stale captured values.
Try partial compilation for its hook-bearing graphs and record any required
fallback explicitly. Never call a flag alone evidence of CUDA graph execution.

Compiler/Triton caches and best checkpoints live under ignored
`experiments/runs/`. Record cold compilation separately from warmed step
time. A compiled comparison has its own protocol and output directory;
do not combine it with eager results for ranking.

## Preflight results

Six analytic/CUDA tests passed, including fresh compiled checkpoint reloads
for Magma+Muon and AdaFisher on both attentions. The complete launch also runs
all 74 previous CPU/CUDA tests before its first optimizer run (80 total).

Actual-model warmed speedup: AdamW 2.02x/2.03x, Magma+Muon
1.46x/1.45x, AdaFisher 1.06x/1.45x (softmax/Sparsemax). Compiled allocated
peaks were 96--116 MiB; optimizer steps are still eager. See
`results/preflight/throughput.json` for cold costs and paired measurements.

Recompilation telemetry covers AdamW, Magma+Muon and AdaFisher on both
attentions, 1220 updates and five validations per case. Training has one
initial FX graph in fullgraph mode. First validation adds the no-grad batch32
and tail batch7 variants. After each mode/shape warms and records, three
additional train/validation cycles produce no new traces or CUDA graph nodes.
Counts stabilize at FX3/CUDA4 for AdamW/Magma+Muon and FX22/CUDA91 for
AdaFisher's partial graphs. `results/preflight/recompiles.log` contains the
guard failures; `recompilation.json` contains the before/after counters.

Preflight emitted `Not enough SMs to use max_autotune_gemm mode` on the
RTX3050, without CUDA Graph execution errors. Hook graph breaks occur once
while AdaFisher's partial compilation is assembled, not on every training
step. Every long run records its own evaluation-time compiler counters.

## Full run

New output: `results/rtx3050_all/`. The old eager completed AdaGrad result and
unfinished second run remain under `experiments/patience_benchmark/` and
ignored `experiments/runs/patience_benchmark/`.
The full compiled launch passed all80 tests. Its live state is in
`results/rtx3050_all/progress.log`; each completed run updates `REPORT.md`,
`runs.jsonl` and both summary formats. Do not treat the partial leaderboard
as the final 24-method result.

Use the unchanged stopping defaults: checks every250, patience8,
min_delta=.0001, minimum1000, divergence+.1 CE for3 checks, cap20000.
Keep the prior validation-selected rates, paired seeds0/1/2 and minibatch
prefixes. Initialize/reset the compiler per model with persistent disk cache;
keep cold compilation included in reported total/training seconds and record
first train/inference forward costs separately.

```bash
.venv/bin/python -u -m experiments.compiled_benchmark --all
.venv/bin/python -m experiments.compiled_benchmark --all --validate-only
```
