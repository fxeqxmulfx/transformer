# Compile the complete GPU training step

The user requested compilation of the entire computational path, including
optimizer updates. The previous model-only compiled job was interrupted; its
six completed runs, partial DASH curve and best checkpoints are preserved.

- [x] Write eager/full-step contracts for all24 optimizers before benchmarking.
- [x] Use a functional GPTMini forward/loss and compiled `torch.func.grad_and_value`.
- [x] Include every optimizer's moments, DASH roots/grafting, Muon auxiliary
      Adam, global guards, decay and Magma scales/masks in `fullgraph=True`.
- [x] Replace changing Python step integers with device tensor counters.
- [x] Preserve the original CPU Magma mask stream by precomputing it and
      indexing the transferred plan inside the compiled step.
- [x] Obtain AdaFisher's genuine fresh activation/derivative factors through
      zero perturbations at each Linear output, differentiated with the loss.
      Check against the frozen hook implementation before claiming equivalence.
- [x] Compile input-window gathering and validation loss accumulation too.
- [x] Verify losses, gradients, updates, all dense states, masks, Fisher factors,
      tied-weight routing, checkpoint reload and steady compilation counters.
- [x] Measure real CUDA Graph recording, warm speed and cold compilation cost.
- [x] Launch the separate all24 protocol after all85 CPU/CUDA tests passed.
- [x] Finish the two-attention, three-seed patience protocol and checkpoint audits.

Logging, file I/O, checkpoint serialization and host decisions about stopping
are Python orchestration. All tensor computations in each GPU training and
validation batch must pass through a full compiler graph. Do not claim that
Torch compiles filesystem operations or hide eager tensor fallbacks.

Keep all previous measured Python packages unchanged. New sources/output are
under `experiments/full_compile_benchmark/`; checkpoints/caches stay under
ignored `experiments/runs/`. Keep float32, deterministic algorithms, TF32 off,
the previous validation-selected rates and the existing patience defaults.

The all24 job finished on 2026-10-01 under `results/rtx3050_all/`: all 144 runs
and all 144 compiled CUDA checkpoint audits passed. `validation.json` records
the replayed stopping decisions, paired initializations/minibatch streams,
validation-only checkpoint selection and unchanged previous measurements.
There were 118 patience stops, 26 budget caps, no failures and no recovered
runs. The source and preflight fingerprints match the recorded protocol.
Completed runs store evaluation-time compiler counters in `runs.jsonl`:
every run has FX4 and zero graph breaks; CUDA Graph counts are four, or five
for DASH EVD. The earlier fixed-budget and partial model-only compiled
results remain separate and unchanged.

## Final comparison

AMSGrad without weight decay has the lowest mean best-checkpoint test CE on
both attentions. All entries below use three seeds; the spread is sample SD.

| Attention | Method | Test CE mean +/- SD | Mean training seconds |
| --- | --- | ---: | ---: |
| Softmax | AMSGrad | 1.627871 +/- 0.000749 | 61.3 |
| Softmax | Adam | 1.629838 +/- 0.003155 | 62.3 |
| Softmax | AdamNC | 1.630171 +/- 0.006281 | 68.8 |
| Sparsemax | AMSGrad | 1.646002 +/- 0.017073 | 57.9 |
| Sparsemax | AdamNC | 1.647939 +/- 0.011943 | 73.8 |
| Sparsemax | DASH EVD | 1.655525 +/- 0.007876 | 227.0 |

The differences between AMSGrad and the runner-up are about 0.002 CE;
three seeds do not establish a reliable advantage across initializations.
This is the result of the recorded early-stopping protocol, rather than a
general convergence claim about minibatch GPT training. The complete all24
rankings are in [REPORT.md](results/rtx3050_all/REPORT.md), and checkpoint
audit results are in [validation.json](results/rtx3050_all/validation.json).
Both guarded methods accepted zero proposed directions in every run; their
recorded test results match the paired SGD cases.

## Verified results

Four preflight tests covered all24 methods on both attentions, their gradients,
updates and every original state buffer, plus paired mask streams and modern
Sparsemax reverse AD. Eight train/validation configurations had zero graph
breaks and no growth in traces/recordings after mode/shape warmup. A separate
training test exercised both attentions and Magma+Muon/AdaFisher stopping,
serialization and fresh exact best-checkpoint re-evaluation. All passed.

Actual model: 2 layers, 4 heads, width128, FFN512, context64, batch32,
65 characters, float32. Six methods on both attentions were paired against
their frozen eager implementation: 20 warmups and 200 timed updates each.
Every full compiled case retained FX1, zero graph breaks and fixed CUDA
Graph counts (one node, two for EVD). Full-step warm speedups:

| Method | Softmax | Sparsemax |
| --- | ---: | ---: |
| AdamW | 1.97x | 2.30x |
| Magma+Muon | 3.00x | 3.37x |
| guarded DASH NDB | 2.73x | 2.80x |
| DASH EVD | 1.74x | 1.82x |
| DASH Chebyshev | 2.69x | 2.72x |
| AdaFisher | 2.77x | 2.97x |

Cold first full-step costs: 13--71 seconds. The warm comparisons exclude
these costs; the main benchmark's train/total timing includes them.
`Not enough SMs to use max_autotune_gemm mode` is the remaining preflight
warning. Mutable-input CUDA Graph skips were eliminated by marking stable
state/gradient/parameter addresses. Tensor sets were replaced by equivalent
dictionary membership before tracing Magma. No eager numerical fallback is
permitted: compilation uses `fullgraph=True` for all methods.

```bash
.venv/bin/python -u -m experiments.full_compile_benchmark --all
.venv/bin/python -m experiments.full_compile_benchmark --all --validate-only
```
