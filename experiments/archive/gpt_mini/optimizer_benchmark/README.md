# Eighteen optimizers at a fixed budget

The first optimizer comparison of the gpt_mini codebase, on an RTX 3050
Laptop GPU. [GPTMini](../gpt_mini.py) of width 128, with two layers of four
heads, an FFN of width 512 and a context of 64 characters, trains on the
Tiny Shakespeare characters (split 90/5/5 in order) in float32, under
softmax attention and again under causal sparsemax. Eighteen methods
compete: SGD, AdaGrad, Adam, AdamW, AMSGrad under three momentum schedules,
AdamX, AdamNC, Muon and safeguarded Muon, DASH with four inverse-root
solvers and safeguarded DASH-NDB, AdaFisher and AdaFisherW. Each trains at
the rate its validation loss selected from three (250 updates, seed 0),
then for 1000 updates on seeds 0, 1 and 2; the ranking is the mean test
cross entropy of the last iterate.

AdamW had the lowest mean under both attentions: 1.76013 ± 0.01341 under
softmax and 1.78993 ± 0.00632 under sparsemax; the report claims no
general ordering from margins this close. Both safeguards accepted no
proposal and trained as SGD.

- [`PLAN.md`](PLAN.md): the protocol, declared before training.
- [`results/rtx3050/REPORT.md`](results/rtx3050/REPORT.md): both rankings;
  the raw runs (`runs.jsonl`), the checks and the provenance beside it.
- [`results/pilot_unshifted_sparsemax/`](results/pilot_unshifted_sparsemax):
  the first launch, stopped when a test found cancellation in its sparsemax,
  and excluded from the ranking.
- `results/tests_cpu.log`, `results/tests_gpu.log`: the tests run before
  training.

[`Transformer.OptimizerBenchmark`](../../../../src/Transformer/OptimizerBenchmark.lean)
encodes `results/rtx3050/runs.jsonl` in Lean
([`scripts/optimizer_benchmark_data.py`](../../../../scripts/optimizer_benchmark_data.py))
and proves AdamW's lead in the logged means (`adamw_mean_winner`) and that
a safeguard rejecting every proposal trains as SGD on the same batches
(`guardedBatchRun_eq_sgd`).
The [Magma benchmark](../magma_benchmark) extends the comparison to 24
methods.
