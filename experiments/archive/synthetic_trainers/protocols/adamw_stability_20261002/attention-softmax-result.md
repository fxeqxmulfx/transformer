# Complete fresh 300,000-update softmax control

The fresh control of the frozen sparsemax-first pair completes its full budget.
It retains native AdamW, GPTMini with 436,104 parameters, mod-193 division,
25% training equations, model/data seeds 0/0, batch 512 with 48-example epoch
tails, LR 0.0003, decay 0.1 on every parameter, warmup 10 and no clipping.
The unchanged criterion requires every canonical observation in updates
250,000–300,000 to have joint train/held-out complete-RHS accuracy at least 99%.

| Complete control measurement | Value |
| --- | ---: |
| Completed updates | 300,000 |
| Canonical / neighbor / tensor-diagnostic observations | 1,201 / 2,400 / 3,600 |
| Dense pre-update gradient observations | 300,000 |
| Final train / held-out complete-RHS accuracy | 100% / 100% |
| Final held-out answer CE | 1.2687003352088038e-7 nats |
| Required memorization plateau | 4,500–27,250, 92 observations |
| First long joint confirmation | 95,750–100,500, 20 observations |
| Observed training / wall seconds at confirmation | 2,491.83 / 3,092.41 |
| Failed final-window observations | 2 / 201 |
| Minimum final-window held-out accuracy | 49.179620%, at 274,000 |
| Final sampled joint target streak | 275,750–300,000, 98 observations |
| Stable grokking / persistent final performance | Both false |
| Training / wall / tensor-diagnostic seconds | 7,466.26 / 9,260.63 / 85.35 |
| Actual training-example exposures / epochs | 146,273,904 / 15,789.4974 |
| Peak CUDA allocated / reserved bytes | 120,240,128 / 140,509,184 |

All six sampled post-confirmation failure episodes recover. Their first
failures are 105,250, 109,000, 110,750, 154,000, 274,000 and 275,500;
sampled updates until first recovery are respectively 500, 250, 1,250,
250, 250 and 250. The late held-out minima are 49.179620% and 81.768135%.
The five complete 10k final-window bins have failures 0/40, 0/40, 2/40,
0/40 and 0/41, with episode onsets 0, 0, 2, 0 and 0. Frequencies are not
monotonically decreasing: eleven complete quiet bins from 160,000 to 270,000
precede two new episodes. This is a finite descriptive observation, not an
estimate of future stability or a statistical population trend.

Recovery does not erase either frozen final-window failure. The first late
failure follows a failing short-batch neighbor and persists after full batches.
The second occurs at a canonical short batch, with a passing full-batch
before-probe and failing full-batch after-probe. After canonical recovery at
275,750, probe 275,751 fails again. EOS remains correct in these triplets.
The [preserved late witness](softmax-late-tail-recoveries.md) retains the exact
gradients and neighbors; no causal claim or continuous recovery follows.
The dense maximum gradient L2 is 1,727.980957 at 178,641 in a full batch.

Every one of the 1,201 non-time canonical records matches the frozen earlier
300k softmax reference with the same seeds and split. This is a successful
observed-metric A/A check, not an independent confirmation, a comparison of
all tensors, or a statement about unobserved intermediate dynamics.

The [portable archive](../../baselines/adamw_attention_adamw-softmax_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002/summary.json)
preserves all measurement and raw-log bytes, derived CSVs, hashes and two
standalone PNG/PDF figures. The [complete review receipt](attention-softmax-complete-metrics/validation.json)
retains the two recovery derivations, exact verification program and native
CPU audit. All eleven final parameter/moment states are finite at step
300,000, with the exact native AdamW group, no maximum buffer, betas
(0.9, 0.98), epsilon 1e-8, decay 0.1 and LR 0.0003. The audit performs no
optimizer updates and initializes no CUDA context. Both PNGs and actual PDF
rasterizations have been visually reviewed. All 43 Python, seventeen native
training, nine Lean, two paper and prior audit fingerprints remain intact.

```bash
python3 -m experiments.synthetic_trainers.stability_report \
  experiments/synthetic_trainers/baselines/adamw_attention_adamw-softmax_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002 --verify
```

The ignored raw final checkpoint must be transferred separately for weight
inspection; its SHA256 is
`516bf695bb98cefeef080ce3bc8e5e24ecb53bb3a861e94600c9351eccbbb0bb`.
Both original trainer and replacement archive-worker sessions have now
terminated with consumed exit code 0. The whole pair is complete and requires
its separate committed comparison/review receipt before any next scientific
freeze. No eligible persistent timing ratio or independent benchmark gate
is established by this control. All six fresh benchmark confirmations and
later architecture/complementary studies remain outstanding.
