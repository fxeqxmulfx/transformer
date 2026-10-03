# Complete 300,000-update sparsemax candidate

The first case of the prospectively frozen normalizer pair is complete.
It retains native AdamW, GPTMini with 436,104 parameters, mod-193 division,
the 25% split (9,264 train / 27,792 exhaustive held-out equations), model/data
seeds 0/0, batch 512 with 48-example epoch tails, LR 0.0003, decay 0.1,
warmup 10, no clipping and all original phase/persistence criteria.
Only the attention normalizer changes to the existing causal-simplex Euclidean
projection. This is an exploratory single case, not independent confirmation.
The fresh paired softmax control is running its full frozen budget.

| Complete candidate measurement | Value |
| --- | ---: |
| Completed updates | 300,000 |
| Canonical / neighbor / tensor-diagnostic observations | 1,201 / 2,400 / 3,600 |
| Dense pre-update gradient observations | 300,000 |
| Final train / held-out complete-RHS accuracy | 100% / 34.056563% |
| Correct held-out equations | 9,465 / 27,792 |
| Final held-out EOS accuracy | 100% |
| Final held-out answer CE | 5.221931 nats |
| Required memorization plateau | 49,750–255,750, 825 observations |
| First held-out 99% / long joint confirmation | Neither occurs |
| Failed final-window observations, 250,000–300,000 | 201 / 201 |
| Minimum final-window held-out accuracy | 2.511514%, at 250,750 |
| Stable grokking / persistent final performance | Both false |
| Training / wall / tensor-diagnostic seconds | 8,771.17 / 12,503.70 / 44.72 |
| Actual training-example exposures / epochs | 146,273,904 / 15,789.4974 |
| Peak CUDA allocated / reserved bytes | 119,802,880 / 157,286,400 |

Late held-out improvement from 2.723805% at 250,000 to 34.056563% at the final
step remains visible. It does not reach the frozen 99% target within the
user's update limit. The failure is on numeric answers; EOS is fully correct.
A valid memorization plateau alone does not establish subsequent generalization
or persistent grokking. The maximum pre-update gradient L2 is 120.361588 at
290,160, in a full 512-example batch. These gradients do not identify a cause.

The five fixed 10,000-update tail bins have failures 40/40, 40/40, 40/40,
40/40 and 41/41. With no first long joint confirmation, post-confirmation
episode counts, recovery durations and onset-frequency rates are unavailable,
not zero; the whole post-onset timeline has null metrics. The final sampled
99% streak has zero observations. The empty standard `collapse-neighbors`
figure has zero supported post-confirmation triplets; all 2,400 actual
neighboring evaluations remain in the archive.

The [complete portable archive](../../baselines/adamw_attention_adamw-sparsemax_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002/summary.json)
contains the frozen plan, measurements, all raw logs and matching CSV files,
checksums and two standalone PNG/PDF figures. System Python recomputes the
report without importing Torch. Original measurement/log bytes match the
archive, and canonical JSONL matches the report. All 43 frozen Python,
seventeen native training, nine Lean and two manuscript fingerprints, plus
the prior Lean audit receipt, remain unchanged.

The [additional metrics and review receipt](attention-sparsemax-complete-metrics/validation.json)
preserve both complete recovery derivations, the verification program and
independent native checkpoint audit. On CPU, the actual final model and native
moments are finite; all eleven parameter states have step 300,000, no maximum
buffer, betas (0.9, 0.98), epsilon 1e-8, LR 0.0003 and all-parameter decay 0.1.
Each trainable parameter appears once. The inspection performs no optimizer
update and initializes no CUDA context. Both PNGs and their actual PDF
rasterizations have been visually reviewed.

```bash
python3 -m experiments.synthetic_trainers.stability_report \
  experiments/synthetic_trainers/baselines/adamw_attention_adamw-sparsemax_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002 --verify
```

Transfer the ignored raw model/optimizer checkpoint separately for weight
inspection or the retained CPU audit. Its SHA256 is
`c3f296b2f9ffd388451f4beefb358b184ad32da404658424c6685c9a935ac601`.
The [earlier routing probe](sparsemax-routing-gradient-probe.md) uses different
weights from update 280,000; do not substitute the final checkpoint. Lean's
fixed-score convex row result does not establish joint-training success.

The scientific pair remains incomplete. Complete all 300,000 softmax updates,
verify/review/commit the whole pair and consume both terminal sessions.
No timing ratio or repeatable architecture improvement is certified here.
The all-six primary benchmark gate and later declared studies remain required.
