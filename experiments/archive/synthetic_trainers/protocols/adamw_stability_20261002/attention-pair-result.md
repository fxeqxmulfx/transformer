# Complete frozen softmax/sparsemax comparison

Both fresh native AdamW runs complete 300,000 updates. Sparsemax runs first,
then the fresh softmax control, with the same corpus, seeds, 436,104-parameter
GPTMini, optimizer, batch stream and criterion. Only normalization changes.
This is one exploratory initialization and split; all failures remain archived.

| Measurement | Softmax | Sparsemax |
| --- | ---: | ---: |
| Final train / held-out complete-RHS accuracy | 100% / 100% | 100% / 34.056563% |
| Memorization plateau | 4,500–27,250 | 49,750–255,750 |
| First long joint 99% confirmation | 95,750–100,500 | None |
| Failed final-window observations | 2 / 201 | 201 / 201 |
| Stable grokking / persistent performance | Both false | Both false |
| Sampled post-confirmation episodes | 6, all recovered | Unavailable |
| Training / wall seconds | 7,466.26 / 9,260.63 | 8,771.17 / 12,503.70 |

Sparsemax's final held-out difference is −65.943437 percentage points.
Neither prospective descriptive improvement rule passes. Eligible persistent
time-to-target support is zero, and both ratios are null. Total execution costs
do not establish speed to a persistent target. Sparsemax's unavailable recovery
rate is not zero; softmax's recovered episodes do not repair its two tail failures.

The [portable full pair](../../baselines/adamw_attention_softmax_sparsemax_mod193_fraction25_lr0003_budget300k_pair_20261002/normalizer-pair.json)
preserves both full case archives, the byte-exact historical reference, plans,
outcome/frequency CSVs and comparison/recovery PNG/PDF figures. Each case has
1,201 canonical / 2,400 neighbor / 3,600 tensor-diagnostic / 300,000 dense
gradient observations. All duplicated case artifacts and original raw bytes
match. All 1,201 non-time softmax canonical records equal the earlier reference
with the same seeds/split: an observed A/A check, not independent replication
or tensor identity. Frequencies and recovery durations remain descriptive.

The [review/terminal receipt](attention-pair-result-validation.json) records
NoTorch recomputation, intact 43 Python/seventeen native training/nine Lean/
two manuscript/audit pins, both finite final native CPU audits, actual reviewed
PNG/PDF hashes and consumed exit-code-0 sessions 47567/25387, PIDs 139507/145323.
The [completion evidence](attention-pair-result-evidence/artifact-hashes.json)
retains the completed raw state, final trainer record and worker metadata/log.
New figures were reviewed as PNGs and rasterizations of the actual PDFs;
unchanged sparsemax/reference figures retain their earlier reviews.
Verification adds no optimizer updates and needs no raw checkpoints:

```bash
python3 - <<'PY'
from experiments.synthetic_trainers.attention_report import verify_pair
verify_pair('experiments/synthetic_trainers/baselines/adamw_attention_softmax_sparsemax_mod193_fraction25_lr0003_budget300k_pair_20261002')
PY
```

Lean proves causal-simplex Euclidean projection for fixed scores at their
original scale. It does not prove convex joint training, optimizer stability
or generalization. The measured negative sparsemax outcome respects that scope.

The stable primary gate remains closed. This complete failed control permits
the prepared conditional schedule calibration after a separate committed
freeze: fresh constant softmax first, then cosine annealing 150,000–250,000
to 10% rate, with the same 300,000-update cap and criterion per case. Native
AdamW's lower rate also reduces per-update decoupled decay; these effects are
not isolated. All six fresh crossed benchmark confirmations still precede
scientific architecture comparisons. No schedule execution or independent
confirmation is claimed by this result commit.
