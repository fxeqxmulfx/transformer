# Independent AdamW / GPTMini confirmation

Setting: mod-7 division, adapting *Convexifying Transformers*, Section 4.
All 6 prospectively frozen repeats completed 120 updates.
Stable grokking: 0/6. Repeatable benchmark: False.

| Initialization / data seed | Final held-out | Long confirmation, training s | Tail failures | Stable grokking |
| --- | ---: | ---: | ---: | --- |
| 4 / 2 | 14.2857% | Not reached | 3 | False |
| 5 / 2 | 14.2857% | Not reached | 3 | False |
| 6 / 2 | 4.7619% | Not reached | 3 | False |
| 4 / 3 | 23.8095% | Not reached | 3 | False |
| 5 / 3 | 9.5238% | Not reached | 3 | False |
| 6 / 3 | 19.0476% | Not reached | 3 | False |

Measured target timing support is 0/6.
Missing events remain empty. The CSV retains each outcome, phase flag, exposure, costs and memory.
New initialization seeds are crossed with new data splits; these are not independent IID runs.
Means and sample SD are descriptive. They do not establish a confidence interval or universal speedup.
The unchanged calibration criterion applies to every complete trajectory, including all later failures.
All repeats must pass before an architecture comparison is ready. A final rebound or first target does not replace persistence.
The complete calibration evidence is included; checkpoints remain local and their fingerprints are retained.
Finite scheduled observations do not identify an internal algorithm, a cause, or behavior beyond the budget.

Next action: `retain_CPU_pipeline_fixture; no_scientific_learning_claim`.

Scientific run: False. Explicit CPU fixture: True.
