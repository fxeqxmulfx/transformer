# Complete primary AdamW calibration: negative phase and persistence

`adamw-short-lr001` completed its frozen 150,000 updates. Final exhaustive
train and held-out accuracy are both 100%, but the recipe fails the prospective
stable-grokking gate: it has no required low-held-out memorization plateau,
and seven observations fail inside the final 100,000–150,000 window.
This is one exploratory calibration, not an independent confirmation.

The [complete archive](../../baselines/adamw_stability_adamw_short_lr001_seed0_data0_20261002/summary.json)
retains all 601 canonical observations, 1,200 exhaustive neighbor probes,
1,800 full tensor diagnostics and 150,000 pre-update gradient records,
their complete CSV tables, and standalone PNG/PDF curves. Both
[overview](../../baselines/adamw_stability_adamw_short_lr001_seed0_data0_20261002/plots/stability-overview.png)
and [collapse-neighbor plot](../../baselines/adamw_stability_adamw_short_lr001_seed0_data0_20261002/plots/collapse-neighbors.png)
were visually inspected. The [validation](adamw-result-validation.json)
confirms offline verification without PyTorch, unchanged original source
bytes, all frozen fingerprints and empty eligible timing support.

| Final-window update | Train accuracy | Held-out accuracy |
| --- | ---: | ---: |
| 104,000 | 90.0129% | 87.9940% |
| 108,000 | 98.1529% | 97.8737% |
| 125,250 | 84.6005% | 81.7440% |
| 129,000 | 45.5112% | 43.9003% |
| 141,000 | 77.1048% | 76.2672% |
| 143,750 | 55.4768% | 52.9639% |
| 146,500 | 60.1160% | 57.3024% |

194 of 201 final-window observations pass. The final rebound to 100% does
not remove these failures. First canonical held-out target is 1,250; the
first long joint confirmation spans 1,250–6,000, costing 181.81 training
and 194.91 wall seconds. These observed event costs remain available, but
eligible time-to-target is null with support zero because persistence fails.
Rapid early generalization is distinct from the absent memorization phase.

There are 13 canonical held-out failures after the legacy confirmation
(all also after long confirmation). All concern numeric answers; EOS fails
at none of those canonical points. All 13 have both immediate neighbors;
12 fail at both neighbors, none is isolated to the canonical observation,
and none recovers by the immediately following update. These sampled
associations do not identify a cause. The dense trace's largest norm is
160.7858 at 133,040 after a 48-example epoch tail; accuracy was not evaluated
at that update, so it is not a measured accuracy-collapse trigger.

The run consumes 69,840,000 examples and uses 423,816 parameters. Total costs
are 4,022.04 training and 4,284.18 wall seconds, with 44.04 measured diagnostic
seconds. Peak CUDA allocation/reservation are 117,470,208/144,703,488 bytes.
Costs include this frozen instrumentation and are descriptive on the GTX 1050;
no optimizer speed ratio is certified from a failed persistent target.

This GPTMini/AdamW adaptation of *Convexifying Transformers*, Section 4 uses
the prospectively specified betas (0.9, 0.98), bias correction, no maximum
buffer, learning rate 0.001 and all-parameter decay 0.1. The separately
[documented CSV runtime repair](csv-verification-repair.md) retains the
original frozen files and identical verification outcomes; it changes only
how often the common CSV column set is computed.

The paired raw AMSGradW control is now running its own full budget. Finish
and archive it, verify the complete comparison, then execute the already
frozen mod-193 AdamW task adaptation requested by the user. This mod-97
primary recipe cannot launch independent confirmation; no scientific
architecture comparison or complementary-task campaign has begun.
