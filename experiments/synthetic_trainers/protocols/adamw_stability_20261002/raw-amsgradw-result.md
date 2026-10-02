# Complete paired raw AMSGradW control: negative phase and persistence

`amsgradw-short-lr001` completed all 150,000 frozen updates. Final exhaustive
train/held-out accuracy is 100%/99.9785%, with one incorrect held-out numeric
answer. The required low-held-out memorization plateau is absent, and ten
of 201 observations fail the joint 99% target in the final 50,000 updates.
The final rebound therefore does not meet the prospective persistence gate.

The [complete archive](../../baselines/adamw_stability_raw_amsgradw_short_lr001_seed0_data0_20261002/summary.json)
retains all 601 canonical observations, 1,200 exhaustive neighbor probes,
1,800 full tensor diagnostics and 150,000 pre-update gradient records, with
complete CSV tables and standalone PNG/PDF figures. Both the
[overview](../../baselines/adamw_stability_raw_amsgradw_short_lr001_seed0_data0_20261002/plots/stability-overview.png)
and [neighbor curves](../../baselines/adamw_stability_raw_amsgradw_short_lr001_seed0_data0_20261002/plots/collapse-neighbors.png)
were visually inspected. The [validation](raw-amsgradw-result-validation.json)
checks the archive without PyTorch, original log/measurement bytes, local
checkpoint SHA256, and all frozen source/manuscript fingerprints.

| Final-window update | Train accuracy | Held-out accuracy |
| --- | ---: | ---: |
| 102,250 | 87.9510% | 83.9347% |
| 107,250 | 88.3591% | 84.3643% |
| 112,250 | 87.2423% | 84.4502% |
| 117,250 | 93.2345% | 90.6357% |
| 123,000 | 89.8411% | 87.1778% |
| 128,000 | 98.4536% | 96.7354% |
| 132,500 | 58.9132% | 58.1400% |
| 137,750 | 93.3419% | 89.8840% |
| 143,750 | 95.4253% | 93.5567% |
| 148,500 | 95.7474% | 93.4064% |

191 of 201 final-window observations pass. First canonical held-out 99%
is at 5,250; the legacy two-observation confirmation is at 6,000. The first
long joint confirmation is much later, spanning 52,750–57,500 and costing
1,595.81 training / 1,692.87 wall seconds. These observed event costs are
retained, while eligible persistent-target timing is null with support zero.

After the legacy confirmation, 64 canonical observations fail held-out 99%.
All fail numeric answers and none fails EOS; 63 also fail at both immediate
neighbors. Every failure has neighbor support, none is isolated to its
canonical observation, and none recovers by the next update. This count
includes failures before the long confirmation. The largest dense gradient
norm is 56.1920 at update 74,350, a 48-example epoch tail; accuracy was not
evaluated at that update. These associations do not identify a cause.

The control has 423,816 parameters and consumes 69,840,000 examples. Full
costs are 4,256.08 training / 4,511.30 wall seconds, including 41.62 measured
diagnostic seconds. Peak CUDA allocation/reservation are
119,166,464/146,800,640 bytes. Final held-out answer/EOS cross-entropies are
0.0106473/0.000332590 nats. Costs describe this single instrumented GTX 1050
calibration, not a repeatable optimizer speedup.

This is the paired GPTMini adaptation of *Convexifying Transformers*,
Section 4, using the repository's raw AMSGradW: betas (0.9, 0.999), a maximum
second moment, no bias correction, epsilon 1e-8 and all-parameter decay 0.1.
It is distinct from native bias-corrected AdamW with `amsgrad=True`.
The complete primary AdamW calibration also fails phase and persistence.
Neither mod-97 case can open independent confirmation. The separately frozen
mod-193 AdamW adaptation is now running; confirmation and architecture work
still require the unchanged gates.
