# Original CM46 local-mixture results

CM46 retains 35/35 original primary and auxiliary successes in full-batch,
minibatch and forward-only modes, for both initializer policies. Full and mini
improve Fibonacci raw accuracy, but validation CE increases in every mode.
The earlier CM45 shared mixture therefore remains the better CE result.
All 35 original cases and 35 replays finish with exit0; every changed parent
metric reproduces exactly. Independent complete reproduction follows the
authored findings commit before task296 approval.

| Fibonacci mode, 4704 validation records | CM45 raw errors | CM46 raw errors | CM45 CE | CM46 CE |
| --- | ---: | ---: | ---: | ---: |
| Full batch | 109 | 101 | 0.1401069095 | 0.1726145666 |
| Minibatch | 103 | 100 | 0.1419598638 | 0.1751058800 |
| Forward only | 100 | 100 | 0.1490438375 | 0.1772798206 |

The respective new raw correct counts are 4603, 4604 and 4604. Original field
accuracy stays 4701/4704, or 99.9362%, in every mode. Observed CE remains
9.626594307e-9. Both policies contain the same three changed choices and
102 inherited choices. No original primary or aggregate raw-accuracy
regression occurs; all three unique changed endpoints have a CE regression,
reported separately in both policies. Every other task and all baselines
remain in the complete summary.

The fixed gate is each existing latent block; the anchor is the reviewed
CM45 shared mixture, with radius one half and the same 91-map bank. For the
prior component, local initialization changes 91 groups in full/mini and 89
in forward. Full retains step11; mini retains initialization. Their final
source-excluded diagnostic CEs are 0.04052040, 0.04438710 and 0.04835203,
respectively. These lower training diagnostics do not translate into lower
validation CE. This is consistent with excessive local flexibility; the
experiment does not establish a unique cause or a generalization guarantee.

Across the six prior/supplement components, there are 288 new gradient batches,
182 optimizer updates and 7887 local-row updates. Full contributes 128 batches,
22 updates and 781 changed rows; mini contributes 160 batches, 160 updates and
7106 changed rows. Forward uses zero gradients and optimizer updates. Total
gradient-record visits are 677376; forward-scoring visits are 7366464, retaining
original multiplicities, all candidate evaluations and the 96-record mini tail.
All inherited discovery and earlier optimizer costs remain separately visible.

All 35 conventional transformer endpoints and 336 depth outcomes are retained,
without retraining. Historical DiffGPT's recorded three-seed 35/35 remains a
separate result. The frozen conventional Fibonacci endpoint has 95.1318% raw
accuracy and CE0.234544, compared with CM46's 97.85-97.87% and CE0.173-0.177.
This comparison does not claim matched training compute or a new DiffGPT run.

Preparation takes 16.22 seconds. All original cases total 86.37 seconds and
their replays 132.79 seconds; summary generation takes 2.73 seconds. Peak
observed RSS is 2098864 KiB during the original cases and 3251716 KiB, about
3.10 GiB, during replay. The complete eligible case forecasts 22243261440
buffer bytes and 196129411584 scalar-work units across all retained components
and metrics. Every process uses the 64 GiB address-space limit. Timing is an
observation on this run, not a speed comparison.

The first detached launcher disappeared before creating an input-check log,
freeze or original output directory; no original case began. Its PID/status
files and the unknown termination status are retained under task296's
`launch-v1-interrupted`. The unchanged script then ran through a persistent
exec session. The completed inventory includes 332 hashed files, including
preparation outputs and the compact report. No failed numerical case was removed.

Reproduction uses producer `1b42f41136dfa55a93e4cbcd45d0549f1f4e3f17`, reviewed
component `a8deaa63`, Lean `c60816fa`, and parent findings `f0a19727`.
The frozen source fingerprint is
`87c9fdf6af27264a1686025337a5add5ee48daf38b4cec2452aecc5cbfa5ed6e`;
the executable hash is
`8472b0bca6f1dc2bd07fa8aecfc726d402f4bf99dc75410403f1ee073d29474b`.
The original inventory hash is
`5ca5956bf64c6cbd14bea6a1d6694d555bef6821b6ab4c87d19910f443ee573b`;
the summary hash is
`04400a75ef233b5a12d65589ad5e1f7dcb731e53b3d2d660b98c464b4bc60d07`;
the report hash is
`21fa5d07222ff21901310d00116a33574938ed9c51c41a241c43f19fc8a03c57`.

Local evidence is on storage under `transformer-runs/gated-context-compare/`;
the executable, request and source snapshot are under
`transformer-runs/freezes/gated-context-compare-v1/`. The commands in the
parent README regenerate the complete comparison and report. Exact-commit
review must run a fresh complete preparation, all cases and replays, compare
every JSON artifact and report, and rerun source/native/Lean/index/axiom checks.

The next model change should constrain local flexibility or use stronger
training-only validation of it, while keeping ordinary CE and the proved
complex bounds. A smaller shared set of fixed gates is one testable option.
The current result is retained in full; it is not an across-metric improvement
over the shared CM45 model.
