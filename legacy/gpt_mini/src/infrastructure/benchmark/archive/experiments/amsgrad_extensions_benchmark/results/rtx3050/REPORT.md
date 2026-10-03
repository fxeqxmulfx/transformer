# AMSGradW and AMSGradMD on GPTMini

Completed new runs: 18/18. All24 baseline: 144/144.

The table uses the exact best validation checkpoint of each run.
Test CE is mean +/- sample SD over three seeds; lower is better.
The main ranking includes every usable method, including original MD.

Float32, RTX3050, full compiled steps and CUDA Graphs; the original
paired data/models/patience settings are preserved. Training seconds
include cold compilation and exclude validation/checkpoint time.

## New variants and the existing AMSGrad reference

| Attention | Method | Test CE mean +/- SD | Rate | Updates mean | Train seconds mean |
| --- | --- | ---: | ---: | ---: | ---: |
| softmax | amsgradw | 1.625375 +/- 0.001731 | 0.0003 | 14000 | 60.2 |
| softmax | amsgrad | 1.627871 +/- 0.000749 | 0.0003 | 14333 | 61.3 |
| softmax | amsgradmd | 1.629636 +/- 0.012977 | 0.0003 | 13500 | 63.4 |
| softmax | amsgradmd_guarded | 1.635253 +/- 0.001809 | 0.1 | 13583 | 68.2 |
| sparsemax | amsgradw | 1.638935 +/- 0.016796 | 0.0003 | 13500 | 59.9 |
| sparsemax | amsgrad | 1.646002 +/- 0.017073 | 0.0003 | 12917 | 57.9 |
| sparsemax | amsgradmd | 1.647135 +/- 0.011880 | 0.0003 | 13833 | 68.0 |
| sparsemax | amsgradmd_guarded | 1.655872 +/- 0.018814 | 0.1 | 16417 | 85.1 |

Rates are: the W rate, raw MD direction rate, and guarded MD
fallback tau, respectively. MD gain rate is0.001; auxiliary rate is0.0003.
The guarded proposal itself uses direction rate0.0003. See PLAN.md.

## Guard behavior

| Attention | Seed | Accepted proposals | Acceptance fraction | Halved zero steps |
| --- | ---: | ---: | ---: | ---: |
| softmax | 0 | 60 | 0.003636 | 0 |
| softmax | 1 | 79 | 0.006723 | 0 |
| softmax | 2 | 55 | 0.004400 | 0 |
| sparsemax | 0 | 43 | 0.002774 | 0 |
| sparsemax | 1 | 34 | 0.002000 | 0 |
| sparsemax | 2 | 26 | 0.001552 | 0 |

## All measured methods: softmax

| Rank | Method | Seeds | Test CE mean +/- SD | Stop reasons |
| ---: | --- | ---: | ---: | --- |
| 1 | amsgradw | 3 | 1.625375 +/- 0.001731 | {'patience': 3} |
| 2 | amsgrad | 3 | 1.627871 +/- 0.000749 | {'patience': 3} |
| 3 | amsgradmd | 3 | 1.629636 +/- 0.012977 | {'patience': 3} |
| 4 | adam | 3 | 1.629838 +/- 0.003155 | {'patience': 3} |
| 5 | adamnc | 3 | 1.630171 +/- 0.006281 | {'patience': 3} |
| 6 | rmsprop | 3 | 1.632257 +/- 0.009166 | {'patience': 3} |
| 7 | amsgradmd_guarded | 3 | 1.635253 +/- 0.001809 | {'patience': 3} |
| 8 | dash_ndb | 3 | 1.641189 +/- 0.004554 | {'patience': 3} |
| 9 | dash_cn | 3 | 1.643084 +/- 0.009876 | {'patience': 3} |
| 10 | dash_evd | 3 | 1.643382 +/- 0.009595 | {'patience': 3} |
| 11 | dash_chebyshev | 3 | 1.643744 +/- 0.010546 | {'patience': 3} |
| 12 | dash_ndb_guarded | 3 | 1.644122 +/- 0.007724 | {'patience': 3} |
| 13 | sgd | 3 | 1.644122 +/- 0.007724 | {'patience': 3} |
| 14 | muon_guarded | 3 | 1.644122 +/- 0.007724 | {'patience': 3} |
| 15 | adafisherw | 3 | 1.649149 +/- 0.006353 | {'max_steps': 2, 'patience': 1} |
| 16 | muon | 3 | 1.650845 +/- 0.004786 | {'patience': 3} |
| 17 | adamw | 3 | 1.651835 +/- 0.013151 | {'patience': 3} |
| 18 | magma_adam | 3 | 1.653967 +/- 0.007823 | {'patience': 3} |
| 19 | adafisher | 3 | 1.660088 +/- 0.009782 | {'max_steps': 2, 'patience': 1} |
| 20 | magma_rmsprop | 3 | 1.661038 +/- 0.013209 | {'patience': 2, 'max_steps': 1} |
| 21 | adagrad | 3 | 1.666385 +/- 0.012950 | {'patience': 3} |
| 22 | magma_muon | 3 | 1.669633 +/- 0.013691 | {'patience': 3} |
| 23 | magma_adamw | 3 | 1.676261 +/- 0.010415 | {'patience': 3} |
| 24 | magma_sgd | 3 | 1.687310 +/- 0.003103 | {'patience': 2, 'max_steps': 1} |
| 25 | adamx | 3 | 1.703904 +/- 0.001723 | {'max_steps': 3} |
| 26 | amsgrad_geometric | 3 | 1.763921 +/- 0.018316 | {'max_steps': 3} |
| 27 | amsgrad_inverse | 3 | 1.847515 +/- 0.053071 | {'max_steps': 3} |

## All measured methods: sparsemax

| Rank | Method | Seeds | Test CE mean +/- SD | Stop reasons |
| ---: | --- | ---: | ---: | --- |
| 1 | amsgradw | 3 | 1.638935 +/- 0.016796 | {'patience': 3} |
| 2 | amsgrad | 3 | 1.646002 +/- 0.017073 | {'patience': 3} |
| 3 | amsgradmd | 3 | 1.647135 +/- 0.011880 | {'patience': 3} |
| 4 | adamnc | 3 | 1.647939 +/- 0.011943 | {'patience': 3} |
| 5 | dash_evd | 3 | 1.655525 +/- 0.007876 | {'patience': 3} |
| 6 | amsgradmd_guarded | 3 | 1.655872 +/- 0.018814 | {'patience': 3} |
| 7 | dash_chebyshev | 3 | 1.658134 +/- 0.017073 | {'patience': 3} |
| 8 | adafisherw | 3 | 1.658383 +/- 0.008424 | {'patience': 2, 'max_steps': 1} |
| 9 | adam | 3 | 1.659101 +/- 0.015078 | {'patience': 3} |
| 10 | rmsprop | 3 | 1.659338 +/- 0.016670 | {'patience': 3} |
| 11 | magma_rmsprop | 3 | 1.659435 +/- 0.020650 | {'patience': 3} |
| 12 | adamw | 3 | 1.661099 +/- 0.008038 | {'patience': 3} |
| 13 | dash_ndb | 3 | 1.661942 +/- 0.015671 | {'patience': 3} |
| 14 | dash_cn | 3 | 1.662389 +/- 0.003119 | {'patience': 3} |
| 15 | magma_muon | 3 | 1.663846 +/- 0.013197 | {'patience': 3} |
| 16 | adafisher | 3 | 1.663897 +/- 0.009730 | {'patience': 2, 'max_steps': 1} |
| 17 | magma_adam | 3 | 1.664848 +/- 0.013043 | {'patience': 3} |
| 18 | adagrad | 3 | 1.667063 +/- 0.008146 | {'patience': 3} |
| 19 | muon | 3 | 1.667377 +/- 0.014844 | {'patience': 3} |
| 20 | muon_guarded | 3 | 1.673107 +/- 0.012765 | {'patience': 3} |
| 21 | sgd | 3 | 1.673107 +/- 0.012765 | {'patience': 3} |
| 22 | dash_ndb_guarded | 3 | 1.673107 +/- 0.012765 | {'patience': 3} |
| 23 | magma_adamw | 3 | 1.677485 +/- 0.013682 | {'patience': 3} |
| 24 | magma_sgd | 3 | 1.689913 +/- 0.008868 | {'patience': 3} |
| 25 | adamx | 3 | 1.708388 +/- 0.016640 | {'max_steps': 3} |
| 26 | amsgrad_geometric | 3 | 1.744752 +/- 0.016446 | {'max_steps': 3} |
| 27 | amsgrad_inverse | 3 | 1.817552 +/- 0.009055 | {'max_steps': 3} |

## Lean proof scope

AMSGradW convergence needs L < decay*epsilon and gives a history-dependent
weighted regularized equilibrium, generally not zero original gradient.
Original AMSGradMD has a proved counterexample. Guarded MD has deterministic
stationarity/loss convergence; strong convexity adds weight convergence.
These hypotheses are not established for minibatch GPTMini. The Python
multi-block guard is a documented extension of the single-matrix Lean model.
All proposal/gain/auxiliary buffers use AMSGrad; no Adam is hidden in routing.
Three seeds and the rate screen do not establish a general superiority.
See LEAN_AUDIT.md, metadata.json, screening.jsonl and validation.json.
