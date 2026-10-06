# Experiments

A folder per experiment, and its `README.md` says what the runs ask, how
they differ, and what their archived runs found. A folder with an
`experiment.py` is written in the language of the
[lab](../python/README.md): it composes a model and the conditions of its
benchmark runs, and `./make.py check|show|run|report|profile
experiments/<name>` reads it. Its runs train into `runs/<label>/` beside
it, which git ignores.

| Experiment | Runs | What it asks |
| --- | ---: | --- |
| [`basis`](basis) | 180 | Does a two-layer GPTMini of width 64 pass depth, recall and parity in the easy mode, while the hard mode takes width 128 and six layers? At what batch and rate do the runs go fastest? |
| [`basis_sparsemax`](basis_sparsemax) | 222 | Where does sparsemax fail where softmax passes the basis, and do a smaller QKNorm starting scale or ScaledDot repair it while retaining sparse attention? |
| [`basis_qknorm`](basis_qknorm) | 150 | Does sparsemax need query/key L2 normalization when its learned head gain and starting score dispersion are controlled? |
| [`basis_ansr`](basis_ansr) | 9 | Does ANSR train the unchanged softmax GPTMini on Basis depth, recall and parity, and how does low p_self compare with high p_self and AdamW? |
| [`mod97_grokking`](mod97_grokking) | 12 | Do the openai/grok transformer and GPTMini generalize x / y mod 97 long after fitting it? |
| [`mod97_stability`](mod97_stability) | 6 | Once GPTMini generalizes x / y mod 97, does it stay generalized, under AdamW and raw AMSGradW? |
| [`mod193_stability`](mod193_stability) | 7 | Mod 193 under AdamW: archived normalizer/schedule recipes and fresh sparsemax starting-scale confirmation |
| [`mqar_sparsemax`](mqar_sparsemax) | 33 | Does sparsemax attention learn associative recall where softmax does not? |
| [`shakespeare_amsgradw`](shakespeare_amsgradw) | 6 | GPTMini on Tiny Shakespeare under AMSGradW, softmax against sparsemax |
| [`shakespeare_zoo`](shakespeare_zoo) | 162 | GPTMini on Tiny Shakespeare under 27 optimizer recipes |
| [`synthetic_amsgradw`](synthetic_amsgradw) | 61 | Which tasks of the synthetic suite does GPTMini learn, and which does it only memorize? |
| [`synthetic_scaling`](synthetic_scaling) | 27 | Double descent in width on noisy parity, and copy across model sizes |

[`quartet_sr_evaluation`](quartet_sr_evaluation) predates the lab: a NumPy
script that measures the error of FP4 stochastic rounding with upward E4M3
scales.

[`archive/`](archive) keeps the records of the three codebases the lab
replaced: the reference GPTMini with its Tiny Shakespeare optimizer
benchmarks, the synthetic trainers, and the convex MQAR comparison.
