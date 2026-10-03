# Experiments

A folder per experiment, and its `README.md` says what the runs ask, how
they differ, and what their archived runs found. A folder with an
`experiment.py` is written in the language of the
[lab](../python/README.md): it composes a model and the conditions of its
benchmark runs, and `./make.py check|show|run|report experiments/<name>`
reads it. Its runs train into `runs/<label>/` beside it, which git ignores.

| Experiment | Runs | What it asks |
| --- | ---: | --- |
| [`mod97_grokking`](mod97_grokking) | 12 | Do the openai/grok transformer and GPTMini generalize x / y mod 97 long after fitting it? |
| [`mod97_stability`](mod97_stability) | 6 | Once GPTMini generalizes x / y mod 97, does it stay generalized, under AdamW and raw AMSGradW? |
| [`mod193_stability`](mod193_stability) | 5 | The same on mod 193 under AdamW, and with sparsemax or an annealed rate |
| [`mqar_sparsemax`](mqar_sparsemax) | 33 | Does sparsemax attention learn associative recall where softmax does not? |
| [`shakespeare_amsgradw`](shakespeare_amsgradw) | 6 | GPTMini on Tiny Shakespeare under AMSGradW, softmax against sparsemax |
| [`shakespeare_zoo`](shakespeare_zoo) | 162 | GPTMini on Tiny Shakespeare under 27 optimizer recipes |
| [`synthetic_amsgradw`](synthetic_amsgradw) | 61 | Which tasks of the synthetic suite does GPTMini learn, and which does it only memorize? |
| [`synthetic_scaling`](synthetic_scaling) | 27 | Double descent in width on noisy parity, and copy across model sizes |

Two folders predate the lab and run as scripts:

| Folder | What it measures |
| --- | --- |
| [`quartet_sr_evaluation`](quartet_sr_evaluation) | the error of FP4 stochastic rounding with upward E4M3 scales, in NumPy |
| [`loopexp`](loopexp) | how quantization error propagates through looped and flat pre-norm GPT, configured by flags |

[`archive/`](archive) keeps the records of the three codebases the lab
replaced: the reference GPTMini with its Tiny Shakespeare optimizer
benchmarks, the synthetic trainers, and the convex MQAR comparison.
