# Experiments

New experiments and diagnostics use Rust/Kernite under the 2026-11-07 user
override in `AGENTS.md`. The Python commands below describe the historical lab.

A folder per experiment, and its `README.md` says what the runs ask, how
they differ, and what their archived runs found. A folder with an
`experiment.py` is written in the language of the
[lab](../python/README.md): it composes a model and the conditions of its
benchmark runs, and `./run.py check|show|run|report|profile
experiments/<name>` reads it. Its runs train into `runs/<label>/` beside
it, which git ignores.

| Experiment | Runs | What it asks |
| --- | ---: | --- |
| [`answer_contrasts`](answer_contrasts) | 24 saved states; no training | Pooling by cubic answer exposes accurate answer probes and useful measured combined contrasts; conservative L1 and weight-norm budgets still fail. |
| [`flat_approx`](flat_approx) | 24 saved states; no training | Quantitative recovery diagnostic: all native correctness counts reproduce, but the fixed full-character spectral budget fails on every held-out query; direct contrast budgets are useful for most trained queries. |
| [`rms_backward`](rms_backward) | 48 | Does separating the finite-epsilon radial RMS backward from its tangent term restore correct gradients and known-solution recovery without changing the forward function? |
| [`context_gains`](context_gains) | 24 | Can input-dependent post-RMS gains recover conditional amplitudes that a correctly differentiated pre-RMS path learns poorly? |
| [`adaptive_residual`](adaptive_residual) | 8 | Does a conditional post-RMS gain improve first99% validation over the current corrected, fixed-scale GPT on two matched seeds at P97/P193? |
| [`basis_residual`](basis_residual) | 24 | Does the same conditional gain help original Basis depth, recall and parity at small/easy and large/hard settings, with matched seeds and Schedule-Free AdamW? |
| [`basis_control`](basis_control) | 2 | Does the historically successful GPTMini solve hard recall under the same Schedule-Free AdamW recipe as current GPT? |
| [`attention_temperature`](attention_temperature) | 2 | Learned score gains fail the new twice-best cost cap: both seeds have0% exact validation at24,400 updates; later100,000/62,900 states are retained, learning closed and final tests unopened. |
| [`residual_scaled_long`](residual_scaled_long) | 2 | Both complete-state scaled continuations cross 99% validation at 24k versus 32k/37k for ordinary addition; final test 99.227%/98.711%; parents remain unchanged. |
| [`residual_gradient_paths`](residual_gradient_paths) | 36 | Known-solution CPU controls: post-RMS gates recover in 12/12 cases; pre-RMS gates in 4/12; depth scaling keeps transport gain at least 1/2 in the linear witness. |
| [`residual_generalization`](residual_generalization) | 16 | Closed two-hour cohort: ordinary residuals reach99.96–100% test; scaling generalizes earlier at23k; depth attention remains near chance within its allowance. All16 cases and execution failures retained. |
| [`modular_composition`](modular_composition) | 48 | Can the normalized source RoPE transformer learn flat-spectrum mixed arithmetic and composed modular chains with Schedule-Free AdamW? All48 completed; full curves and replay discrepancies retained. |
| [`modular_composition_long`](modular_composition_long) | 10 | All10 reached100000 updates in1h44m43s: division groks, mixed pairs remain near chance; full-state and gradient audits retained. |
| [`basis`](basis) | 180 | Does a two-layer GPTMini of width 64 pass depth, recall and parity in the easy mode, while the hard mode takes width 128 and six layers? At what batch and rate do the runs go fastest? |
| [`basis_sparsemax`](basis_sparsemax) | 222 | Where does sparsemax fail where softmax passes the basis, and do a smaller QKNorm starting scale or ScaledDot repair it while retaining sparse attention? |
| [`basis_qknorm`](basis_qknorm) | 150 | Does sparsemax need query/key L2 normalization when its learned head gain and starting score dispersion are controlled? |
| [`basis_tensor`](basis_tensor) | 57 | Does the jointly convex complete-data tensor stack learn all Basis recipes with ordinary AdamW at the original GPTMini softmax's measured FLOPs? |
| [`basis_ansr`](basis_ansr) | 9 | Does ANSR train the unchanged softmax GPTMini on Basis depth, recall and parity, and how does low p_self compare with high p_self and AdamW? |
| [`convex_atomic`](convex_atomic) | 38 | Does answer-only convex atomic sparsemax training escape uniform and saturated error floors, attain its positive-error minimum and transfer to Basis with generated physical heads? |
| [`convex_binding`](convex_binding) | 18 | Does a factorized neighboring-token key encoder remove recall's binding obstruction while retaining convex atomic training and free original values? |
| [`mod97_grokking`](mod97_grokking) | 12 | Do the openai/grok transformer and GPTMini generalize x / y mod 97 long after fitting it? |
| [`mod97_size_search`](mod97_size_search) | 12 | What is the smallest tested dense GPTMini that groks mod 97 within 10,000 full epochs? |
| [`mod97_scalar_mlp`](mod97_scalar_mlp) | 10 | A perceptron reading x and y as two numbers can represent x / y mod 97: why does gradient descent not fit it? A spurious minimum, slowness, divergence, or a step past the curvature? |
| [`mod97_representation_controls`](mod97_representation_controls) | 36 | Which difficulty comes from scalar versus one-hot operands, scalar versus vector answers, MSE versus cross-entropy, or decay on identical division splits and samplers? |
| [`rotate90_analytic`](rotate90_analytic) | 9 | Does SiLU or a near-budget SwiGLU branch improve the published Rotate90 classifier with the same shared E, split, sampler and AdamW settings? |
| [`rotate90_swiglu_stability`](rotate90_swiglu_stability) | 9 | Do smaller SwiGLU steps prevent the observed loss of fit at matched update and nominal optimizer-time horizons? |
| [`rotate90_power_learning`](rotate90_power_learning) | 175 | Compare flat-spectrum and character-separation predictions with current Peri-RMSNorm SwiGLU learning, including larger-model stability with Schedule-Free AdamW under Python 3.15 / PyTorch 2.15 max-autotune. |
| [`rotate90_alias_escape`](rotate90_alias_escape) | 50 | Test actual-gradient alias symmetry and whether decay, frozen parameters or perturbations let narrow Schedule-Free checkpoints escape merged-answer solutions. |
| [`rotate90_alias_budget`](rotate90_alias_budget) | 24 | Separate longer training, reduced decay and greater width after the alias interventions; six width-64 controls and two width-16 seeds cross 99%. |
| [`rotate90_transfer`](rotate90_transfer) | 15 | Actual transfer predicts all 120 next-step loss signs, but the early forecast and width ranking fail. Forty-five stopped-base controls verify the exact averaging clock; all six successful wide checkpoints eventually lose accuracy despite initially improving loss. |
| [`rotate90_forcing`](rotate90_forcing) | 30 | All six wide P97/P193 continuations exceed 99% without decay after update 400; zero-gradient controls finish at 2.95--5.15%. Actual-source response and span invariance are proved; all three P17 endpoint-replay predictions fail. |
| [`grokking_progress`](grokking_progress) | 4 | Does a causal held-out symmetry projection reveal rule formation in ordinary softmax transformers before accuracy rises, while rejecting memorization and immediate learning? |
| [`grokking_internals`](grokking_internals) | 1 | Which frozen probes, answer/EOS gradients, spectra, neuron statistics, head ablations, computation-stage interactions, archived AdamW directions, finite-step curvature, same-gradient moment resets and exact current-logit certificates track ordinary transformer grokking? |
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
