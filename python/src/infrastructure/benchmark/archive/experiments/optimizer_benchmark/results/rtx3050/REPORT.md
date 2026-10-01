# GPTMini optimizer comparison on Tiny Shakespeare

This report records the original 18-method experiment. The subsequent
[Magma follow-up](../../../magma_benchmark/results/rtx3050/REPORT.md)
includes all 24 measured methods and finds Magma+Muon has the lowest
three-seed mean for both attention types. Original measurements below
retain their provenance and have not been replaced.

Device: **NVIDIA GeForce RTX 3050 Laptop GPU**, 3.68 GiB CUDA-visible memory.
PyTorch 2.14.0+cu130; CUDA runtime 13.0; float32, TF32 disabled.

## Observed winners

- **softmax: adamw**, final test cross-entropy 1.76013 ± 0.01341, character perplexity 5.813, 8.2 seconds per final run.
- **sparsemax: adamw**, final test cross-entropy 1.78993 ± 0.00632, character perplexity 5.989, 9.6 seconds per final run.

The ± value is sample standard deviation across seeds, not a confidence interval. These are the lowest observed three-seed means in this experiment; close scores do not establish a reliable or general optimizer ordering.

## Protocol

- Reference: unmodified `experiments/gpt_mini.py`; model configuration: `{'vocab_size': 65, 'n_layers': 2, 'n_heads': 4, 'd_model': 128, 'd_ff': 512, 'max_seq_len': 64, 'rope_theta': 10000.0}`.
- Character tokenizer: 65 characters. Contiguous 90/5/5 split; boundaries `[1003853, 1059623, 1115393]`. Dataset SHA256: `53493bf304b639aba6a47400da75c58ce32372b378fd2cd9d16ee987aeb12e4e`.
- Batch 32; unique matrix initialization Normal(0, 0.02), with the source temperatures and embedding/unembedding tying preserved.
- Each method and attention type receives three declared learning rates, 250 updates per rate on seed 0, selected only by final validation loss.
- Final runs: 1000 updates each, seeds `[0, 1, 2]`. Same initial parameters and minibatch plans across methods and attention types.
- Held-out losses evaluate all complete nonoverlapping context windows. Test data is evaluated only in final runs, after learning-rate selection.
- No AMP, gradient clipping, dropout, early stopping or compilation. Report last-iterate loss. Training time excludes held-out evaluation; optimizer time is measured with CUDA events.
- Sparsemax is the causal Euclidean projection of the unchanged QK/RoPE score row onto the simplex, matching `Transformer.GPTMini.Convex.Attention`. XSA and FFN remain unchanged.
- Tests passed before GPU training: 41; no failures or skips. See `tests.log`. Source hashes, environment and protocol hash are in `metadata.json`.

## softmax results

| Method | LR | Test CE ± SD | Char. PPL | Train s | Optimizer ms/step | Peak MiB | Guard accepted | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adamw | 0.001 | 1.76013 ± 0.01341 | 5.813 | 8.2 | 1.007 | 145 | — | 3/3 |
| adam | 0.0003 | 1.76436 ± 0.00949 | 5.838 | 7.9 | 0.617 | 145 | — | 3/3 |
| amsgrad | 0.0003 | 1.76445 ± 0.00963 | 5.838 | 8.2 | 0.752 | 145 | — | 3/3 |
| muon | 0.03 | 1.76803 ± 0.00932 | 5.859 | 12.6 | 5.249 | 142 | — | 3/3 |
| adamnc | 0.01 | 1.79868 ± 0.00712 | 6.042 | 7.9 | 0.626 | 145 | — | 3/3 |
| dash_cn | 0.001 | 1.80401 ± 0.01121 | 6.074 | 35.6 | 28.118 | 146 | — | 3/3 |
| dash_evd | 0.001 | 1.80403 ± 0.01119 | 6.074 | 26.5 | 18.982 | 233 | — | 3/3 |
| dash_chebyshev | 0.001 | 1.80409 ± 0.01114 | 6.074 | 67.0 | 59.291 | 146 | — | 3/3 |
| dash_ndb | 0.001 | 1.80746 ± 0.00380 | 6.095 | 37.8 | 30.263 | 146 | — | 3/3 |
| muon_guarded | 0.1 | 1.85975 ± 0.00619 | 6.422 | 13.5 | 6.172 | 142 | 0.0% | 3/3 |
| sgd | 0.1 | 1.85975 ± 0.00619 | 6.422 | 7.7 | 0.199 | 145 | — | 3/3 |
| dash_ndb_guarded | 0.1 | 1.85975 ± 0.00619 | 6.422 | 38.3 | 30.823 | 146 | 0.0% | 3/3 |
| adafisher | 0.001 | 1.88625 ± 0.00620 | 6.595 | 11.9 | 3.379 | 142 | — | 3/3 |
| adafisherw | 0.001 | 1.88684 ± 0.00859 | 6.598 | 11.9 | 3.576 | 142 | — | 3/3 |
| adagrad | 0.03 | 1.91586 ± 0.02837 | 6.793 | 8.3 | 0.799 | 145 | — | 3/3 |
| adamx | 0.003 | 1.98792 ± 0.01824 | 7.300 | 8.2 | 0.872 | 145 | — | 3/3 |
| amsgrad_geometric | 0.003 | 1.99193 ± 0.01929 | 7.330 | 8.0 | 0.739 | 145 | — | 3/3 |
| amsgrad_inverse | 0.003 | 2.18394 ± 0.07320 | 8.881 | 7.9 | 0.643 | 145 | — | 3/3 |

Fastest completed method: **sgd**, 7.7 s for the same final update budget.

## sparsemax results

| Method | LR | Test CE ± SD | Char. PPL | Train s | Optimizer ms/step | Peak MiB | Guard accepted | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adamw | 0.001 | 1.78993 ± 0.00632 | 5.989 | 9.6 | 0.899 | 145 | — | 3/3 |
| adam | 0.0003 | 1.79137 ± 0.01484 | 5.998 | 9.3 | 0.523 | 145 | — | 3/3 |
| muon | 0.03 | 1.79271 ± 0.00529 | 6.006 | 14.0 | 5.259 | 142 | — | 3/3 |
| amsgrad | 0.0003 | 1.80118 ± 0.00369 | 6.057 | 9.4 | 0.641 | 145 | — | 3/3 |
| adamnc | 0.01 | 1.81425 ± 0.00081 | 6.136 | 9.2 | 0.476 | 145 | — | 3/3 |
| dash_evd | 0.001 | 1.83175 ± 0.00851 | 6.245 | 27.2 | 18.418 | 233 | — | 3/3 |
| dash_chebyshev | 0.001 | 1.83345 ± 0.00489 | 6.255 | 67.1 | 58.109 | 147 | — | 3/3 |
| dash_cn | 0.001 | 1.83430 ± 0.00546 | 6.261 | 36.9 | 27.919 | 147 | — | 3/3 |
| dash_ndb | 0.001 | 1.83653 ± 0.01149 | 6.275 | 39.4 | 30.355 | 147 | — | 3/3 |
| adagrad | 0.03 | 1.87897 ± 0.01405 | 6.547 | 9.2 | 0.511 | 145 | — | 3/3 |
| dash_ndb_guarded | 0.3 | 1.90579 ± 0.08004 | 6.725 | 39.4 | 30.513 | 147 | 0.0% | 3/3 |
| sgd | 0.3 | 1.90579 ± 0.08004 | 6.725 | 9.0 | 0.143 | 145 | — | 3/3 |
| muon_guarded | 0.3 | 1.90579 ± 0.08004 | 6.725 | 14.9 | 6.080 | 142 | 0.0% | 3/3 |
| adafisher | 0.001 | 1.93403 ± 0.02556 | 6.917 | 13.0 | 3.278 | 142 | — | 3/3 |
| adafisherw | 0.001 | 1.93792 ± 0.02202 | 6.944 | 13.1 | 3.441 | 142 | — | 3/3 |
| amsgrad_geometric | 0.003 | 1.97968 ± 0.01289 | 7.240 | 9.2 | 0.504 | 145 | — | 3/3 |
| adamx | 0.003 | 1.98208 ± 0.00226 | 7.258 | 9.6 | 0.771 | 145 | — | 3/3 |
| amsgrad_inverse | 0.003 | 2.10723 ± 0.01781 | 8.225 | 9.4 | 0.661 | 145 | — | 3/3 |

Fastest completed method: **sgd**, 9.0 s for the same final update budget.

## Implementation and proof scope

The local convergence theorems have explicit assumptions about fixed objectives, smoothness, step bounds or convex projected online problems. Stochastic minibatches and a nonconvex GPT training loss are an empirical test, not a verification of those hypotheses. A sparse convex attention-row projection does not make joint training convex.

| Method | Local formalization / role |
| --- | --- |
| sgd | reference baseline; SGD rule formalized |
| adagrad | AdaGrad rule and monotone metric formalized |
| adam | paper Adam without debiasing; counterexamples formalized |
| adamw | practical bias-corrected reference; decay 0.01 |
| amsgrad | constant-momentum deterministic convergence extension |
| amsgrad_inverse | source inverse-momentum regret and offline convergence |
| amsgrad_geometric | source geometric-momentum regret and offline convergence |
| adamx | AdamX regret bound; inverse momentum |
| adamnc | AdamNC regret bound; inverse momentum and second-moment averaging |
| muon | paper finite Newton-Schulz recipe; unguarded training can cycle |
| muon_guarded | formalized global direction safeguard |
| dash_evd | exact regularized spectral inverse-root specification |
| dash_ndb | corrected finite chained NDB solver |
| dash_cn | corrected finite fourth-root coupled Newton solver |
| dash_chebyshev | corrected cosine fit, sample guard and Clenshaw scaling |
| dash_ndb_guarded | finite NDB/grafting with formalized global training safeguard |
| adafisher | corrected factors, bias correction and deterministic convergence |
| adafisherw | formalized decoupled-decay update; general convergence not proved |

- AMSGrad/AdamX/AdamNC retain paper raw moments without Adam debiasing. Numerical epsilon is 1e-8. Inverse momentum is 0.9/t; geometric momentum is 0.9·0.99^(t−1); their step is base LR/sqrt(t). AdamNC second moments are the actual mean of past g². Constant AMSGrad/Adam use β1=0.9, β2=0.999. AdamW uses bias correction and decay 0.01.
- Muon: zero-initialized raw momentum 0.95·M+G, Nesterov input 0.95·M_new+G, five printed Newton–Schulz steps in float32, and shape scale 0.2·sqrt(max(m,n)). The tied embedding and temperatures use bias-corrected Adam with 0.05 times the Muon LR; decay is zero.
- DASH: block size 32 including residual blocks; left/right EMA β=0.99; damping 1e-4. Blockwise Adam grafting uses raw β1=0.9, β2=0.999 and no debiasing. NDB chains two actual six-step solves; CN uses eight fourth-root steps; both have checked Rayleigh scaling with a Frobenius/row certificate and output rescaling. Chebyshev uses degree 60, 1,000 cosine nodes, sample repair and corrected original-scale regularization/output.
- The Muon/DASH training guard is the literal joint-parameter alignment/length check with σ=0.5. A rejected candidate becomes the current minibatch gradient, while all optimizer histories are retained. Acceptance counts expose this behavior. No smoothness constant is certified for this stochastic experiment.
- AdaFisher: genuine Linear-input and Linear-output-backpropagation squared sums; fresh-factor EMA γ=0.8; separate min/max normalization with the proved constant-range zero extension; damping 0.001; β1=0.9 and positive-time bias correction. The tied embedding uses its unembedding Linear factors and combined parameter gradient; temperatures use the identity-factor fallback. Shared-weight curvature is a modeling approximation. AdaFisherW adds decoupled decay 0.01; other AdaFisher decay is zero.

## Artifacts

- `runs.jsonl`: every screening/final run, failures, loss curves, hashes and timing.
- `selected_rates.json`: separate validation-selected rates for each attention type.
- `summary.csv`, `summary.json`: all final three-seed measurements and observed winners.
- `metadata.json`, `tests.log`, `progress.log`: environment, exact source/data fingerprints, tests and progress.
- Seed-0 final checkpoints: `experiments/runs/optimizer_benchmark/rtx3050/` (gitignored).

Reproduce from the repository root:

```bash
.venv/bin/python -m experiments.compare_optimizers
```

## Parameter-group audit and recipe differences

The primary ranking includes every measured variant. Matrix-root convergence,
convex online regret, deterministic training convergence, and empirical GPT
quality are different claims. The unguarded Muon/DASH rows and AdaFisherW
have no general training convergence theorem. AdaGrad has component results;
AdamX/AdamNC have convex online regret bounds.

Muon is a real hybrid in this experiment: eight attention/FFN matrices
(393,216 parameters) use Muon, while the single tied embedding/head matrix
and two head-temperature vectors (8,328 parameters) use bias-corrected Adam.
The tied matrix is updated once. There are no trainable RMSNorm scales or
Linear biases in this GPTMini. The selected Muon LR is 0.03 for both attention
types, and the auxiliary LR is 0.0015, with betas (0.9, 0.999), epsilon 1e-8
and zero decay. Adam here is equivalent to AdamW with zero weight decay.

**Recipe difference:** Muon section 2.2 proposes sharing learning rate and
weight decay between Muon and AdamW after RMS scaling. The fixed auxiliary
LR ratio 0.05 and zero decay are experimental settings of this benchmark,
not an exact reproduction of that shared-LR/weight-decay recipe. Therefore
these results do not establish how the paper's exact Muon recipe ranks.
Source: `papers/arXiv-2502.16982/2-analysis.tex`, section 2.2.

DASH uses matrix preconditioning plus blockwise Adam grafting, including
embeddings. Its temperature vectors are reshaped to columns and use the
matrix inverse-fourth-root rule. This is an explicit model adapter:
section 4 of the paper uses an inverse-square-root rule for one-dimensional
normalization weights, while its DASH-A experiments use Adam there. Our
model has no trainable normalization weights, and these temperature updates
are not that special vector recipe. Source:
`papers/arXiv-2602.02016/dash.tex`, section 4 and Appendix A experiments.

AdaFisher uses measured Linear input/output-derivative factors. For the tied
embedding it combines the parameter gradient with unembedding factors;
shared-weight curvature is an approximation. Temperature parameters use
the constant-factor fallback, which leaves damping after the corrected
min-max normalization. AdamW and AdaFisherW apply their configured decay
to all unique parameters, including the temperatures.

Four additional post-run tests passed: actual-model parameter routing and
unique tied weights; auxiliary updates against independent Torch AdamW
with zero decay; two-step Nesterov and five-step NS against an independent
scalar polynomial calculation; and the joint hybrid guard with preserved
optimizer state. See `parameter_group_tests.log`, `parameter_group_audit.json`
and `experiments/test_optimizer_parameter_groups.py`. The frozen measured
implementations were not changed. The original 41 tests passed before
training; these four tests were added during the final parameter-group audit.

## Artifact verification

Independent checks confirmed 216 unique complete runs, 36 complete
attention/method groups, validation-only learning-rate selection, identical
initial weights and minibatch plans, matching source/data/protocol hashes,
and agreement between raw measurements and every summary mean/sample SD.
All 36 seed-0 checkpoints are retained. Both winning checkpoints reproduce
the recorded seed-0 test loss after reloading on CUDA.

Both guarded methods rejected every candidate in all three seeds and both
attention types. They therefore used the minibatch gradient at every update,
matching SGD's recorded losses. Their seed-0 checkpoints were verified
bit-for-bit identical to SGD for each attention type. In this experiment
the safeguard provided no adaptive update advantage and retained the cost
of computing discarded candidates.

See `validation.json` and `checkpoint_validation.json`. The corrected full
run was interrupted after 62 final runs and resumed with the same source and
protocol fingerprint; completed IDs were skipped. Training times exclude
the interruption and the discarded incomplete attempt. The initial
unshifted-Sparsemax pilot is retained separately and excluded from the
ranking. The latest resume test log is separately retained; its metadata
also preserves the initial test summary and the superseded log hash.

## Lean verification and limits of explaining the winner

`src/Transformer/OptimizerBenchmark.lean` imports 16 proved theorems auditing
these results and analyzing the safeguard and decoupled decay. They have no
`sorry` dependencies. No remaining `sorry` was found in AMSGrad, AdamBeyond,
Muon, DASH, AdaFisher or Optimization; unrelated project proof debt is not an
explanation of this ranking.

`scripts/optimizer_benchmark_data.py` extracts all 108 final test losses into
`OptimizerBenchmark/Basic.lean`. Each rational equals the logged JSON value
parsed as binary64, rather than a rounded report decimal. The generator
checks unique attention/method/seed keys, completed budgets, shared
initialization and batch hashes, protocol, frozen measured source hashes,
and all 36 summary means. The Lean kernel verifies the table's arithmetic;
the GPU execution and honesty of the source log remain outside that proof.
The exact mean is computed before the report's final binary64 rounding.

`Ranking.lean` proves that AdamW has a strictly smaller exact mean than every
other measured variant for each attention mode. Its margin over Adam lies in
`(0.0042, 0.0043)` for softmax and `(0.0014, 0.0015)` for Sparsemax. Adam beats
AdamW at softmax seed 1 and Sparsemax seeds 0 and 2. Thus the mean winner does
not dominate every paired run. Both guarded variants' encoded losses equal
SGD's on every seed; equal losses alone do not establish identical weights.

`GuardFallback.lean` proves the conditional trajectory identity: a stateful
safeguard rejecting all proposals up to a finite horizon has the same weight
trajectory as SGD, for the same initialization, batch losses and rate.
Auxiliary histories can still change. This is a real-arithmetic recurrence
theorem; recorded zero acceptance and seed-0 checkpoint equality are separate
executable evidence, not a Lean verification of floating-point autograd.

`AdamWAnalysis.lean` proves the first-step moment-correction formula and the
exact isolated decay displacement. On genuine smooth, strongly convex
quadratics, decay can either improve or worsen the unregularized objective.
With the selected Adam/AdamW rates, universal AdamW superiority is already
false after one update at a nonzero minimizer. The counterexample concerns
one update on another objective; it does not refute this experiment's
1000-update measurements. Adam and AdamW here differ simultaneously in
debiasing, decay and selected learning rate, so these data do not isolate a
cause of AdamW's advantage. Controlled ablations would be needed to separate
those effects. Convergence under explicit assumptions also does not order
finite-budget held-out losses.

Validation: full `lake build`, `lake env lean scripts/Axioms.lean`,
`python3 scripts/index.py`, and
`python3 scripts/optimizer_benchmark_data.py --check`. The project audit
retains 169 existing `sorry`, zero proved declarations resting on `sorry`,
zero extra axioms, zero vacuous statements and zero placeholder definitions.
New modules build without warnings. See `lean_build.log`, `lean_axioms.log`,
`lean_index.log` and `lean_validation.json`.
