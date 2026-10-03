# AMSGradW and AMSGradMD on GPTMini

Requested on 2026-10-01. Preserve the completed all24 experiment and every
measured Python source. Add an independent experiment using its functional
model, complete-step compiler, data split, initialization and stopping rules.

- [x] Read local AMSGrad/MD manuscripts and the actual Lean recurrences.
- [x] Build `Transformer.AMSGradW` and `Transformer.MagnitudeDirection`.
- [x] Audit the transitive axioms of the convergence/counterexample theorems.
- [x] Write and pass numerical contracts before any training benchmark.
- [x] Check all new methods on both attentions against eager losses,
      gradients, weights and every optimizer buffer on CUDA.
- [x] Verify actual CUDA Graph recording, stable compiler counters, and
      best-checkpoint restoration with the complete compiled step.
- [x] Screen three rates per method and attention for 250 updates on seed0,
      using validation only. Never evaluate the test split during screening.
- [x] Freeze sources, settings, selected rates and screening artifacts.
- [x] Run all three methods with Softmax/Sparsemax and seeds0,1,2 to the
      previous validation patience/budget stop (18 runs).
- [x] Replay stopping and re-evaluate every selected checkpoint on CUDA.
- [x] Save the comparison with the existing AMSGrad and all24 rankings.

## Algorithms and predeclared parameter choices

Every callback uses raw moments, beta1=0.9, beta2=0.999, epsilon=1e-8,
zero initial histories, and no bias correction. AMSGradW uses decay0.01
on every unique parameter, outside the moment gradient; its rate grid is
0.0003,0.001,0.003, as for the existing AMSGrad. Update semantics match
`AMSGradW.trainingStep`.

AMSGradMD applies the Lean `amsgradMDProposal` to the eight hidden linear
matrices. Both effective gain vectors initially equal one, and each fixed
Frobenius sphere radius is that matrix's actual initialization norm. The
model always sees the fused weight. Direction rates are screened over
0.0003,0.001,0.003; the raw-gain rate is fixed at0.001. The tied embedding
and attention temperatures use ordinary AMSGrad at0.0003, once per unique
tensor. There is no Adam, bias correction, decay, or embedding normalization.
These are deliberate GPTMini choices; this is the explicitly specified Lean
AMSGrad variant of Appendix A, not the manuscript's Adam-gain recipe.

`amsgradmd_guarded` keeps exactly the same AMSGrad proposal (direction
rate0.0003, gain rate0.001, auxiliary rate0.0003) and all updated histories.
It checks the complete fused displacement of the joint parameter vector.
For tau, let d=(old-candidate)/tau; accept only finite, nonzero MD blocks
with <gradient,d> >= sigma*||gradient||^2 and ||d|| <= ||gradient||,
where sigma=0.25. Otherwise use the genuine gradient fallback, halving
only a matrix step that would hit zero. Rebalance each rejected MD
representation by scaling all its row gains equally, preserving fused
weights, column gains and row ratios. Screen tau over0.1,0.3,1.0, as in
the previous guarded/SGD recipes. The implied guard L=sigma/tau is a
parameter, not a proved smoothness bound for GPTMini. The global product
guard and ordinary auxiliary callbacks are an explicit multi-block Python
extension of Lean's single-matrix recurrence; record their certificates.

## Proof scope

AMSGradW has26 theorems and no sorry. Its deterministic full-gradient
convergence theorem assumes L < decay*epsilon and eta*decay <=1, and
converges to a history-dependent diagonally regularized equilibrium.
It does not generally approach stationarity of the unregularized loss.
The tiny experimental epsilon/decay do not certify the domination condition.

Original AMSGradMD has a proved strongly convex one-dimensional
counterexample. Only the guarded fused-step variant has a stationarity/loss
convergence theorem; convergence of weights to a minimum additionally needs
strong convexity. Its proof uses a fixed smooth lower-bounded objective in
exact real arithmetic. Neither proof establishes convergence of minibatch,
float32, nonconvex GPTMini or global smoothness for Sparsemax.

## Shared measurement protocol

RTX3050 Laptop4GB; GPTMini2 layers,4 heads,width128,FFN512,context64,
batch32; float32, deterministic algorithms, TF32 off. The original
TinyShakespeare90/5/5 split and private per-seed initialization/minibatch
streams remain identical. Fullgraph Inductor reduce-overhead compiles
gathering, model, reverse AD, all moment/factor/guard/repair computations,
parameter updates and held-out reductions. No eager numerical fallback.
Host I/O and validation stopping orchestration remain Python.

Validation every250 updates, patience8, min_delta0.0001, min_steps1000,
max_steps20000, divergence delta0.1/patience3. Select exact minimum
validation loss, restore that checkpoint and evaluate test once after stop.
Report all methods that produced usable results, and distinguish partial
runs, failed/recovered runs, screening, cold compilation and warm execution.

## Recorded preflight and selection

All13 new CPU/CUDA tests passed before screening. Every method and attention
matched eager loss/gradients/weights/all histories and passed stable-graph
and checkpoint-reload checks. The7 principal Lean convergence and
counterexample theorems depend only on propext, Classical.choice and
Quot.sound; the full audit has zero rests/extra axioms.

All18 screening candidates completed successfully with no test evaluation.
On both attentions validation selected W rate0.0003, original MD direction
rate0.0003, and guarded MD fallback tau0.1. The main18-run protocol is
frozen in results/rtx3050/metadata.json; previous measurements are unchanged.

## Completed comparison

All18 main runs and all18 fresh compiled CUDA checkpoint checks passed on
2026-10-01. All18 runs stopped by validation patience; none failed,
recovered, or reached the budget cap. Every main run retained FX4, CUDA4,
zero graph breaks and stable tracing/recording counters after mode warmup.
Validation/test loss reproduced to1e-9 after loading every saved model.
All initial models/minibatch plans match the previous benchmark. Previous
all24 sources, raw results and audit artifacts remain unchanged.

AMSGradW has the smallest mean best-checkpoint test CE on both attentions
in the combined27-method ranking (162 runs). Values are three-seed mean
and sample standard deviation, not a statistical superiority claim.

| Method | Softmax test CE | Sparsemax test CE |
| --- | ---: | ---: |
| AMSGradW | 1.625375 +/-0.001731 | 1.638935 +/-0.016796 |
| Previous AMSGrad | 1.627871 +/-0.000749 | 1.646002 +/-0.017073 |
| Original AMSGradMD | 1.629636 +/-0.012977 | 1.647135 +/-0.011880 |
| Guarded AMSGradMD | 1.635253 +/-0.001809 | 1.655872 +/-0.018814 |

Guarded MD accepted0.155%--0.672% of complete proposals over its main runs;
the nonsingular gradient fallback performed most updates. No matrix step
had to be halved for a zero landing. See per-run optimizer diagnostics for
final gains and sphere errors. The benchmark is empirical: the fixed-loss
Lean hypotheses have not been established for stochastic GPTMini.

Artifacts: results/rtx3050/REPORT.md (all27 rankings), runs.jsonl,
screening.jsonl, metadata.json, combined_summary.json and validation.json.
Ignored best models/curves are under experiments/runs/amsgrad_extensions_benchmark/rtx3050/.

```bash
.venv/bin/python -u -m experiments.amsgrad_extensions_benchmark
.venv/bin/python -m experiments.amsgrad_extensions_benchmark --validate-only
```
