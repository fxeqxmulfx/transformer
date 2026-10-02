# Complete mod-193 AdamW calibration: negative phase and persistence

The user-directed harder arithmetic task completed all 150,000 frozen updates.
Final exhaustive train/held-out accuracy is 100%/100%, but the required
memorization plateau is absent and seven of 201 final-window observations
fail the joint 99% target. Increasing the prime did not produce the required
stable-grokking benchmark on this calibration.

The [individual archive](../../baselines/adamw_stability_mod193_lr001_seed0_data0_20261002/summary.json)
and [complete single-recipe stage](../../baselines/adamw_stability_mod193_calibration_20261002/REPORT.md)
preserve all 601 canonical evaluations, 1,200 exhaustive neighbor probes,
1,800 full tensor diagnostics and 150,000 gradient records, complete CSV
tables and standalone PNG/PDF figures. The
[validation](larger-modulus-result-validation.json) checks the complete
archives without PyTorch, original plan/measurement/log bytes, local checkpoint
SHA256, byte-identical nested archives, earlier source prefixes and all 31
frozen source/manuscript fingerprints. Both trainer and serial worker completed;
session 46645 is terminal with exit code 0.

| Final-window update | Train accuracy | Held-out accuracy |
| --- | ---: | ---: |
| 102,750 | 95.1749% | 94.6082% |
| 111,750 | 96.3245% | 95.4609% |
| 116,500 | 98.7586% | 98.6669% |
| 121,750 | 98.1595% | 98.1002% |
| 123,500 | 97.0315% | 96.5728% |
| 132,250 | 98.8504% | 98.7316% |
| 148,000 | 81.9570% | 79.8953% |

194 of 201 final-window observations pass. First train 99% is at 39,500,
with held-out already 98.5320%; first held-out 99% is at 40,000. No preceding
observation meets the required train ≥99% / held-out ≤10% condition.
The first long joint confirmation spans 52,250–57,000, costing 1,436.21
training / 1,779.09 wall seconds. These observed costs are retained, while
eligible persistent-target timing is null with support zero.

After legacy confirmation at 40,250, 20 canonical held-out observations fail
99%. All fail numeric answers and none fails EOS. All have neighbor support:
13 stay below target at both neighbors, two are isolated at the canonical
observation and five recover by the following update. Categories can overlap;
this count includes failures before long confirmation. The
[first final-window failure](first-tail-failure-mod193.md) follows a full batch.
Neither this neighborhood nor the gradient trace identifies a cause. The
largest norm is 575.0655 at 147,757 after a full 512-example batch; accuracy
was not evaluated at that update.

The model has 436,104 parameters and consumes 75,113,536 examples. Complete
costs are 3,750.27 training / 4,648.78 wall seconds, with 40.38 diagnostic
seconds. Peak CUDA allocation/reservation are 119,638,016/161,480,704 bytes.
Final held-out answer/EOS cross-entropies are 8.73021e-6/1.24241e-8 nats.
The reviewed overview and neighbor plots retain all scheduled data. The
original stage-comparison figure has overlapping update labels and is preserved
unchanged; generate a readable sidecar after the analysis-only plot correction.

This adapts *Convexifying Transformers*, Section 4. Only the prime changes
relative to primary mod-97, but vocabulary, total parameters, common-seed
initial weights, epoch length/tails, exposure and evaluation cost change as
documented in [the frozen protocol](larger-modulus-protocol.md). One split and
initialization cannot establish repeatability or isolate arithmetic difficulty.
CPU preparation/verification overlapped parts of this run; these measured
costs are descriptive and do not certify a causal optimizer/task speed ratio.

No six-case confirmation is eligible from this recipe. The
[prepared 25% fraction](fraction25-preparation.md) preserves the vocabulary,
parameters and common initialization while testing whether fewer observed
equations create the missing memorization phase. Its CPU checks establish
implementation only. After preserving this full negative result, move the
verified CSV optimization into the ordinary verifier and prospectively freeze
the next scientific intervention with every criterion unchanged. All six new
confirmations, the targeted paired architecture change and scientific
complementary mechanics remain outstanding.
