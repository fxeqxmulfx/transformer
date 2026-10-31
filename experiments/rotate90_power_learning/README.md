# Powered modular division: Rotate90 learning and the spectral explanation

Can the current Peri-RMSNorm SwiGLU learn `(a/b)^K mod P`, and what does the
flat-spectrum theory predict about it?

Completed on 2026-10-31: **175 runs**, including **48 Schedule-Free AdamW runs**.
The latest 39 runs exclusively use Schedule-Free AdamW, as requested. Earlier
AdamW records are retained. Larger one-block models reliably learn the main
P97/P193 tasks with much less seed variation than width 16. The P17 control
still fits the training table without generalizing well at width 512.

**Larger models, Schedule-Free AdamW only.** These are averaged-weight heldout
accuracies after 5000 updates, lr=.003, betas=(.9,.98), decay=1, and 10-step
external warmup. Each row uses seeds 11, 17 and 23, varying initialization,
split and batch order together. These are not initialization-only variance
estimates. Comparisons at the same prime/seed share operand splits.
Late SD is the largest within-run standard deviation across the seven complete
heldout evaluations at steps 3500–5000. Values are percentages or percentage
points; [results.json](results.json) also records exact values and seed SDs.

| P, K | Width / blocks | Mean heldout | Range over 3 seeds | Largest late SD |
| --- | --- | ---: | ---: | ---: |
| 97, 2 | 16 / 1 | 31.46 | 31.14–31.64 | 3.287 |
| 97, 2 | 256 / 1 | 99.90 | 99.87–99.91 | 0.147 |
| 97, 2 | 512 / 1 | 99.94 | 99.91–99.96 | 0.131 |
| 97, 2 | 256 / 4 | 99.46 | 99.18–99.94 | 0.303 |
| 193, 2 | 64 / 4 | 99.95 | 99.91–99.98 | 0.048 |
| 193, 2 | 256 / 1 | 99.98 | 99.97–100.00 | 0.010 |
| 193, 2 | 512 / 1 | 99.99 | 99.98–100.00 | 0.021 |
| 193, 2 | 256 / 4 | 99.99 | 99.97–100.00 | 0.250 |
| 193, 3 | 16 / 1 | 38.34 | 11.79–58.14 | 3.495 |
| 193, 3 | 256 / 1 | 99.90 | 99.84–99.94 | 0.021 |
| 193, 3 | 512 / 1 | 99.99 | 99.98–99.99 | 0.016 |
| 193, 3 | 256 / 4 | 99.97 | 99.90–100.00 | 0.192 |
| 17, 1 | 64 / 1 | 8.58 | 5.88–12.50 | 0.853 |
| 17, 1 | 512 / 1 | 11.76 | 9.56–13.97 | 0.000 |

The narrow P193/K3 case is sensitive to the seed; narrow P97/K2 repeatedly
settles near one-third accuracy. Both improve greatly with width. Four blocks
sometimes introduce late dips after reaching 100%, so best and final accuracy
are both retained. Increasing depth is not uniformly helpful. Both P17 sizes
reach 100% training accuracy: poor generalization within this budget despite
fitting the training table.

The seed-11 Schedule-Free rate screen used .001, .003 and .01 on the three
original failing configurations. Width 16 remained below 32% on P97/K2 and
below 12% on P193/K3 at every tested rate. W64/D4 P193/K2 improved from 46.92%
at .001 to 99.98% at .003 and .01, with the same decay, betas and budget.

Evaluation uses averaged weights. For seed 11 and lr=.003, **heldout** accuracies
at the training-iterate versus averaged weights are respectively 31.72%/31.14%
for W16/D1 P97/K2, 100%/99.98% for W64/D4 P193/K2, and 11.78%/11.79% for
W16/D1 P193/K3. Both columns concern heldout examples, not training accuracy.

**Theory comparison and correction.** The
[revised note](../../review/flat_spectrum.md#powered-division-and-the-current-rotate90-model-2026-10-31)
keeps the proved completion theorems and corrects their application to training.

| Claim or prediction | Finding |
| --- | --- |
| A character readout of powered division has rank one | Exact outer-product identity for every natural K; numerical checks cover P17/97/193 and K1/2/3/5. This available route does not depend on model width. |
| Theorem E predicts ideal character-kernel recovery | All nine prime/seed masks are connected and have perfect matchings, with saved integer witnesses. Given a fixed faithful readout, ideal completion predicts exact nonzero-table recovery even at P17. |
| Recovering a low-rank feature identifies the answer | Requires a separating code. On the output group of order m, frequencies R have ambiguity size gcd(m,R). A nonfaithful or constant rank-one feature can merge answers. |
| Dominant embedding energy predicts accuracy | Not universally: a centered four-label example puts over 99% of energy in parity, yet small faithful features permit 100% decoding. |
| The actual model bounds classification margins | Final RMSNorm gives `abs(L_c-L_c') <= sqrt(W)*norm(E_c-E_c')`. Every reported alias-pair bound is checked directly. |

The feature-code separation, positive-margin and low-rank-score arguments are
now formalized in `Transformer.Grokking.FlatSpectrum.FeatureCodes`. The frequency-gcd
formula and the actual-readout norm bound remain mathematical arguments in the note.
The further [actual-network construction](../../review/flat_spectrum.md#actual-network-expressivity-for-every-exponent)
is formalized in `Transformer.Rotate90.Power`: for every prime and natural K,
width 16 with hidden size 43 has weights that classify the entire valid table,
including zero numerators and K=0. This holds at every positive finite depth,
with all RMSNorms and the tied embedding retained. The weights depend on the
fixed task exponent; it is not an additional model input. Arbitrary finite
padding is allowed. Consequently, even the W16 failures above do not establish
insufficient capacity. Neither result proves that Schedule-Free finds these
weights or generalizes from the observed split.

The prospective rule in [hypotheses.json](hypotheses.json) was committed before
the fresh-seed retry. It predicts quotient confusion when at least 95% of
centered output-embedding energy lies on frequencies divisible by an alias size
g. It passes **16 of 17 applicable cases**. The earlier W16/D1 P193/K2 run is
a counterexample: quotient accuracy is 92.87%, below the fixed 95% threshold.
All four applicable fresh-seed confirmation cases and all seven applicable
Schedule-Free cases pass. Most runs have g=1, for which the rule makes no
accuracy prediction. A separating output code does not show that the hidden
state computes the correct features. The account of optimization remains incomplete.

The initial AdamW discoveries illustrate the aliases: W16/D1 P97/K2 gives
32.11% exact accuracy but 99.98% after cubing prediction and target; W64/D4
P193/K2 gives 31.67% versus 99.70%; W16/D1 P193/K3 gives 11.95% versus 100%
after power 8. See [alias_diagnostics.json](alias_diagnostics.json).

**Earlier screen and scope.** Five shapes (16/1, 64/1, 128/1, 64/2, 64/4)
cross P17/97/193 and K1/2/3. AdamW changes one factor at a time: lr
.001/.003/.01, decay 0/.1/1, betas (.9,.98)/(.9,.999)/(.5,.98). Three full-table
K3 controls measure fitting; one 100-update pilot measures compilation. Nine
narrow-model optimizer combinations, three K5 runs and twelve fresh-seed
AdamW runs complete the earlier stage. Base W64/D1 final heldout results:

| P | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: |
| 17 | 14.71% | 72.79% | 10.29% | 9.56% |
| 97 | 100% | 99.94% | 99.89% | 100% |
| 193 | 100% | 99.99% | 99.88% | 100% |

K5 retains all nonzero output classes at these primes. W16/D1 P193/K1 improves
from 42.01% to 99.95% when decay changes from 1 to 0 at lr=.003: the failure
is not a capacity impossibility. Repeating the original setting gives 41.28%,
illustrating that fixed seeds do not imply bitwise-identical long CUDA paths.
Full-table K3 fitting ends at 100%, 99.9785% and 99.9865% for P17/97/193.

**Protocol and validation.** The current `grok-simple.py` source is pinned by
SHA256 in [the source fixture](../../python/tests/fixtures/peri_rotary_source.json).
It uses tied E, Peri-RMSNorm SwiGLU, no biases or norm gains, H=(8W+1)//3,
and the source initializer. The older `rotate90_analytic` forward is different.
CPU source-fixture weights/logits match exactly; independent forward/gradient,
finite-difference and compiled-continuation checks pass.

Runtime: **Python 3.15.2, PyTorch 2.15.0+cu132, CUDA 13.2, RTX 3050 Laptop**.
Float32 forward/loss/backward and inference use
`torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)`.
TorchInductor reports too few SMs for its GEMM autotuning search; the requested
compilation mode still runs. Schedule-Free uses the existing revised
1.4.1-equivalent implementation with eager optimizer updates; historical AdamW
uses native fused updates. The largest model has 2,195,456 parameters.

Except fitting controls, runs use half the exhaustive table, batch 512, 5000
updates, shuffled epochs with short tails, 10-step warmup and complete-split
evaluation every 250 steps. Zero numerators are included, zero denominators
excluded. Every label independently satisfies `b^K*c = a^K mod P`. Class counts
and majority baselines are recorded. Budgets match updates, not FLOPs or epochs.
Heldout scores guide this exploratory study; three seeds are descriptive evidence.

Completed budgets total 870,100 updates. Summed recorded training time is
1360.02 seconds; summed run wall time is 3269.98 seconds, including preparation,
compilation and evaluation but excluding process startup and the failed attempt.
These are execution records, not an isolated throughput benchmark. One W512/D1
P193/K2 process failed in CUDA capture before its first checkpoint. An unchanged
retry reached 100%; see [attempt_failures.json](attempt_failures.json).

The full Python suite passed: 398 lab tests and 104 optimizer tests (12 and 14
skipped in that CPU invocation). Three focused compiled CUDA tests passed,
including Schedule-Free averaging. Checkpoint replay checks source hashes and
exact labels, comparing losses within 5e-5 relative / 5e-6 absolute. One narrow
Schedule-Free case differs by one heldout decision on CPU, with a top-logit gap
of 4.77e-7. Compiled CUDA replay matches the original accuracy exactly; maximum
CPU/CUDA logit difference is 1.50e-5. The strict mismatch remains visible in
[cuda_roundoff_replay.json](cuda_roundoff_replay.json) and the final audit.

Portable records: [summary.csv](summary.csv), [curves.csv](curves.csv),
[results.json](results.json), [theory_results.json](theory_results.json),
[checkpoint_audit.json](checkpoint_audit.json). Raw checkpoints stay in the root
experiment's ignored `runs/`, outside the disposable worktree. The larger-model
specification is [large_model_hypotheses.json](large_model_hypotheses.json).

Reproduce a Schedule-Free case from the repository root:

```sh
./run.py check experiments/rotate90_power_learning
./run.py run experiments/rotate90_power_learning large-w512d1-p193k2-s11
./run.py report experiments/rotate90_power_learning
```

From `python/`, rebuild analyses with `uv run --locked python` on
`../experiments/rotate90_power_learning/summarize.py`, `audit.py` and `theory.py`.
If a new CPU/CUDA discrepancy appears, `replay_roundoff.py` checks the original
compiled device before another audit. Both hypothesis specifications are retained.
