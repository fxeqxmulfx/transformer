# Sparsemax first frozen final-window violation

This is a partial scientific witness from the unchanged native AdamW /
sparsemax GPTMini mod-193, 25% adaptation of *Convexifying Transformers*,
Section 4. The complete 300,000-update candidate and its fresh paired softmax
control remain required. This witness does not close either run or select a
later architecture or primary benchmark.

At the first prospectively frozen final-window observation, update 250,000,
train accuracy is 100%, held-out complete-RHS/numeric-answer accuracy is
2.7238054116%, and held-out EOS accuracy is 100%. Held-out answer cross entropy
is 10.6402723173 nats; the EOS loss is 1.1217e-8 nats. The exhaustive pools
contain 9,264 train and 27,792 held-out equations. Exposure is 13,157.93955
epochs. This observation and its two immediate neighbors use full 512-example
batches under the unchanged short-final policy.

| Actual observation | Train accuracy | Held-out accuracy | Held-out EOS | Applied LR | Pre-update gradient L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 249,999, before | 100% | 2.723805% | 100% | 0.0003 | 1.20363e-6 |
| 250,000, canonical | 100% | 2.723805% | 100% | 0.0003 | 1.20624e-6 |
| 250,001, after | 100% | 2.734600% | 100% | 0.0003 | 1.23889e-6 |

The frozen final-tail criterion requires joint train/held-out accuracy at least
99% at **every** canonical observation from 250,000 through 300,000, inclusive
(201 observations). Its first held-out observation fails. Therefore this
candidate cannot meet that frozen final-window criterion even if it reaches
the target later. This is a finite necessary-condition violation, not a claim
about future accuracy, global convergence or the cause of failed generalization.
Keep the full budget and all later observations, targets and recoveries.

The [captured prefix](sparsemax-first-tail-witness/canonical-prefix.jsonl)
preserves all 1,001 canonical observations from zero through 250,000. The
[boundary witness](sparsemax-first-tail-witness/boundary-witness.json) retains
both complete neighboring evaluations and all three actual gradient/rate and
per-tensor diagnostic records. The unchanged frozen root manifest and native
case plan, [prefix assessment](sparsemax-first-tail-witness/assessment.json),
source/audit validation and checksums are included. All 43 frozen Python,
seventeen native training and nine Lean source fingerprints remain intact.

There is no long joint confirmation in this prefix. Its memorization plateau
runs from 49,750 through 250,000 with 802 canonical observations. The low
held-out boundary is not a collapse from previously confirmed generalization;
post-confirmation episode counts and failure-frequency windows are unavailable,
not zero. Sparsemax's Lean theorem specifies Euclidean routing at fixed Q/K;
it supplies no guarantee of joint learning or algorithmic generalization.

The entire stored prefix, boundary records, plans and assessment reproduce
byte-for-byte in a second direct capture. The assessment also independently
recomputes without importing Torch. The exact executed recorder snapshot is
retained. This is a partial record: complete scientific budget/history is
false, and no scientific comparison ratio or improvement is claimed.

While the same live raw case remains available, recapture into a fresh path:

```bash
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.sparsemax_first_tail_capture /tmp/sparsemax-first-tail-witness
```

Finish and archive the full candidate, then the automatically queued unchanged
softmax control. Follow the primary control's unchanged six-case gate; any
conditional schedule selection requires both full reviewed/committed results.
