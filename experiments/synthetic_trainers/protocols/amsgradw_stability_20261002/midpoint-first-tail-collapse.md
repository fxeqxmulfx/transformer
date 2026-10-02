# Midpoint first final-tail collapse

The raw AMSGradW / GPTMini 0.0002 calibration, an explicit adaptation of
*Convexifying Transformers*, Section 4, fails final persistence at canonical
update 112,750. It had already completed 20 joint ≥99% observations over
95,750–100,500. All 51 canonical tail observations from 100,000 through
112,500 pass; the next observation falls to train 1.2672% / held-out 0.7947%.
This is a collapse after confirmed generalization. The required earlier
memorization plateau was also absent. Both independent-confirmation gates fail.
The unchanged full 150,000-update budget remains in progress.

| Update | Batch | Train / held-out | Gradient L2 | Raw second-moment L2 | Raw maximum-moment L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 112,499 | 512 | 100% / 99.2483% | 0.1538 | 0.002564 | 18.2884 |
| 112,500 | 48 | 100% / 99.2483% | 0.2289 | 0.002565 | 18.2884 |
| 112,501 | 512 | 100% / 99.2483% | 0.2770 | 0.002570 | 18.2884 |
| 112,749 | 512 | 1.2672% / 0.7947% | 0.3553 | 23.7806 | 31.9410 |
| 112,750 | 48 | 1.2672% / 0.7947% | 1.2102 | 23.7568 | 31.9410 |
| 112,751 | 512 | 1.2672% / 0.7947% | 0.3450 | 23.7330 | 31.9410 |

EOS is 100% on both exhaustive splits at all six points. The failing scores
therefore concern numeric answers. The failure already exists after a full
512-example batch before the canonical observation and persists afterward.
It is not confined to the canonical observation or its 48-example batch.
The larger question of batch-size effects remains unresolved.

Gradients concern the sampled batch before its update; moment buffers,
parameters, and temperatures are read after it. Complete split scores are
evaluated after the update. The first known failure lies between observations
112,501 and 112,749, leaving 248 intervening updates without these probes.
Inverse temperatures change from approximately 1.1976–1.2522 to 1.1957–1.2450.
Small sampled endpoint gradients do not imply that every intervening gradient
was small: the second-moment norm rises by several orders of magnitude.

The frozen raw update uses `v_t = beta2*v_(t-1) + (1-beta2)*g_t^2`, with
`beta2=0.999` and coordinatewise squares. In real arithmetic, triangle
inequality and `||g^2||_2 <= ||g||_2^2` give

```text
||v_end||_2 <= beta2^n * ||v_start||_2 + (1-beta2^n) * max_t ||g_t||_2^2.
```

Substituting the recorded endpoints and `n=248` gives an approximate lower
bound of 10.40 for some intervening gradient norm. This is an inference under
the stated recurrence using rounded Float32 norm estimates, not a directly
observed gradient or a certified floating-point bound. It does not locate the
gradient, identify the trigger, or prove a causal direction. The unchanged
optimizer source and frozen recipe support the recurrence assumption.

The [record](midpoint-first-tail-collapse.json) retains all six original score
and diagnostic records, derived joint norms, the incomplete assessment through
112,750, source-prefix and manifest SHA256 values, and explicit assumptions for
the calculation. Prefix schedule and scores were checked against live raw
histories; 452 canonical observations precede or include the first tail failure.
Full-budget archives and curves will supersede this partial result. Preserve
this failure when selecting a fresh controlled adaptation after completion.
