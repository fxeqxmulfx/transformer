# Larger-modulus AdamW task adaptation

The user asked to make the task harder on 2026-10-02. The next controlled
calibration changes modular division from prime 97 to prime 193, retaining
the primary AdamW optimizer, 50% train fraction, seeds 0/0, width 128, two
layers, four heads, batch 512 with short epoch tails, learning rate 0.001,
decay 0.1, ten-update warmup, 150,000 updates, and unchanged phase/persistence
criterion. Prime is the only differing training-configuration field.

This is an explicit task adaptation of the mod-97/mod-15 experiments in
*Convexifying Transformers*, Section 4. The local manuscript does not claim
that prime 193 is harder or produces grokking. A larger finite task with the
same transformer block capacity is the hypothesis tested here.

| Derived quantity | Mod 97 control | Mod 193 adaptation |
| --- | ---: | ---: |
| Complete legal operand pairs | 9,312 | 37,056 |
| Exhaustive train / held-out equations | 4,656 / 4,656 | 18,528 / 18,528 |
| Atomic-token vocabulary | 239 | 335 |
| Total parameters | 423,816 | 436,104 |
| Transformer block parameters | 393,224 | 393,224 |
| Tied embedding parameters | 30,592 | 42,880 |
| Full batches / final batch per epoch | 9 / 48 | 36 / 96 |
| Examples consumed at 150,000 updates | 69,840,000 | 75,113,536 |
| Prompt tokens / supervised targets | 5 / 2 | 5 / 2 |

Domain size rises by 3.9794 times. The vocabulary adds 12,288 embedding
parameters (about 2.9% of the old total); blocks and their dimensions are
unchanged. Epoch lengths, exposure and exhaustive evaluation cost also change.
This control therefore does not isolate arithmetic difficulty from vocabulary,
initialization, data volume, exposure, or parameter changes. Shared seed numbers
across primes do not imply identical common initial weights. Report complete
costs and memory rather than assuming a matched-budget or timing speedup.

The [finite corpus inspection](larger-modulus-preparation.json) independently
checks every equation by modular inverse and multiplication, unique legal
pairs, exhaustive disjoint splits, and all legal numerator/denominator/answer
classes in both partitions. Both corpora use the tracked complete oracle and
licensed author-compatible token format; no data download is needed. This
coverage is not a learning guarantee or a length/novel-token transfer result.

A real [38-update CPU smoke](larger-modulus-cpu-smoke.json) tests the full
width-128 model, exhaustive scoring, native AdamW moments, every-update
gradient trace, the 96-example tail at update 37, and return to a full batch
at 38. Its portable archive verifies without PyTorch. It is implementation
evidence only, not a scientific stability or grokking result. The current
GPU optimizer pair continues without a second GPU trainer.

Freeze the new manifest before scientific training. The custom launcher
requires the complete verified parent mod-97 optimizer comparison before
starting, so both original 150,000-update budgets and failures remain intact:

```bash
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_larger_modulus
```

Use `--check-only` to inspect the frozen manifest without requiring parent
completion or starting training. The new plan must preserve source, local
paper, environment and instrumentation fingerprints from the parent controls.
Archive the full outcome, all dense and sampled histories, CSV and PNG/PDF
figures; verify and commit it even if the intended effect fails. Only a complete
passing calibration can open the unchanged independently frozen six-case
confirmation gate. No architecture comparison is launched by this preparation.
