# Sparsemax frozen-checkpoint routing and gradient diagnostic

This read-only CPU diagnostic uses the actual native AdamW / sparsemax
scientific checkpoint at update 280,000. Its 436,104 trainable parameters,
native moment counters, corpus and unchanged frozen source/proof provenance
are checked. No optimizer is constructed or stepped, and CUDA is not initialized.
The original 300,000-update run and fresh paired softmax control continue.

The exhaustive canonical observation at this checkpoint has train accuracy
99.946028% and held-out complete-RHS accuracy 6.789724%, with EOS accuracy
100%. The two CPU derivative probes below use training batches, not held-out
predictions. They do not measure generalization or additional training.

The saved sampler selects a full batch of 512 for stream update 280,001 and
the next 48-example epoch tail for stream update 280,003. Both derivatives
use **fixed weights from update 280,000**. In particular, the latter is not
the actual later GPU derivative: the intervening optimizer updates are omitted.

| Fixed-checkpoint batch | Layer | Singleton support among directly supervised attention rows | Q gradient L2 | K gradient L2 | V gradient L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 examples | 0 | 37.3047% (1,528/4,096) | 0.156415 | 0.141573 | 0.500288 |
| 512 examples | 1 | 3.7598% (154/4,096) | 0.0876594 | 0.152796 | 0.396178 |
| 48 examples | 0 | 37.2396% (143/384) | 0.00252598 | 0.00280834 | 0.00965632 |
| 48 examples | 1 | 3.9063% (15/384) | 0.00134392 | 0.00205163 | 0.00728430 |

Rows aggregate heads and the two directly supervised positions (numeric answer
and EOS), excluding the first four prefix positions. Full prefix support
histograms are separately retained, including the forced singleton first
position. Every weight is nonnegative, future-position mass is exactly zero,
and the maximum row-sum error is at most 2.9803e-7. All singleton-support
score gradients are exactly zero under the implemented support Jacobian.
Q/K gradients and all-score gradient norms are nevertheless nonzero in both
layers and batches. Thus these observations do not support complete routing
gradient freezing; they do not establish the cause of low held-out accuracy.

This is compatible with the existing Lean result: each sparsemax row minimizes
the causal-simplex Euclidean objective for **fixed Q/K scores**. The theorem
does not imply convex joint training, successful generalization, a useful
gradient for every inactive score, or agreement of CPU and GPU rounding.
No Lean source changes or new full-tree axiom-audit claim are made here.

The [archive](sparsemax-routing-gradient-probe/routing-gradient-probes.json)
retains selected corpus indices, sampler metadata, per-layer support histograms,
gradients, temperatures, checkpoint hash, frozen manifest and exact executed
script. Actual instrumented logits, loss, accuracy and every parameter gradient
match uninstrumented CPU execution exactly for both batches. A second replay
of the saved checkpoint reproduces the entire result except its local file
path. The [receipt](sparsemax-routing-gradient-probe/validation.json) and artifact
checksums retain these checks; the CPU sessions finish with code zero.
This standalone diagnostic does not add a unit test or replace the integrated
280-test CPU suite, full scientific histories or strict persistence gate.

The ignored local checkpoint must be transferred separately to reproduce the
same derivative measurement. Run into fresh output and snapshot paths:

```bash
env CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.probe_sparsemax_routes \
  --checkpoint-source experiments/runs/adamw_stability_20261002/bootstrap/sparsemax-routing-gradient-checkpoint-final.pt \
  --snapshot /tmp/sparsemax-routing-replay.pt \
  --output /tmp/sparsemax-routing-replay
```

The checkpoint SHA256 is
`cf729a88c7f5cb49d25e3f7e012aa4c0c84f90a890687f8b92fac4a29366d0e6`.
Without the saved file, this command cannot reproduce the archived weights.
Complete and review both scientific budgets before selecting the next recipe.
