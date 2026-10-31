module

public import Transformer.Rotate90.Training.Sampling.Escape

/-!
# Sampled training can break a full-table alias invariant

Source: `review/flat_spectrum.md`, *Sampled updates can break aliases*. At a
collapsed tied embedding, the actual minibatch gradient equals the difference
between uniform probability and empirical target frequency, times the common
hidden vector. Every input lookup, residual block, SiLU gate and RMSNorm remains
in the differentiated function. Hidden derivatives cancel as common logit shifts.

The exact mean squared transverse singleton signal is 2/card(V) times the hidden
coordinate squared, even though a complete division table has zero embedding
gradient there. A valid singleton produces opposite target/competitor gradient
signs, and one source Schedule-Free AdamW step separates their rows. An explicit
nonzero width-16, hidden-43 witness works at every positive finite depth and in
both query and evaluation buffers. All general hypotheses have examples.

These are local escape and finite-sample statements, not a proof of typical
random-start convergence, generalization, or escape from every subgroup code.
-/
