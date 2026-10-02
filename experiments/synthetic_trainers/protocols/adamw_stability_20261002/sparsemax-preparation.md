# Sparsemax preparation against the existing Lean specification

On 2026-10-02 the user requested sparsemax after the current native AdamW
300,000-update run and directed the implementation to the existing Lean code.
This is CPU/proof preparation; no scientific sparsemax run has started.

[Basic.lean](../../../../src/Transformer/GPTMini/Convex/Basic.lean) defines
`sparseWeights` as the unique minimizer of
`sum(a²)/4 - sum(a*score)/2` on the causal simplex: nonnegative weights,
sum one, and zero above the diagonal. `routing_projection_identity` shows
that four times this objective is `sum((a-score)²) - sum(score²)`.
Consequently ordinary Euclidean sparsemax uses the original scores without
any additional scaling. Existence and uniqueness are proved for every finite
real score row, with the diagonal supplying a feasible point.

[Attention.lean](../../../../src/Transformer/GPTMini/Convex/Attention.lean)
connects this minimizer to the original content scores, QK normalization,
RoPE and learned temperature, then retains the epsilon-regularized XSA
projection. It proves nonnegative weights, sum one, causal zeros, projection
optimality and bounds on the value mixture/head. Model.lean retains the
original stack, ReLU² FFN and tied embeddings. TrainingBoundary.lean gives
an exact cross-entropy Jensen counterexample for the trainable one-neuron
FFN; convexity of row inference provides no joint-training guarantee.

All 24 theorems in these four modules were individually audited with
`#print axioms`; every result uses only `propext`, `Classical.choice` and
`Quot.sound`. The [audit input](sparsemax-preparation/lean-audit.lean),
[actual output](sparsemax-preparation/lean-axioms.txt), and
[verification metadata](sparsemax-preparation/validation.json) retain the
checked sources and command. No Lean statement or proof was changed.

The new [Python adapter](../../sparsemax_attention.py) uses sorted simplex
projection and the support derivative from the repository's existing
convex_mqar implementation. It preserves every initialized parameter object,
state-dict name, tied embedding and RNG state; only attention normalization
changes. The exact real minimizer is a noncomputable Lean definition. Its
floating-point algorithm and backward pass are checked separately.

Six [CPU tests](../../tests/test_sparsemax_attention.py) pass: an independent
oracle enumerates all simplex faces and checks their KKT conditions; other
checks cover sparse examples, translation/precision, finite-difference
derivatives, causal prefix/future behavior, optimizer bindings, and exact
original outputs/all parameter gradients when softmax is restored. These
checks are implementation evidence, not learning or stability results.

The [tagged trainer](../../attention_training.py) reuses the unchanged modular
training loop, factory initialization, native optimizer, sampling, diagnostics
and checkpoint format. Its config records `attention_normalization` before
resume validation, and both paths pin the adapter and factory source hashes.
Five additional CPU checks pass: the full-width softmax trajectory matches the
original trainer exactly; sparsemax resumed from 10 to 20 updates matches an
uninterrupted run in model/native buffers, sampling and dense diagnostics;
cross-normalizer and ordinary-core resume fail before checkpoint loading;
exceptions restore the original functions; invalid normalizers/optimizers and
budgets above 300,000 are rejected. This tiny resume is a pipeline fixture,
not a scientific budget extension. The
[training receipt](sparsemax-training-validation.json) also checks the exact
mod-193 scientific initialization on CPU: 436,104 parameters, identical initial
softmax/sparsemax tensors and RNG, original corpus fingerprints, and finite
forward/backward on eight real training equations. No GPU context is created.
All 39 source fingerprints of the still-running stage remain unchanged.

Finish, verify, review and commit the current 300,000-update result before
scientific sparsemax training. Freeze a paired softmax/sparsemax comparison
with the same native AdamW, corpus, seeds, rate, decay, warmup, batching,
observations, criterion and total 300,000-update cap. Retain every budget and
failure; report complete quality/losses, episode frequency/recovery, measured
time and peak memory. The user-directed pair may be exploratory while the
all-six independent benchmark gate remains closed; it cannot certify a
repeatable architecture improvement. Preserve that gate and the outstanding
complementary mechanics. Dense attention multiplication remains dense, so
sparse weights alone imply no runtime speedup.
