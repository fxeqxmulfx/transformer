# gpt-mini Python reference

`gpt_mini.py` is restored byte for byte from `reference/model.py` at commit
`f11b6e27d3cfe6813a2876bdeec38565fc54258c`. It is the Python source referenced
by the `Transformer.GPTMini` Lean formalization.

It uses PyTorch and contains the original small forward-pass demonstration.
Restoring the file does not run that demonstration or start training.

## Lean replacement

`src/Transformer/GPTMini/Convex/Model.lean` defines a modified forward pass
with causal sparsemax attention. Each weight row is the unique solution
of a convex quadratic program. The model keeps the original parameter
records, RoPE, QK normalization, XSA, residuals, FFN, and tied embeddings.

This establishes convexity of the attention inference problem.
`Convex/TrainingBoundary.lean` proves an exact cross-entropy Jensen violation
for a trainable one-neuron instance of the original ReLU² FFN. Joint training
convexity for the stack remains a separate requirement. Different
parameterizations or penalties are not excluded by that counterexample.

The restored Python file contains the historical architecture. The
replacement is currently defined in Lean.
