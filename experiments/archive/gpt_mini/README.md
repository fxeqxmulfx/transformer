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

## Tiny Shakespeare benchmarks

The gpt_mini codebase (`legacy/gpt_mini/src/`, last in commit `2aec5b9`)
compared optimizers on a small GPTMini trained on the Tiny Shakespeare
characters, under softmax and under sparsemax attention, on an RTX 3050
Laptop GPU. Each benchmark kept its plan and its measured results, and each
built on the one before:

| Benchmark | What changed | Lowest mean test cross entropy |
| --- | --- | --- |
| [`optimizer_benchmark`](optimizer_benchmark) | 18 optimizers, 1000 updates each | AdamW under both attentions |
| [`magma_benchmark`](magma_benchmark) | six methods more, Magma among them | Magma over Muon under both |
| [`patience_benchmark`](patience_benchmark) | the 24 stopped on validation patience | stopped after 1 of 144 runs |
| [`compiled_benchmark`](compiled_benchmark) | the model compiled | stopped after 6 of 144 runs |
| [`full_compile_benchmark`](full_compile_benchmark) | the whole training step compiled | AMSGrad under both |
| [`amsgrad_extensions_benchmark`](amsgrad_extensions_benchmark) | AMSGradW and AMSGradMD added | AMSGradW under both |

The lab trains the last protocol as
[`experiments/shakespeare_zoo`](../../shakespeare_zoo) and its AMSGradW arm
as [`experiments/shakespeare_amsgradw`](../../shakespeare_amsgradw), and
cites `gpt_mini.py` as the reference GPTMini.

## Synthetic trainers

The [synthetic trainer suite](../synthetic_trainers/README.md) includes MQAR,
composed lookup, prefix languages, and RASP-family sequence and arithmetic tasks
with free generation. The [plan](../synthetic_trainers/PLAN.md) records the goal of
evaluating mini GPT architecture changes by quality at a fixed budget and time
to reach a predefined quality target.
