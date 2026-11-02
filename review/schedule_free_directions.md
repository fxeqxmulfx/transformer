# Schedule-Free direction diagnosis

Task166 follows the user's request to reason before spending laptop time.
The [prediction and derivation](../experiments/basis_control/analysis/schedule_free_directions.md)
precede the saved-state measurements. Six closed endpoints supply one next
training minibatch each. There are no optimizer updates, new learning curves
or final-test predictions. Python3.15.2 and PyTorch2.15.0 use one CPU thread.

The proposed common mechanism is rejected at these endpoints. In all three
states with zero exact validation sequences, averaging opposes less than
0.008% of the fresh gradient's predicted training-loss decrease, against a
declared50% threshold. No individual parameter tensor in those three states
passes that threshold either. The only tensor that passes it belongs to the
already successful historical GPTMini, with a tiny fresh directional loss
contribution. Presence alone therefore would not establish a generalization
failure.

| Saved state | Updates | Exact validation | Averaging opposition to fresh predicted descent | Predicted full decrease / current batch loss |
| --- | ---: | ---: | ---: | ---: |
| Historical GPTMini, seed0 | 12,200 | 99.02% | 0.0535% | 8.098 |
| Historical GPTMini, seed1 | 100,000 | 0% | 0.0071% | 5.078 |
| Current GPT, seed0 | 100,000 | 0% | 0.0050% | 8.585 |
| Current GPT, seed1 | 123,550 | 99.02% | 0.1365% | 0.063 |
| Learned temperature, seed0 | 100,000 | 0% | 0.0053% | 4.219 |
| Learned temperature, seed1 | 62,900 | 81.25% | 0.0697% | 0.602 |

The last column is a post-measurement analytical check, not an additional
predeclared hypothesis. Cross-entropy is nonnegative. Whenever
`L(y) + dot(grad L(y), delta) < 0`, that first-order prediction cannot be the
actual loss at `y+delta`. Its remainder must be at least
`-L(y)-dot(grad L(y),delta)`. Four endpoints meet this condition, including
the successful historical GPTMini. This establishes substantial nonlinearity
at the proposed step, not that the actual step increases loss or delays
grokking. A larger-rate sweep is not justified by these directional numbers.

The algebraic witnesses explain why inspecting the fresh coefficient alone
would also mislead: for a constant Adam direction without warmup or decay,
the working-weight increment after the first step is0.55 times the base step
at beta1=0.9. A reversing direction can instead be opposed by the average.
The tests check the constant-direction closed form, the reversal's change
from descent to ascent, preservation of diagnostic inputs and a stateful
FP64 optimizer update with decay. These are numerical checks of the stated
algebra, not a new Lean convergence theorem.

The probe verifies closed checkpoint hashes, frozen model/optimizer/loss and
sampler operation hashes, exact recipes and task fingerprints. Dispatch
additions since the historical protocols were reviewed atccf91e8e: the same
Transformer/CanonicalResidual classes and supervised objectives remain in
use. Per-parameter contributions, sampler indices and all checked hashes are
retained in the [result](../experiments/basis_control/analysis/schedule_free_directions.json).
The current CPU gradient and FP64 decomposition are not a bit-exact replay
of a compiled CUDA step. Endpoints differ in age and progress, and a single
minibatch cannot describe the complete learning trajectory.

A second invocation reproduces the entire six-state JSON byte-for-byte. It
takes 24.17 elapsed seconds (23.27 CPU user seconds and 1.61 system seconds),
while the mandatory CPU test suite is also running; this is a diagnostic
cost record, not an isolated throughput benchmark. No GPU study is launched.

The complete result SHA256 is
`c3e63d6d56c6adb590f11aab7258e535c2d6f7ad4dc2e6ccc97b95fc13061330`.
The two focused tests pass in 0.883 seconds. The required `./run.py test`
checks 515 lab tests in 923.811 seconds with 13 CUDA skips, then 104 optimizer
tests in 14.079 seconds with 14 skips. Reference archive checks also pass.
This is same-assistant verification; no review is delegated.

No optimizer or architecture change is promoted. The next distinct question
is whether normalization makes the depth-scaled projection initialization
change effective parameter learning rates while removing its forward scale.
That requires its own mathematical prediction and cheap check before any
training; it is not inferred from the failed averaging hypothesis.
