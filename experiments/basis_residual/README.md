# Current GPT and conditional residual gains on Basis

Task160 asks whether the conditional gain candidate also improves the
repository's original Basis tasks. Modular division measures one kind of
generalization; these runs add depth, associative recall, and parity. No
Basis learning result is claimed until its source-pinned run is measured.

| Setting | Small/easy | Large/hard |
| --- | --- | --- |
| Width / layers / heads | 64 / 2 / 4 | 128 / 6 / 4 |
| Depth | E2, lengths32–64; final test at128 | E4, lengths32–64; validation at128 |
| Recall | 256 symbols, 8 bindings, 8 queries | 256 symbols, 16 writes, 8 overwrites, 8 queries |
| Parity | 1–16 bits, generate answer and EOS | Same task, larger model |
| Model seeds | 0 and1 | 0 and1 |

Each setting compares the current `diff_gpt` GPT at `df1e9f7` with its
conditional post-RMS gain candidate. The candidate adds one zero-initialized
bias-free row per branch; its multiplier is
`2*sigmoid(W_gate*RMS(branch_input))`. The common initial weights, logits,
gradients and RNG state are identical. See
[`adaptive_residual`](../adaptive_residual) for exact source correspondence
and [`context_gains`](../context_gains) for the finite-solution gradient test.

The task generators, tokens, labels, length distributions and full split
fingerprints must equal `lab.domain.basis.BASIS`. Both variants use data
seed1, 20,000 training examples, 512 validation and 512 test examples. The
batch seed is model seed plus10,000, as in Basis. Rates, training batches,
50-update warmup and observations every3,200 examples are unchanged.
The original recipe used AdamW and GPTMini on CPU. This comparison uses
only Schedule-Free AdamW, betas(.9,.98), decay.1, the actual normalized
source GPT, Python3.15/Torch2.15, FP32 CUDA fullgraph max-autotune and math
SDPA. Evaluation batch64 fits the local4GiB GPU; historical Basis used256.

The stopping score is **sequence accuracy**, at the first observation
reaching99%. Hard depth uses `validation/length-128`; the other settings
use ordinary validation. Parity is scored through free generation,
including EOS. The current complete Schedule-Free endpoint is retained;
no best checkpoint is selected. Test remains sealed until learning closes.
The initial100,000-update chunk is a continuation point, not an impossibility
result. Promising uncrossed cases need longer training before final test.
The second seed reverses model order in each task/size pair.

Original Basis samples can repeat inputs across splits. We keep those
splits intact, disclose their overlap, and separately report test examples
whose inputs never occurred in training. For parity, the input key is its
prompt, not the teacher-forced answer. Length transfer, sampled test and
novel-input results are distinct. CPU endpoint replay is identified as such;
the canonical validation crossing comes from the recorded CUDA history.

The mathematical depth hierarchy has specific precision, head-count and
RoPE assumptions. It does not directly prove a depth lower bound for these
four-head, exact-angle, finite-context runs. Likewise, poor finite training
is not a proof that a model cannot represent the task.

Run `./run.py check experiments/basis_residual`, then
`./run.py basis-residual freeze`, `run`, and `report`. The frozen protocol
pins all lab/optimizer code, experiment scripts, runtime and full data
fingerprints. Complete-state checkpoints preserve model, optimizer,
sampler, epoch remainders and gradient traces. Report every pair and every
unresolved case; retain the current production model if the candidate
regresses on a Basis task. Final test never chooses the architecture.
