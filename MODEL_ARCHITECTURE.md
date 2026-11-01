# Transformer architecture for modular generalization

Run `./run.py site` for the [interactive local diagram](sites/model_architecture/README.md).

Use a **Q/K-normalized, causal RoPE transformer with parameter-free Peri-RMSNorm,
SwiGLU, centered XSA and additive residuals**. The most supported starting point
is width 128, four blocks and four heads. Ordinary addition is the default for
the strongest completed accuracy result; fixed post-normalization branch scaling
is the measured option for an earlier generalization transition.

This recommendation concerns the local modular-division experiments. It is not
a global optimum for language, unseen primes, longer contexts or every task.

| Residual option | Coefficient after each branch output RMSNorm | Evidence | Use |
| --- | --- | --- | --- |
| `add` | 1 | All four P97/P193 controls reached stable validation; 99.957–100% final test | Default accuracy baseline |
| `scaled` | 1/sqrt(2L), about 0.353553 at L=4 | P97 validation 97.595%/98.368% at update 23,000; ordinary addition 34.966%/27.234% at that update | Earlier-onset alternative |
| `gated` | 2 sigmoid(s), s initialized to zero independently for each branch | Post-RMS amplitude recovery 12/12 synthetic cases; actual GPT integration/regressions pass | Experimental learned branch strength |

The scaled endpoints scored 97.766%/98.368% on final test and were capped before
the 99.9% stability target. Their eventual final accuracy and time to that target
are unknown. Learned gates have no full-transformer grokking comparison yet.
The implemented `gated` mode starts at coefficient 1; it does **not** multiply
the gate by 1/sqrt(2L). That combination would be a separate untested variant.

## Exact reference configuration

| Component | Setting |
| --- | --- |
| Width / blocks / heads | 128 / 4 / 4 |
| Head width | 32 |
| SwiGLU hidden width | floor((8 × 128 + 1)/3) = 341 |
| Vocabulary | Exactly P residues for the division benchmark; P=97 or 193 |
| Context capacity | 33; the measured division input contains only two tokens |
| Token embedding / readout | Learned embedding, shared with the output projection |
| Positions | Geometric RoPE on Q and K in every block, base 10,000 |
| Attention weights | Causal softmax, including the current position |
| Q/K normalization | Per-head RMSNorm before RoPE; no learned gain |
| Branch normalization | RMSNorm before and after both attention and FFN |
| Embedding / final normalization | Parameter-free RMSNorm |
| Norm epsilon in measured FP32 execution | dtype epsilon = 2^-23, about 1.19209e-7 |
| XSA | Centered clipped projection; own-value L2 clipping at 1e-12 |
| Linear biases / norm affine parameters / dropout | None / none / zero |
| Parameters, `add` or `scaled` | 798,336 at P97; 810,624 at P193 |
| Additional parameters for `gated` | 8 scalar logits, one per attention/FFN branch |

The two epsilons serve different operations: RMSNorm adds epsilon to mean-square
energy; XSA clamps an own-value L2 norm. Preserve the actual source convention.
Casting the whole model to another dtype changes default RMS epsilon; the
measured results are FP32, not a validation of mixed-precision training.

## Forward pass

Write N(x)=x/sqrt(mean(x²)+epsilon). With shared embedding matrix E:

```text
h = N(E[tokens])

repeat L = 4 blocks:
    a = N(Attention_XSA_RoPE(N(h)))
    h = h + alpha_attention * a

    f = N(W_out [SiLU(W_gate N(h)) * (W_value N(h))])
    h = h + alpha_ffn * f

logits = N(h) @ E.T
answer_logits = logits[:, -1, :]
```

The skip stream retains its identity coefficient. Each branch coefficient is
applied **after its output normalization**. Scaling the output projection's
initial weights is a different operation: the following RMSNorm largely removes
that amplitude change when epsilon is small.

Inside one attention head:

```text
q_i, k_i, v_i = projections of the normalized branch input
q_i = RoPE(position_i, RMS(q_i))
k_i = RoPE(position_i, RMS(k_i))
p_ij = softmax_j(q_i · k_j / sqrt(32)), restricted to j <= i
u_i = v_i / max(norm(v_i), 1e-12)
y_i = sum_j p_ij v_j - dot(sum_j p_ij v_j, u_i) * u_i
output = W_attention_out(concat_heads(y_i))
```

The implementation evaluates the clipped XSA expression in centered coordinates,
retains its below-epsilon remainder, and explicitly zeros the first-position
attention output. A persistent repeated-prefix guard shares equal states only
while all required earlier heads are above clipping; it cannot reactivate.
Use the existing implementation for those numerical details.

RoPE pairs channels i and i+16 within each 32-dimensional head. Its angle at
position t is t × 10000^(-2i/32), for i=0,…,15. These are general rotations,
not a fixed 90-degree transform or a single periodic position encoding.

## Construction in diff_gpt

This matches the implementation at GPT commit `379a9f2`:

```python
from diff_gpt.domain.model.residual_gpt import ResidualGPT

model = ResidualGPT(
    vocab_size=97,
    n_embd=128,
    n_head=4,
    n_layer=4,
    block_size=33,
    mixing="add",             # "scaled" for the measured earlier-onset option
    centered_xsa=True,
    rope_base=10_000.0,
    rope_layers=None,          # RoPE in every block
    pad_vocab_size_to=1,       # exactly 97 output classes for this task
    use_checkpoint=False,     # matches the small measured training model
).float()
```

Specify `mixing` explicitly: `ResidualGPT` currently defaults to `conditioned_qk`,
which is a different depth-aggregation architecture. The lab's
`ResidualTransformer` defaults to `add` and supports the measured `scaled` option.
The optional `gated` extension currently lives in diff_gpt, not that lab block.

Initialization uses N(0, 0.02) for embedding/linear weights, then redraws attention
and FFN output projections with standard deviation 0.02/sqrt(2L). Preserve both
the initialization and, for `scaled`, the separate post-RMS branch coefficient.
At fixed P, total parameters for addition are
P×d + L×(4d² + 3d×floor((8d+1)/3)); learned gates add 2L.

## Training conditions that accompany the recommendation

Use Schedule-Free AdamW with lr 0.003, betas (0.9, 0.98), epsilon 1e-8 and
weight decay 0.1. Apply ten inclusive linear warmup updates, then a constant
rate; do not also enable the optimizer's built-in warmup. Batch size is 256.
The synthetic mechanism study used a different recipe; its zero decay and
lr 0.01 are not the validated full-transformer settings.

For division, inputs are [a,b], b nonzero; the target is a×b^(-1) mod P.
Apply cross-entropy to the final position only. The constructor's optional
`targets` argument computes loss at every position, so use the benchmark's
final-answer loss rather than passing a single answer through that API.

Retain the 50%/25%/25% train/validation/test split and paired data/batch seed 11;
the measured initialization seeds were 11 and 17. Evaluate Schedule-Free's
averaged weights, and save the training weights, base iterate, moments, averaging
clock and sampler together. Keep the final test sealed during learning.

Evaluate every 1,000 updates. A stable success requires both train and validation
accuracy >=99.9% at three consecutive observations, all at or after update 5,000.
Ordinary P97 controls stopped at 40k/45k updates and P193 at 19k/11k. These numbers
describe these runs, not a sufficient epoch budget for every depth or seed.
Long plateaus require complete-state continuations within a declared budget.

Use the measured Python 3.15/Torch 2.15 FP32 path and compile with
`mode="max-autotune", fullgraph=True, dynamic=False`. For this two-token workload,
the lab's `ResidualMaxAutotune(attention_backend="math")` selects math SDPA inside
the compiled graphs. An outer-only SDPA context did not change cached graphs.
This backend choice is a short-context performance result, not a general
recommendation for long contexts. Compilation alone does not confer generalization.

## Interpretation and limits

Keep token Q/K normalization. The matched residual cohort retained it throughout,
so it does not newly isolate its causal effect. Likewise, it retained XSA, Peri
norms and SwiGLU rather than proving each individually optimal. Depth attention
remained near chance within its allotted runs; it is a research alternative,
not the selected default and not proved incapable of learning.

The new Lean gradient results explain why an amplitude gate before RMS can be
poorly conditioned and why an identity skip alone does not guarantee a nonzero
Jacobian. The scalar depth-scaled witness uses 1/D, whereas this measured
transformer uses 1/sqrt(2L); its lower bound does not transfer to arbitrary
nonlinear transformer blocks. Flat-spectrum representation bounds also do not
choose the optimizer, sufficient training time, width or depth.

For tasks with more operands, use explicit tokens for operations and structure,
and benchmark the required composition depth and input lengths. Context capacity
33 is not evidence of generalization to length 33. The repeated-token length
collision and boundary-token repair remain a separate queued task (#149).
Width 128/depth 4 is the supported reference, not a claim that larger models
cannot help or that width must grow exponentially with context.

Sources: [full-transformer comparison](experiments/residual_generalization/README.md),
[synthetic mechanisms](experiments/residual_gradient_paths/README.md),
[formal gradient statements](src/Transformer/Residual/GradientPaths.lean),
[lab residual implementation](python/src/lab/infrastructure/nn/residual_transformer.py),
and diff_gpt `domain/model/{gpt,block,mha,feed_forward,rope,residual_gpt}.py` at
commit `379a9f2`.
