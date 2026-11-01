# Transformer architecture for modular generalization

Run `./run.py site` for the [interactive local diagram](sites/model_architecture/README.md).

Use a **Q/K-normalized, causal RoPE transformer with parameter-free Peri-RMSNorm,
SwiGLU, centered XSA and fixed additive residual scaling**. The reference size is
width 128, four blocks and four heads. Each normalized attention/FFN branch is
multiplied by 1/sqrt(2L), about 0.353553 at four blocks. `diff_gpt.GPT` now has one
architecture: there are no residual-mixing, legacy-XSA or per-layer RoPE switches.
Its RMS backward also preserves small finite-epsilon radial gradients.

This choice concerns the local modular-division experiments. It is not a global
optimum for language, unseen primes, longer contexts or every task. Historical
comparisons remain evidence, not selectable production architectures:

| Historical comparison | Coefficient after output RMSNorm | Evidence |
| --- | --- | --- |
| Ordinary addition | 1 | All four P97/P193 controls reached stable validation; 99.957–100% final test |
| Fixed scaling, selected architecture | 1/sqrt(2L) | Both P97 seeds crossed 99% validation at 24k updates; ordinary controls first crossed at 32k/37k |
| Learned gates, removed experimental option | 2 sigmoid(s), initially 1 | Post-RMS amplitude recovery 12/12 synthetic cases; no full-transformer grokking comparison |

The original scaled runs were capped at 23k. Their complete-state continuations
needed only 1,000 more updates: validation 99.270%/99.098%, final test 99.227%/98.711%.
They stopped at the user's first 99% validation target, using 25%/35% fewer updates
than the paired ordinary controls. The test split was reused from the earlier
report; seed 17's test remains below 99%. Time to the stricter 99.9% stability
target and eventual final accuracy remain unknown. Learned gates have no
full-transformer grokking comparison yet.
The historical gate started at coefficient 1; no gate-times-depth-scale combination was tested.

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
| Parameters | 798,336 at P97; 810,624 at P193 |
| Residual coefficient | Fixed 1/sqrt(2L); no learned gate |
| RMS backward | Separate tangent/radial terms, computed in FP64 and returned in input dtype |

The two epsilons serve different operations: RMSNorm adds epsilon to mean-square
energy; XSA clamps an own-value L2 norm. Preserve the actual source convention.
Torch 2.15 uses FP32 opmath epsilon for FP16/BF16/FP32 and FP64 epsilon for
FP64. Forward/dtype regressions pass, but the measured grokking runs are FP32.
This does not establish mixed-precision training quality.

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

The current production constructor is:

```python
from diff_gpt.domain.model.gpt import GPT

model = GPT(
    vocab_size=97,
    n_embd=128,
    n_head=4,
    n_layer=4,
    block_size=33,
    rope_base=10_000.0,
    pad_vocab_size_to=1,       # exactly 97 output classes for this task
    use_checkpoint=False,     # matches the small measured training model
).float()
```

The former `ResidualGPT` class and its architecture switches were removed.
Use the historical Git revision to replay an old architecture. The lab retains
explicit comparison blocks and frozen results for research. Checkpoint verification
uses a new architecture marker so old ordinary-model weights are not silently
interpreted as this scaled model.

Initialization uses N(0, 0.02) for embedding/linear weights, then redraws attention
and FFN output projections with standard deviation 0.02/sqrt(2L). Preserve both
the initialization and the separate post-RMS branch coefficient.
At fixed P, total parameters for addition are
P×d + L×(4d² + 3d×floor((8d+1)/3)).

## RMS backward repair

For m=mean(x²), r=1/sqrt(m+epsilon), and p=x*mean(x*g)/m, the real RMS VJP is
`r*(g-p) + epsilon*r³*p`. At x=0 it is `g/sqrt(epsilon)`. The implementation
computes this decomposition in FP64 and returns the input dtype; native forward
values are preserved. Lean proves the identities and conditioning ratio.

In 48 matched synthetic runs, all four unit-scale pre-RMS amplitude cases
recovered with this backward; all four native controls stayed stuck. Post-RMS
control curves matched exactly. Scale-1000 cases still failed: tiny true gradients,
upstream rounding and optimizer conditioning remain. An exploratory GPU profile
measured about 9% forward/backward overhead. The full-transformer grokking curves
below predate this backward repair; fixed-weight division regressions pass, but
faster grokking after retraining is not established.

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
Task154 separately stopped the continued scaled models at their first 99%
validation crossing, at 24k. This is not the preceding 99.9% stability rule.

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
[fixed-scale continuations](experiments/residual_scaled_long/README.md),
[synthetic mechanisms](experiments/residual_gradient_paths/README.md),
[RMS recovery benchmark](experiments/rms_backward/README.md),
[formal gradient statements](src/Transformer/Residual/GradientPaths.lean),
[RMS backward identities](src/Transformer/Residual/RMSBackward.lean),
[lab residual implementation](python/src/lab/infrastructure/nn/residual_transformer.py),
and current diff_gpt `domain/model/{gpt,block,mha,feed_forward,rope,rms_norm}.py`.
Historical architecture comparisons used commit `379a9f2`; the RMS repair is
commit `7314dec`; the single-architecture migration is `df1e9f7`.
