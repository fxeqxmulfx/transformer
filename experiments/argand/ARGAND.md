# Argand

## Hyperparameters

| Hyperparameter | Value | Meaning |
|---|---|---|
| `c` | `64` in the example | Number of complex elements in a token vector. |
| `H` | `4` in the example | Number of attention heads. |
| `layers` | Model configuration | Number of distinct layers. |
| `repeat` | `1` in the short screen | Number of passes through the same layers. |
| `ffn_expansion` | `4` | Ratio of the FFN intermediate width to `c`. |
| `eps` | `1e-6` | Offset added to the vector norm in `complex_rmsnorm`. |
| `rope_base` | `10000` | Base of the LeRoPE frequency schedule. |

### Initialization

For Gaussian-initialized weights, sample the real and imaginary parts
separately. In $\mathcal N(0,\sigma^2)$, $\sigma$ is the standard deviation
of each part.

| Parameter or state | Initial value |
|---|---|
| Embedding and tied output table `E` | $\mathcal N(0,0.02^2)$ per part. |
| Attention matrices `W_Q`, `W_K`, `W_V`, `W_O` | $\mathcal N(0,0.02^2)$ per part. |
| FFN matrices `W_g`, `W_u`, `W_F` | $\mathcal N(0,0.02^2)$ per part. |
| Skip and write vectors `w_skip`, `w_write` in each sublayer | $0+0i$ in every entry. |
| LeRoPE log-frequency adjustments $\alpha_m$ | $0$ for every $m$, shared across heads and layers. |
| `s_init` of the learned score multiplier `s` | $1+0i$ in every layer. |
| SGD velocity $v_0$ | $0$ for every real coordinate. |
| WSD learning rate at the first update | $0$. |

Both gates start at $1+0i$, giving ordinary residual addition at
initialization. Each stress run uses one of the seeds `0`, `1` or `2` to
initialize `E` and the attention and FFN matrices.
[Kernite parameter initialization](src/engine/weights.rs)

### Objective, optimizer and WSD

| Setting | Value | Meaning |
|---|---|---|
| `optimizer` | `sgd-momentum` | SGD with Polyak momentum. |
| Momentum $\beta$ | `0.9` | Weight of the previous velocity. |
| Penalty $\lambda$ | `1e-4` | Coefficient of $(\lambda/2)\lVert\theta\rVert_2^2$ in the objective. |
| `weight_decay` | `0` | No decoupled decay. |
| Gradient norm limit | `1.0` | Upper limit on the global Euclidean gradient norm. |
| WSD peak learning rate $\eta_{\mathrm{peak}}$ | `1e-3` | Rate during the stable phase. |
| WSD final learning rate $\eta_{\mathrm{final}}$ | `1e-8` | Rate on the final scheduled update. |

Every update uses the full gradient of the regularized training objective

$$
F(\theta) = \frac{1}{N}\sum_{j=1}^{N}\ell_j(\theta)
  + \frac{\lambda}{2}\lVert\theta\rVert_2^2 .
$$

Here $\ell_1,\dots,\ell_N$ are the cross-entropy losses at the $N$
labeled positions of the training split. For $g_k=\nabla F(\theta_k)$
and learning rate $\eta_k$, the update is

$$
\tilde g_k = \frac{g_k}{\max(1,\lVert g_k\rVert_2)}, \qquad
v_{k+1} = 0.9 v_k + \tilde g_k, \qquad
\theta_{k+1} = \theta_k - \eta_k v_{k+1}.
$$

Microbatches of the suite's batch size visit the training split in
order. Each divides its loss gradient by $N$, so their gradients sum to
the data term. The penalty gradient $\lambda\theta_k$ is added once,
before clipping. At the first warmup update, the zero rate keeps the
weights fixed while the velocity receives the clipped gradient.
Checkpoints store the weights and velocity.
[Lean full-gradient run](../../src/Transformer/Grokking/FiniteRecovery/ArgandFullGradient.lean)

WSD (Warmup–Stable–Decay) uses the task's full `max_updates` budget $B$.
Let $k$ be the number of completed updates, $W=\lfloor 0.02B\rfloor$ and
$D=\lfloor 0.8B\rfloor$.

- **Warmup.** For $0\le k<W$, the rate rises linearly from zero as $\eta_{\mathrm{peak}}k/W$.
- **Stable.** For $W\le k<D$, the rate is $\eta_{\mathrm{peak}}$.
- **Decay.** For $D\le k<B$, let $u=(k-D)/(B-1-D)$. The rate is $\eta_{\mathrm{peak}}+(\eta_{\mathrm{final}}-\eta_{\mathrm{peak}})(1-\cos(\pi u))/2$.

For $B=100\,000$, the short screen's 1,000-update limit covers at most half
of the 2,000-update warmup.

The runner uses Kernite 0.8.0 and has no configuration flags. The
residual, score-initialization, minibatch, AdamW and AMSGradW variants
in the README ran in `experiments/phasor_value_transport` at commit
7399c3bb or earlier.
[Suite budgets](../convex_stress/suite.json) · [Optimizer kernel](src/engine/kernels.rs)

## Embedding

The embedding table `E` contains one learned complex vector for each
vocabulary token. The table has shape `vocab_size × c`. Each row contains
`c` complex numbers, or `2c` real parameters.

For a sequence of `T` tokens, select the corresponding rows from `E` to obtain
`emb = E[tokens]` of shape `T × c`.

The length of a complex vector is its number of complex elements.
`complex_rmsnorm` divides each element by the vector's Euclidean norm plus
`eps`, then multiplies it by the square root of the vector's length.

Apply `complex_rmsnorm` to each row of `emb`. The resulting `norm_emb` has
shape `T × c`. Row `n` is the normalized embedding of the token at position
`n`.

Use `norm_emb` as the initial residual state `x` of shape `T × c`.

## Skip and write gates

Each attention and FFN sublayer learns two complex vectors `w_skip` and
`w_write`, each of length `c`. Both gates use the same normalized input
`x_in`. For each token, multiply the matching complex entries of its
row and a gate vector, sum the products, and add $1+0i$. This gives one
complex number per token for each gate.

~~~text
skip_write_sublayer(x, w_skip, w_write, F, scale, ...):
    x_in = complex_rmsnorm(x)          # T × c
    g_skip = 1 + x_in @ w_skip         # T
    g_write = 1 + x_in @ w_write       # T
    y = complex_rmsnorm(F(x_in, ...))  # T × c
    return g_skip[:, None] * x + scale * (g_write[:, None] * y)
~~~

The skip gate multiplies the incoming residual `x`. The write gate
multiplies the normalized block output `y` before scaling the residual
write. Each gate applies its token's complex number to every entry of
that token vector. The sublayer returns a complex matrix of shape `T × c`.
The two sublayers add $4c$ complex parameters per layer.

## complex_attention

The input `x_in` is the normalized attention input of shape `T × c`.

**Step 1.** `W_Q`, `W_K`, and `W_V` are learned complex matrices of shape
`c × c`.

~~~text
x_in × W_Q → Q, T × c
x_in × W_K → K, T × c
x_in × W_V → V, T × c
~~~

Each entry of the result sums the elementwise products of a row of `x_in`
and a column of the learned matrix. Multiplying one row of `x_in` by all
columns gives a vector of length `c`.

**Step 2.** Split the columns of each of `Q`, `K`, and `V` into `H` equal
groups. Each group retains all `T` rows and has `d = c/H` columns. Call the
matrices for head `i` `Qi`, `Ki`, and `Vi`. With `c=64` and `H=4`, we get
`d=16`.

**Step 3.** Apply `complex_rmsnorm` to the rows of `Qi` and `Ki`.

**Step 4.** Rotate the complex numbers in `Qi` and `Ki`. Row `n` determines
the number of steps, `n−1`. The zero-based column index `m` determines the
angle per step. The first table uses rounded increments.

| Column `m` | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Angle increment | 1 | 0.562 | 0.316 | 0.178 | 0.1 |
| Row 1, zero steps | 0·1=0 | 0·0.562=0 | 0·0.316=0 | 0·0.178=0 | 0·0.1=0 |
| Row 2, one step | 1·1=1 | 1·0.562=0.562 | 1·0.316=0.316 | 1·0.178=0.178 | 1·0.1=0.1 |
| Row 3, two steps | 2·1=2 | 2·0.562=1.124 | 2·0.316=0.632 | 2·0.178=0.356 | 2·0.1=0.2 |

LeRoPE multiplies the base increment $\mathrm{rope\_base}^{-m/d}$ by
$e^{\alpha_m}$. The parameter $\alpha_m$ is a learned real number for
column `m`.

| Column `m` with `d=16` | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Angle increment | $10000^{-0/16}e^{\alpha_0}$ | $10000^{-1/16}e^{\alpha_1}$ | $10000^{-2/16}e^{\alpha_2}$ | $10000^{-3/16}e^{\alpha_3}$ | $10000^{-4/16}e^{\alpha_4}$ |
| Row 1, zero steps | $0\cdot10000^{-0/16}e^{\alpha_0}=0$ | $0\cdot10000^{-1/16}e^{\alpha_1}=0$ | $0\cdot10000^{-2/16}e^{\alpha_2}=0$ | $0\cdot10000^{-3/16}e^{\alpha_3}=0$ | $0\cdot10000^{-4/16}e^{\alpha_4}=0$ |
| Row 2, one step | $1\cdot10000^{-0/16}e^{\alpha_0}$ | $1\cdot10000^{-1/16}e^{\alpha_1}$ | $1\cdot10000^{-2/16}e^{\alpha_2}$ | $1\cdot10000^{-3/16}e^{\alpha_3}$ | $1\cdot10000^{-4/16}e^{\alpha_4}$ |
| Row 3, two steps | $2\cdot10000^{-0/16}e^{\alpha_0}$ | $2\cdot10000^{-1/16}e^{\alpha_1}$ | $2\cdot10000^{-2/16}e^{\alpha_2}$ | $2\cdot10000^{-3/16}e^{\alpha_3}$ | $2\cdot10000^{-4/16}e^{\alpha_4}$ |

The table gives the angle $\vartheta$ in radians. Multiply each element of
`Qi` and `Ki` by $\cos\vartheta+i\sin\vartheta$. After the rotation,
conjugate `Qi`.

**Step 5.** Multiply `Qi × Kiᵀ` to obtain the complex matrix `Si` of shape
`T × T`. Each entry sums the elementwise products of a row of `Qi` and a
column of `Kiᵀ`. Pairing one row of `Qi` with every row of `Ki` gives a
vector of length `T`.

**Step 6.** Divide each entry of `Si` by `sqrt(d)` to obtain `Si′`.

**Step 7.** Each layer has a learned complex multiplier `s` initialized to
`s_init`. Multiply each entry in row `n` of `Si′` by `s log(n)` to
obtain `Si″`. Count rows starting at one.

**Step 8.** Split `Si″` into its real and imaginary matrices. Set the
entries strictly above the diagonal in both matrices to `−inf`, then apply
`softmax` row by row. Combine the two results into the complex matrix `Si‴`
of shape `T × T`.

~~~text
Si‴ = softmax(mask(real(Si″)), axis=-1) + i * softmax(mask(imag(Si″)), axis=-1)
~~~

**Step 9.** Multiply `Si‴` by `Vi` to obtain `ATTNi` of shape `T × d`.

**Step 10.** Concatenate the `ATTNi` matrices along their columns to obtain
`ATTN` of shape `T × c`.

**Step 11.** Multiply `ATTN` by the learned complex matrix `W_O` of shape
`c × c`. The attention output has shape `T × c`.

## complex_ffn

The input `x_in` is the normalized FFN input of shape `T × c`. The
intermediate vectors have length `m = ffn_expansion * c`.
With the example width, this gives `m=256`.

**Step 1.** Multiply `x_in` by the learned complex matrices `W_g` and `W_u`
of shape `c × m` to obtain `G` and `U`, both of shape `T × m`.

**Step 2.** Multiply the complex numbers at matching entries of `G` and
`U`.

**Step 3.** Multiply the result by the learned complex matrix `W_F` of shape
`m × c` to obtain the output of shape `T × c`.

## Forward pass and loss

Apply `complex_rmsnorm` to the final residual state `x`. The output
projection uses the learned rows of the same table `E`. Each column of `logits` corresponds to one
vocabulary token.

~~~text
cross_entropy(logits, tokens):
    log_probs = log_softmax(logits, axis=-1)
    losses = []
    for n = 1, ..., T - 1:
        target = tokens[n + 1]
        losses.append(-log_probs[n, target])
    return mean(losses)

emb = E[tokens]                         # T × c
norm_emb = complex_rmsnorm(emb)         # T × c
x = norm_emb                           # T × c

sin_cos = lerope()
L = layers * repeat
scale = 1 / sqrt(2 * L)

for i in range(repeat):
    for j in range(layers):
        x = skip_write_sublayer(
            x, w_skip_attn[j], w_write_attn[j], complex_attention_j, scale, sin_cos
        )
        x = skip_write_sublayer(
            x, w_skip_ffn[j], w_write_ffn[j], complex_ffn_j, scale
        )

x = complex_rmsnorm(x)

logits_re = real(x) @ real(E).T          # T × vocab_size
logits_im = imag(x) @ imag(E).T          # T × vocab_size
logits = (logits_re + logits_im) / 2
result = softmax(logits, axis=-1)
loss = cross_entropy(logits, tokens)
~~~

The LeRoPE tables use `c=64` and `d=16`. The short screen used each test
case's model dimensions and supervised positions.
[Run settings](README.md#training-and-replay) · [Residual gates](src/engine/residual.rs)
