# Argand

## Hyperparameters

| Hyperparameter | Value | Meaning |
|---|---|---|
| `c` | `64` in the example | Number of complex elements in a token vector. |
| `H` | `4` in the example | Number of attention heads. |
| `layers` | Model configuration | Number of distinct layers. |
| `repeat` | `1` in the short screen | Number of passes through the same layers. |
| `S` | `4` | Number of mHC residual streams. |
| `ffn_expansion` | `4` | Ratio of the FFN intermediate width to `c`. |
| `eps` | `1e-6` | Offset added to the vector norm in `complex_rmsnorm`. |
| `rope_base` | `10000` | Base of the LeRoPE frequency schedule. |
| `write_gain` | `2` | Scale applied to the sigmoid write amplitudes. |
| `mix_diagonal_bias` | `2` | Offset added to the diagonal of the mixing logits. |
| `sinkhorn_cycles` | `20` | Number of column and row normalization cycles. |

### Initialization

For Gaussian-initialized weights, sample the real and imaginary parts
separately. In $\mathcal N(0,\sigma^2)$, $\sigma$ is the standard deviation
of each part.

| Parameter or state | Initial value |
|---|---|
| Embedding and tied output table `E` | $\mathcal N(0,0.02^2)$ per part. |
| Attention matrices `W_Q`, `W_K`, `W_V`, `W_O` | $\mathcal N(0,0.02^2)$ per part. |
| FFN matrices `W_g`, `W_u`, `W_F` | $\mathcal N(0,0.02^2)$ per part. |
| First 12 columns of each `W_mhc` for amplitude logits | $\mathcal N(0,(0.01/\sqrt{Sc})^2)$ per part. |
| Last 12 columns of each `W_mhc` for angles | $0+0i$ in every entry. |
| LeRoPE log-frequency adjustments $\alpha_m$ | $0$ for every $m$, shared across heads and layers. |
| `s_init` of the learned score multiplier `s` | $1+0i$ in every layer. |
| AMSGradW buffers $m_0$, $v_0$, $v_{\max,0}$ | $0$ for every real coordinate. |
| WSD learning rate at the first update | $0$. |

For $S=4$ and $c=64$, the amplitude weights of `W_mhc` have standard
deviation $0.000625$. The initial read gates, write gates, and mixing
coefficients equal their real amplitudes computed from `X` because all
their angles start at zero.

Each stress run uses one of the seeds `0`, `1` or `2`. A generator seeded
with `seed` initializes `E` and the attention and FFN matrices. A separate
generator seeded with `seed + 20_000` initializes the mHC amplitude weights.
[Kernite parameter initialization](src/phasor_v2/weights.rs)

### Optimizer and WSD

| Setting | Value | Meaning |
|---|---|---|
| `optimizer` | AMSGradW | Uses the running maximum of the second moment. |
| `beta1` | `0.9` | Weight of the previous first moment. |
| `beta2` | `0.98` | Weight of the previous second moment. |
| Optimizer epsilon | `1e-8` | Added to the denominator after the square root. |
| `weight_decay` | `0.1` | Decoupled decay with multiplier $1-0.1\eta$ on the pre-update weight. |
| Gradient norm limit | `1.0` | Upper limit on the global Euclidean gradient norm. |
| WSD peak learning rate $\eta_{\mathrm{peak}}$ | `1e-3` | Rate during the stable phase. |
| WSD final learning rate $\eta_{\mathrm{final}}$ | `1e-8` | Rate on the final scheduled update. |

Both moments use the gradient after clipping. The maximum is taken over the
raw second moments and retained throughout training. On update $t$, counted
from one, bias correction uses $m_t/(1-\beta_1^t)$ and
$v_{\max,t}/(1-\beta_2^t)$.

WSD (Warmup–Stable–Decay) uses the task's full `max_updates` budget $B$.
Let $k$ be the number of completed updates, $W=\lfloor 0.02B\rfloor$ and
$D=\lfloor 0.8B\rfloor$.

- **Warmup.** For $0\le k<W$, the rate rises linearly from zero as $\eta_{\mathrm{peak}}k/W$.
- **Stable.** For $W\le k<D$, the rate is $\eta_{\mathrm{peak}}$.
- **Decay.** For $D\le k<B$, let $u=(k-D)/(B-1-D)$. The rate is $\eta_{\mathrm{peak}}+(\eta_{\mathrm{final}}-\eta_{\mathrm{peak}})(1-\cos(\pi u))/2$.

For $B=100\,000$, the short screen's 1,000-update limit covers at most half
of the 2,000-update warmup.

Select AMSGradW in Kernite with `ARGAND_OPTIMIZER=amsgradw` and set
`ARGAND_AMSGRADW_BETA2=0.98`.
[Suite budgets](../convex_stress/suite.json) · [AMSGradW run settings and results](README.md#argand-amsgradw-with-a-larger-second-moment-coefficient)

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

Stack `S` identical copies of `norm_emb` to form the initial state `X` of
shape `T × S × c`. Each position has `S` complex vectors of length `c`, one
for each stream.

## mHC with complex gates and independent phases

An mHC sublayer reads one vector per token from the `S` streams in `X`,
passes it through attention or the FFN, and writes the result back into the
`S` streams.

### Step 1. Compute the gate parameters

For each position `n`, concatenate the `S` vectors in `X[n]` into one
complex vector of length `Sc`. Apply `complex_rmsnorm` to this vector and
multiply it by the learned complex matrix `W_mhc` of shape
`(S * c) × (2S + S²)`.

The projection produces `2S + S²` complex numbers per token. List their
coordinates in the order `Re(r₁), Im(r₁), Re(r₂), Im(r₂), …`. For `S=4`,
the 24 complex numbers yield 48 real coordinates for the logits and angles.

| Coordinates | Purpose |
|---|---|
| 1–4 | Read amplitude logits for the four streams |
| 5–8 | Write amplitude logits for the four streams |
| 9–24 | Mixing amplitude logits arranged as a `4 × 4` matrix |
| 25–28 | Four read angles |
| 29–32 | Four write angles |
| 33–48 | Sixteen mixing angles arranged as a `4 × 4` matrix |

Each attention and FFN sublayer has its own `W_mhc` matrix. Recompute the
gates from the current `X` each time the sublayer runs.

### Step 2. Form the complex coefficients

Compute two complex numbers for each stream at each position.

$$
g_{\mathrm{read}} = \operatorname{sigmoid}(\ell_{\mathrm{read}})\,e^{i\phi_{\mathrm{read}}}
\qquad
g_{\mathrm{write}} =
\mathrm{write\_gain}\cdot\operatorname{sigmoid}(\ell_{\mathrm{write}})\,e^{i\phi_{\mathrm{write}}}.
$$

On an Argand diagram, multiplying by a gate scales each complex element
by the gate's amplitude and rotates it by the gate's angle.

To mix the streams, arrange the `S²` mixing logits as an `S × S` matrix and
add `mix_diagonal_bias` to each diagonal entry. Exponentiate each entry,
then apply `sinkhorn_cycles` cycles of Sinkhorn normalization. In each
cycle, divide each column by its sum and then each row by its sum. The
resulting real amplitude matrix `P[n]` has shape `S × S`.

Compute a complex weight for each pair of streams.

$$
R[n,a,b] = P[n,a,b]\,e^{i\theta[n,a,b]}.
$$

Here `b` is the source stream and `a` is the output stream. Each of the `S²`
connections has its own angle $\theta[n,a,b]$, computed from the token state
at position `n`.

### Step 3. Read from the streams

At each position `n`, multiply each of the `S` vectors `X[n,b]` by its
complex read gate and add the results.

$$
z[n] = \sum_{b=1}^{S} g_{\mathrm{read}}[n,b]\,X[n,b].
$$

The resulting `z` has shape `T × c`. Normalize each row, pass the result
through attention or the FFN, and normalize the block output.

~~~text
x_in = complex_rmsnorm(z)          # T × c
y = complex_rmsnorm(F(x_in, ...))  # T × c
~~~

`F` is the attention or FFN block of the current layer.

### Step 4. Mix the streams and write the block output

For each output stream `a`, combine the input streams using the weights
from row `a` of `R[n]`. Add the block output multiplied by `scale` and by
the complex write gate for stream `a`.

$$
X_{\mathrm{new}}[n,a]
=
\sum_{b=1}^{S} R[n,a,b]\,X[n,b]
+
\mathrm{scale}\cdot g_{\mathrm{write}}[n,a]\,y[n].
$$

Compute all `S` output streams from the same input `X`. Steps 1–4 define
`mhc_sublayer(X, W_mhc, F, scale, ...)`, which returns `X_new` of shape
`T × S × c`.

## complex_attention

The input `x_in` is the `T × c` complex matrix from step 3 of the mHC
sublayer.

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

The input `x_in` is the `T × c` complex matrix from step 3 of the FFN's mHC
sublayer. The intermediate vectors have length `m = ffn_expansion * c`.
With the example width, this gives `m=256`.

**Step 1.** Multiply `x_in` by the learned complex matrices `W_g` and `W_u`
of shape `c × m` to obtain `G` and `U`, both of shape `T × m`.

**Step 2.** Multiply the complex numbers at matching entries of `G` and
`U`.

**Step 3.** Multiply the result by the learned complex matrix `W_F` of shape
`m × c` to obtain the output of shape `T × c`.

## Forward pass and loss

After the final layer, average the `S` streams at each position and
normalize the resulting vectors. The output projection uses the learned
rows of the same table `E`. Each column of `logits` corresponds to one
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
X = stack([norm_emb] * S, axis=1)        # T × S × c

sin_cos = lerope()
L = layers * repeat
scale = 1 / sqrt(2 * L)

for i in range(repeat):
    for j in range(layers):
        X = mhc_sublayer(X, W_mhc_attn[j], complex_attention_j, scale, sin_cos)
        X = mhc_sublayer(X, W_mhc_ffn[j], complex_ffn_j, scale)

x = mean(X, axis=1)                     # T × c
x = complex_rmsnorm(x)

logits_re = real(x) @ real(E).T          # T × vocab_size
logits_im = imag(x) @ imag(E).T          # T × vocab_size
logits = (logits_re + logits_im) / 2
result = softmax(logits, axis=-1)
loss = cross_entropy(logits, tokens)
~~~

The LeRoPE tables use `c=64` and `d=16`. The short screen used each test
case's model dimensions and supervised positions. The Kernite runner
selects this initialization with `ARGAND_SCORE_INIT=1`.
[Run settings](README.md#argand-score-initialization-at-1) · [mHC implementation](src/phasor_v2/complex_mhc.rs)
