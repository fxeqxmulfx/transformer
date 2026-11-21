# A block of the main model: LeRoPE against other positional schemes

Request, 2026-11-20: check that the current block with LeRoPE is optimal. Asked what that
means, the user answered: it approximates better than the alternative variants of the block, by
function classes in Lean, as `review/complex_ffn.md` did for the FFN. ReRoPE and its variants are
out of this note by the user's later instruction. Lean: `Transformer.LeRoPEBlock`
(`src/Transformer/LeRoPEBlock/`), independent of `Transformer.ReRoPEBlock`.

## Sources

- DiffGPT, `~/python_projects/gpt` at `5a37c51`, `research/main_model/README.txt`,
  *Architecture*: `q_i = LeRoPE_i(R_d(W_Q z_i))`, `k_i = LeRoPE_i(R_d(W_K z_i))`,
  `p_ij = softmax_{j ≤ i}((ln N_i / d)(Re Z_ij + β (Im Z_ij)² / d))`, `N_i = i + 1`, `d = D/H`;
  LeRoPE's angle `i 10000^{−2m/d} e^{α_m}`, the `α_m` shared by Q, K, heads and layers;
  `β = a/√(1 + a²)`, one zero-initialized coordinate per layer and head.
- LeRoPE, Karypis et al., arXiv:2607.10134v1, §3.1, Eq. (5): the learned frequency
  `θ̂_m = e^{α_m} ω_m` is positive and reaches every positive target
  (`Transformer.RoPE.LeRoPE.learned_frequency_surjective`).
- The alternatives are the arms of `experiments/alibi_kerple` (no rotation and a distance bias
  added to the logit, or none) and the fixed-frequency RoPE the LeRoPE frequencies start from.

## The block

Everything but the attention acts on one position at a time: the mHC routers and their Sinkhorn
maps, the RMS normalizations, `W_O`, the FFN and the writes. So at the query `i` the block is
`Φ(x_i, (attend_{j ≤ i}(F_h(i, x_i, x_j, i − j), v_h(x_j)))_h)` (`block`), with `x_j` all that the
block reads at the position `j`. A positional scheme is a choice of the scores `F`; `Φ`, the
projections, the values and the number of heads range over everything, so every statement below
covers the main model's own `Φ`. `leropeLogit` is the main model's head;
`leropeLogit_sub` checks that it is LeRoPE's turn of the query by `i` and of the key by `j`.

How to read a verdict: `A ⊆ B` is "every block of `A` is a block of `B`" (exact containment) or
within any `ε` of one on any finite set of inputs (containment up to closure); a separation is a
target in `B` that every block of `A` misses by a positive constant.

## Results

LB1, no positional encoding is blind to the order of the keys (`Basic`). A block whose scores
ignore the distance, as the `nope` arm's do, is unchanged when the keys `j < i` are permuted
(`block_noPos_perm`), for every `Φ`, every scores and every values. So a block of that class
within `δ` of any target `T` on a set holding `x` and its permuted copy has
`‖T x − T x'‖ ≤ 2δ` (`norm_sub_le_of_noPos`).

LB2, LeRoPE's π head sees it (`PiHead`, `PiSwap`, `NoPE`). One channel of the main model can
have the frequency `π` (`α` is free). A head whose query and key are one `q` for every token,
supported on that channel, with `β = 0`, scores the distance `δ` by `ln(i + 1) κ (−1)^δ`,
`κ = qᵀq/d` (`leropeLogit_pi`). Swapping the values of two keys changes an attention by
`(e^{s_a} − e^{s_b}) / ∑ e^{s}` times `u_a − u_b` (`attend_sub_attend_swap`). At the query
`n + 2` the first two keys lie at distances of opposite parity and score `±κ ln N`, `N = n + 3`,
so swapping them changes the head by at least `(1 − N^{−2κ}) / N ‖u_a − u_b‖`
(`norm_piHead_sub_swap`). Hence every block with no positional encoding, with any `Φ`, any number
of heads, any scores and any values, misses the π head, on `x` and its swap, by
`δ ≥ (1 − N^{−2κ}) / (2N) ‖u_a − u_b‖` (`gap_noPos_piHead`). At `κ = 1` and `N = 128` this is
about `0.0039 ‖u_a − u_b‖`: the π head spreads its weight over half of the keys, so this proves
that the order is seen, not that it is seen well.

LB3, LeRoPE loses nothing to it (`TwoPi`, `Limit`). Positions are natural numbers, and the
frequency `2π` turns a natural distance by a multiple of `2π`: it leaves its plane unrotated, as
the frequency `0` does (`rotary_replace_zero`). LeRoPE's learned scales reach `2π`. So every
rotary block with frequencies `θ ≥ 0` (RoPE's grid, partial RoPE with its unrotated planes, no
rotation at all) is a LeRoPE block with the same `Φ`, projections, `β`, `d` and values, on every
sequence and every query (`exists_pos_block_eq`, with LeRoPE's own parametrization in
`exists_alpha_block_eq`). The block with no positional encoding is the case `θ = 0`
(`exists_lerope_eq_noPos`). By continuity small frequencies give it up to `ε` on any finite set
of sequences too (`exists_lerope_close`), the reading that survives a perturbation.

## Verdict, so far

| alternative | LeRoPE's blocks contain it | it contains LeRoPE's | where |
| --- | --- | --- | --- |
| no positional encoding (`nope`) | yes, exactly | no: the π head | LB1–LB3 |
| RoPE, fixed grid; partial RoPE | yes, exactly (`α = 0`; unrotated planes at `2π`) | not separated in this class | LB3; rounded model below |
| distance biases (ALiBi, KERPLE, −ln(1 + r)) | not decided | not decided | task results below |
| Hermitian score (`β = 0`) | yes: the main block at `β = 0` | not decided | `phaseLogit` |
| constant gain (plain softmax) instead of SSMax's `ln N` | not decided | no, for one head's retrieval: bounded scores fade at long context, `s ln n` with `2 s B > 1` recalls | SA3, SA5 below |

So the block with LeRoPE is strictly better than a block with no positional encoding, and loses
nothing to RoPE, partial RoPE or any rotary scheme with frequencies `θ ≥ 0`. The class does not
decide the other rows, and nothing here says that LeRoPE is optimal.

## Why the class does not decide the other rows

With an arbitrary `Φ` and finite inputs only an exact invariance of the smaller class separates.
RoPE, ALiBi and KERPLE blocks have none: the distance enters every score, and a head that
separates the finitely many inputs followed by an arbitrary `Φ` can produce any target on them.
Hand argument, not in Lean. So the rows below separate by what the class leaves out (rounding,
bounded or restricted `Φ`, depth, length), not by function classes of the blocks as stated.

## What the library already proves about them

All of it is in the rounded model of `review/mrcr.md` (precision `s = 0` unless said), and each
statement is about a construction, not about every transformer.

- RoPE's grid against LeRoPE, one head. A RoPE head of the first layer, whose planes all rotate at
  frequencies outside `2πℤ`, weighs keys arbitrarily far (`Transformer.MRCR.exists_one_le_weight_grid`),
  so it does not attend to the token before alone. LeRoPE learns a frequency `2π` and an unrotated
  plane, and then the weight is `1` at the token before and `0` elsewhere
  (`Transformer.MRCR.exists_lerope_partial`). This separates the two in the rounded model; with
  the real softmax the far keys come back within any margin for both, and nothing is separated.
- ALiBi and KERPLE against LeRoPE, five layers on the spelled asks of MRCR. Five rounded layers
  with ALiBi, KERPLE or SSMax of either answer them (`Transformer.MRCR.PhraseKeyLogits`). The head
  that reads the token before with a margin does not exist with RoPE, YaRN or LeRoPE at any
  frequencies (`Transformer.MRCR.not_readsPrev_lerope`), and the construction fails with them. Only
  the construction is ruled out: LeRoPE's unrotated plane reads the token before without a margin
  (`exists_lerope_partial`), and the five layers are not built on it.
- The scale, for one head whose output is the answer (`review/flat_spectrum.md`). Softmax with
  bounded scores answers the worst dictionary of values `±1` with an error above any `ε < 2` from
  some number of keys on (`Transformer.Grokking.FlatSpectrum.Recall.exists_lt_softmaxAttention_recall_error`,
  SA3), while the scale `s log n` fades if `2 s B < 1` and recalls if `2 s B > 1`
  (`oneHot_logScaleAttention_threshold`, SA5). The main model has `s = 1/d` on raw scores up to
  `d` (Peri-RMS: `‖q‖ ‖k‖ ≤ d`), so `2 s B` can reach `2`, above the threshold: a statement of what
the head can do, not of what training finds.

The FFN branch of the same block is in `review/complex_ffn.md`.

## Not claimed

- The real `Φ`. The witness is the π head's own output (`Φ` = that head), against an arbitrary `Φ`
  on the alternative's side. The main model's own `Φ` (RMS, `W_O`, the mHC writes, the FFN) is one
  fixed map; whether it keeps the head's swap visible depends on how it treats that direction, and
  nothing here bounds that.
- That the main model realizes the π head exactly. It needs a query and a key that agree across
  tokens on the direction `W_Q`, `W_K` read, supported on the channels whose frequency is `π`
  (reachable: `α` is free), and `κ = qᵀq/d` below `1` (Peri-RMS: `‖R_d(·)‖² ≤ d`). The statement
  takes `q` constant by hypothesis; a head close to it is what the model would learn.
- Depth. A stack of blocks with no positional encoding can leak the position through the causal
  mask's counts into later layers; the statements are about one block acting on arbitrary token
  sequences.
- Per-head frequencies. LeRoPE's `α` is shared by Q, K, heads and layers; a scheme with its own
  frequencies in each head is not covered by the containments.
- Training, length generalization, floating point (the exact `2π` of LB3 is a statement over the
  reals) and the gpt runs.
- The size of the gap: `0.0039 ‖u_a − u_b‖` at `N = 128`.

## Open

The distance biases, the Hermitian score and the constant gain are not decided by a class
comparison here. A decision needs a restriction the main model has and the class lacks, such as a
bound on `Φ`'s Lipschitz constant, a number of layers, or the rounded model; the cohort
`experiments/alibi_kerple` measures the same question on the GPU, and its findings are pending.

## Reading a slot (added)

Request, 2026-11-21: in pricing functions of `n` variables (`review/flat_spectrum.md`,
*Functions of n variables*), bound from below and from above what one head of the main model can
do to read the operand at a given distance. The positional scheme is LeRoPE alone, SSMax's gain
kept; abstract statements are welcome, special cases are not wanted. Lean: `SlotFejer`,
`SlotWeight`, `SlotDFT`, `SlotOneSided`, `SlotOneSidedHead`, `SlotGeometric`, `SlotDFTShift` and
`SlotAdjacent` of `src/Transformer/LeRoPEBlock/`.

A head reads the key at the distance `t` by putting its softmax weight there. When the keys
around the slot carry the same token, or the head's keys ignore the token, only the way the score
depends on the distance tells the slot from its neighbours, and the question is how much weight
that can buy.

RS1, a head keeps a bounded share on a slot, whatever its frequencies. The score of a LeRoPE head
at the distance `t` is a sum of bands, one per rotated plane, `X_l cos(tθ_l) + Y_l sin(tθ_l)`,
each at most `(|q_l|² + |k_l|²)/2` (`dotProduct_rotary_neg`, `abs_band_le`). Over the window of
distances `c + s − s'`, `s, s' < n`, a band sums to its value at `c` times
`|Σ_{s<n} e^{isθ}|²`, a Fejér factor between `0` and `n²` (`sum_sum_band`, `sq_add_sq_sum_le`).
So the window cannot lie below the slot everywhere: its `n²` margins `ℓ(c) − ℓ(c + s − s')` sum
to at most `n²` times the amplitude, at every frequency and with any number of bands
(`sum_sum_dotProduct_rotary_sub_le`). With the term `β (Im Z)²/d` and a gain `g ≥ 0` the sum is
at most `n² g ((|q|² + |k|²)/2 + |β| |q|² |k|²/d)` (`sum_sum_phaseLogit_sub_le`).

Margins become weight by counting keys. With `n = R + 1`, the row `s` of the window hits `R + 1`
distinct keys, so its `R` pairs off the diagonal carry at most `Z − e^{ℓ(c)}` of the partition
function `Z`. Jensen's inequality for `exp` along each row, then over the rows
(`card_mul_exp_mean_le`), gives: if the margins sum to at most `(R + 1)² B`, the slot keeps at
most `1 / (1 + R e^{−B(1 + 1/R)})` of the weight (`depthWeight_le_of_sum_sum_le`).

For the main model, Q/K RMS bounds `|q|²` and `|k|²` by `d`, and SSMax's gain is `ln N / d`,
`N = i + 1`. So `B = (1 + |β|) ln N`, and a slot whose `R` keys on each side carry its token has

```
weight on the slot ≤ 1 / (1 + R N^{−(1 + |β|)(1 + 1/R)}),
```

whatever the frequencies (`depthWeight_leropeLogit_le`). A head whose keys ignore the token has,
on every input, the weights of the constant sequence of the query's token, so the bound holds for
it everywhere. At `β = 0` the middle of `N = 2R + 1` keys keeps at most `9/10` at `R = 1` (the
example of the theorem), `0.72` at `R = 16`, and the bound falls to `2/3` as `R` grows.

The same two steps price a head with any gain `λ` in place of SSMax's and squared lengths at most
`A`, since the statements take any `g ≥ 0`: at `β = 0`, a weight `1 − ε` on the slot needs

```
λ A ≥ (R / (R + 1)) ln(R (1 − ε) / ε),
```

a logarithm in the number of neighbours and in `1/ε`, not a power of the length. Solved for `λ A`
by hand; Lean states the weight bound it comes from.

RS2, the cap is nearly reached (`SlotDFT`). Give a head `R` bands at the frequencies of the
discrete Fourier transform on `2R + 1` points, `θ_l = 2π (l + 1)/(2R + 1)` (`dftFreq`), the query
`1` on every channel and the key `1` turned by `R`, at `β = 0` and `d = 2R`; both have squared
length `d`, as after Q/K RMS. At the query `2R` the distance `δ` scores `ln N / (2R)` times
`2 Σ_l cos((R − δ) θ_l)`, `N = 2R + 1`, and Dirichlet's kernel sums these cosines to `−1/2` at
every distance but `R` (`two_mul_sin_mul_sum_cos`, `sum_cos_dft`). So the middle key scores
`ln N` and the other `2R` keys `−ln N / (2R)`, whatever the tokens (`leropeLogit_dft`), and the
middle key keeps

```
1 / (1 + 2R N^{−1−1/(2R)})
```

of the weight (`depthWeight_dft`), more than half for every `R` (`half_lt_depthWeight_dft`).
Against the cap of RS1 at `β = 0` (`depthWeight_dft_mem`): `0.72` against `0.9` at `R = 1`;
`0.535` against `0.72` at `R = 16`, the main model's `16` bands on `33` keys; `1/2` against `2/3`
as `R` grows. With a free gain `λ` and squared lengths `A` the same head pays
`λ A = (2R / (2R + 1)) ln(2R (1 − ε)/ε)` for the weight `1 − ε` (by hand from
`leropeLogit_dft`), against at least `(R / (R + 1)) ln(R (1 − ε)/ε)` for every head; as `ε → 0`
the ratio of the two tends to `2 (R + 1)/(2R + 1) ≤ 4/3`.

RS3, the ends of the context. RS1's window needs `R` keys of the slot's token on each side. The
key next to the query has one key on its near side and the first key of the context none on its
far side, so RS1 caps neither, and these are the slots a head reads when the operand sits at an
end of the sequence. Weights on the keys of one side do the window's job.

Put weights `w ≥ 0` summing to one on other keys, at the offsets `e` from the slot, on either
side. Gibbs' inequality, `Σ e^a ≥ e^{Σ w a + H}` with the entropy `H = −Σ w ln w`
(`exp_add_entropy_le`), turns margins into weight: if the margins `ℓ(t) − ℓ(t + e)` average to at
most `M` under `w`, the slot keeps at most `1 / (1 + e^{H − M})` (`depthWeight_le_of_weighted`).
A band's margins average to its value at the slot times `1 − C` minus its quadrature times `S`,
where `C + iS = Σ w e^{ieθ}`, and value and quadrature are the two legs of the band's amplitude,
so the average is at most `V` times the amplitude once `|1 − Σ w e^{ieθ}| ≤ V` at every frequency
(`sum_weight_band_sub_le`). For a head of the main model, Q/K RMS and SSMax's gain make
`M = (V + |β|) ln N` (`sum_weight_phaseLogit_sub_le`), and a slot whose weighted keys carry its
token keeps at most

```
1 / (1 + e^H N^{−(V + |β|)})
```

whatever the frequencies and the query's token (`depthWeight_leropeLogit_le_of_weights`).

The two numbers pull against each other: spreading `w` over more keys raises `H`, but the defect
`V` stays above `1`. For offsets on one side, `1 − Σ w z^e` is a polynomial equal to `1` at
`z = 0`, so the mean of `ln |1 − Σ w z^e|` over the unit circle is at least `0` (Jensen's
formula); the polynomial vanishes at `z = 1`, where the weights sum to one, so its modulus
exceeds `1` elsewhere on the circle and `V > 1` (by hand). Geometric weights strike the balance:
`(1 − r) r^j` at the offsets `j + 1 ≤ T` and the rest, `r^T`, at `T + 1` (`geomWeight`,
`sum_geomWeight`). With `z = e^{iφ}` the defect factors,
`1 − Σ w z^{j+1} = (1 − z) Σ_{j≤T} (rz)^j` (`defect_eq_norm_sq`, `one_sub_sum_geomWeight`), and
`(1 + r)|1 − z| ≤ 2|1 − rz|` on the circle (`one_add_mul_norm_one_sub_le`), so
`V = 2(1 + r^{T+1})/(1 + r)` (`defect_geomWeight_le`), while
`H ≥ (1 − r^T)(r − ln(1 − r))` (`le_entropy_geomWeight`). A key whose `T + 1` keys on one side
carry its token keeps at most

```
1 / (1 + e^{(1 − r^T)(r − ln(1 − r))} N^{−(2(1 + r^{T+1})/(1 + r) + |β|)})
```

for every `r ∈ (0, 1)`: older keys for the key next to the query
(`depthWeight_leropeLogit_le_geom`), newer keys for the first key of the context
(`depthWeight_leropeLogit_le_geom_newer`).

At `β = 0` and the best `r` (by hand), on `33` keys the key next to the query keeps at most
`0.676` (`T = 30`, `r = 0.91`) and the first key `0.668` (`T = 31`). Next to the query the cap is
`0.656` on `100` keys, `0.653` on `300`, `0.657` on `1000` (`r = 0.996`), `0.669` on `10⁴`,
`0.688` on `10⁶` and `0.705` on `10⁹`: it is lowest near `300` keys and creeps back to `1`, the
best `r` having `1 − r` of order `ln ln N / N` and the cap a deficit of order `1 / ln ln N` (by
hand). On `33` keys one side already caps the end keys below RS1's `0.72` for the middle key; the
middle key itself has only `16` keys on each side, and one side of them gives `0.82`, worse than
RS1.

From below, turn RS2's key by `t` instead of `R`. Now `t − δ` runs over `[−2R, 2R]`, past the
window of `sum_cos_dft`, but the cosines of the discrete Fourier transform repeat with the period
`N = 2R + 1`, so they still sum to `−1/2` at every `0 < |t − δ| ≤ 2R`
(`sum_cos_dft_of_abs_le`). The distance `t` scores `ln N` and the other `2R` distances
`−ln N/(2R)` (`leropeLogit_dft_shift`), and the key at the distance `t` keeps RS2's
`1 / (1 + 2R N^{−1−1/(2R)})` (`depthWeight_dft_shift`), at every `t ≤ 2R`. Both sides together,
at `β = 0` with the main model's `16` bands on `33` keys: the key next to the query keeps
`0.535` under the DFT head and at most `0.676` under any head (`depthWeight_adjacent_mem`), the
first key `0.535` and at most `0.668` (`depthWeight_first_mem`).

### Not claimed

- Sharp constants. The factor `1 + |β|` is crude, and so is RS3's `|β|`: the score itself is at
  most `(5/4) ln N` at `|β| ≤ 1` (`leropeLogit_abs_le`). A sharper factor for the margins is not
  stated, and RS3 uses a lower bound on the entropy, not the entropy itself.
- Long contexts at the ends. RS3's cap holds for any number of bands and creeps back to `1` as
  `N` grows. With the main model's `16` bands the phases of all bands return together at some
  distance (Dirichlet's simultaneous approximation), which drives the weight on any slot towards
  `1/2`, but only from around `10⁵⁰` keys (by hand); nothing in between is stated.
- Few bands. The DFT head needs `R` bands for `2R + 1` keys, so the main model's `16` bands
  cover `33` keys. On a longer context the caps of RS1 and RS3 still hold for every head, but
  nothing here is built from below them.
- That the main model realizes the DFT head. Its query and key are the same for every token, which
  needs `W_Q` and `W_K` to read a direction on which the tokens agree, as for the π head.
- Content. The bound is for keys of one token around the slot. A head whose key depends on the
  token can single out a slot whose token differs from its neighbours'; reading an arbitrary
  operand needs a large weight for every assignment of tokens, and that is not stated.
- Readout. Weight is not output: a value read with the weight `w` is mixed with its neighbours'
  values, and how much of that a nonlinear `Φ` undoes is not bounded here.
