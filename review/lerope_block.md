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
