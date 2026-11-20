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
  LeRoPE's angle `i 10000^{−2m/d} e^{α_m}`, the `α_m` shared by Q, K, heads and layers (so any
  positive frequency is reachable); `β = a/√(1 + a²)`, one zero-initialized coordinate per layer
  and head.
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

How to read a verdict: `A ⊆ B` is "every block of `A` is within any `ε` of a block of `B` on any
finite set of inputs" (containment up to closure); a separation is a target in `B` that every
block of `A` misses by a positive constant. With an arbitrary `Φ`, only an exact invariance of
`A` can give a separation: an injective head followed by an arbitrary `Φ` interpolates anything
on a finite set. So the rows below separate by invariances, or say that they do not.

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

LB3, LeRoPE loses nothing to it (`Limit`). LeRoPE's frequencies are positive, so a block with no
positional encoding is not literally a LeRoPE block; but the logit, the attention and, for a
continuous `Φ`, the block are continuous in the frequencies (`continuous_block`), and at zero
frequencies the head has no rotation (`leropeLogit_zero`). So for the main model's own `Φ`,
projections, `β`, `d` and values, every finite set of sequences and every `ε > 0`, there are
positive frequencies with the LeRoPE block within `ε` of the block with none
(`exists_lerope_close`).

## Verdict, so far

| alternative | LeRoPE contains it | it contains LeRoPE | lemmas |
| --- | --- | --- | --- |
| no positional encoding (`nope`) | yes, up to closure | no: the π head | LB1–LB3 |
| fixed-frequency RoPE | not yet | not yet | |
| distance biases (ALiBi, KERPLE, −ln(1 + r)) | not yet | not yet | |
| Hermitian score (`β = 0`) | not yet | not yet | |
| constant gain instead of SSMax's `ln N` | not yet | not yet | |

Against a block with no positional encoding LeRoPE's class is strictly better, in the sense of
the reading above.

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
- Training, length generalization and the gpt runs.
- The size of the gap: `0.0039 ‖u_a − u_b‖` at `N = 128`.

## Open

The remaining rows of the table: fixed-frequency RoPE, the additive distance biases of the
`alibi_kerple` arms, the Hermitian score and a constant gain. Each is one step.
