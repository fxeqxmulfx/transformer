# A block of the main model: LeRoPE against ReRoPE

Request, 2026-11-20: take the whole block of DiffGPT's main model, change only LeRoPE to
ReRoPE, and derive whether it approximates better or worse, as `review/complex_ffn.md`
did for the FFN. Lean: `Transformer.ReRoPEBlock` (`src/Transformer/ReRoPEBlock/`).

## Sources

- DiffGPT, `~/python_projects/gpt` at `5a37c51`, `research/main_model/README.txt`,
  *Architecture*: `q_i = LeRoPE_i(R_d(W_Q z_i))`, `k_i = LeRoPE_i(R_d(W_K z_i))`,
  `p_ij = softmax_{j ≤ i}((ln N_i / d)(Re Z_ij + β (Im Z_ij)² / d))`, `N_i = i + 1`;
  LeRoPE's angle `i 10000^{−2m/d} e^{α_m}`, the `α_m` shared by Q, K, heads and layers;
  `β = a/√(1 + a²)`, one zero-initialized coordinate per layer and head.
- Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708
  (`papers/kexue-9708-rerope`), section 融合: ReRoPE scores the distance `d` as `min(d, w)`,
  Leaky ReRoPE as `d` inside the window and `w + (d − w)/k` beyond it.

## The block

Everything but the attention acts on one position at a time: the mHC routers and their
Sinkhorn maps, the RMS normalizations, `W_O`, the FFN and the writes. So at the query `i`
the block is `Φ(x_i, (attend_{j ≤ i}(F_h(i, x_i, x_j, ψ(i − j)), v_h(x_j)))_h)`
(`block`), with `x_j` all that the block reads at the position `j` and `ψ` the position a
head gives a distance: `ψ δ = δ` for LeRoPE, `min(δ, w)` for ReRoPE. The main model's
score is `headScore`; `headScore_sub` checks that it is LeRoPE's turn of the query by `i`
and of the key by `j`. The two blocks compared differ only in `ψ`; `Φ`, `F`, `v` and the
frequencies range over everything, so every statement below covers the main model's `Φ`.

## Results

R1, inside the window (`Basic`). At a query `i < w` every distance is below the window,
so the two blocks with the same `Φ`, scores and values are one function
(`block_rePosition_of_lt`).

R2, ReRoPE forgets the order of the far keys (`Basic`). A permutation of the keys at
distances `≥ w` leaves every ReRoPE block unchanged (`block_rePosition_perm`). So a ReRoPE
block within `δ` of any target `T` on a set holding `x` and its permuted copy has
`‖T x − T(permute τ x)‖ ≤ 2δ` (`norm_sub_le_of_rePosition`, through
`ComplexFFN.gap_of_invariant`).

R3, a LeRoPE head that sees that order (`PiHead`, `FarOrder`). The π head is one head of
the main model with every frequency `π`, `β = 0` and a constant query and key `q`. It
scores the distance `t` by `κ ln(i + 1) cos(π t)`, `κ = qᵀq/d` (`headScore_pi`); after
the head RMS `κ` is just below `1`. Swapping the values of two keys changes an attention by
`(e^{s_a} − e^{s_b}) / ∑ e^{s}` times `u_a − u_b` (`attend_sub_attend_swap`). At the
query `i + 1` the first two keys score `±κ ln N`, `N = i + 2`, so swapping them changes
the π head by at least `(1 − N^{−2κ}) / N ‖u_a − u_b‖` (`norm_piHead_sub_lePosition`). At
the query `w + 1` both keys lie beyond the window, so every ReRoPE block, with any `Φ`,
heads, scores, values and frequencies, misses the π head on `x` and its swap by
`δ ≥ (1 − N^{−2κ}) / (2N) ‖u_a − u_b‖`, `N = w + 2` (`gap_rePosition_piHead`). At
`κ = 1` and `w = 128` this is about `0.0038 ‖u_a − u_b‖`. The bound is small because the
π head spreads its weight over half of the keys; a head peaked at one far distance would
move more weight (by hand, not formalized).

R4, free scores (`FarOrder`). If the scores may be any function of the position, the
ReRoPE blocks are the LeRoPE blocks that read their positions through the clamp
(`rePosition_class_subset`); R3 makes the inclusion strict.

## Verdict, so far

Inside the window: equal. Beyond it ReRoPE is strictly weaker as an information
channel: it cannot see the order of the far keys, and with free scores its class is
strictly inside LeRoPE's. This does not settle the question for the main model's rotary
scores: LeRoPE turns a key by `δ θ` and cannot clamp, so a ReRoPE head beyond the window
need not be a LeRoPE head.

## Open

- Whether LeRoPE blocks with rotary scores approximate ReRoPE blocks beyond the window.
- Nothing here concerns training, length generalization or the gpt runs.
