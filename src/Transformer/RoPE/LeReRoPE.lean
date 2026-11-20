module

public import Transformer.RoPE.LeReRoPE.Adjoint
public import Transformer.RoPE.LeReRoPE.Clamp
public import Transformer.RoPE.LeReRoPE.Covariance
public import Transformer.RoPE.LeReRoPE.Fold

/-!
# LeReRoPE: LeRoPE with ReRoPE

In no paper. LeRoPE (Karypis et al., arXiv:2607.10134v1; `Transformer.RoPE.LeRoPE`) learns one
scale per band of RoPE's frequencies; ReRoPE (Su, kexue.fm/archives/9708;
`Transformer.RoPE.ReRoPE`) clamps the relative position at a window. LeReRoPE puts them together.

`Transformer.RoPE.LeReRoPE.Clamp` treats LeRoPE's dominant band, `≤ 0` throughout the training
window and positive beyond it (§7). With a window no longer than the training length ReRoPE keeps
the band `≤ 0` at every distance, and at its trained values inside the window; Leaky ReRoPE keeps
it so `k` times further and then lets it turn positive.

`Transformer.RoPE.LeReRoPE.Fold` carries LeRoPE's frequency gradient (§3.2, Eq. (6) and
Lemma 3.1) to the positions of ReRoPE: it is the derivative of the same surrogate with the offset
`δ` rotated by `ψ(δ)`, and under the clamp the frequency sees the far pairs only through their
summed phasor, at the single offset `w`. `Transformer.RoPE.LeReRoPE.Adjoint` reads the same gradient
off the rotated chunks and their gradients, as a kernel has them: `Σ_i a_i Im(x̃_i^* dx̃_i)`, a sum
over positions; under ReRoPE the near queries and keys enter at `s` and `t`, the far queries at
`w`, and the unrotated far keys not at all.

`Transformer.RoPE.LeReRoPE.Covariance` carries LeRoPE's Lemma B.3, the gradient as
`−Σ_s Cov_{t∼p_s}(v_tᵀ g_s, S_st)`, to the positions `ψ`. Under the clamp every pair at least `w`
apart has the sine term of the offset `w`: a pair's weight in the gradient stops growing at the
window, and the gradient's bound carries `w` where LeRoPE's carries the distance `s − t`.
-/
