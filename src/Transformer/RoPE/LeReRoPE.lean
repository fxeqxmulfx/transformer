module

public import Transformer.RoPE.LeReRoPE.Adjoint
public import Transformer.RoPE.LeReRoPE.Clamp
public import Transformer.RoPE.LeReRoPE.Covariance
public import Transformer.RoPE.LeReRoPE.Fold
public import Transformer.RoPE.LeReRoPE.Window
public import Transformer.RoPE.LeReRoPE.WindowKernel
public import Transformer.RoPE.LeReRoPE.WindowLoss

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

`Transformer.RoPE.LeReRoPE.Window` lets a head learn its window `w` and slope `s`: the distance
`d` sits at `min(d, w) + s·max(d − w, 0)`, ReRoPE's position at `s = 0`, Leaky ReRoPE's at
`s = 1/k` and RoPE's at `s = 1`. Two far distances are `s(d − d')` apart whatever the window:
the window moves every far pair alike, `∂p/∂w = 1 − s`, and only the slope's derivative, the
excess `d − w`, tells far distances apart. One pass computes the map, as it computes Leaky
ReRoPE's.

`Transformer.RoPE.LeReRoPE.WindowLoss` differentiates every loss of the heads' positions in a
head's window and slope. The slope's gradient weighs each far pair by its excess `d − w`; the
window's is `1 − s` times the loss's derivative along the indicator of the far pairs, the same
for every far distance. A window wider than every distance of its head gets no gradient, and
neither does its slope; at `s = 1` the window gets none at all. Nothing in the loss pulls an open
window back.

`Transformer.RoPE.LeReRoPE.WindowKernel` reads both gradients off the rotated chunks, as a kernel
has them. One pass rotates the far queries to `w + s(t − w)` and the far keys to `s j`; the slope's
gradient weighs the far queries' torques by their excess `t − w` and the far keys' by `j`, and the
window's is `1 − s` times the far queries' torques, summed over the head's bands.
-/
