module

public import Transformer.RoPE.LeReRoPE.Clamp
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
summed phasor, at the single offset `w`.
-/
