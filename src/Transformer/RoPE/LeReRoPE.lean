module

public import Transformer.RoPE.LeReRoPE.Clamp

/-!
# LeReRoPE: LeRoPE with ReRoPE

In no paper. LeRoPE (Karypis et al., arXiv:2607.10134v1; `Transformer.RoPE.LeRoPE`) learns one
scale per band of RoPE's frequencies; ReRoPE (Su, kexue.fm/archives/9708;
`Transformer.RoPE.ReRoPE`) clamps the relative position at a window. LeReRoPE puts them together.

`Transformer.RoPE.LeReRoPE.Clamp` treats LeRoPE's dominant band, `≤ 0` throughout the training
window and positive beyond it (§7). With a window no longer than the training length ReRoPE keeps
the band `≤ 0` at every distance, and at its trained values inside the window; Leaky ReRoPE keeps
it so `k` times further and then lets it turn positive.
-/
