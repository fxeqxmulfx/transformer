module

public import Transformer.RoPE.ReRoPE2D.Defs
public import Transformer.RoPE.ReRoPE2D.Basic

/-!
# 2D ReRoPE for the axial RoPE of vision encoders

In no paper. ReRoPE (Su, kexue.fm/archives/9708; `Transformer.RoPE.ReRoPE`) clamps RoPE's
relative position at a window. This carries it to the axial 2D RoPE of vision encoders, Kimi-K3's
MoonViT and GLM-5.3-Flash's, whose patches have a row and a column.

`Transformer.RoPE.ReRoPE2D.Defs` defines axial RoPE, whose score is a row score plus a column
score, and the two encoders' layouts. `Transformer.RoPE.ReRoPE2D.Basic` clamps each axis's offset
on both sides, since a vision encoder attends both ways and Su's one-sided `min(d, w)` leaves the
negative offsets unbounded. Inside the window box the score is axial RoPE's. With the key above and
to the left of the query, each axis is Su's merged score. Every score on a grid of any size is an
axial RoPE score of two patches of a grid larger than the windows.
-/
