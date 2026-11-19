module

public import Transformer.MRCR.CountHead
public import Transformer.MRCR.CountLookup
public import Transformer.MRCR.Counting
public import Transformer.MRCR.InnerNeedle
public import Transformer.MRCR.NearMiss
public import Transformer.MRCR.NearMissThreshold
public import Transformer.MRCR.Ordinal
public import Transformer.MRCR.PartScore
public import Transformer.MRCR.Returns
public import Transformer.MRCR.RotaryCopy
public import Transformer.MRCR.Ties

/-!
# MRCR-V2 taken apart

In no paper: the components of MRCR-V2 (8-needle), the long-context benchmark of the Gemini 2.5
report, arXiv:2507.06261v6, Appendix, line 75 of
`8_appendix/1_evaluation_additional_details.tex`: a conversation of asks and answers, keys of
depth three (a topic, a format and a style), eight needles, and a query for the answer to the
`i`-th of them; MRCR itself is arXiv:2409.12640v2, `eval_framework.tex`. Each component is
stated for the attention at the scale `s log n` of SSMax, arXiv:2501.19399v1, §2.2, or for the
rounded transformers of arXiv:2506.16055v3, and `review/mrcr.md` lists what each one covers.

`Transformer.MRCR.NearMiss` and `Transformer.MRCR.NearMissThreshold` give each key its own gap:
near misses, keys that share part of the queried key, are harmless at every length when `s g₁`
exceeds the growth exponent of their number, and spoil recall when it falls below.
`Transformer.MRCR.PartScore` applies it to additive scores over the parts of a key: at depth
three a linear number of near misses triples the scale that recall needs.

`Transformer.MRCR.Ties` and `Transformer.MRCR.InnerNeedle` treat the asks of one key, which are
the same text: attention averages tied needles, a bias that falls with the distance returns the
last of them, and no raw score of the distance returns a needle with a needle on each side.
`Transformer.MRCR.Ordinal` returns the needle with the desired index, counted from the start, by
a formula of depth two and so by a rounded transformer of two layers
(`Transformer.MRCR.Counting` relates its counts to cardinalities).
`Transformer.MRCR.CountHead` and `Transformer.MRCR.CountLookup` do the same with the real softmax
at the scale `s log n`: one head counts the earlier needles of each key, a second compares the
count with the ordinal of the query and returns the needle at every length.

`Transformer.MRCR.Returns` and `Transformer.MRCR.RotaryCopy` treat the copy of the answer by its
position: a RoPE score of the distance comes back near its value at linearly many distances, so
a RoPE head that scores by the distance only loses every position at long range, at every scale.
-/
