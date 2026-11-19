module

public import Transformer.MRCR.NearMiss
public import Transformer.MRCR.NearMissThreshold
public import Transformer.MRCR.PartScore

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
near misses, keys that share part of the queried key, are harmless at every length exactly when
`s g₁` exceeds the growth exponent of their number. `Transformer.MRCR.PartScore` applies it to
additive scores over the parts of a key: at depth three a linear number of near misses triples
the scale that recall needs.
-/
