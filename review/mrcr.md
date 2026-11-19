# MRCR-V2 taken apart

Note for `Transformer.MRCR`, started on 2026-11-19. It takes the long-context benchmark
MRCR-V2 (8-needle) apart into the components that a model has to get right, and lists for each
what is proved in Lean and what is not.

## Sources

- MRCR-V2 (8-needle): the Gemini 2.5 report, arXiv:2507.06261v6, Appendix,
  `8_appendix/1_evaluation_additional_details.tex`, line 75. It nests the dictionary to depth 3
  by adding a style ("write a poem about penguins in an archaic style") and uses 8 needles
  instead of 4. The 128k figure averages contexts up to 128k; the 1M figure is at exactly 1M
  (`3_results_sections/1_methodology.tex`, line 43). The report gives no data.
- MRCR (V1): Michelangelo, arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37. The latent
  structure is a nested dictionary of topic and format that stores "a list of model outputs, in
  the order that they were presented in the context", and the task "requires the model to use
  information about the ordering of multiple needles".
- OpenAI MRCR (huggingface.co/datasets/openai/mrcr) is a different dataset: 2, 4 or 8 identical
  asks, keys of depth 2, graded by `SequenceMatcher`. It does not call itself V2.
- The attention scale `s log n`: SSMax, arXiv:2501.19399v1, §2.2. Rounded transformers and
  the logic `TL[◁#]`: arXiv:2506.16055v3.

## Components

| Component | What the model must do | Lean |
|---|---|---|
| Length | keep the needle visible among `n` tokens | SA5 of [`flat_spectrum.md`](flat_spectrum.md): `s g > 1` recalls, `2 s B < 1` fades, exact error at `2 s B = 1` |
| Near misses | a depth-3 key against distractors that share two of its parts | `Transformer.MRCR.NearMiss`, `NearMissThreshold`, `PartScore` |
| Ties | eight asks with the same key | `Transformer.MRCR.Ties`, `InnerNeedle` |
| Ordinal | the `i`-th of the eight, counted from the start | open |
| Count then look up | both at the scale `s log n` | open |
| Copy | return the whole answer, token by token | open |
| Training | whether a trained model has the gaps | not attempted (the user, 2026-11-19) |

## Near misses (proved)

A distractor that shares part of the queried key may trail it by less than the others. With
each key its own gap `g_j`, the miss of attention at the scale `s log n` is at most
`2 C ∑_{j ≠ j₀} n^{−s g_j}` (`abs_logScaleAttention_sub_le_sum`). With at most `K n^α` near
misses trailing by `g₁` and the rest by `g₂`, `s g₁ > α` and `s g₂ > 1` recall at every length
(`exists_forall_logScaleAttention_recall_nearMiss`); with at least `c n^α` of them and
`s g₁ < α`, the indicator dictionary is answered with an error that tends to `2`
(`exists_lt_logScaleAttention_nearMiss_error`). The threshold of the near misses is the growth
exponent of their number.

For additive scores, `w` per agreeing part of `d` (`partScore`), near misses trail by `w` and
keys that differ everywhere by `d w`. Without near misses `s d w > 1` suffices
(`partScore_recall_of_far`); with a number of near misses linear in the length the threshold is
`s w = 1` (`partScore_recall`, `partScore_nearMiss_error`). At depth 3 that is three times the
scale. Whether the near misses of MRCR-V2 grow linearly with the length depends on how its
distractors are drawn, which the report does not say.

## Ties (proved)

The asks of one key are the same text. A head that scores them by content alone ties them: if
the needles `T` share the largest raw score and the other tokens trail by `g`, attention at the
scale `s log n` is within `2 C (n − |T|) n^{−s g} / |T|` of the average of their values
(`abs_logScaleAttention_sub_average_le`).

Position does not rescue an inner needle. Give a needle at the distance `d` the raw score
`c + h d`, any `h`, and place `k` needles equally spaced by `G` with the last at the distance
`D` (`spaced`). For a needle with a needle on each side, the placements with the last needle at
`D` and at `D + G` do not both favour it: one gives a neighbour a raw score at least its own
(`exists_spaced_le_of_inner`), and there the indicator dictionary of the needle is answered with
a value at most `0`, whatever the other tokens (`exists_spaced_logScaleAttention_indicator_nonpos`).
This holds for every `h`, so for RoPE, ALiBi and no position alike, at every spacing and distance.
The last needle is different: the ALiBi bias `−b d` returns it within
`2 C (k n^{−s b} + (n − 1) n^{−s g})` (`abs_logScaleAttention_sub_last_le`). The first needle is
not covered either way. One layer does not return the `i`-th of eight identical asks,
`1 < i < 8`, at every placement; the count of earlier asks must come from an earlier layer, which is the next row.
