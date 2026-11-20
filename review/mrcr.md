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
| Asks in words | find an ask spelled by several tokens, "a poem about penguins" | `Transformer.MRCR.PhraseKey`, `PhraseKeyRope`, `PhraseKeyTemplatedLimits` (no rounded transformer without positions, nor with RoPE or sinusoidal encodings at rational angles, also on templated conversations), `PhraseKeyPrevCorrect` (`TL[◁#, Y]`, the logic of ALiBi, answers it at depth two), `PhraseKeyLogits` (a transformer of five layers with ALiBi, KERPLE or SSMax of either answers it, the logit divided by `1`, `√d` or `d`), `PhraseKeySandwich` (not with Sandwich: its bias comes back near `0` far away, where a key beats the token before), `PhraseKeyRotary` (not with RoPE, YaRN or LeRoPE at any frequencies: a RoPE score converges only if it is constant, and a far key comes back within any margin of the token before), `PhraseKeyReRope` (with the distance clamped at a window, as in ReRoPE, five layers do, also under SSMax and Su's `log n` scale), `ReRopeHeads`, `ReRopeFading` (with the real softmax a ReRoPE head reads inside the window under SSMax with `s g > 1` and under Su's scale, and fades under the plain softmax), `PhraseKeyStyleLogits` (keys of depth three, a style word after the topic: six layers with ALiBi or ReRoPE, alone, under SSMax or with Su's scale), `ReRopeStyleRecall` (with the real softmax under SSMax, three layers: four ReRoPE heads, rounded, read the last four tokens exactly, then the heads of `CountLookup`), `RotaryPrevHead` (with every plane rotated, keys arbitrarily far keep a weight at least `1` at precision `0`, so a head of the first layer does not attend to the token before alone), `RotaryPartialHead` (with a plane left unrotated, as in partial RoPE or LeRoPE at the frequency `2π`, a head attends to the token before alone, without a margin); RoPE or Sandwich by other means, and SSMax of the dot product alone, open |
| Ties | eight asks with the same key | `Transformer.MRCR.Ties`, `InnerNeedle` |
| Ordinal | the `i`-th of the eight, counted from the start | `Transformer.MRCR.Ordinal`, `Counting` |
| Count then look up | both at the scale `s log n` | `Transformer.MRCR.CountHead`, `CountLookup`, `ReRopeStyleRecall` (on the keys of asks in words) |
| Copy | return the whole answer, token by token | `Transformer.MRCR.Returns`, `RotaryCopy`, `ReRopeFading` (copy by position: RoPE loses every position at long range, ReRoPE every position beyond the window; copy by content open) |
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
`1 < i < 8`, at every placement; the count of earlier asks must come from an earlier layer.

## Ordinal (proved, rounded transformers)

A row carries a key, a value, a query and an ordinal; an ask and its answer share one token, as
in `Transformer.CRASP.exists_rtfr_answers`. `v` answers the row `i` (`NthAnswer`) when an earlier
row has the key of its query, the value `v` and exactly `o i` rows of that key before it. The
formula

    ⋁_{q, r} (query = q ∧ ordinal = r ∧ ∃ j < i [key = q ∧ value = v ∧ #(key = q) = r + 1])

has depth two (`nthAnswers_mem`) and holds exactly when `v` answers (`sat_nthAnswers`), so a
rounded transformer of two layers whose attention ignores the query recognizes the task at every
length (`exists_rtfr_nthAnswers`). The first layer counts, the second compares the count with
the ordinal. Counted from the end the task is out of reach already at the last needle: no rounded
transformer recognizes the latest binding at every length
(`Transformer.CRASP.not_recallsLatest`). The next section takes the same two steps with the real
softmax at the scale `s log n`.

## Count then look up (proved, real softmax)

After a BOS the rows `t` carry keys `κ t` and values; `occ κ t` counts the rows up to `t` with the
key of `t`, `t` included. The first head, at the row `t`, attends causally to the BOS and the rows
up to `t` with the raw score `1` at the BOS and at the rows of the key of `t`, and reads `1` at the
BOS only. The `occ κ t + 1` leading tokens tie, so at the scale `s₁ log n`, `s₁ ≥ 1`, the head
returns `1 / (occ κ t + 1)` within `(t + 2)^{1 − s₁}` (`abs_countHead_sub_le`). Rows of one key
have distinct counts (`occ_lt_occ`).

The second head, at the query for the key `q` and the ordinal `r`, scores the row `t` by
`[κ t = q] − K_r (c_t − 1/(r + 2))²` with `K_r = ((r + 2)(r + 3))²` and the BOS by `0`; the
score is bilinear in `(e_κ, 1, c, c²)` against features of the query, and the feedforward layer
supplies `c²`. Every count `o ≠ r + 1` puts `1/(o + 1)` at least `1/((r + 2)(r + 3))` from
`1/(r + 2)` (`one_le_lookupWeight_mul_sq`), so the needle leads by
`g = 1 − 2 K_r 2^{1 − s₁}` and the second head is within `2 C m (m + 1)^{−s₂ g}` of its value
(`abs_lookupHead_sub_le`). For every bound `R` on the ordinal and every `s₂ > 1` some `s₁` returns
the needle to every accuracy from some number of rows on, whatever the keys and the values
(`exists_forall_lookupHead_recall`). The first rows keep an error of `2^{1 − s₁}` at every length,
so `2 K_R 2^{1 − s₁}` must stay below `1`: the proof takes `2^{s₁ − 1} > 4 K_R s₂ / (s₂ − 1)`,
whose least integer for eight needles (`R = 8`) at `s₂ = 2` is `s₁ = 18`, and the weight `K_r` of
the second head grows like `r⁴`. The query token is left out of the second head; its score would
be at most `0`.

## Asks in words (proved: no rounded transformer without positions or with RoPE at rational angles, also templated; a formula of `TL[◁#, Y]`, and transformers with ALiBi, KERPLE, SSMax or RoPE clamped at a window; keys of depth three with ALiBi or ReRoPE, rounded or with the real softmax under SSMax)

MRCR-V2 spells a key in words: "write a poem about penguins in an archaic style". The ordinal
above gives an ask and its answer one token. `Transformer.MRCR.PhraseKey` spells the ask
"`f` about `t`" in three tokens, lets the token after it answer it, and closes the conversation
with a query for the format, the topic and the number of such asks before the one wanted
(`PhraseAnswer`). No future-masked rounded transformer accepts exactly the conversations that
`v` answers, at every length, whatever its depth, width and precision (`not_phraseAnswers`).

The proof reads a string of bits as words, `b` the format and `a` the word `about`, and closes
it with the topic, the answer and the query (`bitInstance`). An ask can only end at the topic,
so the conversation spells it iff the bits end in `ba` (`phraseAnswer_bitInstance_iff`). The
fixed suffix moves into the formula at no cost in depth (`Transformer.CRASP.DefinableL.append`),
the words are read through the bits (`Transformer.CRASP.RTfr.comap`), and the resulting formula
of `TL[◁#]` over the bits would tell `u·aba` from `u·baa` for every `u`, which none does
(`Transformer.CRASP.exists_models_iff_secondLast`).

`Transformer.MRCR.PhraseKeyRope` does the same for RoPE and sinusoidal encodings at angles that
are rational multiples of `π`, which round to `TL[◁#, MOD]`
(`Transformer.CRASP.exists_mem_TLClMod_of_rope`, `exists_mem_TLClMod_of_sinusoidal`): no such
transformer accepts exactly the conversations that `v` answers (`not_phraseAnswers_rope`,
`not_phraseAnswers_sinusoidal`). Write each bit as a block of `M` words, `M` a multiple of the
moduli of the formula: `MOD` is then constant on a block, `MOD_0^r` tests the number of the
block, and the formula becomes one of `TL[◁#]` over the bits, one level deeper
(`Transformer.CRASP.ModBlocks`, `ModBlocksCorrect`). The last block, which an ask shares with
the close, is replaced by the close and read at a dummy last bit, so that the ask is spelled iff
the bit before the dummy is `b`. Hence `Σ*bΣ` is in `TL[◁#, MOD]` at no depth
(`Transformer.CRASP.not_definableMod_secondLast`), while it is in `TL[◁#, Y]` at depth `0`:
`TL[◁#, Y] ⊄ TL[◁#, MOD]` (`exists_definableY_not_definableMod`), which arXiv:2506.16055v3 does
not state.

Templated conversations defeat the same transformers (`Transformer.MRCR.PhraseKeyTemplated`,
`PhraseKeyTemplatedLimits`). A templated conversation is a list of turns, each the ask
"`f` about `t`" answered by one or more tokens, closed by a query (`Templated`); among those
whose query is answered, no formula of `TL[◁#, MOD]` accepts exactly those that `v` answers
(`not_phraseAnswers_templated_mod`), and so no rounded transformer without positions, with RoPE
or with sinusoidal encodings at rational angles does (`not_phraseAnswers_templated`,
`not_phraseAnswers_templated_rope`, `not_phraseAnswers_templated_sinusoidal`), given two formats,
two topics and two answers. The blocks start at a topic: the block of a bit is the topic `t₁`,
answers, and "`f_c` about", where `f_c` is `f₀` for `b` and `f₁` for `a`; so the key of an ask
straddles two blocks. The conversation opens with "`f₁` about", which moves into the formula
(`Transformer.CRASP.FormP.leftQuot`, the left quotient, at the same depth and free of `Y` if the
formula is), and closes by the topic `t₀`, an answer `v`, the ask "`f₀` about `t₀`", an answer
`v'` and the query for the first ask "`f₀` about `t₀`": that is the ask before the close if the
last block is of `b`, answered by `v`, and the one in the close otherwise, answered by `v'`
(`phraseAnswer_closing_iff`).

`TL[◁#, Y]`, into which ALiBi rounds, answers the query at depth two
(`Transformer.MRCR.PhraseKeyPrev`, `PhraseKeyPrevCorrect`, `definableY_phraseAnswer`), on every
string. `Y² Q_f ∧ Y Q_about ∧ Q_t` marks the end of an ask (`askEnd`), and its count up to a
position `j` is the number of asks that start before the index `j − 2` (`val_countL_askEnd`).
`Y³ Q_f ∧ Y² Q_about ∧ Y Q_t ∧ Q_v` marks the answer `v` after an ask (`answerHere`); there the
count of `askEnd` is `r + 1` iff `r` such asks stand before that one, since `about` follows the
format word (`count_add_two_of_askAt`, `needle`). At the end, the query for "`f` about `t`" with
`r` asks before it and a count of the needle of at least `1` (`queryForm`), over all queries,
accept exactly the conversations that `v` answers (`models_phraseAnswerForm`).

A rounded transformer of five layers answers it as well, at a precision and width that do not
depend on the length (`Transformer.MRCR.PhraseKeyLogits`). arXiv:2506.16055v3 compiles only
`TL[◁#]`, into transformers without positions (`thm:TLCl_to_rtfr`), and not `Y`; here a logit of
the positions computes `Y` (`Transformer.CRASP.LTfr`, `Transformer.CRASP.Grams`). Three layers
read the last four tokens into a gram: a reading layer gives a key the dot product `b` when its
gram differs from that of the query and `0` when it is the same, and the logit sends the query
position and, if its gram differs, the one before it to `0`, and every other position to at most
`−1` (`ReadsPrev`). Floor rounding keeps those one or two positions, and the feed-forward map
reads the two grams off their sum (`layer_read`). Over the grams `Y^m Q_a` is a formula of
`TL[◁#]` (`symAt`), so the answer formula is one of `TL[◁#]` of depth two (`phraseAnswerG_mem`,
`mem_lang_phraseAnswerG`), and two more layers run the transformer of `thm:TLCl_to_rtfr` for it
with every dot product `−1`: the logits of a query are then all negative or all `0`, and
attention averages, as without positions. The construction needs `b < 2^(p−1)`. A logit divided
by `√d` or `d` needs a `b` that grows with the width, and a true comparison of constants
conjoined to the formula raises the precision past it (`pad`, `exists_recognizes`). The logit
can be:

- ALiBi, `eq:alibi`: of slope `1` as a `PTfr` (`exists_alibi_phraseAnswer`), or of a slope `a`
  with the logit divided by `c d`, for any positive `c d` such as `1`, `√d` or `d`
  (`exists_scale_alibi_phraseAnswer`);
- KERPLE, arXiv:2205.09921v2, `eq:krpe`, with the logarithmic kernel or the power kernel of any
  exponent `r₂ > 0`, divided by `c d` (`exists_kerpleLog_phraseAnswer`,
  `exists_kerplePower_phraseAnswer`);
- SSMax, arXiv:2501.19399v1, `eq:ssmax`, of each of these with `s log 2 ≥ 1`, which multiplies
  the whole logit, bias included, by `s log n` (`exists_ssmax_alibi_phraseAnswer`,
  `exists_ssmax_kerpleLog_phraseAnswer`, `exists_ssmax_kerplePower_phraseAnswer`).

Sandwich, arXiv:2212.10356v2, Eq. (6), cannot be one of them (`Transformer.MRCR.PhraseKeySandwich`).
Its bias `(Σ_i cos(tθ_i) − d̄/2)/h` is `0` at the distance `0` and negative elsewhere, but it does not
decay: `exists_returns` over enough positions leaves returns `t` as large as wanted with every
phase `tθ_i` near a multiple of `2π` (`exists_return_ge`), and there the bias beats its value at
the distance `1` (`exists_sandwich_lt`). A key at such a distance gets a larger logit than the
same key at the distance `1`, with or without SSMax (`sandwichLogit_lt`,
`ssmax_sandwichLogit_lt`), so no head reads the token before alone, and `ReadsPrev` fails for
every divisor, compression ratio and dot product (`not_readsPrev_sandwich`,
`not_readsPrev_ssmax_sandwich`). Only the construction is ruled out: whether several layers with
Sandwich read the token before otherwise is open.

Nor can RoPE be one of them, at any frequencies (`Transformer.MRCR.PhraseKeyRotary`). A head
whose query and key carry a constant pair in each rotated plane adds to the dot product a RoPE
score of the distance (`ropeLogit`), and such a score converges only if it is constant
(`ropeScore_eq_of_tendsto`): returns `d` as far as wanted bring the score at `t + d` within any
margin of the score at `t`. So the key at the distance `1 + d` scores within `1/(2g)` of the key
at the distance `1`, and `ReadsPrev` fails for every frequency, coefficient, divisor, temperature
`g > 0` and dot product (`not_readsPrev_ropeLogit`). Changing the frequencies does not help:
YaRN rescales them and multiplies the whole logit by its temperature, arXiv:2309.00071v3,
Eq. (14) and Definition 2 (`yarnLogit`, `not_readsPrev_yarn`), and LeRoPE learns them,
arXiv:2607.10134v1, Eq. (5) (`not_readsPrev_lerope`). Again only the construction is ruled out.

Clamping the distance repairs it (`Transformer.MRCR.PhraseKeyReRope`). ReRoPE (Su,
kexue.fm/archives/9708, formalized in `Transformer.RoPE.ReRoPE`; arXiv:2309.00071v3, §2.3, cites
it without its formula) clamps RoPE's relative position at a window. `reropeLogit` takes the RoPE
score at the position of Su's matrix (rerope), `min (i − j) W`
(`reropeLogit_eq_rectifiedPosition`). It is the dot product plus Su's merged score of a constant
pair in one plane (`Transformer.MRCR.PhraseKeyReRopeSu`, `reropeLogit_eq_merge`). The dot product
itself is not rotated, as in partial RoPE, while in Su's ReRoPE every plane rotates. From the
window on the logit no longer depends on the distance (`reropeLogit_of_le`), so it converges.
One frequency `θ` with `θ (W + 2) ≤ π` and the sine coefficient `−β / sin θ`, `β = b / c ≥ 3`,
score `0`, `−β` and at most `−2β cos θ ≤ −4β/3` at the clamped distances `0`, `1` and `2, …, W`
(`sin_two_mul_le_sin`, `two_thirds_le_cos`), so `ReadsPrev` holds at every window `W ≥ 2`
(`readsPrev_rerope`) and five layers answer spelled asks, the logit divided by any positive
`c d`, with or without SSMax (`exists_rerope_phraseAnswer`, `exists_ssmax_rerope_phraseAnswer`).
Su's own `log n` runs multiply each query by `max(1, log_T n)`, `T` the training length; the
rotation keeps the factor, so it multiplies the whole logit. It is not SSMax: it is `1` up to `T`.
Every factor of at least `1` keeps `ReadsPrev` (`readsPrev_mul`), so five layers answer with it
at every `T` (`exists_logScale_rerope_phraseAnswer`). RoPE's grid `10000^{−2m/d}`
for `d = 128` has a frequency near `1.2 · 10^{−4}`, enough for windows up to about `2.7 · 10^4`
(a hand computation, not in Lean). The coefficients are parameters of the logit, like ALiBi's
slope. That a rounded query and key carry the constant pair is not modelled.

With the real softmax the plain softmax and SSMax part ways (`Transformer.MRCR.ReRopeHeads`,
`Transformer.MRCR.ReRopeFading`). A rounded transformer drops every key whose logit is at most
`−1`, at every length; with the real softmax the far keys add up. One frequency with the
coefficients `A cos(θ k)`, `A sin(θ k)` scores the distance `t` by `A cos(θ (t − k))`
(`ropeScore_peak`), and for `k < W` and `θ W ≤ π` every other clamped distance trails by
`g = A (1 − cos θ)` (`reropePeak_gap`). At the scale `s log n` the head returns the value at the
distance `k` within `2 C i (i + 1)^{−s g}` (`abs_logScaleAttention_rerope_sub_le`), so for
`s g > 1` at every length (`exists_forall_rerope_recall`). Su's scale is SSMax with
`s = 1 / log T` from the training length on (`softmaxAttention_logScale_eq`), so it reads at
every length when `g > log T` (`exists_forall_logScale_rerope_recall`). The plain softmax fades
at every frequency, coefficient and window: a RoPE score is bounded by `∑ (|a_m| + |b_m|)`
(`abs_ropeScore_le`), so the dictionary that marks one key is answered with an error of at least
`2 i / (e^{2B} + i)` (`two_mul_le_one_sub_softmax_rerope`), which tends to `2`
(`exists_lt_softmax_rerope_error`). SSMax with `2 s B < 1` fades too
(`exists_lt_logScale_rerope_error`). Under SSMax such heads make three layers that answer the
query of depth three (below).

Keys of depth three take one layer more (`Transformer.MRCR.PhraseKeyStyle`,
`PhraseKeyStyleCorrect`, `PhraseKeyStyleGrams`, `PhraseKeyStyleLogits`). MRCR-V2 adds a style to
the key, "write a poem about penguins in an archaic style". Here an ask is the format word,
`about`, the topic word and the style word, and the query names all three and the number of such
asks before the one wanted (`StyleAskAt`, `StyleAnswer`). The formula of depth two reads one
token further back (`styleAskEnd`, `styleAnswerHere`) and answers the query
(`models_styleAnswerForm`, `definableY_styleAnswer`). Over grams of five tokens it is one of
`TL[◁#]` (`mem_lang_styleAnswerG`), so six layers answer with every logit that reads the token
before (`exists_styleAnswer`): ALiBi (`exists_alibi_styleAnswer`), and ReRoPE's clamped logit
alone, under SSMax and with Su's scale (`exists_rerope_styleAnswer`,
`exists_ssmax_rerope_styleAnswer`, `exists_logScale_rerope_styleAnswer`). As at depth two, each
word of the key is one token, and the theorems hold for every number of needles, eight included.

With the real softmax three layers answer the query of depth three under SSMax
(`Transformer.MRCR.ReRopeSlots`, `ReRopeStyleKeys`, `ReRopeStyleRecall`). A ReRoPE head reads
the distance `k < W` from the position `k` on. Before that position the peak would lie before the
first token; the scores rise towards it by at least `g` per step (`cos_add_add_one_le`,
`reropeEarly_gap`), so the head reads the first token (`abs_slotHead_sub_le`). For values in
`{0, 1}` and `s g > 4` the error stays below `1/4` at every position, and a ReLU layer of width
two rounds the reading to the token itself (`clampUnit_slotHead`). Four such slots at the
distances `4, 3, 2, 1`, with `W > 4` and `θ W ≤ π`, and one ReLU unit per key give each position
the one-hot key of the ask that ends just before it (`rowKey_eq_some`, `keyUnit_eq`,
`keyVec_eq`). The dot product of two such vectors is the raw score of the counting head
(`sum_keyVec_mul`, `styleCount_eq`), the position after the ask wanted has the count `r + 1` of
its key (`occ_rowKey`), and the lookup head of `CountLookup` returns the token there
(`styleLookup_eq`). So for every `s₂ > 1` some `s₁` makes the three layers return the answer
within every `ε` from some length on (`exists_forall_styleLookup`), for every bound on the index
asked, eight needles included. One ReRoPE head reads any distance inside its window, so one layer
does the work of the four that read the token before in the rounded transformers. Layer 1 attends
over the tokens without a BOS, and the lookup head leaves out the query token, as in
`CountLookup`. Under Su's scale the three layers are not treated: below the training length it is
the plain softmax, and the counting head is proved for `s₁ ≥ 1` only. Under the plain softmax
every ReRoPE head fades (`exists_lt_softmax_rerope_error`), but that no transformer with it
answers is not proved.

Without a margin RoPE still fails when every plane rotates (`Transformer.MRCR.RotaryPrevHead`).
At precision `s = 0` a key whose logit is negative gets the weight `round(exp z) = 0`, so a head
need not keep far keys at `−1`, only below `0`. But a RoPE score whose frequencies all lie outside
`2πℤ` has bounded partial sums, since `2 sin(θ/2)` times the sum of a plane telescopes
(`neg_le_sum_ropeScore`), while a score below `0` from some distance on would be at most `−δ/2`
at the linearly many returns of `exists_returns`. So the score is at least `0` at distances as
large as wanted (`exists_nonneg_ropeScore`), and there the weight is at least `1`
(`exists_one_le_weight_rope`), with RoPE's grid (`exists_one_le_weight_grid`) and with YaRN's
frequencies and temperature (`exists_one_le_weight_yarn`). This holds for each pair of a query
and a key vector, so for a head of the first layer, whose vectors depend on the token alone. A
plane left unrotated, as in partial RoPE, adds a constant term and escapes the argument.

And it does escape (`Transformer.MRCR.RotaryPartialHead`). At the frequency `θ₀ = arccos(3/5)`
with the coefficients `3, 4`, and an unrotated plane with the coefficient `−5`, the score is
`5 cos((t − 1) θ₀) − 5` (`ropeScore_partial`): `0` at the distance `1` and below `0` at every
other, since `θ₀` is no rational multiple of `π` by Niven's theorem
(`cos_int_mul_partialAngle_ne_one`). The weights are `1` at the token before and `0` elsewhere
(`val_round_exp_ropeScore_partial`), and with values in `{0, 1}` the head returns the value of the
token before (`sum_val_round_partial`). Partial RoPE has such planes, and LeRoPE gets one by
learning the frequency `2π` (`exists_lerope_partial`). The head has no margin, so the five layers
above, which read with one, do not take it as it stands.

Not proved:

- RoPE by other means. The construction above fails at every set of frequencies
  (`PhraseKeyRotary`), and with every plane rotated even a head without a margin gives far keys a
  weight (`RotaryPrevHead`); deeper heads and readings through mixtures are not treated. Partial
  RoPE has an exact head without a margin (`RotaryPartialHead`), but the five layers need a
  reader without a margin, which is not built. The rounding to `TL[◁#, MOD]` needs
  `θ_c ∈ π ℚ`, and for the usual frequencies `10000^{−2c/d}` no logic and no lower bound for
  every RoPE transformer is in Lean.
- SSMax of the dot product alone, `s log n · q·k` plus a bias that it does not scale. The reading
  layers above need the logit at the position before to vanish at every length, which a fixed
  `b` does not give when only the dot product grows with `log n`; this variant is not treated.

## Copy by position (proved, RoPE and ReRoPE)

MRCR scores the reproduction of an answer of up to 512 tokens written after a random string
(arXiv:2409.12640v2, `eval_framework.tex`, line 52). Written token by token, the copy can follow
the answer by its position: the source of each written token lies at the same distance from the
writing token throughout. A head that copies so scores the visible tokens by their distance, and
SA5 returns the token at that distance at every length if the score has a gap there at every
length.

A positional RoPE head scores the distance `t` by `∑_i (a_i cos θ_i t + b_i sin θ_i t)`
(`ropeScore`; `re_sum_rotary_eq_ropeScore` derives it from the rotation of arXiv:2309.00071v3,
§2.1, Eq. (5)). Such a score has no gap at long range. Cut the torus of the phases into
`(N + 1)^K` boxes; by the pigeonhole principle at least `H / (N + 1)^K − 1` of the distances below
`H` bring every phase within `2π / (N + 1)` of a multiple of `2π` (`exists_returns`), and each
moves the score by at most `2π ∑_i (|a_i| + |b_i|) / (N + 1)` (`abs_ropeScore_add_sub_le`). So
every position has linearly many near misses (`le_card_nearReturns`), and the near-miss theorem
with the exponent `1` gives: for all frequencies, coefficients and scales `s`, from some length
on, the indicator dictionary of every position is answered with an error above every `ε < 2`
(`exists_lt_logScaleAttention_ropeScore_error`). This covers the previous-token head at the
distance `1`. The proof needs `4 (N + 1)^K` tokens with `N + 1 > 2π s ∑_i (|a_i| + |b_i|)`,
far beyond 1M for `K = 64`, so it says what fails at every length, not at the lengths of MRCR-V2.

ReRoPE fails otherwise (`Transformer.MRCR.ReRopeFading`). Its score does not depend on the
distance from the window on, so the keys at distances `≥ W` tie, and tied keys share the weight
(`two_mul_le_one_sub_softmaxAttention_of_tie`). A positional head, whatever factor multiplies
its logit (`1`, `s log n` or Su's `max(1, log_T n)`), answers the dictionary that marks one of
them with an error of at least `2 (i − W) / (i − W + 1)`
(`two_mul_le_one_sub_softmax_rerope_far`). That bound holds at every length `i ≥ W`, not only
from some length on. Inside the window SSMax copies by position (`exists_forall_rerope_recall`).

Open: copying by content. An induction head finds the source by the token before it
(`Transformer.Grokking.FlatSpectrum.Recall.Induction`); in natural text that token recurs, the
matches tie and attention averages them (`abs_logScaleAttention_sub_average_le`). What a head
needs to copy 512 tokens of one of eight similar answers, and how errors of the copy compound
under sampling, is not proved.
