module

public import Transformer.MRCR.CountHead
public import Transformer.MRCR.CountLookup
public import Transformer.MRCR.Counting
public import Transformer.MRCR.InnerNeedle
public import Transformer.MRCR.NearMiss
public import Transformer.MRCR.NearMissThreshold
public import Transformer.MRCR.Ordinal
public import Transformer.MRCR.PartScore
public import Transformer.MRCR.PhraseKey
public import Transformer.MRCR.PhraseKeyGrams
public import Transformer.MRCR.PhraseKeyLogits
public import Transformer.MRCR.PhraseKeyPrev
public import Transformer.MRCR.PhraseKeyPrevCorrect
public import Transformer.MRCR.PhraseKeyReRope
public import Transformer.MRCR.PhraseKeyRope
public import Transformer.MRCR.PhraseKeyRotary
public import Transformer.MRCR.PhraseKeySandwich
public import Transformer.MRCR.PhraseKeyTemplated
public import Transformer.MRCR.PhraseKeyTemplatedLimits
public import Transformer.MRCR.Returns
public import Transformer.MRCR.RotaryCopy
public import Transformer.MRCR.RotaryPrevHead
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

`Transformer.MRCR.PhraseKey` spells an ask in words, "`f` about `t`", as MRCR-V2 does. Without
positional encoding no rounded transformer answers the query at every length: over the words of
a string of bits it would have to accept the conversations whose bits end in `ba`, and no formula
of `TL[◁#]` tells `u·aba` from `u·baa` for every `u` (`Transformer.CRASP.SecondLast`).
`Transformer.MRCR.PhraseKeyRope` does the same for RoPE and sinusoidal encodings at rational
angles: on blocks of one period a formula of `TL[◁#, MOD]` is one of `TL[◁#]`, so positions read
modulo a fixed number do not help (`Transformer.CRASP.SecondLastMod`).
`Transformer.MRCR.PhraseKeyTemplated` and `Transformer.MRCR.PhraseKeyTemplatedLimits` extend
both to templated conversations, turns of an ask and its answer closed by an answered query: in
blocks that start at a topic the key of an ask straddles two blocks, and the opening ask moves
into the formula (`Transformer.CRASP.LeftQuotient`). ALiBi rounds to `TL[◁#, Y]`, whose `Y` reads
the previous token, and there a formula of depth two answers the query
(`Transformer.MRCR.PhraseKeyPrev`, `Transformer.MRCR.PhraseKeyPrevCorrect`). A rounded
transformer of five layers answers it at every length too: three layers read the last four
tokens into a gram, and two run that formula over the grams. It does so with ALiBi, with either
kernel of KERPLE, and with SSMax of any of these, the logit divided by `1`, `√d` or `d`
(`Transformer.MRCR.PhraseKeyGrams`, `Transformer.MRCR.PhraseKeyLogits`). Sandwich's bias does not
decay: it comes back near its maximum at distances without bound, where it beats its value at the
token before, so no head with it reads that token alone and the construction has no Sandwich
version (`Transformer.MRCR.PhraseKeySandwich`). Nor does a RoPE score of the distance decay, at
any frequencies: it converges only if it is constant, and a far key comes back within any margin
of the token before, so the construction has no version with RoPE's grid, YaRN's rescaled grid
and temperature, or LeRoPE's learned frequencies (`Transformer.MRCR.PhraseKeyRotary`). Clamping
the distance at a window, as ReRoPE does, makes the score constant from the window on, and then
five layers answer the query at every window `W ≥ 2`, under SSMax too
(`Transformer.MRCR.PhraseKeyReRope`). Without a margin, at precision `0`, a negative logit
already drops a key, yet RoPE with every plane rotated still gives keys arbitrarily far a weight
(`Transformer.MRCR.RotaryPrevHead`). SSMax that scales the dot product alone, and RoPE or
Sandwich by other means, are left open.

`Transformer.MRCR.Returns` and `Transformer.MRCR.RotaryCopy` treat the copy of the answer by its
position: a RoPE score of the distance comes back near its value at linearly many distances, so
a RoPE head that scores by the distance only loses every position at long range, at every scale.
-/
