module

public import Transformer.Grokking.FlatSpectrum.Recall.Defs
public import Transformer.Grokking.FlatSpectrum.Recall.Basic
public import Transformer.Grokking.FlatSpectrum.Recall.Width
public import Transformer.Grokking.FlatSpectrum.Recall.Sign
public import Transformer.Grokking.FlatSpectrum.Recall.Coyote
public import Transformer.Grokking.FlatSpectrum.Recall.RetNet
public import Transformer.Grokking.FlatSpectrum.Recall.Attention
public import Transformer.Grokking.FlatSpectrum.Recall.Training
public import Transformer.Grokking.FlatSpectrum.Recall.Softmax
public import Transformer.Grokking.FlatSpectrum.Recall.SoftmaxRecall
public import Transformer.Grokking.FlatSpectrum.Recall.SoftmaxScale
public import Transformer.Grokking.FlatSpectrum.Recall.Fading
public import Transformer.Grokking.FlatSpectrum.Recall.QueryFree
public import Transformer.Grokking.FlatSpectrum.Recall.LogScale
public import Transformer.Grokking.FlatSpectrum.Recall.Threshold
public import Transformer.Grokking.FlatSpectrum.Recall.RawGap
public import Transformer.Grokking.FlatSpectrum.Recall.Induction
public import Transformer.Grokking.FlatSpectrum.Recall.InductionExact
public import Transformer.Grokking.FlatSpectrum.Recall.InductionPhases
public import Transformer.Grokking.FlatSpectrum.Recall.CausalCopy
public import Transformer.Grokking.FlatSpectrum.Recall.CausalFading
public import Transformer.Grokking.FlatSpectrum.Recall.CausalRecall
public import Transformer.Grokking.FlatSpectrum.Recall.CausalKeyToken
public import Transformer.Grokking.FlatSpectrum.Recall.CausalPhases
public import Transformer.Grokking.FlatSpectrum.Recall.PositionBias
public import Transformer.Grokking.FlatSpectrum.Recall.ALiBiWindow
public import Transformer.Grokking.FlatSpectrum.Recall.NearerTokens
public import Transformer.Grokking.FlatSpectrum.Recall.FittedSlope
public import Transformer.Grokking.FlatSpectrum.Recall.LogBias
public import Transformer.Grokking.FlatSpectrum.Recall.PowerBias
public import Transformer.Grokking.FlatSpectrum.Recall.PowerThreshold
public import Transformer.Grokking.FlatSpectrum.Recall.Subkernel
public import Transformer.Grokking.FlatSpectrum.Recall.Parity
public import Transformer.Grokking.FlatSpectrum.Recall.Monomials
public import Transformer.Grokking.FlatSpectrum.Recall.Polynomial
public import Transformer.Grokking.FlatSpectrum.Recall.PolynomialCount
public import Transformer.Grokking.FlatSpectrum.Recall.Quadratic
public import Transformer.Grokking.FlatSpectrum.Recall.TwoLayers
public import Transformer.Grokking.FlatSpectrum.Recall.TwoLayersRecall
public import Transformer.Grokking.FlatSpectrum.Recall.Family
public import Transformer.Grokking.FlatSpectrum.Recall.AffineFamily
public import Transformer.Grokking.FlatSpectrum.Recall.MultiQuery
public import Transformer.Grokking.FlatSpectrum.Recall.Unseen

/-!
# Associative recall has a flat spectrum

Source: `review/flat_spectrum.md`, *Associative recall has a flat spectrum* (added), for the
task and the models of Arora et al., *Zoology: Measuring and Improving Recall in Efficient
Language Models* (§3 Definition `def: general-AR`, §4 equation `eq: coyote-recursion` and
Claim 1, Appendix `app:retnet-proof`); the modules of `Recall/` cite its arXiv version.

A context binds `m` keys to values, a dictionary `π`, and the query `q` asks for `π q`. For a
readout `φ` of the answer the recall kernel is `K(q, π) = φ(π q)`, with `m` rows and a column for
each dictionary (`Defs`). For a centered readout its rows are orthogonal of equal length, so `K`
is flat: rank and effective rank `m`. Theorem A for flat kernels of any shape then says that `K`
is a minimum of M from `Ω` exactly when `φ(π q) = 0` at every unseen pair, so never from a proper
`Ω` for a readout without zeros (`Basic`).

Flatness prices the width. A readout `g(q) + ∑_{r < d} a_r(q) b_r(π)` of cut width `d` misses
recall by at least the fraction `1 − d / m` of the variance of `φ`, exact recall has `d ≥ m`,
and one-hot keys reach `d = m` (`Width`). A sign or an argmax that is always right has `d ≥ m`
as well (`Sign`). A Coyote layer of dimension `d` read linearly at the query has cut width `d`,
and one of dimension `m` recalls `m` bindings (`Coyote`); the RetNet output `C[N,:] z^N` has cut
width `d`, at any precision of its `d × d` state (`RetNet`); attention without softmax has cut
width the dimension of its projections, and one more with linear biases (`Attention`). The pure
factorization of width `k < m` keeps a squared loss on the full table, and training on part of
the table does not return `K` (`Training`).

Softmax attention is not such a readout (*Softmax attention escapes the cut* of the note). The
attention of Yang, Cadilhac and Chiang, *Knee-Deep in C-RASP: A Transformer Depth Hierarchy*,
§2.1, averages the values with the weights `exp s_j` (`Softmax`). With scores that pair the
query with the key it recalls every number of bindings to every accuracy at projection
dimension two, so it has no cut width below the number of keys (`SoftmaxRecall`), and it
pays in the scale of its scores: bounded by `B`, they leave a squared error of at least the
fraction `((m − 1) / (e^{2B} + m − 1))²` of the variance of `φ` (`SoftmaxScale`). The worst
dictionary of values `±1` is answered with an error of at least `2 (m − 1) / (e^{2B} + m − 1)`:
recall by the sign has `m − 1 < e^{2B}`, and at a fixed `B` that error tends to `2`, the
opposite value (`Fading`). With scores
that ignore the query, as in the transformers that paper builds from formulas, a linear readout
read by its sign recalls at most `d + 1` bindings (`QueryFree`). The scale `s log n` over `n`
tokens, which long-context systems take (*The Mean-Field Dynamics of Transformers*, §6), grows
by itself: raw scores that lead by a fixed `g` with `s g > 1` recall at every large number of
keys, and raw scores bounded by `B` with `2 s B < 1` fade (`LogScale`). For one-hot raw scores `±B`
the worst error is exactly `2m / (n^{2sB} + m)`: at the threshold `2 s B = 1` every answer keeps
its sign, and the error tends to `1` (`Threshold`). The lead is limited by the geometry of the
raw scores `⟨u_i, v_j⟩`: if every key leads every other by more than `a b` at its own query,
with `‖u_i‖ ≤ a` and `‖v_j‖ ≤ b`, the keys are affinely independent, at most `d + 1` in
dimension `d`; the diff-gpt model needs a raw lead above `1` and has it for at most `d + 1` keys
of head width `d` (`RawGap`). A binding split into a key token and a value token is recalled by
two layers, as by an induction head: layer 1 copies the key to the value token, layer 2 recalls
from the copy, and with both scales above `1` they recall at every large number of keys
(`Induction`). Computed exactly, the induction head is one-hot attention at the scale
`s₂ a`, where `a = (n^{s₁} − 1) / (n^{s₁} + n − 1)` is the lead of the copy, and its worst error
over the dictionaries of values `±1` is `(3m + 1) / (n^{s₂ a} + 2m + 1)` (`InductionExact`).
The lead tends to `0`, `1/2` or `1` as `s₁` is below, at or above `1`, and the head recalls when
`s₂` times that limit exceeds `1` and fades when it is below: `s₁ = 1` with `s₂ > 2` recalls, and
`s₁ < 1` fades at every `s₂` (`InductionPhases`). With a causal mask, a value token copies its key
as the cyclic head over its prefix of `L` tokens, with the weight `L^{s₁} / (L^{s₁} + L − 1)`: the
copy of a key is set by its position, not by the length of the context (`CausalCopy`). Layer 2
fades at every key whose copy `w_q` has `s₂ w_q < 1`, and below `s₁ = 1` at every key from some
key on (`CausalFading`); it recalls a key with `s₂ w_q > 1` that its key token copies less than
its value token does (`CausalRecall`), and loses one that its key token copies more, as the first
key without a beginning-of-sequence token at every `s₁` (`CausalKeyToken`). Over the keys,
below `s₂ = 1` every key fades, `s₁ = 1` with `s₂ ≥ 2` recalls every key but the first without
such a token, and above `s₁ = 1` and `s₂ = 1` the late keys are recalled (`CausalPhases`).
A bias of the positions, as ALiBi adds `−m r` at the distance `r`, only adds to the miss at a
token of least bias, so no bias lowers the worst error over the positions of a key
(`PositionBias`). For one-hot raw scores `±B` a fixed ALiBi slope `m` at the scale `s log n`
leaves a window of radius `(2sB/m) log n`: the keys inside it are recalled, at a bounded
distance at every positive lead, and the keys outside it are lost (`ALiBiWindow`). A bias that
does not increase with the distance makes the `R` nearer tokens rivals of the key at the distance
`R`, so the keys beyond `n^ρ` are lost below `2 s B = ρ`, at every ALiBi slope (`NearerTokens`).
The slope `log(1 + 1/D)` fitted to a span `D ≤ n^ρ` reaches that threshold: it recalls every key
of the span when `s g > ρ`, and loses the keys beyond `κ (D + 1) log n`, `κ > 2 s B`
(`FittedSlope`). The bias `−log(1 + r)` reaches it at every distance at once: every bias weighs
some distance `R` by at least `(R + 1) H_n`, this one weighs every distance by exactly that, and it
recalls every key with `R + 1 ≤ n^ρ` when `s g > ρ` (`LogBias`). The bias `−α log(1 + r)`,
`α ≥ 0`, has the threshold `τ = α ρ + max(0, 1 − α)` at the distance `n^ρ`: its prior weighs at
most `α / (α − 1)` above `α = 1` and `n^{1−α} / (1 − α)` below it (`PowerBias`), the keys within
`n^ρ` are recalled when `s g > τ`, and the keys beyond it are lost when `2 s B < τ`
(`PowerThreshold`). A wrong coefficient raises the threshold `ρ` of `α = 1` by `(α − 1) ρ` above
it and by `(1 − α)(1 − ρ)` below it.

The width bound is not special to recall: on any queries and contexts that carry a flat kernel
with squared singular value `c`, a readout of cut width `d` has squared error at least
`c (min(|m|, |n|) − (d + 1))`, and `d` replaces `d + 1` for rows of sum zero (`Subkernel`).

Readouts that are polynomial in the numbers that cross the cut are bounded by a second flat
kernel (*Polynomial readouts and two layers* of the note). The parities of the bindings, a row
for each of the `2^m` sets of keys, form a flat kernel of rank `2^m` (`Parity`). A readout of
degree `Δ` in `D` crossing numbers has cut width at most `C(D + Δ, D)` (`Monomials`,
`Polynomial`), and if it recalls `m` bindings exactly then the parity kernel is a product
through the `C(D + Δ m, D)` monomials of degree `Δ m`, so `2^m ≤ C(D + Δ m, D)`: linear in `m`,
`(4/3)^m ≤ (2Δ + 1)^{2D}`, and `m ≤ 5 D` at degree two; a hidden layer with the activation
`t^p` has degree `p` (`PolynomialCount`). A product of two readouts of degree one is of degree
two, with cut width at most `D + D²` (`Quadratic`). Two Coyote layers read at the query are of
degree two in `2d` numbers of the context (`TwoLayers`): exact recall has `m ≤ 10 d`, dimension
`m` suffices, and the error and the argmax are bounded at cut width `2d + 4d²`
(`TwoLayersRecall`).

The bounds do not need every dictionary (*Families of dictionaries and the multi-query task*
of the note). On a family in which every key takes every value equally often and every two
keys take every pair of values equally often, the kernel `φ(D_i q)` is flat, so the regression
bound and the exact bound hold on the family (`Family`). The `p²` affine dictionaries
`q ↦ a q + b` over a field of `p` elements are such a family, and the constant dictionaries,
which are not, are recalled at cut width one (`AffineFamily`). A dictionary and a query are an
input of the multi-query task of the Zoology paper, with distinct keys and the answer `π q` at
its last triple, so a model of that task is bounded as well (`MultiQuery`).

The first paragraph says that recall is not a minimum of M, and not what the minimum is. With
one unseen pair `(q, π)` it is known (*One unseen cell* of the note). For two keys or more and a
centered readout without zeros, a minimum of M from the other pairs exists, and every minimum
answers the unseen pair with `−θ φ(π q)`, `g / (2c − g) ≤ θ ≤ g / (c − g)`, where
`c = |ν|^{m−1} ∑_v |φ(v)|²` and `g = ∑_{p ≠ q} |φ(π p)|²`: the opposite phase. For a readout of
modulus one, `(m − 1) / (2 |ν|^m − (m − 1)) ≤ θ ≤ (m − 1) / (|ν|^m − (m − 1))` (`Unseen`).
-/
