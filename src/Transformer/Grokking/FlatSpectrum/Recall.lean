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
public import Transformer.Grokking.FlatSpectrum.Recall.QueryFree
public import Transformer.Grokking.FlatSpectrum.Recall.Subkernel
public import Transformer.Grokking.FlatSpectrum.Recall.Parity
public import Transformer.Grokking.FlatSpectrum.Recall.Monomials
public import Transformer.Grokking.FlatSpectrum.Recall.Polynomial
public import Transformer.Grokking.FlatSpectrum.Recall.PolynomialCount
public import Transformer.Grokking.FlatSpectrum.Recall.Quadratic
public import Transformer.Grokking.FlatSpectrum.Recall.TwoLayers
public import Transformer.Grokking.FlatSpectrum.Recall.TwoLayersRecall

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
dimension two (`SoftmaxRecall`), and it pays in the scale of its scores: bounded by `B`, they
leave a squared error of at least the fraction `((m − 1) / (e^{2B} + m − 1))²` of the variance
of `φ` (`SoftmaxScale`). With scores that ignore the query, as in the transformers that paper
builds from formulas, a linear readout read by its sign recalls at most `d + 1` bindings
(`QueryFree`).

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
-/
