module

public import Transformer.Grokking.FlatSpectrum.Recall.Defs
public import Transformer.Grokking.FlatSpectrum.Recall.Basic
public import Transformer.Grokking.FlatSpectrum.Recall.Width
public import Transformer.Grokking.FlatSpectrum.Recall.Sign
public import Transformer.Grokking.FlatSpectrum.Recall.Coyote
public import Transformer.Grokking.FlatSpectrum.Recall.RetNet
public import Transformer.Grokking.FlatSpectrum.Recall.Attention
public import Transformer.Grokking.FlatSpectrum.Recall.Training

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
-/
