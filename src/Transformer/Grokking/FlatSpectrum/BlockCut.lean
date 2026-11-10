module

public import Transformer.Grokking.FlatSpectrum.BlockCut.Defs
public import Transformer.Grokking.FlatSpectrum.BlockCut.Basic
public import Transformer.Grokking.FlatSpectrum.BlockCut.Sign
public import Transformer.Grokking.FlatSpectrum.BlockCut.Exact

/-!
# The block languages across a cut

Source: `review/flat_spectrum.md`, *The block languages across a cut* (added), for the
languages `L_k` and the depth hierarchy of Yang, Cadilhac and Chiang, *Knee-Deep in C-RASP: A
Transformer Depth Hierarchy* (§2.4 Equation `eq:altplus`, §4 Theorem `thm:rtfr_depth_hierarchy`);
the modules of `BlockCut/` cite its arXiv version.

`L_k` holds the words of `k` alternating blocks of `a` and `b` that start with `a`. Cut a word
into a prefix `u` and a suffix `v`, and let a readout `g(v) + ∑_{r < d} a_r(v) b_r(u)` of cut
width `d` tell whether `u ++ v ∈ L_k`.

The blocks add across the cut, and the two blocks at the cut are one when their letters agree
(`Defs`). A word that is not empty is in `L_k` exactly when it starts with `a` and has `k`
blocks, so the suffix acts on two numbers of the prefix, its number of blocks and its last
letter (`Basic`). Five numbers of the prefix then decide every `L_k` by the sign of a readout,
at every cut (`Sign`). The indicator of `L_k` is the identity on `k / 2` prefixes and suffixes,
a flat kernel, so a readout of cut width `d` misses it by `k / 2 − (d + 1)` in squared error;
and no transformer of depth `k`, of any width, recognizes `L_{k+1}`: the hierarchy of the paper
is in depth, at bounded cut width (`Exact`).
-/
