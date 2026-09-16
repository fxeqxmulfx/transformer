/-
# Knee-Deep in C-RASP: a transformer depth hierarchy

Formalization of Yang, Huang, Chiang — arXiv:2506.16055v3, "Knee-Deep in
C-RASP: A Transformer Depth Hierarchy" (COLM 2025).

The paper studies `TL[◁#]` (equivalently C-RASP) and `TL[◁#, ▷#]`, temporal
logics whose only non-Boolean device is counting how many earlier (or later)
positions satisfy a subformula.  Their *depth* — the nesting of counting
operators — matches the depth of a fixed-precision future-masked transformer,
and the main results are that this depth induces a strict hierarchy: the
languages `A_k = (a⁺b⁺)^{k/2}` need depth exactly `k`.

| Module | Contents |
| --- | --- |
| `CRASP.Defs` | the syntax and semantics of `TL[◁#, ▷#]`, its depth, and its fragments |
| `CRASP.Basic` | the derived connectives and the first properties of satisfaction |
| `CRASP.Parikh` | Parikh vectors, intervals, affix restrictions, and the §4 vocabulary |
| `CRASP.Alternating` | the separating family `A_k` and its 𝒥-expression characterization |
| `CRASP.Blocks` | the blocks of a word of `L_k`: their number, its last letter, its prefixes, and appending blocks |
| `CRASP.BoundedExists` | `∃ j ≤ i`, `∃ j < i` and `∃ j > i`, written as counts |
| `CRASP.Indicator` | sums and indicators `ψ ? 1 : 0` in a comparison, at no cost in depth |
| `CRASP.Subsequence` | the depth-`k` past-only formula testing a subsequence of length `k` |
| `CRASP.SubsequenceTwoSided` | the depth-`(k+1)` formula testing a subsequence of length `2k+1` from its middle |
| `CRASP.PiecewiseTestable` | `𝒥`-expressions, `k`-piecewise testability, and `A_k` |
| `CRASP.NeutralLetter` | a neutral letter preserves `k`-piecewise testability |
| `CRASP.DepthZero` | what a depth-0 formula reads: the letter at its position and its PNPs |
| `CRASP.Middle` | in the middle of an affix restriction, a depth-0 formula reads the letter alone |
| `CRASP.Commutative` | `lem:TLCP_commutative`: depth-1 formulas define languages commutative on the middle |
| `CRASP.CroppingUnsound` | the cropping lemmas fail: a count also reads the positions before the interval |
| `CRASP.ReductionUnsound` | the Reduction Lemma fails: no depth-1 formula checks the affix `ab` |
| `CRASP.Locality` | a past-only formula reads the prefix up to its position; the formulas under its counts |
| `CRASP.Strip` | formulas constant, letter by letter, on a strip of prefix vectors past a fixed prefix |
| `CRASP.Affine` | past a prefix, a count of constant formulas is affine in the numbers of `a`s and `b`s |
| `CRASP.Shrink` | `lem:cropping_oneway` repaired: a strip shrinks until given formulas are constant on it |
| `CRASP.LowerBound` | a depth-`(k+1)` formula confuses `L_{k+2}` with `L_{k+4}`, without `lem:reduction` |
| `CRASP.Depth` | the Cropping and Reduction Lemmas refuted, and `thm:TLCl_depth` |
| `CRASP.Prediction` | the next-token prediction problem for `L_{k+3}`, solved at depth `k + 1` |
| `CRASP.TLCDepth` | the two-sided Cropping Lemma refuted, and `thm:TLC_depth` |
| `CRASP.Extensions` | the sugar of Appendix A.3 as an extended syntax, and its elimination |
| `CRASP.Fixed` | fixed-precision numbers, rounding, and their two characterizing bounds |
| `CRASP.Transformers` | future-masked rounded transformers and the equivalence with `TL[◁#]` |
| `CRASP.MajTwo` | `MAJ²`, majority quantification over two variables |
| `CRASP.MajTwoDepthOne` | why its closed depth-`1` formulas cannot read the last symbol |
| `CRASP.MajTwoEquiv` | its translations to and from `TL[◁#, ▷#]`, and `LTC⁰` |
| `CRASP.Positional` | `TL[◁#]^pos`, the extension by `MOD` and `Y` |
| `CRASP.PositionalEmbedding` | `TL[◁#]` inside `TL[◁#]^pos` and both its fragments, at the same depth |
| `CRASP.YNormalForm` | its `Y`-normal form, and the transformation into it |
| `CRASP.YNormalFormEquiv` | the transformation preserves meaning, once guarded as the paper's is not |
| `CRASP.Spread` | the string map `e^r w₁ e^{r−1} ⋯ wₙ e^{r−1}` of the reduction, read by blocks |
| `CRASP.PositionalReductionAtom` | the translation of the reduction on `Y`-atoms, and the block size it needs |
| `CRASP.PositionalReduction` | the translation on formulas and terms, at the same depth |
| `CRASP.PositionalReductionAtomEquiv` | the translation of a `Y`-atom reads the letter its position carries |
| `CRASP.PositionalReductionCount` | a count on `f(w)` is the first block plus the counts of the block formulas |
| `CRASP.PositionalReductionEquiv` | the translation preserves meaning; the paper's block size is too small |
| `CRASP.PositionalDepth` | its reduction to `TL[◁#]`, and its hierarchy |
| `CRASP.PositionalTransformers` | sinusoidal, RoPE and ALiBi position encodings |
| `CRASP.PositionalHierarchy` | what they simulate, and their depth hierarchies |

Two things the paper carries are deliberately absent.  `lem:bb` and
`lem:piecewise_testable_depth_majtwo` sit inside `\iffalse` blocks in the
source and so are not part of it; so do `thm:mnf`, `thm:tlmod_to_rtfr` and
`thm:TLCmod_to_rtfr` of Appendix F, and `lem:find_half_planes_oneway` of §4.4.
The step from `MAJ²` to `FO[<]`-uniform `LTC⁰` circuits is not formalized
either: it is a statement about circuits, which this development does not
model.
-/

import Transformer.CRASP.Defs
import Transformer.CRASP.Basic
import Transformer.CRASP.Parikh
import Transformer.CRASP.Alternating
import Transformer.CRASP.Blocks
import Transformer.CRASP.BoundedExists
import Transformer.CRASP.Indicator
import Transformer.CRASP.Subsequence
import Transformer.CRASP.SubsequenceTwoSided
import Transformer.CRASP.PiecewiseTestable
import Transformer.CRASP.NeutralLetter
import Transformer.CRASP.DepthZero
import Transformer.CRASP.Middle
import Transformer.CRASP.Commutative
import Transformer.CRASP.CroppingUnsound
import Transformer.CRASP.ReductionUnsound
import Transformer.CRASP.Locality
import Transformer.CRASP.Strip
import Transformer.CRASP.Affine
import Transformer.CRASP.Shrink
import Transformer.CRASP.LowerBound
import Transformer.CRASP.Depth
import Transformer.CRASP.Prediction
import Transformer.CRASP.TLCDepth
import Transformer.CRASP.Extensions
import Transformer.CRASP.Fixed
import Transformer.CRASP.Transformers
import Transformer.CRASP.MajTwo
import Transformer.CRASP.MajTwoDepthOne
import Transformer.CRASP.MajTwoEquiv
import Transformer.CRASP.Positional
import Transformer.CRASP.PositionalEmbedding
import Transformer.CRASP.PositionalTransformers
import Transformer.CRASP.YNormalForm
import Transformer.CRASP.YNormalFormEquiv
import Transformer.CRASP.Spread
import Transformer.CRASP.PositionalReductionAtom
import Transformer.CRASP.PositionalReduction
import Transformer.CRASP.PositionalReductionAtomEquiv
import Transformer.CRASP.PositionalReductionCount
import Transformer.CRASP.PositionalReductionEquiv
import Transformer.CRASP.PositionalDepth
import Transformer.CRASP.PositionalHierarchy
