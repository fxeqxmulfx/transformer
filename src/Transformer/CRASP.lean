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
| `CRASP.PiecewiseTestable` | `𝒥`-expressions, `k`-piecewise testability, and `A_k` |
| `CRASP.Depth` | the Cropping and Reduction Lemmas and `thm:TLCl_depth` |
| `CRASP.TLCDepth` | their two-sided companions and `thm:TLC_depth` |
| `CRASP.Extensions` | the sugar of Appendix A.3 as an extended syntax, and its elimination |
| `CRASP.Fixed` | fixed-precision numbers, rounding, and their two characterizing bounds |
| `CRASP.Transformers` | future-masked rounded transformers and the equivalence with `TL[◁#]` |
| `CRASP.MajTwo` | `MAJ²`, majority quantification over two variables |
| `CRASP.MajTwoDepthOne` | why its closed depth-`1` formulas cannot read the last symbol |
| `CRASP.MajTwoEquiv` | its translations to and from `TL[◁#, ▷#]`, and `LTC⁰` |
| `CRASP.Positional` | `TL[◁#]^pos`, the extension by `MOD` and `Y` |
| `CRASP.PositionalDepth` | its `Y`-normal form, its reduction, and its hierarchy |
| `CRASP.PositionalTransformers` | sinusoidal, RoPE and ALiBi position encodings |
| `CRASP.PositionalHierarchy` | what they simulate, and their depth hierarchies |

Two things the paper carries are deliberately absent.  `lem:bb` and
`lem:piecewise_testable_depth_majtwo` sit inside `\iffalse` blocks in the
source and so are not part of it; so do `thm:mnf`, `thm:tlmod_to_rtfr` and
`thm:TLCmod_to_rtfr` of Appendix E, and `lem:find_half_planes_oneway` of §4.3.
The step from `MAJ²` to `FO[<]`-uniform `LTC⁰` circuits is not formalized
either: it is a statement about circuits, which this development does not
model.
-/

import Transformer.CRASP.Defs
import Transformer.CRASP.Basic
import Transformer.CRASP.Parikh
import Transformer.CRASP.Alternating
import Transformer.CRASP.PiecewiseTestable
import Transformer.CRASP.Depth
import Transformer.CRASP.TLCDepth
import Transformer.CRASP.Extensions
import Transformer.CRASP.Fixed
import Transformer.CRASP.Transformers
import Transformer.CRASP.MajTwo
import Transformer.CRASP.MajTwoDepthOne
import Transformer.CRASP.MajTwoEquiv
import Transformer.CRASP.Positional
import Transformer.CRASP.PositionalTransformers
import Transformer.CRASP.PositionalDepth
import Transformer.CRASP.PositionalHierarchy
