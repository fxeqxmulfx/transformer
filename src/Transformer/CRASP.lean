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
| `CRASP.PiecewiseTestable` | `𝒥`-expressions, `k`-piecewise testability, and `A_k` |
| `CRASP.Depth` | the Cropping and Reduction Lemmas and `thm:TLCl_depth` |
| `CRASP.TLCDepth` | their two-sided companions and `thm:TLC_depth` |
| `CRASP.Extensions` | the sugar of Appendix A.3 as an extended syntax, and its elimination |
| `CRASP.Fixed` | fixed-precision numbers, rounding, and their two characterizing bounds |
| `CRASP.Transformers` | future-masked rounded transformers and the equivalence with `TL[◁#]` |
| `CRASP.MajTwo` | `MAJ²`, majority quantification over two variables |
| `CRASP.MajTwoEquiv` | its translations to and from `TL[◁#, ▷#]`, and `LTC⁰` |
| `CRASP.Positional` | `TL[◁#]^pos`, the extension by `MOD` and `Y` |
| `CRASP.PositionalDepth` | its `Y`-normal form, its reduction, and its hierarchy |

Two things the paper carries are deliberately absent.  `lem:bb` and
`lem:piecewise_testable_depth_majtwo` sit inside `\iffalse` blocks in the
source and so are not part of it.  And the transformers with sinusoidal, RoPE
and ALiBi position encodings of Appendix E, together with the step from `MAJ²`
to `FO[<]`-uniform `LTC⁰` circuits, are not yet formalized.
-/

import Transformer.CRASP.Defs
import Transformer.CRASP.Basic
import Transformer.CRASP.Parikh
import Transformer.CRASP.PiecewiseTestable
import Transformer.CRASP.Depth
import Transformer.CRASP.TLCDepth
import Transformer.CRASP.Extensions
import Transformer.CRASP.Fixed
import Transformer.CRASP.Transformers
import Transformer.CRASP.MajTwo
import Transformer.CRASP.MajTwoEquiv
import Transformer.CRASP.Positional
import Transformer.CRASP.PositionalDepth
