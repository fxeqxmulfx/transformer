/-
# Thinking Like Transformers

Formalization of Weiss, Goldberg, Yahav — arXiv:2106.06981v2, "Thinking Like
Transformers" (ICML 2021).

The paper proposes RASP, a programming language whose primitives are the
primitives of a transformer encoder, and argues that reading a task as a RASP
program bounds the layers and heads a transformer needs for it.  It states no
theorems; what is formalized here is the semantics of §3, the two identities
§3.1 asserts about the built-ins, the `selector_width` implementation of
Figure 8, the worked programs, and the compilation claim of §3.1 together
with the expressiveness consequence §4 draws from it.

| Module | Contents |
| --- | --- |
| `RASP.Defs` | sequences, selection matrices, and the three core operations |
| `RASP.Basic` | their row-level API, `length` as a program, and the histogram |
| `RASP.SelectorWidth` | and that Figure 8 computes the width, with and without BOS |
| `RASP.Programs` | `reverse` and `frac_as`, the two programs written out in §3 |
| `RASP.Sort` | and the sorting program, whose ranks are a permutation |
| `RASP.Compilation` | a program's syntax, its heads and layers, and what layer 0 can see |
-/

import Transformer.RASP.Defs
import Transformer.RASP.Basic
import Transformer.RASP.SelectorWidth
import Transformer.RASP.Programs
import Transformer.RASP.Sort
import Transformer.RASP.Compilation
