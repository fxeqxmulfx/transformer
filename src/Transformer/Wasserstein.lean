/-
# The 2-Wasserstein distance

`W_2` on a pseudometric measurable space (`Wasserstein/Basic.lean`), with the
two bounds the papers' mass arguments run on: mass across a gap bounds it
from below (`Wasserstein/LowerBound.lean`), and collapsing the mass outside a
set onto one point bounds it from above (`Wasserstein/Collapse.lean`).

Sources: arXiv:2411.04551v3, §1 (`W_2` throughout); arXiv:2604.01978v1,
`prop:satisfying_MF`, `prop: poc`.
-/

import Transformer.Wasserstein.Basic
import Transformer.Wasserstein.LowerBound
import Transformer.Wasserstein.Collapse
