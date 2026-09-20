/-
# Sphere packing — how many `δ`-separated unit vectors fit

The R'enyi centers of §5 of 2411.04990v2 are, once the indexing is stripped
away, a `δ`-separated set of unit vectors (`Packing.Basic`), so their number is
the packing number of `𝕊^{d-1}` at scale `δ`.  The two volume ledgers
(`Packing.Upper`, `Packing.Lower`) bound that number in both directions, and
`Packing.Count` turns them into the matching powers of `δ`, which
`Packing.Renyi` re-reads at the R'enyi scale `δ = c β^{-1/2}` for
`Causal.renyi_count`.
-/

import Transformer.Causal.Packing.Basic
import Transformer.Causal.Packing.Upper
import Transformer.Causal.Packing.Lower
import Transformer.Causal.Packing.Count
import Transformer.Causal.Packing.Renyi
