/-
# The paraboloid embedding: definitions

The `LookUp` primitive of the *Append-only Lookup Machine* (Percepta,
`Percepta-Core/transformer-vm`) turns exact key matching into a linear
argmax by lifting keys onto a paraboloid.

The published construction embeds a **scalar** key `k` as `(2k, -k²)` and
queries as `(q, 1)`, so a head of dimension `2` performs exact matching.  We
define the general form directly: `m`-dimensional keys lift to
`k ↦ (2k, -‖k‖²)`, queries to `q ↦ (q, 1)`, and a head of dimension `m + 1`
realizes the inner product

  `score q k = ⟪2k, q⟫ + (-‖k‖²)·1 = 2⟪k, q⟫ - ‖k‖²`.

The published construction is the case `m = 1`, isolated here as `sScore`.

The API lives in `Transformer.ALM.Basic`.
-/

import Transformer.Basic
import Mathlib.Analysis.InnerProductSpace.Basic

namespace Transformer
namespace ALM

variable {m : ℕ}

/-- The attention score realized by the paraboloid embedding
`k ↦ (2k, -‖k‖²)` against the query embedding `q ↦ (q, 1)`:

  `score q k = ⟪2k, q⟫ + (-‖k‖²)·1 = 2⟪k, q⟫ - ‖k‖²`. -/
noncomputable def score (q k : EucSpace m) : ℝ :=
  2 * inner (𝕜 := ℝ) k q - ‖k‖ ^ 2

/-- The scalar paraboloid score `sScore q k = 2kq - k²`, i.e. `score` in
dimension `m = 1` restricted to integer keys.  This is what the `LookUp`
primitive actually compiles to. -/
noncomputable def sScore (q k : ℤ) : ℝ := 2 * (k : ℝ) * (q : ℝ) - (k : ℝ) ^ 2

end ALM
end Transformer
