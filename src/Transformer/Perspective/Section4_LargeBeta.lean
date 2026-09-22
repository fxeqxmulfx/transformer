/-
# §5 — A single cluster for large β

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §5 of the survey, which contains:

* `Theorem thm: beta.interval` — clustering when `β ≥ C(d) n²`.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section3_SmallBeta

open scoped BigOperators
open MeasureTheory

namespace Transformer
namespace Perspective

open Perspective

variable (d n : ℕ)

/-- **Theorem (thm: beta.interval).** *Cluster collapse at high β.*

Fix `d, n ≥ 2`.  There exists a constant `C = C(d) > 0` (depending only on
`d`) such that whenever `β ≥ C(d) n²`, the conclusion of `thm: beta.tiny` holds
for both `SA` and `USA`:

For Lebesgue-almost any initial sequence `X₀ ∈ (𝕊^{d-1})^n`, there exists
`x⋆ ∈ 𝕊^{d-1}` such that `lim_{t→∞} x_i(t) = x⋆` for all `i ∈ [n]`, for the
solution of `SA` and for that of `USA` (`clusteringSetUSA`).

*Almost any* is not *any*: `antipodalPair_not_mem_clusteringSet` exhibits, for
every `β`, an initial sequence outside `𝒮_β` — an antipodal pair is stationary
for `SA`, so its two particles never meet.  The exceptional set is therefore
non-empty at every `β`, however large, and the quantifier has to be the
almost-everywhere one, read against the uniform law as in §4.

Not proved here.

Source: arXiv:2312.10794v5, §5, `thm: beta.interval`. -/
theorem beta_interval
    (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∃ C : ℝ, 0 < C ∧ ∀ β : ℝ, C * (n : ℝ)^2 ≤ β →
      ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
        ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet d n β ∧ X₀ ∈ clusteringSetUSA d n β := by
  sorry

/-- The hypotheses of `beta_interval` are satisfiable: `d = n = 2`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

end Perspective
end Transformer
