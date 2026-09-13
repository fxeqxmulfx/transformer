/-
# §5 — A single cluster for large β

This file formalizes §5 of the survey, which contains:

* `Theorem thm: beta.interval` — clustering when `β ≥ C(d) n²`.
-/

import Transformer.Basic
import Transformer.Section1_IPS
import Transformer.Section3_SmallBeta

open scoped BigOperators

namespace Transformer
namespace SectionLargeBeta

open SectionIPS

variable (d n : ℕ)

/-- **Theorem (thm: beta.interval).** *Cluster collapse at high β.*

Fix `d, n ≥ 2`.  There exists a constant `C = C(d) > 0` (depending only on
`d`) such that whenever `β ≥ C(d) n²`, the conclusion of `thm: beta.tiny` holds
for both `SA` and `USA`:

For Lebesgue-almost any initial sequence `X₀ ∈ (𝕊^{d-1})^n`, there exists
`x⋆ ∈ 𝕊^{d-1}` such that `lim_{t→∞} x_i(t) = x⋆` for all `i ∈ [n]`. -/
theorem beta_interval
    (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∃ C : ℝ, 0 < C ∧ ∀ β : ℝ, C * (n : ℝ)^2 ≤ β →
      ∀ (X₀ : SphereTuple d n),
        ∃ x_star : SSphere d,
          ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → SectionIPS.SA d n β X →
            ∀ i : Idx n,
              Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace d) - x_star))
                Filter.atTop (nhds 0) := by
  sorry

end SectionLargeBeta
end Transformer
