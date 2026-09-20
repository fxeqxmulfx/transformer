/-
# Measure-to-measure interpolation — the 2-Wasserstein distance

The statements of arXiv:2411.04551v3 are about `W_2` on `𝒫(𝕊^{d-1})`, which is
the general construction of `Transformer.Wasserstein` read on the sphere.  The
distance there is the ambient `‖x - y‖` restricted to `𝕊^{d-1}`, which is what
the paper's Monge identity `lem: monge` is written in.

Source: arXiv:2411.04551v3, §1 (`W_2` throughout).
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Wasserstein

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- `γ` is a *coupling* of `μ` and `ν`: a measure on `𝕊^{d-1} × 𝕊^{d-1}` whose
two marginals are `μ` and `ν`.  Source: arXiv:2411.04551v3, §1. -/
abbrev IsCoupling (μ ν : Measure (SSphere d)) (γ : Measure (SSphere d × SSphere d)) : Prop :=
  Wasserstein.IsCoupling μ ν γ

/-- The quadratic transport costs `∫ d(x, y)² dγ` of the couplings `γ` of `μ`
and `ν`.  Source: arXiv:2411.04551v3, §1. -/
noncomputable abbrev transportCosts (μ ν : Measure (SSphere d)) : Set ℝ :=
  Wasserstein.transportCosts μ ν

/-- **The 2-Wasserstein distance on `𝒫(𝕊^{d-1})`**:

  `W_2(μ, ν) = (inf_{γ ∈ Π(μ, ν)} ∫ d(x, y)² dγ)^{1/2}`.

Source: arXiv:2411.04551v3, §1. -/
noncomputable abbrev W2 (μ ν : Measure (SSphere d)) : ℝ :=
  Wasserstein.W2 μ ν

/-- Every transport cost is nonnegative, the cost being an integral of a
nonnegative function. -/
theorem transportCosts_nonneg (μ ν : Measure (SSphere d)) :
    ∀ c ∈ transportCosts d μ ν, 0 ≤ c :=
  Wasserstein.transportCosts_nonneg μ ν

/-- The costs are bounded below, by `0`. -/
theorem bddBelow_transportCosts (μ ν : Measure (SSphere d)) :
    BddBelow (transportCosts d μ ν) :=
  Wasserstein.bddBelow_transportCosts μ ν

/-- `W_2` is nonnegative, being a square root. -/
theorem W2_nonneg (μ ν : Measure (SSphere d)) : 0 ≤ W2 d μ ν :=
  Wasserstein.W2_nonneg μ ν

/-- **One coupling bounds `W_2` from above.**  This is the only property of the
definition the paper's proofs use.  Source: arXiv:2411.04551v3, §1. -/
theorem W2_le_of_coupling (μ ν : Measure (SSphere d))
    (γ : Measure (SSphere d × SSphere d)) (hγ : IsCoupling d μ ν γ) :
    W2 d μ ν ≤ Real.sqrt (∫ p, dist p.1 p.2 ^ 2 ∂γ) :=
  Wasserstein.W2_le_of_coupling γ hγ

/-- The hypothesis of `W2_le_of_coupling` is satisfiable: a Dirac mass on the
diagonal couples a Dirac mass with itself. -/
example (x : SSphere 1) :
    IsCoupling 1 (Measure.dirac x) (Measure.dirac x) (Measure.dirac (x, x)) := by
  constructor <;> simp [measurable_fst, measurable_snd]

/-- **The diagonal coupling**: `W_2(μ, μ) = 0` for a probability measure `μ`. -/
theorem W2_self (μ : Measure (SSphere d)) [IsProbabilityMeasure μ] :
    W2 d μ μ = 0 :=
  Wasserstein.W2_self μ

end Interpolation
end Transformer
