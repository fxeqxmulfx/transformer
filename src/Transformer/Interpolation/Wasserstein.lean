/-
# Measure-to-measure interpolation — the 2-Wasserstein distance

Mathlib carries no Wasserstein distance, and the statements of
arXiv:2411.04551v3 are about one, so it is defined here, on the sphere, where
the paper needs it: the square root of the infimum of the quadratic transport
cost over the couplings of the two measures,

  `W_2(μ, ν)² = inf_{γ ∈ Π(μ, ν)} ∫ d(x, y)² dγ(x, y)`.

The infimum is a `sInf` over the set of costs, which is bounded below by `0`
and, for probability measures, nonempty — the product coupling is one.  That
is all the paper's proofs use of the definition: an upper bound on `W_2` from
one exhibited coupling.

Source: arXiv:2411.04551v3, §1 (`W_2` throughout).
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- `γ` is a *coupling* of `μ` and `ν`: a measure on `𝕊^{d-1} × 𝕊^{d-1}` whose
two marginals are `μ` and `ν`.  Source: arXiv:2411.04551v3, §1. -/
def IsCoupling (μ ν : Measure (SSphere d)) (γ : Measure (SSphere d × SSphere d)) : Prop :=
  Measure.map Prod.fst γ = μ ∧ Measure.map Prod.snd γ = ν

/-- The quadratic transport costs `∫ d(x, y)² dγ` of the couplings `γ` of `μ`
and `ν`.  Source: arXiv:2411.04551v3, §1. -/
noncomputable def transportCosts (μ ν : Measure (SSphere d)) : Set ℝ :=
  { c | ∃ γ : Measure (SSphere d × SSphere d), IsCoupling d μ ν γ ∧
      ∫ p, dist p.1 p.2 ^ 2 ∂γ = c }

/-- **The 2-Wasserstein distance on `𝒫(𝕊^{d-1})`**:

  `W_2(μ, ν) = (inf_{γ ∈ Π(μ, ν)} ∫ d(x, y)² dγ)^{1/2}`.

Source: arXiv:2411.04551v3, §1. -/
noncomputable def W2 (μ ν : Measure (SSphere d)) : ℝ :=
  Real.sqrt (sInf (transportCosts d μ ν))

/-- Every transport cost is nonnegative, the cost being an integral of a
nonnegative function. -/
theorem transportCosts_nonneg (μ ν : Measure (SSphere d)) :
    ∀ c ∈ transportCosts d μ ν, 0 ≤ c := by
  rintro c ⟨γ, -, rfl⟩
  exact integral_nonneg fun p => by positivity

/-- The costs are bounded below, by `0`. -/
theorem bddBelow_transportCosts (μ ν : Measure (SSphere d)) :
    BddBelow (transportCosts d μ ν) :=
  ⟨0, transportCosts_nonneg d μ ν⟩

/-- `W_2` is nonnegative, being a square root. -/
theorem W2_nonneg (μ ν : Measure (SSphere d)) : 0 ≤ W2 d μ ν :=
  Real.sqrt_nonneg _

/-- **One coupling bounds `W_2` from above.**  This is the only property of the
definition the paper's proofs use.  Source: arXiv:2411.04551v3, §1. -/
theorem W2_le_of_coupling (μ ν : Measure (SSphere d))
    (γ : Measure (SSphere d × SSphere d)) (hγ : IsCoupling d μ ν γ) :
    W2 d μ ν ≤ Real.sqrt (∫ p, dist p.1 p.2 ^ 2 ∂γ) :=
  Real.sqrt_le_sqrt (csInf_le (bddBelow_transportCosts d μ ν) ⟨γ, hγ, rfl⟩)

/-- The hypothesis of `W2_le_of_coupling` is satisfiable: a Dirac mass on the
diagonal couples a Dirac mass with itself. -/
example (x : SSphere 1) :
    IsCoupling 1 (Measure.dirac x) (Measure.dirac x) (Measure.dirac (x, x)) := by
  constructor <;> simp [measurable_fst, measurable_snd]

/-- **The diagonal coupling**: `W_2(μ, μ) = 0` for a probability measure `μ`.
The map `x ↦ (x, x)` pushes `μ` to a coupling of `μ` with itself carrying no
cost, which is what makes `W2` a distance-like quantity rather than an
artefact of the `sInf`. -/
theorem W2_self (μ : Measure (SSphere d)) [IsProbabilityMeasure μ] :
    W2 d μ μ = 0 := by
  have hdiag : Measurable (fun x : SSphere d => (x, x)) := by fun_prop
  have hcoup : IsCoupling d μ μ (Measure.map (fun x : SSphere d => (x, x)) μ) := by
    constructor <;>
      rw [Measure.map_map (by fun_prop) hdiag] <;> simp [Function.comp_def]
  have hle := W2_le_of_coupling d μ μ _ hcoup
  have hcost : ∫ p, dist p.1 p.2 ^ 2 ∂(Measure.map (fun x : SSphere d => (x, x)) μ) = 0 := by
    have hmeasf : AEStronglyMeasurable (fun p : SSphere d × SSphere d => dist p.1 p.2 ^ 2)
        (Measure.map (fun x : SSphere d => (x, x)) μ) := by fun_prop
    rw [integral_map hdiag.aemeasurable hmeasf]
    simp
  rw [hcost, Real.sqrt_zero] at hle
  exact le_antisymm hle (W2_nonneg d μ μ)

end Interpolation
end Transformer
