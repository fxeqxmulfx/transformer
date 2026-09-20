/-
# The 2-Wasserstein distance

Mathlib carries no Wasserstein distance, and two of the papers formalized here
are about one, so it is defined once: the square root of the infimum of the
quadratic transport cost over the couplings of the two measures,

  `W_2(μ, ν)² = inf_{γ ∈ Π(μ, ν)} ∫ d(x, y)² dγ(x, y)`.

The infimum is a `sInf` over the set of costs, which is bounded below by `0`
and, for probability measures, nonempty — the diagonal coupling is one when
`μ = ν`, and the product coupling always is.  That is all the papers' proofs
use of the definition: an upper bound on `W_2` from one exhibited coupling.

The ambient space is any pseudometric measurable space.
arXiv:2411.04551v3 reads it on `𝕊^{d-1}` and arXiv:2604.01978v1 on `ℝ^d`
against measures carried by the sphere; the distance is the same chordal
`‖x - y‖` in both, so there is one object here and not two.

Sources: arXiv:2411.04551v3, §1 (`W_2` throughout); arXiv:2604.01978v1,
`prop:satisfying_MF`, `prop: poc`.
-/

import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Map

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Wasserstein

variable {X : Type*} [MeasurableSpace X] [PseudoMetricSpace X]

/-- `γ` is a *coupling* of `μ` and `ν`: a measure on `X × X` whose two
marginals are `μ` and `ν`. -/
def IsCoupling (μ ν : Measure X) (γ : Measure (X × X)) : Prop :=
  Measure.map Prod.fst γ = μ ∧ Measure.map Prod.snd γ = ν

/-- The quadratic transport costs `∫ d(x, y)² dγ` of the couplings `γ` of `μ`
and `ν`. -/
noncomputable def transportCosts (μ ν : Measure X) : Set ℝ :=
  { c | ∃ γ : Measure (X × X), IsCoupling μ ν γ ∧ ∫ p, dist p.1 p.2 ^ 2 ∂γ = c }

/-- **The 2-Wasserstein distance**:

  `W_2(μ, ν) = (inf_{γ ∈ Π(μ, ν)} ∫ d(x, y)² dγ)^{1/2}`. -/
noncomputable def W2 (μ ν : Measure X) : ℝ :=
  Real.sqrt (sInf (transportCosts μ ν))

/-- Every transport cost is nonnegative, the cost being an integral of a
nonnegative function. -/
theorem transportCosts_nonneg (μ ν : Measure X) :
    ∀ c ∈ transportCosts μ ν, 0 ≤ c := by
  rintro c ⟨γ, -, rfl⟩
  exact integral_nonneg fun p => by positivity

/-- The costs are bounded below, by `0`. -/
theorem bddBelow_transportCosts (μ ν : Measure X) :
    BddBelow (transportCosts μ ν) :=
  ⟨0, transportCosts_nonneg μ ν⟩

/-- `W_2` is nonnegative, being a square root. -/
theorem W2_nonneg (μ ν : Measure X) : 0 ≤ W2 μ ν :=
  Real.sqrt_nonneg _

/-- **One coupling bounds `W_2` from above.**  This is the only property of the
definition the papers' proofs use. -/
theorem W2_le_of_coupling {μ ν : Measure X} (γ : Measure (X × X))
    (hγ : IsCoupling μ ν γ) :
    W2 μ ν ≤ Real.sqrt (∫ p, dist p.1 p.2 ^ 2 ∂γ) :=
  Real.sqrt_le_sqrt (csInf_le (bddBelow_transportCosts μ ν) ⟨γ, hγ, rfl⟩)

/-- The hypothesis of `W2_le_of_coupling` is satisfiable: a Dirac mass on the
diagonal couples a Dirac mass with itself. -/
example (x : ℝ) : IsCoupling (Measure.dirac x) (Measure.dirac x) (Measure.dirac (x, x)) := by
  constructor <;> simp [measurable_fst, measurable_snd]

section Diagonal

/-! The diagonal coupling needs `d(·,·)` to be measurable for the product
σ-algebra, which is what these two instances buy; the definitions above do not
need them, and neither do the papers' uses of the upper bound. -/
variable [OpensMeasurableSpace X] [SecondCountableTopology X]

/-- **The diagonal coupling**: `W_2(μ, μ) = 0` for a probability measure `μ`.
The map `x ↦ (x, x)` pushes `μ` to a coupling of `μ` with itself carrying no
cost, which is what makes `W2` a distance-like quantity rather than an
artefact of the `sInf`. -/
theorem W2_self (μ : Measure X) [IsProbabilityMeasure μ] : W2 μ μ = 0 := by
  have hdiag : Measurable (fun x : X => (x, x)) := by fun_prop
  have hcoup : IsCoupling μ μ (Measure.map (fun x : X => (x, x)) μ) := by
    constructor <;>
      rw [Measure.map_map (by fun_prop) hdiag] <;> simp [Function.comp_def]
  have hle := W2_le_of_coupling _ hcoup
  have hcost : ∫ p, dist p.1 p.2 ^ 2 ∂(Measure.map (fun x : X => (x, x)) μ) = 0 := by
    have hmeasf : AEStronglyMeasurable (fun p : X × X => dist p.1 p.2 ^ 2)
        (Measure.map (fun x : X => (x, x)) μ) := by fun_prop
    rw [integral_map hdiag.aemeasurable hmeasf]
    simp
  rw [hcost, Real.sqrt_zero] at hle
  exact le_antisymm hle (W2_nonneg μ μ)

/-- The hypothesis of `W2_self` is satisfiable. -/
example : IsProbabilityMeasure (Measure.dirac (0 : ℝ)) := inferInstance

end Diagonal

end Wasserstein
end Transformer
