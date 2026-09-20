/-
# Perceptrons and attention's mean-field landscape — atomicity on the circle

Formalization of `thm: circle` and `thm: circle.gelu` of arXiv:2601.21366v2,
§3.1: in `d = 2` the stationary measures of the coupled energy are purely
atomic with finite support — for the ReLU because the potential is not
analytic, for an analytic activation because a *strict* SOPD critical point
cannot spread out.

**What the source says and what is carried here.**

* "purely atomic and has finite support" is `IsFinitelyAtomic`: `μ` is a finite
  combination of Dirac masses.  For a probability measure this is the same as
  being carried by a finite set, and it is the source's own phrasing.

* `σ(s) = s_+` is substituted into the statement, and `φ` is left as *any*
  primitive of `2σ` (`hasDerivAt_reluSq` exhibits one): that is the standing
  convention `φ' = 2σ` of §2, and no statement depends on the constant.

* "real-analytic `σ`" is `AnalyticOnNhd ℝ σ Set.univ`; "`v_ϑ` not real-analytic
  on `𝕊¹`" is `¬ IsAnalyticOnSphere`, see `Analytic.lean`.

Source: arXiv:2601.21366v2, `thm: circle`, `thm: circle.gelu`.
-/

import Transformer.Perceptron.Analytic
import Transformer.Perceptron.Dirac

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### Purely atomic measures with finite support -/

/-- **`μ` is purely atomic with finite support**: a finite combination of
Dirac masses.

Source: arXiv:2601.21366v2, `thm: circle`. -/
def IsFinitelyAtomic (μ : Perspective.ProbSphere d) : Prop :=
  ∃ (F : Finset (SSphere d)) (m : SSphere d → ℝ≥0∞),
    (μ : Measure (SSphere d)) = ∑ z ∈ F, m z • Measure.dirac z

/-- A Dirac mass is purely atomic with finite support. -/
theorem isFinitelyAtomic_diracProb (x : SSphere d) :
    IsFinitelyAtomic (Perspective.diracProb d x) :=
  ⟨{x}, fun _ => 1, by
    show Measure.dirac x = ∑ z ∈ ({x} : Finset (SSphere d)), (1 : ℝ≥0∞) • Measure.dirac z
    rw [Finset.sum_singleton, one_smul]⟩

/-! ### A concrete frame of `ℝ²`

`basePoint 1` is the first standard basis vector of `ℝ²`; `secondAxis` is the
second.  The witnesses below need a great circle through the atom, hence an
orthonormal pair. -/

/-- The second standard basis vector of `ℝ²`. -/
noncomputable def secondAxis : EucSpace 2 := EuclideanSpace.single (1 : Fin 2) (1 : ℝ)

@[simp] theorem norm_secondAxis : ‖secondAxis‖ = 1 := by
  simp [secondAxis]

@[simp] theorem norm_basePoint_one : ‖((basePoint 1 : SSphere 2) : EucSpace 2)‖ = 1 :=
  mem_sphere_zero_iff_norm.mp (basePoint 1).2

@[simp] theorem inner_basePoint_secondAxis :
    inner (𝕜 := ℝ) ((basePoint 1 : SSphere 2) : EucSpace 2) secondAxis = 0 := by
  simp [basePoint, secondAxis, EuclideanSpace.inner_single_left]

/-! ### The two theorems -/

/-- **Theorem (thm: circle).**  Let `d = 2`, `β > 0` and `σ(s) = s_+`, and let
the weights `ϑ` be such that `v_ϑ` is not real-analytic on `𝕊¹`.  Then any
`μ ∈ P(𝕊¹)` satisfying the stationarity condition `eq: steady.state` is purely
atomic and has finite support.

The footnote of the source ("this discards `v_ϑ ≡ 0` and the weight symmetries
making `v_ϑ` a quadratic trigonometric polynomial") is not a further
hypothesis: those are cases in which `v_ϑ` *is* analytic.

Not proved here.

Source: arXiv:2601.21366v2, `thm: circle`. -/
theorem circle_isFinitelyAtomic (β : ℝ) (hβ : 0 < β) (φ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * max s 0) s) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsStationary β (fun s => max s 0) ω a μ) :
    IsFinitelyAtomic μ := by
  sorry

/-- The hypotheses of `circle_isFinitelyAtomic` are satisfiable: `β = 1`, the
ReLU primitive `φ(s) = (s_+)²`, the one-neuron weights `ω = e₀`,
`a_0 = -basePoint 1` — whose potential is not analytic on `𝕊¹`, by
`not_isAnalyticOnSphere_relu` — and `μ = δ_{basePoint 1}`, which is stationary
because that neuron is inactive at its atom. -/
example :
    (0 : ℝ) < 1 ∧ (∀ s : ℝ, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
      ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
          (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
      IsStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨one_pos, hasDerivAt_reluSq,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis
      inner_basePoint_secondAxis,
    isStationary_relu_pin 1 0 (basePoint 1)⟩

/-- **Theorem (thm: circle.gelu).**  Let `d = 2`, `β > 0`, fix weights `ϑ` and
a real-analytic `σ : ℝ → ℝ`.  If `μ ∈ P(𝕊¹)` is a *strict* SOPD Wasserstein
critical point of `E_{β,ϑ}` in the sense of `eq:strict2order`, then `μ` is
purely atomic and has finite support.

Equivalently: a SOPD critical point with infinite support is degenerate.

Not proved here.

Source: arXiv:2601.21366v2, `thm: circle.gelu`. -/
theorem circle_gelu_isFinitelyAtomic (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (hσ : AnalyticOnNhd ℝ σ Set.univ) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (hμ : IsStrictSOPD β φ σ ω a μ) :
    IsFinitelyAtomic μ := by
  sorry

/-- The hypotheses of `circle_gelu_isFinitelyAtomic` are satisfiable: `β = 1`,
`φ = id`, the constant activation `σ ≡ 1/2` — real-analytic — and the pinned
Dirac mass of `isStrictSOPD_pin`, a strict SOPD critical point with `κ = 1/2`. -/
example :
    (0 : ℝ) < 1 ∧ (∀ s : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) s) s) ∧
      AnalyticOnNhd ℝ (fun _ : ℝ => (2 : ℝ)⁻¹) Set.univ ∧
      IsStrictSOPD 1 (fun s => s) (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨one_pos, fun s => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := s), fun _ _ => analyticAt_const,
    isStrictSOPD_pin 1 0 (basePoint 1)⟩

end Perceptron
end Transformer
