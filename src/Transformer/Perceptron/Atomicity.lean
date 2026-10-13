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

* "real-analytic `σ`" is `AnalyticOnNhd ℝ σ Set.univ`; its strict SOPD
  conclusion is proved in `Section3_StrictAtomicity.lean`. The non-analytic
  ReLU hypothesis is `¬ IsAnalyticOnSphere`, see `Analytic.lean`.

Source: arXiv:2601.21366v2, `thm: circle`, `thm: circle.gelu`.
-/

import Transformer.Perceptron.Analytic
import Transformer.Perceptron.Dirac
import Mathlib.MeasureTheory.Measure.Support
import Mathlib.MeasureTheory.Measure.Dirac.Basic

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

/-- **A probability measure with finite topological support is a finite
combination of Dirac masses, with its actual singleton masses.**
Source: arXiv:2601.21366v2, §3.1, the meaning of finite atomicity
in `thm: circle` and `thm: circle.gelu`.
The ambient dimension is arbitrary; the sphere has a Borel measure
space with measurable singletons and second-countable topology. -/
theorem isFinitelyAtomic_of_finite_topological_support
    (μ : Perspective.ProbSphere d) (hfinite : (μ : Measure (SSphere d)).support.Finite) :
    IsFinitelyAtomic μ := by
  classical
  have hcarry : ∀ᵐ x ∂(μ : Measure (SSphere d)), x ∈ hfinite.toFinset := by
    filter_upwards [Measure.support_mem_ae (μ := (μ : Measure (SSphere d)))] with x hx
    exact hfinite.mem_toFinset.mpr hx
  exact ⟨hfinite.toFinset, fun x => (μ : Measure (SSphere d)) {x},
    Measure.ae_mem_finset_iff.mp hcarry⟩

/-- The finite-support input is realized on the actual nonempty
one-dimensional sphere by its Dirac probability measure. -/
example : (Perspective.diracProb 1 (basePoint 0) : Measure (SSphere 1)).support.Finite := by
  apply (Set.finite_singleton (basePoint 0)).subset
  intro x hx
  exact Set.mem_singleton_iff.mpr (Interpolation.eq_of_mem_support_dirac hx)

/-- **A finite Dirac presentation has finite topological support.**
Source: arXiv:2601.21366v2, §3.1, the meaning of finite atomicity
in `thm: circle` and `thm: circle.gelu`. Together with the converse
above this verifies that the calculus step yields the source's
actual measure-theoretic conclusion. Zero-mass atoms are permitted. -/
theorem finite_support_of_finitely_atomic (μ : Perspective.ProbSphere d)
    (hμ : IsFinitelyAtomic μ) : (μ : Measure (SSphere d)).support.Finite := by
  classical
  obtain ⟨F, m, heq⟩ := hμ
  have hsub (G : Finset (SSphere d)) :
      (∑ z ∈ G, m z • Measure.dirac z).support ⊆ (G : Set (SSphere d)) := by
    induction G using Finset.induction_on with
    | empty => simp
    | @insert z G hz ih =>
      rw [Finset.sum_insert hz, Measure.support_add]
      intro x hx
      rcases hx with hx | hx
      · have hxz : x = z := Interpolation.eq_of_mem_support_dirac
          (Measure.smul_absolutelyContinuous.support_mono hx)
        subst x
        exact Finset.mem_insert_self z G
      · exact Finset.mem_insert_of_mem (ih hx)
  rw [heq]
  exact F.finite_toSet.subset (hsub F)

/-- The finite-presentation input is satisfied by an actual Dirac
probability measure on a nonempty sphere. -/
example : IsFinitelyAtomic (Perspective.diracProb 1 (basePoint 0)) :=
  isFinitelyAtomic_diracProb (basePoint 0)

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

/-! ### ReLU atomicity -/

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

end Perceptron
end Transformer
