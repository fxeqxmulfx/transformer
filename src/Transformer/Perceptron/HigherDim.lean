/-
# Perceptrons and attention's mean-field landscape — higher dimensions

Formalization of `thm: any.d` of arXiv:2601.21366v2, §3.1: in `d ≥ 2` the
landscape is more complicated, but stationary measures are always singular
with respect to the uniform law, and atomicity remains generic.

**What the source says and what is carried here.**

* `σ_d`, the uniform measure on `𝕊^{d-1}`, is the tree's single device for
  the uniform law: a rotation-invariant probability measure,
  `Metastability.IsUniformOn`.  Such a measure is unique, so "`σ_d(supp μ) = 0`"
  is carried as "`ν(supp μ) = 0` for every rotation-invariant probability
  measure `ν`", and that quantifier sits in the conclusion, not among the
  hypotheses: the tree never constructs `σ_d`, and a hypothesis it cannot
  witness would be a hypothesis nobody could discharge.

* "In particular, `μ` is singular with respect to `σ_d`" is
  `mutuallySingular_of_measure_support_eq_zero`, proved from that conclusion
  taken as a hypothesis: the complement of the support is `μ`-null.

* The parameter space `ℝ_{>0} × (ℝ^{d+1})^d` is `Params d`, and `U_μ ⊂ ℝ_{>0} ×
  (ℝ^{d+1})^d` open and dense is `IsOpen U`, `U ⊆ {β > 0}` and
  `{β > 0} ⊆ closure U` — density in the subspace `{β > 0}`, which is what the
  source's inclusion says, and openness in it is openness outright since it is
  itself open.

* "the restriction of `μ` to the active regions is purely atomic with at most
  countably many atoms" is: the active region minus some countable set is
  `μ`-null.

Source: arXiv:2601.21366v2, `thm: any.d`.
-/

import Transformer.Perceptron.Atomicity
import Transformer.Metastability.InitialUniform

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-- The parameter space `ℝ_{>0} × (ℝ^{d+1})^d` of `thm: any.d` (ii)–(iii): an
inverse temperature `β`, and for each of the `d` neurons an output scalar
`ω_j` and an input vector `a_j`. -/
abbrev Params (d : ℕ) : Type := ℝ × (Idx d → ℝ) × (Idx d → EucSpace d)

/-- **"In particular, `μ` is singular with respect to `σ_d`".**  A measure
whose support is `ν`-null is mutually singular with `ν`: the complement of the
support carries no mass of `μ`.

Source: arXiv:2601.21366v2, `thm: any.d` (i). -/
theorem mutuallySingular_of_measure_support_eq_zero {μ ν : Measure (SSphere d)}
    (h : ν μ.support = 0) : μ ⟂ₘ ν :=
  ⟨μ.supportᶜ, (Measure.isOpen_compl_support (μ := μ)).measurableSet,
    Measure.measure_compl_support, by rwa [compl_compl]⟩

/-- The hypothesis of `mutuallySingular_of_measure_support_eq_zero` is
satisfiable: the zero measure has empty support. -/
example : (0 : Measure (SSphere d)) (Measure.support (0 : Measure (SSphere d))) = 0 := by
  simp

/-- **Theorem (thm: any.d) (i).**  Let `d ≥ 2` and `β > 0`.  If `μ` is
stationary for `σ(s) = s_+` with weights whose potential `v_ϑ` is not
real-analytic, or `μ` is a strict SOPD critical point for a real-analytic `σ`,
then `σ_d(supp μ) = 0`.

The uniform law is quantified over inside the conclusion; see the module
docstring.  Singularity of `μ` with respect to it then follows, by
`mutuallySingular_of_measure_support_eq_zero`.

Not proved here.

Source: arXiv:2601.21366v2, `thm: any.d` (i). -/
theorem any_d_measure_support_eq_zero (d : ℕ) (hd : 2 ≤ d) (β : ℝ) (hβ : 0 < β)
    (φ σ : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hcrit : (σ = fun s => max s 0) ∧ ¬ IsAnalyticOnSphere φ ω a ∧
        IsStationary β σ ω a μ ∨
      AnalyticOnNhd ℝ σ Set.univ ∧ IsStrictSOPD β φ σ ω a μ) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  sorry

/-- The hypotheses of `any_d_measure_support_eq_zero` are satisfiable, in both
of the two alternatives: at `d = 2` the pinned ReLU perceptron of
`isStationary_relu_pin` is stationary with a non-analytic potential, and the
pinned perceptron of `isStrictSOPD_pin` is a strict SOPD critical point at the
real-analytic `σ ≡ 1/2`. -/
example :
    ((fun s => max s 0) = fun s : ℝ => max s 0) ∧
      ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
          (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) ∧
      IsStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨rfl,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis
      inner_basePoint_secondAxis,
    isStationary_relu_pin 1 0 (basePoint 1)⟩

example :
    AnalyticOnNhd ℝ (fun _ : ℝ => (2 : ℝ)⁻¹) Set.univ ∧
      IsStrictSOPD 1 (fun s => s) (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨fun _ _ => analyticAt_const, isStrictSOPD_pin 1 0 (basePoint 1)⟩

/-- **Theorem (thm: any.d) (ii).**  If `σ` is real-analytic and `σ(s) ≠ 0` for
`s ≠ 0`, there is an open dense `U_μ ⊂ ℝ_{>0} × (ℝ^{d+1})^d` such that a `μ`
stationary at a parameter of `U_μ` is purely atomic with finite support.

Not proved here.

Source: arXiv:2601.21366v2, `thm: any.d` (ii). -/
theorem any_d_generic_isFinitelyAtomic (d : ℕ) (hd : 2 ≤ d) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (hσ : AnalyticOnNhd ℝ σ Set.univ)
    (hσ0 : ∀ s : ℝ, s ≠ 0 → σ s ≠ 0) (μ : Perspective.ProbSphere d) :
    ∃ U : Set (Params d), IsOpen U ∧ U ⊆ {p | 0 < p.1} ∧ {p | 0 < p.1} ⊆ closure U ∧
      ∀ p ∈ U, IsStationary p.1 σ p.2.1 p.2.2 μ → IsFinitelyAtomic μ := by
  sorry

/-- The hypotheses of `any_d_generic_isFinitelyAtomic` are satisfiable: `d = 2`,
`φ = id` and the constant activation `σ ≡ 1/2`, which is real-analytic and
never vanishes. -/
example :
    (2 : ℕ) ≤ 2 ∧ (∀ s : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) s) s) ∧
      AnalyticOnNhd ℝ (fun _ : ℝ => (2 : ℝ)⁻¹) Set.univ ∧
      ∀ s : ℝ, s ≠ 0 → (fun _ : ℝ => (2 : ℝ)⁻¹) s ≠ 0 :=
  ⟨le_rfl, fun s => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := s),
    fun _ _ => analyticAt_const, fun _ _ => by norm_num⟩

/-- **Theorem (thm: any.d) (iii).**  If `σ(s) = s_+`, there is a dense
`U_μ ⊂ ℝ_{>0} × (ℝ^{d+1})^d` such that, at a parameter of `U_μ` at which `μ`
is stationary, the restriction of `μ` to the active regions
`⋃_j {x : a_j · x > 0}` is purely atomic with at most countably many atoms.

Not proved here.

Source: arXiv:2601.21366v2, `thm: any.d` (iii). -/
theorem any_d_relu_generic_countablyAtomic (d : ℕ) (hd : 2 ≤ d) (φ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * max s 0) s) (μ : Perspective.ProbSphere d) :
    ∃ U : Set (Params d), U ⊆ {p | 0 < p.1} ∧ {p | 0 < p.1} ⊆ closure U ∧
      ∀ p ∈ U, IsStationary p.1 (fun s => max s 0) p.2.1 p.2.2 μ →
        ∃ A : Set (SSphere d), A.Countable ∧
          (μ : Measure (SSphere d))
            ({x : SSphere d | ∃ j : Idx d,
              0 < inner (𝕜 := ℝ) (p.2.2 j) (x : EucSpace d)} \ A) = 0 := by
  sorry

/-- The hypotheses of `any_d_relu_generic_countablyAtomic` are satisfiable:
`d = 2` and the ReLU primitive `φ(s) = (s_+)²`. -/
example : (2 : ℕ) ≤ 2 ∧ ∀ s : ℝ, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s :=
  ⟨le_rfl, hasDerivAt_reluSq⟩

end Perceptron
end Transformer
