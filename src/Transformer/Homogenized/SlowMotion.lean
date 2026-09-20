/-
# Homogenized Transformers — the low-temperature setting

Formalization of `ass:low-temperature` of arXiv:2604.01978v1, *Homogenized
Transformers*, §3: the perturbative regime around the uniform distribution in
which the slow-motion theorem `thm:large_beta_meta` is proved.

The uniform measure `σ_d` on `𝕊^{d-1}` is not constructed here.  It is pinned
down instead, by the property that characterizes it — a probability measure
carried by the unit sphere and invariant under every linear isometry of the
ambient `ℝ^d` is `σ_d`, and there is only one — so a statement quantified over
every measure satisfying `IsUniformAmbient` says exactly what the source says
about `σ_d`.  This is `Metastability.IsUniformOn`'s device, read in ambient
coordinates because §4 and §5 of this paper work with measures on `ℝ^d`
carried by the sphere rather than with measures on a subtype.

That the device is not vacuous is `exists_isUniformAmbient_one`: on the circle
`𝕊^0 = {±e}` the uniform law is the fair two-point measure, and it is proved
to satisfy the three conditions.
-/

import Transformer.Homogenized.MvGenerator
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.MeasureTheory.Measure.WithDensity

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The uniform law on the sphere, in ambient coordinates -/

/-- `σ` is **the uniform law `σ_d` on `𝕊^{d-1}`**, read as a measure on the
ambient `ℝ^d`: a probability measure carried by the unit sphere and invariant
under every linear isometry of `ℝ^d`.

Such a measure is unique, so quantifying over all of them is not a
strengthening; see the module docstring.

Source: arXiv:2604.01978v1, `ass:low-temperature`. -/
def IsUniformAmbient (d : ℕ) (σ : Measure (EucSpace d)) : Prop :=
  IsProbabilityMeasure σ ∧
    σ {y : EucSpace d | ‖y‖ = 1}ᶜ = 0 ∧
      ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map U = σ

/-- Every vector of `ℝ^1` is a multiple of the first basis vector. -/
theorem eq_smul_single_one (v : EucSpace 1) :
    v = v 0 • (EuclideanSpace.single 0 (1 : ℝ)) := by
  ext i
  have : i = 0 := Subsingleton.elim _ _
  subst this
  simp

/-- On `ℝ^1` the norm is the absolute value of the single coordinate. -/
theorem norm_eucSpace_one (v : EucSpace 1) : ‖v‖ = |v 0| := by
  rw [EuclideanSpace.norm_eq]
  simp [Real.sqrt_sq_eq_abs]

/-- **`IsUniformAmbient` is satisfiable.**  On `𝕊^0 = {e, -e} ⊆ ℝ^1` the fair
two-point measure is carried by the sphere and invariant under both linear
isometries of `ℝ^1`, which are `±id`. -/
theorem exists_isUniformAmbient_one : ∃ σ : Measure (EucSpace 1), IsUniformAmbient 1 σ := by
  classical
  set e : EucSpace 1 := EuclideanSpace.single 0 (1 : ℝ) with he
  have hne : ‖e‖ = 1 := by simp [he]
  refine ⟨(2 : ℝ≥0∞)⁻¹ • Measure.dirac e + (2 : ℝ≥0∞)⁻¹ • Measure.dirac (-e), ?_, ?_, ?_⟩
  · constructor
    simp
    rw [ENNReal.inv_two_add_inv_two]
  · have h₁ : e ∉ {y : EucSpace 1 | ‖y‖ = 1}ᶜ := by simp [hne]
    have h₂ : -e ∉ {y : EucSpace 1 | ‖y‖ = 1}ᶜ := by simp [hne]
    simp [Measure.dirac_apply' _ (by measurability : MeasurableSet {y : EucSpace 1 | ‖y‖ = 1}ᶜ),
      Set.indicator_of_notMem h₁, Set.indicator_of_notMem h₂]
  · intro U
    have hU : ‖U e‖ = 1 := by rw [U.norm_map, hne]
    have hcoord : (U e) 0 = 1 ∨ (U e) 0 = -1 := by
      have := norm_eucSpace_one (U e)
      rw [hU] at this
      rcases abs_eq (by norm_num : (0:ℝ) ≤ 1) |>.mp this.symm with h | h
      exacts [Or.inl h, Or.inr h]
    have hUe : U e = e ∨ U e = -e := by
      rcases hcoord with h | h
      · left
        rw [eq_smul_single_one (U e), h, one_smul, ← he]
      · right
        rw [eq_smul_single_one (U e), h, ← he]
        module
    have hall : (∀ v, U v = v) ∨ ∀ v, U v = -v := by
      rcases hUe with h | h
      · left
        intro v
        rw [eq_smul_single_one v, ← he, map_smul, h]
      · right
        intro v
        rw [eq_smul_single_one v, ← he, map_smul, h, smul_neg, ← eq_smul_single_one v]
    rcases hall with h | h
    · have : ⇑U = id := funext h
      rw [this, Measure.map_id]
    · have hfun : ⇑U = fun v : EucSpace 1 => -v := funext h
      rw [hfun, Measure.map_add, Measure.map_smul, Measure.map_smul, Measure.map_dirac,
        Measure.map_dirac, neg_neg]
      all_goals
        first
          | exact add_comm _ _
          | exact measurable_neg
          | exact measurable_neg.aemeasurable

/-! ### The low-temperature assumption -/

/-- **Assumption (ass:low-temperature).**  For every `t ≥ 0` the law `μ(t)` is
absolutely continuous with respect to the uniform measure `σ_d`, and its
density `ρ(t,·) = dμ(t)/dσ_d` is `C¹` with

  `0 < ρ_min ≤ ρ(t,·) ≤ ρ_max`,  `‖∇ρ(t,·)‖_{L^∞(𝕊^{d-1})} ≤ L`.

**What the source says and how it is read here.**

* Absolute continuity together with the density is one condition, `density`:
  naming `ρ` and asking `μ(t) = ρ(t,·) σ_d` says both at once.
* `ρ(t,·) ∈ C¹(𝕊^{d-1})` is carried as `C¹` on the ambient `ℝ^d`, which is how
  the proof of `lemma:Laplace_method` uses it — it expands
  `ρ(Exp_u(v)) = ρ(u) + O(‖v‖)` against an ambient modulus.
* `∇ρ` is the Riemannian gradient, `proj_y ∇ρ̃(y)` by `lem:toolkit_geo_riem`,
  and the `L^∞(𝕊^{d-1})` norm is a bound at every unit vector.  The bounds on
  `ρ` itself are likewise imposed only on the sphere, where they are the
  source's, and are silent off it, where the density is irrelevant to `μ(t)`.
* The source adds that `ρ_min` and `ρ_max` are "independent of `d` and `β`".
  That is not a condition on one instance of this assumption but a uniformity
  in how `thm:large_beta_meta` quantifies; it is recorded there, by placing the
  constants before `d` and `β`.

Source: arXiv:2604.01978v1, `ass:low-temperature`. -/
structure IsLowTemperature (d : ℕ) (σ : Measure (EucSpace d))
    (μ : ℝ → Measure (EucSpace d)) (ρ : ℝ → EucSpace d → ℝ) (ρmin ρmax L : ℝ) : Prop where
  /-- `σ` is the uniform law on `𝕊^{d-1}`. -/
  unif : IsUniformAmbient d σ
  /-- The lower bound on the density is positive. -/
  pos : 0 < ρmin
  /-- `μ(t) ≪ σ_d` with density `ρ(t,·)`. -/
  density : ∀ t, 0 ≤ t → μ t = σ.withDensity fun y => ENNReal.ofReal (ρ t y)
  /-- `ρ(t,·) ∈ C¹`. -/
  smooth : ∀ t, 0 ≤ t → ContDiff ℝ 1 (ρ t)
  /-- `ρ_min ≤ ρ(t,·)` on the sphere. -/
  lower : ∀ t, 0 ≤ t → ∀ y : EucSpace d, ‖y‖ = 1 → ρmin ≤ ρ t y
  /-- `ρ(t,·) ≤ ρ_max` on the sphere. -/
  upper : ∀ t, 0 ≤ t → ∀ y : EucSpace d, ‖y‖ = 1 → ρ t y ≤ ρmax
  /-- `‖∇ρ(t,·)‖ ≤ L` on the sphere, for the Riemannian gradient. -/
  grad : ∀ t, 0 ≤ t → ∀ y : EucSpace d, ‖y‖ = 1 → ‖proj d y (gradient (ρ t) y)‖ ≤ L

/-- **`IsLowTemperature` is satisfiable**, and at a stationary law: the uniform
measure itself, whose density is `ρ ≡ 1`, with `ρ_min = ρ_max = 1` and `L = 0`. -/
example : ∃ (σ : Measure (EucSpace 1)) (μ : ℝ → Measure (EucSpace 1)),
    IsLowTemperature 1 σ μ (fun _ _ => 1) 1 1 0 := by
  obtain ⟨σ, hσ⟩ := exists_isUniformAmbient_one
  refine ⟨σ, fun _ => σ, hσ, one_pos, ?_, ?_, ?_, ?_, ?_⟩
  · intro t _
    simp
  · intro _ _
    exact contDiff_const
  · intro _ _ _ _
    exact le_refl _
  · intro _ _ _ _
    exact le_refl _
  · intro _ _ y _
    simp [proj]

end Homogenized
end Transformer
