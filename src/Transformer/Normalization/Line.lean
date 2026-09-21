/-
# Normalization — the two laws of `thm: convergence` on the line

At `d = 1` the sphere is `𝕊⁰ = {±1}` and the two initial laws of
`thm: convergence` are explicit:

* the uniform law is `uniformSph0 = (δ₊₁ + δ₋₁)/2` (`uniformTuple_sph0`), and
  every rotation-invariant probability measure on `𝕊⁰` charges both points
  (`uniform_sph0_pos`);
* the standard Gaussian charges no point and each open half-line
  (`stdGaussian_one_pos`), so a Gaussian pair has one positive and one negative
  token with positive probability (`not_ae_gaussian_pair`).

These are the measure-theoretic halves of the refutations in `ClusteringLine`.

Source: arXiv:2510.22026v2, §3, `thm: convergence`, its initializations.
-/

import Transformer.Perspective.SphereInvariant
import Transformer.Perspective.Section3_SmallBeta
import Mathlib.Probability.Distributions.Gaussian.Multivariate

open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Normalization

/-- The standard Gaussian on `ℝ¹` charges no point. -/
theorem stdGaussian_one_singleton (x : EucSpace 1) : stdGaussian (EucSpace 1) {x} = 0 := by
  have : ∀ y, stdGaussian (EucSpace 1) ≠ Measure.dirac y := by
    intro y h
    have h1 := variance_dual_stdGaussian (innerSL ℝ (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)))
    rw [h, variance_dirac, innerSL_apply_norm] at h1
    simp at h1
  have := IsGaussian.nullSingletonClass this
  exact measure_singleton x

/-- The standard Gaussian on `ℝ¹` gives positive mass to each open half-line. -/
theorem stdGaussian_one_pos :
    stdGaussian (EucSpace 1) {x | 0 < x 0} ≠ 0 ∧ stdGaussian (EucSpace 1) {x | x 0 < 0} ≠ 0 := by
  set γ := stdGaussian (EucSpace 1)
  have hsymm : γ {x | x 0 < 0} = γ {x | 0 < x 0} := by
    have hmap : γ.map (LinearIsometryEquiv.neg ℝ (E := EucSpace 1)) = γ := stdGaussian_map _
    conv_rhs => rw [← hmap]
    rw [Measure.map_apply (LinearIsometryEquiv.neg ℝ).continuous.measurable
      (measurableSet_lt measurable_const (by fun_prop))]
    congr 1
    ext x
    simp
  have hsum : γ {x | 0 < x 0} + γ {x | x 0 < 0} + γ {0} = 1 := by
    rw [← measure_univ (μ := γ)]
    rw [← measure_union (by
        rw [Set.disjoint_left]; intro x h1 h2; simp at h1 h2; linarith)
        (measurableSet_lt (by fun_prop) measurable_const),
      ← measure_union (by
        rw [Set.disjoint_left]; rintro x (h | h) rfl <;> simp at h)
        (measurableSet_singleton _)]
    congr 1
    ext x
    simp only [Set.mem_union, Set.mem_ofPred_eq, Set.mem_singleton_iff, Set.mem_univ, iff_true]
    rcases lt_trichotomy (x 0) 0 with h | h | h
    · exact Or.inl (Or.inr h)
    · right; ext i; rw [Fin.fin_one_eq_zero i]; simpa using h
    · exact Or.inl (Or.inl h)
  rw [stdGaussian_one_singleton, add_zero, hsymm] at hsum
  have : γ {x | 0 < x 0} ≠ 0 := by
    intro h; rw [h] at hsum; simp at hsum
  exact ⟨this, hsymm ▸ this⟩

/-- The two points of `𝕊⁰`. -/
noncomputable def sph0 (b : Bool) : SSphere 1 :=
  if b then basePoint 0 else Perspective.sphereMap 1 (LinearIsometryEquiv.neg ℝ) (basePoint 0)

theorem sph0_coord (b : Bool) : (sph0 b : EucSpace 1) 0 = if b then 1 else -1 := by
  cases b <;> simp [sph0, basePoint, Perspective.sphereMap]

theorem sq_coord_of_norm_one {θ : EucSpace 1} (hθ : ‖θ‖ = 1) : θ 0 ^ 2 = 1 := by
  have := EuclideanSpace.norm_eq θ
  rw [hθ, Fin.sum_univ_one, Real.norm_eq_abs, sq_abs] at this
  nlinarith [Real.sq_sqrt (sq_nonneg (θ 0)), this]

/-- The hypothesis of `sq_coord_of_norm_one` is satisfiable: the base point. -/
example : ‖(basePoint 0 : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp (basePoint 0).2

/-- `𝕊⁰` has exactly the two points `sph0 true`, `sph0 false`. -/
theorem sph0_cases (x : SSphere 1) : x = sph0 true ∨ x = sph0 false := by
  have h := sq_coord_of_norm_one (mem_sphere_zero_iff_norm.mp x.2)
  have hx : ∀ b, (x : EucSpace 1) 0 = (sph0 b : EucSpace 1) 0 → x = sph0 b := by
    intro b hb
    apply Subtype.ext
    ext i
    rw [Fin.fin_one_eq_zero i, hb]
  rcases sq_eq_one_iff.mp h with h1 | h1
  · exact Or.inl (hx true (by rw [h1, sph0_coord]; rfl))
  · exact Or.inr (hx false (by rw [h1, sph0_coord]; rfl))

theorem sph0_ne : sph0 true ≠ sph0 false := by
  intro h
  have := congrArg (fun x : SSphere 1 => (x : EucSpace 1) 0) h
  simp only [sph0_coord] at this
  norm_num at this

/-- A rotation-invariant probability measure on `𝕊⁰` charges both points. -/
theorem uniform_sph0_pos (σ : Measure (SSphere 1)) [IsProbabilityMeasure σ]
    (hσ : ∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1, σ.map (Perspective.sphereMap 1 U) = σ) (b : Bool) :
    σ {sph0 b} ≠ 0 := by
  have heq := Perspective.measure_singleton_eq_of_invariant 1 σ hσ (sph0 true) (sph0 false)
  have huniv : σ {sph0 true} + σ {sph0 false} = 1 := by
    rw [← measure_union (Set.disjoint_singleton.2 sph0_ne) (measurableSet_singleton _),
      ← measure_univ (μ := σ)]
    congr 1
    ext x
    simpa [or_comm] using sph0_cases x
  intro h0
  cases b
  · rw [heq, h0] at huniv; simp at huniv
  · rw [← heq, h0] at huniv; simp at huniv

/-- The rotation-invariant law on `𝕊⁰`, `(δ_{+1} + δ_{-1})/2`. -/
noncomputable def uniformSph0 : Measure (SSphere 1) :=
  (2⁻¹ : ENNReal) • (Measure.dirac (sph0 true) + Measure.dirac (sph0 false))

instance : IsProbabilityMeasure uniformSph0 := by
  constructor
  simp [uniformSph0, ENNReal.inv_two_add_inv_two]

/-- `uniformSph0` is invariant under every linear isometry of `ℝ¹`. -/
theorem uniformSph0_invariant (U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1) :
    uniformSph0.map (Perspective.sphereMap 1 U) = uniformSph0 := by
  have hm := Perspective.measurable_sphereMap 1 U
  set N := Perspective.sphereMap 1 (LinearIsometryEquiv.neg ℝ (E := EucSpace 1))
  have hneg : Perspective.sphereMap 1 U (sph0 false) = N (Perspective.sphereMap 1 U (sph0 true)) :=
    Subtype.ext (by simp [N, sph0, Perspective.sphereMap])
  have hN : N (sph0 true) = sph0 false := rfl
  have hN' : N (sph0 false) = sph0 true := Subtype.ext (by simp [N, sph0, Perspective.sphereMap])
  rw [uniformSph0, Measure.map_smul _ hm.aemeasurable, Measure.map_add _ _ hm, Measure.map_dirac,
    Measure.map_dirac, hneg]
  rcases sph0_cases (Perspective.sphereMap 1 U (sph0 true)) with h | h
  · rw [h, hN]
  · rw [h, hN', add_comm]

/-- The hypotheses of `uniform_sph0_pos` are satisfiable: `uniformSph0`. -/
example : ∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
    uniformSph0.map (Perspective.sphereMap 1 U) = uniformSph0 :=
  uniformSph0_invariant

/-- `uniformSph0` is the uniform law of `Perspective.UniformTuple`. -/
theorem uniformTuple_sph0 (n : ℕ) :
    Perspective.UniformTuple 1 n (Measure.pi fun _ : Idx n => uniformSph0) :=
  ⟨uniformSph0, inferInstance, uniformSph0_invariant, rfl⟩

/-- A property that fails whenever the first token is positive and the second
negative does not hold almost surely for a standard Gaussian pair on `ℝ¹`. -/
theorem not_ae_gaussian_pair {P : (Idx 2 → EucSpace 1) → Prop}
    (hP : ∀ X₀ : Idx 2 → EucSpace 1, 0 < X₀ 0 0 → X₀ 1 0 < 0 → ¬ P X₀) :
    ¬ ∀ᵐ X₀ ∂(Measure.pi fun _ : Idx 2 => stdGaussian (EucSpace 1)), P X₀ := by
  intro hae
  set B : Set (Idx 2 → EucSpace 1) :=
    Set.univ.pi fun i => if i = 0 then {x | 0 < x 0} else {x | x 0 < 0}
  have hB : (Measure.pi fun _ : Idx 2 => stdGaussian (EucSpace 1)) B ≠ 0 := by
    rw [Measure.pi_pi, Fin.prod_univ_two]
    simpa using mul_ne_zero stdGaussian_one_pos.1 stdGaussian_one_pos.2
  refine hB (measure_mono_null ?_ (ae_iff.1 hae))
  intro X₀ hX₀
  have a := hX₀ 0 (Set.mem_univ _)
  have b := hX₀ 1 (Set.mem_univ _)
  simp at a b
  exact hP X₀ a b

/-- The hypothesis of `not_ae_gaussian_pair` is satisfiable: `P` the complement
of the event. -/
example : ∀ X₀ : Idx 2 → EucSpace 1, 0 < X₀ 0 0 → X₀ 1 0 < 0 →
    ¬ (fun X : Idx 2 → EucSpace 1 => ¬ (0 < X 0 0 ∧ X 1 0 < 0)) X₀ :=
  fun _ h0 h1 h => h ⟨h0, h1⟩

end Normalization
end Transformer
