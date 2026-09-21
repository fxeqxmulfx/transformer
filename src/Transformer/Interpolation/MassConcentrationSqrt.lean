/-
# Measure-to-measure interpolation — mass concentration at rate `√η`

The corrected `lem: mass.concentration.Q1` of arXiv:2411.04551v3.  The paper
claims `W_2 ≤ Cη`, which is false (`not_massConcentrationQ1`); its own proof,
"`μ_0^i(A) = ν_0^i(A ∩ ℚ_1^{d-1}) + ν_0^i(𝕊^{d-1} ∖ ℚ_1^{d-1}) δ_{x_0^i}(A)`",
moves mass `η` a distance at most `2`, which costs `4η` and gives
`W_2 ≤ 2√η`.  That is enough for the continuity argument the lemma serves,
which only needs `W_2 → 0` as `η → 0`.

The point `x_0` has to exist: `diagPoint`, the unit vector `(1, …, 1)/√d`.

Source: arXiv:2411.04551v3, `lem: mass.concentration.Q1` and its proof.
-/

import Transformer.Interpolation.MassConcentration
import Transformer.Wasserstein.Collapse

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

/-- The point `(1, …, 1)/√d` of `𝕊^{d-1}`, for `d ≥ 1`. -/
noncomputable def diagPoint (d : ℕ) (hd : 0 < d) : SSphere d :=
  ⟨WithLp.toLp 2 (fun _ => 1 / √(d : ℝ)), by
    have hd' : (0 : ℝ) < d := Nat.cast_pos.mpr hd
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq]
    simp only [Real.norm_eq_abs, sq_abs, div_pow, one_pow,
      Real.sq_sqrt hd'.le, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [mul_one_div_cancel hd'.ne', Real.sqrt_one]⟩

/-- `(1, …, 1)/√d` lies in the positive quadrant, which is therefore
inhabited in every dimension `d ≥ 1`. -/
theorem diagPoint_mem_positiveQuadrant (d : ℕ) (hd : 0 < d) :
    diagPoint d hd ∈ positiveQuadrant d := by
  intro i
  have : (0 : ℝ) < √(d : ℝ) := Real.sqrt_pos.mpr (Nat.cast_pos.mpr hd)
  simpa [diagPoint] using this

/-- **Lemma (lem: mass.concentration.Q1), corrected.**

If `(φ_η)_# ν_0^i (ℚ_1^{d-1}) = 1 - η` for every `i`, then for every `i`
there is `μ_0^i ∈ 𝒫(ℚ_1^{d-1})` with

  `W_2((φ_η)_# ν_0^i, μ_0^i) ≤ 2√η`.

Deviations from the source, recorded:
* the rate is `2√η`, not `Cη`, which is false (`not_massConcentrationQ1`);
* `φ_η` is only assumed measurable, not Lipschitz and invertible: neither is
  used, so the statement is stronger than the source's hypotheses require;
* `η > 0` is not assumed: `η ≥ 0` follows from the mass condition.

`μ_0^i = ρ|_{ℚ_1} + ρ(ℚ_1ᶜ) δ_{x_0}` with `ρ = (φ_η)_# ν_0^i` — the source
writes `ν_0^i` in place of `ρ`, which does not satisfy the claim when
`φ_η ≠ id` — and `x_0 = diagPoint`.

Source: arXiv:2411.04551v3, `lem: mass.concentration.Q1`. -/
theorem massConcentrationQ1_sqrt (d N : ℕ) (ν : Idx N → Perspective.ProbSphere d) (η : ℝ)
    (φ : SSphere d → SSphere d) (hφ : Measurable φ)
    (hmass : ∀ i : Idx N,
      Measure.map φ (ν i : Measure (SSphere d)) (positiveQuadrant d) = ENNReal.ofReal (1 - η))
    (i : Idx N) : ∃ μ : Perspective.ProbSphere d,
      (μ : Measure (SSphere d)) (positiveQuadrant d)ᶜ = 0 ∧
      W2 d (Measure.map φ (ν i : Measure (SSphere d))) (μ : Measure (SSphere d)) ≤ 2 * √η := by
  set ρ := Measure.map φ (ν i : Measure (SSphere d)) with hρ
  have : IsProbabilityMeasure ρ := (Measure.isProbabilityMeasure_map_iff hφ.aemeasurable).2 inferInstance
  rcases Nat.eq_zero_or_pos d with rfl | hd
  · have h1 := measure_univ (μ := ρ)
    have : (Set.univ : Set (SSphere 0)) = ∅ := by
      ext x
      have hx : ‖(x : EucSpace 0)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
      simp [Subsingleton.elim (x : EucSpace 0) 0] at hx
    rw [this, measure_empty] at h1
    exact absurd h1 zero_ne_one
  set Q := positiveQuadrant d
  have hQ : MeasurableSet Q := measurableSet_positiveQuadrant d
  set x₀ := diagPoint d hd
  have hx₀ : x₀ ∈ Q := diagPoint_mem_positiveQuadrant d hd
  set μ₀ := ρ.restrict Q + ρ Qᶜ • Measure.dirac x₀ with hμ₀
  have hμprob : IsProbabilityMeasure μ₀ := by
    constructor
    rw [hμ₀, Measure.add_apply, Measure.smul_apply, smul_eq_mul, Measure.restrict_apply_univ,
      measure_univ (μ := Measure.dirac x₀), mul_one, measure_add_measure_compl hQ, measure_univ]
  refine ⟨⟨μ₀, hμprob⟩, ?_, ?_⟩
  · show μ₀ Qᶜ = 0
    rw [hμ₀, Measure.add_apply, Measure.smul_apply, Measure.restrict_apply hQ.compl,
      Set.compl_inter_self, measure_empty, Measure.dirac_apply' _ hQ.compl,
      Set.indicator_of_notMem (Set.notMem_compl_iff.2 hx₀)]
    simp
  · show W2 d ρ μ₀ ≤ 2 * √η
    have hD : ∀ x : SSphere d, dist x x₀ ≤ 2 := by
      intro x
      rw [Subtype.dist_eq]
      refine (dist_le_norm_add_norm _ _).trans ?_
      rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp x₀.2]
      norm_num
    have hsq := Wasserstein.W2_sq_le_collapse ρ hQ x₀ hD
    have hQr : ρ.real Q ≥ 1 - η := by
      rw [measureReal_def, hmass i]
      rw [ENNReal.toReal_ofReal']
      exact le_max_left _ _
    have hc : ρ.real Qᶜ ≤ η := by
      rw [measureReal_compl hQ, probReal_univ]; linarith
    have hη : 0 ≤ η := measureReal_nonneg.trans hc
    have hW := W2_nonneg d ρ μ₀
    have h4 : W2 d ρ μ₀ ^ 2 ≤ (2 * √η) ^ 2 := by
      rw [mul_pow, Real.sq_sqrt hη]
      have : ρ.real Qᶜ * 2 ^ 2 ≤ η * 2 ^ 2 := by nlinarith
      linarith [hsq]
    exact (pow_le_pow_iff_left₀ hW (by positivity) two_ne_zero).mp h4

/-- The hypotheses of `massConcentrationQ1_sqrt` are satisfiable: a Dirac mass
at `+1 ∈ ℚ_1^0`, `φ = id`, `η = 0`. -/
example : Measurable (id : SSphere 1 → SSphere 1) ∧
    ∀ i : Idx 1, Measure.map id
      ((fun _ => Perspective.diracProb 1 (basePoint 0)) i : Measure (SSphere 1))
      (positiveQuadrant 1) = ENNReal.ofReal (1 - 0) := by
  refine ⟨measurable_id, fun _ => ?_⟩
  rw [Measure.map_id]
  show Measure.dirac (basePoint 0 : SSphere 1) (positiveQuadrant 1) = _
  rw [Measure.dirac_apply' _ (measurableSet_positiveQuadrant 1),
    Set.indicator_of_mem basePoint_mem_positiveQuadrant]
  simp

end Interpolation
end Transformer
