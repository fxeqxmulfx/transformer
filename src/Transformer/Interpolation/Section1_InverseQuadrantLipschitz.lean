/-
# Quantitative error under the original inverse quadrant phases

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, §3.1,
`lem: first.quadrant`, and §5, inversion after matching.
The original constant-gate neuron is exactly the spherical projection
flow, by identification of their actual Dirac probability laws.
Its proved Lipschitz dependence on the initial point therefore gives
one finite constant for each fixed inverse projection composition.

The constant is chosen before the target family and concentration
radius. It controls the support of every original input probability,
so an accuracy proportional to the desired final accuracy suffices.
The logarithm of that intermediate accuracy differs from the final
one by a fixed additive constant. These estimates retain the actual
inverse phase list and the genuine Wasserstein distance.
In particular, the inverse phase clocks and their control cost remain
fixed while the desired approximation radius tends to zero.
-/

import Transformer.Interpolation.Section1_InverseQuadrantSupports

open Real MeasureTheory Set
open scoped NNReal

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- The actual constant-gate original neuron is the actual projection
particle flow. Source: arXiv:2411.04551v3, §3.1, the two original
flows used for quadrant preprocessing. -/
theorem ridgeFlow_constant_projection (i : Fin d) (w x : SSphere d) (t : ℝ) :
    ridgeFlow (w : EucSpace d) 0 1 x t = sphereProjectionFlow w t x := by
  have he := ridgeLaw_constant_projection i w 1 (Perspective.diracProb d x) t
  rw [one_smul, one_mul, ridgeLaw_diracProb, projectionLaw_diracProb] at he
  have hm := congrArg (fun ν : Perspective.ProbSphere d => (ν : Measure (SSphere d))) he
  exact dirac_eq_dirac_iff.mp hm

/-- Every original projection map has a finite Lipschitz constant,
including at negative clocks. Source auxiliary: arXiv:2411.04551v3,
§3.1 and §5, quantitative error under inversion. -/
theorem exists_lipschitzWith_sphereProjectionFlow (i : Fin d) (w : SSphere d) (t : ℝ) :
    ∃ K : ℝ≥0, LipschitzWith K (sphereProjectionFlow w t) := by
  obtain ⟨K, hK⟩ := exists_lipschitzWith_ridgeFlow (w : EucSpace d) 0 1 t
  refine ⟨K, ?_⟩
  simpa only [ridgeFlow_constant_projection i] using hK

/-- A fixed original inverse quadrant point map has a Lipschitz
constant independent of the targets and input probabilities. Source
auxiliary: arXiv:2411.04551v3, §3.1 and §5, final inverse phases. -/
theorem exists_lipschitzWith_inverse_twoProjection (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) : ∃ K : ℝ≥0,
      LipschitzWith K (fun x => sphereProjectionFlow w₁ (-τ₁) (sphereProjectionFlow w₂ (-τ₂) x)) := by
  obtain ⟨K₁, h₁⟩ := exists_lipschitzWith_sphereProjectionFlow i w₁ (-τ₁)
  obtain ⟨K₂, h₂⟩ := exists_lipschitzWith_sphereProjectionFlow i w₂ (-τ₂)
  exact ⟨K₁ * K₂, h₁.comp h₂⟩

/-- One positive constant controls support error for every original
probability under the fixed inverse quadrant phases. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, stability of the final inversion. -/
theorem exists_inverse_twoProjection_radius_scale (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) : ∃ C : ℝ, 0 < C ∧ ∀ (y : SSphere d) (δ : ℝ), 0 ≤ δ →
      ∀ ν : Perspective.ProbSphere d,
        (ν : Measure (SSphere d)).support ⊆
          Metric.closedBall (sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ y)) δ →
        (phaseSequenceLaw (inversePhaseList (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂)) ν :
          Measure (SSphere d)).support ⊆ Metric.closedBall y (C * δ) := by
  let F (x : SSphere d) := sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x)
  let G (x : SSphere d) := sphereProjectionFlow w₁ (-τ₁) (sphereProjectionFlow w₂ (-τ₂) x)
  obtain ⟨K, hK⟩ := exists_lipschitzWith_inverse_twoProjection i w₁ w₂ τ₁ τ₂
  refine ⟨(K : ℝ) + 1, by positivity, ?_⟩
  intro y δ hδ ν hν z hz
  rw [coe_inverse_twoProjection_phaseLaw] at hz
  obtain ⟨x, hx, rfl⟩ := support_map_subset_image _ _ hK.continuous hz
  have hcancel : G (F y) = y := inverse_twoProjection_point w₁ w₂ y τ₁ τ₂
  have hb := hK.dist_le_mul x (F y)
  change dist (G x) (G (F y)) ≤ (K : ℝ) * dist x (F y) at hb
  rw [hcancel] at hb
  have hr := mul_le_mul_of_nonneg_left (hν hx) K.2
  exact (hb.trans hr).trans
    (mul_le_mul_of_nonneg_right (by linarith : (K : ℝ) ≤ (K : ℝ) + 1) hδ)

/-- The radius for inverse quadrant phases can be chosen linearly in
the requested accuracy. A fixed factor also bounds its logarithmic
cost uniformly over all sufficiently small accuracies. Source:
arXiv:2411.04551v3, §1 and §5, the logarithmic final estimate. -/
theorem exists_linear_inverse_twoProjection_radius (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) : ∃ A : ℝ, 1 ≤ A ∧ ∀ ε : ℝ, 0 < ε → ε ≤ 1 →
      0 < ε / A ∧ ε / A ≤ 1 ∧
      log (1 / (ε / A)) ≤ log A + log (1 / ε) ∧
      ∀ (y : SSphere d) (ν : Perspective.ProbSphere d),
        (ν : Measure (SSphere d)).support ⊆
          Metric.closedBall (sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ y)) (ε / A) →
        (phaseSequenceLaw (inversePhaseList (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂)) ν :
          Measure (SSphere d)).support ⊆ Metric.closedBall y ε := by
  obtain ⟨C, hC, hbound⟩ := exists_inverse_twoProjection_radius_scale i w₁ w₂ τ₁ τ₂
  let A := max C 1
  have hA1 : 1 ≤ A := le_max_right _ _
  have hA : 0 < A := zero_lt_one.trans_le hA1
  refine ⟨A, hA1, ?_⟩
  intro ε hε hε1
  have hδ : 0 < ε / A := div_pos hε hA
  refine ⟨hδ, (div_le_iff₀ hA).mpr (by nlinarith), ?_, ?_⟩
  · have he : 1 / (ε / A) = A * (1 / ε) := by field_simp
    rw [he, log_mul hA.ne' (div_pos one_pos hε).ne']
  · intro y ν hν z hz
    have hb := hbound y (ε / A) hδ.le ν hν hz
    have hscale : C * (ε / A) ≤ ε := by
      calc
        C * (ε / A) ≤ A * (ε / A) :=
          mul_le_mul_of_nonneg_right (le_max_left _ _) hδ.le
        _ = ε := by field_simp
    exact hb.trans hscale

/-- The same linear input tolerance bounds the genuine original
Wasserstein error after the inverse phases, uniformly over a finite
target family. Source auxiliary: arXiv:2411.04551v3, §1 and §5. -/
theorem exists_linear_inverse_twoProjection_W2 (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) : ∃ A : ℝ, 1 ≤ A ∧ ∀ (Y : Idx N → SSphere d) (ε : ℝ),
      0 < ε → ε ≤ 1 → ∀ k, ∀ ν : Perspective.ProbSphere d,
        (ν : Measure (SSphere d)).support ⊆
          Metric.closedBall (sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ (Y k))) (ε / A) →
        W2 d (phaseSequenceLaw (inversePhaseList (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂)) ν :
          Measure (SSphere d)) (Measure.dirac (Y k)) ≤ ε := by
  obtain ⟨A, hA, hbound⟩ := exists_linear_inverse_twoProjection_radius i w₁ w₂ τ₁ τ₂
  refine ⟨A, hA, ?_⟩
  intro Y ε hε hε1 k ν hν
  exact W2_le_radius_of_support _ (Y k) ε hε.le ((hbound ε hε hε1).2.2.2 (Y k) ν hν)

/-- Two distinct actual coordinate axes, nontrivial clocks, antipodal
targets and a positive small tolerance satisfy every inverse-error
hypothesis without restricting targets to the quadrant. -/
example : ∃ (w₁ w₂ : SSphere 3) (Y : Idx 2 → SSphere 3) (τ₁ τ₂ ε : ℝ),
    w₁ ≠ w₂ ∧ τ₁ ≠ τ₂ ∧ 0 < ε ∧ ε ≤ 1 ∧ Function.Injective Y := by
  let p := basePoint 2
  let Y : Idx 2 → SSphere 3 := ![p, antipode 3 p]
  refine ⟨p, ⟨PiLp.single 2 (1 : Fin 3) (1 : ℝ), by simp⟩, Y, 1, 2, 1 / 2,
    ?_, by norm_num, by norm_num, by norm_num, ?_⟩
  · intro he
    have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (0 : Fin 3)) he
    norm_num [p, basePoint] at hh
  · intro k l he
    fin_cases k <;> fin_cases l
    · rfl
    · exact False.elim (antipode_ne 3 p he.symm)
    · exact False.elim (antipode_ne 3 p he)
    · rfl

end Transformer.Interpolation
