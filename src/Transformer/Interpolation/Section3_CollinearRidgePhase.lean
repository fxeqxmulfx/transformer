/-
# Actual ridge phases split collinear barycenter directions

ArXiv:2411.04551v3, §3.2, `lem: perturbation`, and its use in
`prop: separation`. Distinct probabilities with collinear positive means
admit one actual ridge whose small-time laws have distinct directions.
Both equal and unequal means are included. The original weak equation
supplies the result for every weak solution of the selected parameters.

For a ratio in (0,1], equal means use the distinct-measure ridge test;
unequal means use the original scaled-mass test. The same uniform
first-variation determinant controls the actual short-time conclusion.
For any positive ratio, swapping the two laws puts the ratio in (0,1].
Normalized-direction inequality is symmetric, so no solution hypothesis
or geometric assumption is added by that swap.

The source's additional fixed-outside-hulls assertion in the equal-mean
case is false and is not used by these separation statements.
-/

import Transformer.Interpolation.Section3_EqualMeanVariation
import Transformer.Interpolation.Section3_BarycenterDirections

open Real MeasureTheory Set

namespace Transformer.Interpolation

/-- All distinct original quadrant laws at a ratio in (0,1] admit a
single actual ridge with uniform short-time non-collinearity. Source:
arXiv:2411.04551v3, §3.2, the variation in `lem: perturbation`.
The equal-mean case asserts only separation, without the refuted
fixed-outside-hulls clause. -/
theorem exists_collinear_ridge_separation {d : ℕ}
    (μ₀ ν₀ : Perspective.ProbSphere d) (γ : ℝ)
    (hγ₀ : 0 < γ) (hγ₁ : γ ≤ 1) (hne : μ₀ ≠ ν₀)
    (hμ₀ : (μ₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hν₀ : (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hbary : barycenter d μ₀ = γ • barycenter d ν₀) :
    ∃ (i : Fin d) (p u : EucSpace d) (b δ : ℝ), ‖p‖ = 1 ∧ 0 < δ ∧
      ∀ t : ℝ, 0 < t → t < δ →
        ∀ μ ν : ℝ → Perspective.ProbSphere d,
          μ 0 = μ₀ → cauchyPB d (fun _ => reluRidgeParams i p u b) μ →
          ν 0 = ν₀ → cauchyPB d (fun _ => reluRidgeParams i p u b) ν →
          ∀ q : ℝ, barycenter d (μ t) ≠ q • barycenter d (ν t) := by
  by_cases hγeq : γ = 1
  · subst γ
    exact exists_equal_mean_ridge_separation μ₀ ν₀ hne hμ₀ hν₀
      (by simpa only [one_smul] using hbary)
  ·
    have hd := two_le_dim_of_distinct_positiveQuadrant μ₀ ν₀ hμ₀ hν₀ hne
    let i : Fin d := ⟨0, by omega⟩
    obtain ⟨p, hp, hpm, hμhem, hνhem⟩ :=
      exists_transverse_hemisphere_of_positiveQuadrants hd μ₀ ν₀ hμ₀ hν₀
    obtain ⟨u, b, hvariation⟩ := exists_transverse_relu_variation
      μ₀ ν₀ γ hγ₀.le hγeq p (barycenter d ν₀) hp hpm hμhem hνhem
    let θ : TimeParams d := fun _ => reluRidgeParams i p u b
    let F (x : EucSpace d) := max (b - inner ℝ x u) 0 • proj d x p
    let B := 4 * (|b| + ‖u‖) * ‖p‖
    have hF : Continuous F := by
      unfold F proj
      fun_prop
    have hB : 0 ≤ B := by
      dsimp only [B]
      positivity
    have hfield (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d) :
        fullVF d θ ρ s x = F x := fullVF_reluRidgeParams i p u b ρ s x
    have hbound (x : SSphere d) : ‖F x‖ ≤ B := by
      rw [← hfield ν₀ 0 x]
      exact norm_fullVF_reluRidgeParams_le i p u b ν₀ 0 x
    have hm : barycenter d ν₀ ≠ 0 := by
      intro he
      have hh := barycenter_coordinate_pos_of_positiveQuadrant ν₀ hν₀ i
      rw [he] at hh
      exact (lt_irrefl 0) hh
    have hflux : (∫ x : SSphere d, inner ℝ p (F x) ∂(μ₀ : Measure (SSphere d))) ≠
        γ * ∫ x : SSphere d, inner ℝ p (F x) ∂(ν₀ : Measure (SSphere d)) := by
      have he (x : EucSpace d) : inner ℝ p (F x) =
          (1 - inner ℝ x p ^ 2) * max (b - inner ℝ x u) 0 := by
        simpa only [fullVF_reluRidgeParams] using
          inner_fullVF_reluRidgeParams i p u b hp ν₀ 0 x
      simp_rw [he]
      exact hvariation
    obtain ⟨δ, hδ, hsep⟩ := exists_uniform_noncolinear_time_of_flux_variation
      θ F hF B hB hfield hbound μ₀ ν₀ γ (by rwa [abs_of_pos hγ₀])
      hbary p hp hpm hm hflux
    exact ⟨i, p, u, b, δ, hp, hδ, hsep⟩

/-- A positive scalar ratio of any size admits actual short-time
separation of the original normalized barycenters. Source:
arXiv:2411.04551v3, §3.2, the class-splitting step in `prop: separation`.
The selected ridge and one positive time radius precede every time
chosen in that radius. -/
theorem exists_collinear_ridge_direction_separation {d : ℕ}
    (μ₀ ν₀ : Perspective.ProbSphere d) (hne : μ₀ ≠ ν₀)
    (hμ₀ : (μ₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hν₀ : (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hbary : ∃ γ : ℝ, barycenter d μ₀ = γ • barycenter d ν₀) :
    ∃ (p u : EucSpace d) (b δ : ℝ), 0 < δ ∧
      ∀ t : ℝ, 0 < t → t < δ →
        barycenterDirection (ridgeLaw p u b μ₀ t) ≠
          barycenterDirection (ridgeLaw p u b ν₀ t) := by
  obtain ⟨γ, hγ⟩ := hbary
  have hd := two_le_dim_of_distinct_positiveQuadrant μ₀ ν₀ hμ₀ hν₀ hne
  let i₀ : Fin d := ⟨0, by omega⟩
  have hμi := barycenter_coordinate_pos_of_positiveQuadrant μ₀ hμ₀ i₀
  have hνi := barycenter_coordinate_pos_of_positiveQuadrant ν₀ hν₀ i₀
  have hh :=
    congrArg (fun x : EucSpace d => x i₀) hγ
  change barycenter d μ₀ i₀ = γ * barycenter d ν₀ i₀ at hh
  have hγpos : 0 < γ := by nlinarith
  by_cases hγle : γ ≤ 1
  · obtain ⟨i, p, u, b, δ, -, hδ, hsep⟩ :=
      exists_collinear_ridge_separation μ₀ ν₀ γ hγpos hγle hne hμ₀ hν₀ hγ
    refine ⟨p, u, b, δ, hδ, fun t ht htd he => ?_⟩
    have hn := hsep t ht htd (ridgeLaw p u b μ₀) (ridgeLaw p u b ν₀)
      (ridgeLaw_zero _ _ _ _) (cauchyPB_ridgeLaw i p u b μ₀)
      (ridgeLaw_zero _ _ _ _) (cauchyPB_ridgeLaw i p u b ν₀)
    obtain ⟨q, hq⟩ := barycenter_colinear_of_direction_eq _ _ he
    exact hn q hq
  · have hγinv : 0 < γ⁻¹ := inv_pos.mpr hγpos
    have hγinvle : γ⁻¹ ≤ 1 := by
      rw [inv_le_one₀ hγpos]
      exact (lt_of_not_ge hγle).le
    have hbar : barycenter d ν₀ = γ⁻¹ • barycenter d μ₀ := by
      rw [hγ, smul_smul, inv_mul_cancel₀ hγpos.ne', one_smul]
    obtain ⟨i, p, u, b, δ, -, hδ, hsep⟩ :=
      exists_collinear_ridge_separation ν₀ μ₀ γ⁻¹ hγinv hγinvle hne.symm hν₀ hμ₀ hbar
    refine ⟨p, u, b, δ, hδ, fun t ht htd he => ?_⟩
    have hn := hsep t ht htd (ridgeLaw p u b ν₀) (ridgeLaw p u b μ₀)
      (ridgeLaw_zero _ _ _ _) (cauchyPB_ridgeLaw i p u b ν₀)
      (ridgeLaw_zero _ _ _ _) (cauchyPB_ridgeLaw i p u b μ₀)
    obtain ⟨q, hq⟩ := barycenter_colinear_of_direction_eq _ _ he.symm
    exact hn q hq

/-- The genuine distinct equal-mean atomic laws already refuting the
fixed-outside clause realize the original support hypotheses and the
positive ratio one. Thus the enlarged separation statement includes
actual equal-mean data, rather than assuming unequal barycenters. -/
example : ∃ (μ ν : Perspective.ProbSphere 3) (γ : ℝ),
    0 < γ ∧ γ ≤ 1 ∧ μ ≠ ν ∧
    (μ : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 ∧
    (ν : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 ∧
    barycenter 3 μ = γ • barycenter 3 ν ∧
    ∃ q : ℝ, barycenter 3 μ = q • barycenter 3 ν := by
  obtain ⟨μ, ν, hne, hμ, hν, hbary, -⟩ := perturbation_equal_barycenter_counterexample
  refine ⟨μ, ν, 1, zero_lt_one, le_rfl, hne, hμ, hν, ?_, 1, ?_⟩
  · simpa only [one_smul] using hbary
  · simpa only [one_smul] using hbary

end Transformer.Interpolation
