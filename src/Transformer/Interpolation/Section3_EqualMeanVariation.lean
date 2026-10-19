module

public import Transformer.Interpolation.Section3_UniformNoncolinearity
public import Transformer.Interpolation.Section3_PositiveBarycenters
public import Transformer.Interpolation.Section3_RidgeWellPosed
public import Transformer.Interpolation.Section3_PerturbationFalse

/-!
# Separating distinct laws with equal original barycenters

ArXiv:2411.04551v3, §3.2, `lem: perturbation`, equal-mean case,
and the use of that case in `prop: separation`.
The source additionally requires the terminal flow to fix the complement
of the two geodesic hulls. That requirement is false for atomic data,
as `Section3_PerturbationFalse` proves. It is not asserted here.

The separation itself holds with the original distinctness and quadrant
hypotheses. A transverse spherical projection has positive weight on
both supports. Weighted ReLU tests distinguish the two actual measures,
even though their first moments agree. The resulting actual ridge has a
nonzero transverse first variation. The uniform determinant estimate
separates every original weak solution pair at all sufficiently small
positive times. Global weak existence and uniqueness are already proved
for these same actual single-neuron parameters.
-/

@[expose] public section

open Real MeasureTheory Filter Set

namespace Transformer.Interpolation

/-- Distinct original probabilities have a distinguishing transverse
ridge, including when their barycenters are equal. Source:
arXiv:2411.04551v3, §3.2, the variation used in `lem: perturbation`.
No fixed-outside-hulls conclusion is required or assumed. -/
theorem exists_distinct_transverse_relu_variation {d : ℕ}
    (μ ν : Perspective.ProbSphere d) (hne : μ ≠ ν)
    (p m : EucSpace d)
    (hp : ‖p‖ = 1) (hpm : inner ℝ p m = 0)
    (hμ : ∀ᵐ (x : SSphere d) ∂(μ : Measure (SSphere d)),
      0 < inner ℝ (x : EucSpace d) m)
    (hν : ∀ᵐ (x : SSphere d) ∂(ν : Measure (SSphere d)),
      0 < inner ℝ (x : EucSpace d) m) :
    ∃ (u : EucSpace d) (b : ℝ),
      (∫ x, (1 - inner ℝ (x : EucSpace d) p ^ 2) *
        max (b - inner ℝ (x : EucSpace d) u) 0 ∂(μ : Measure (SSphere d))) ≠
      ∫ x, (1 - inner ℝ (x : EucSpace d) p ^ 2) *
        max (b - inner ℝ (x : EucSpace d) u) 0 ∂(ν : Measure (SSphere d)) := by
  apply exists_positive_weighted_sphere_relu_ridge
    (μ : Measure (SSphere d)) (ν : Measure (SSphere d))
    (fun x => 1 - inner ℝ (x : EucSpace d) p ^ 2) (by fun_prop)
    (hμ.mono fun x hx => projection_weight_pos x p m hp hpm hx)
    (hν.mono fun x hx => projection_weight_pos x p m hp hpm hx)
  exact fun he => hne (Subtype.ext he)

/-- A single actual ridge separates distinct equal-mean quadrant laws
at every sufficiently small positive time, uniformly over every original
weak solution pair. Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`.
Only barycenter separation is concluded; the source's false requirement
of fixing points outside the hulls is omitted and remains refuted. -/
theorem exists_equal_mean_ridge_separation {d : ℕ}
    (μ₀ ν₀ : Perspective.ProbSphere d) (hne : μ₀ ≠ ν₀)
    (hμ₀ : (μ₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hν₀ : (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hbary : barycenter d μ₀ = barycenter d ν₀) :
    ∃ (i : Fin d) (p u : EucSpace d) (b δ : ℝ), ‖p‖ = 1 ∧ 0 < δ ∧
      ∀ t : ℝ, 0 < t → t < δ →
        ∀ μ ν : ℝ → Perspective.ProbSphere d,
          μ 0 = μ₀ → cauchyPB d (fun _ => reluRidgeParams i p u b) μ →
          ν 0 = ν₀ → cauchyPB d (fun _ => reluRidgeParams i p u b) ν →
          ∀ q : ℝ, barycenter d (μ t) ≠ q • barycenter d (ν t) := by
  have hd := two_le_dim_of_distinct_positiveQuadrant μ₀ ν₀ hμ₀ hν₀ hne
  let i : Fin d := ⟨0, by omega⟩
  obtain ⟨p, hp, hpm, hμhem, hνhem⟩ :=
    exists_transverse_hemisphere_of_positiveQuadrants hd μ₀ ν₀ hμ₀ hν₀
  obtain ⟨u, b, hvariation⟩ := exists_distinct_transverse_relu_variation
    μ₀ ν₀ hne p (barycenter d ν₀) hp hpm hμhem hνhem
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
      1 * ∫ x : SSphere d, inner ℝ p (F x) ∂(ν₀ : Measure (SSphere d)) := by
    have he (x : EucSpace d) : inner ℝ p (F x) =
        (1 - inner ℝ x p ^ 2) * max (b - inner ℝ x u) 0 := by
      simpa only [fullVF_reluRidgeParams] using
        inner_fullVF_reluRidgeParams i p u b hp ν₀ 0 x
    simp_rw [he, one_mul]
    exact hvariation
  obtain ⟨δ, hδ, hsep⟩ := exists_uniform_noncolinear_time_of_flux_variation
    θ F hF B hB hfield hbound μ₀ ν₀ 1 (by norm_num)
    (by simpa only [one_smul] using hbary) p hp hpm hm hflux
  exact ⟨i, p, u, b, δ, hp, hδ, hsep⟩

/-- The separation can occur before any prescribed positive time,
using actual globally well-posed ridge controls. Source:
arXiv:2411.04551v3, §3.2, the small perturbation step in `prop: separation`.
The time is chosen before all original solution pairs. -/
theorem exists_equal_mean_ridge_time {d : ℕ}
    (μ₀ ν₀ : Perspective.ProbSphere d) (hne : μ₀ ≠ ν₀)
    (hμ₀ : (μ₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hν₀ : (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hbary : barycenter d μ₀ = barycenter d ν₀) (T : ℝ) (hT : 0 < T) :
    ∃ (i : Fin d) (p u : EucSpace d) (b t : ℝ), 0 < t ∧ t < T ∧
      (∀ ρ : Perspective.ProbSphere d, ∃! μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d (fun _ => reluRidgeParams i p u b) μ) ∧
      ∀ μ ν : ℝ → Perspective.ProbSphere d,
        μ 0 = μ₀ → cauchyPB d (fun _ => reluRidgeParams i p u b) μ →
        ν 0 = ν₀ → cauchyPB d (fun _ => reluRidgeParams i p u b) ν →
        ∀ q : ℝ, barycenter d (μ t) ≠ q • barycenter d (ν t) := by
  obtain ⟨i, p, u, b, δ, -, hδ, hsep⟩ :=
    exists_equal_mean_ridge_separation μ₀ ν₀ hne hμ₀ hν₀ hbary
  let t := min T δ / 2
  have ht : 0 < t := half_pos (lt_min hT hδ)
  have htδ : t < δ := (half_lt_self (lt_min hT hδ)).trans_le (min_le_right _ _)
  have htT : t < T := (half_lt_self (lt_min hT hδ)).trans_le (min_le_left _ _)
  exact ⟨i, p, u, b, t, ht, htT,
    exists_unique_cauchyPB_ridge i p u b, hsep t ht htδ⟩

/-- The equal-mean atomic counterexample to the fixed-outside-hulls
clause supplies genuine distinct positive-quadrant data for these
separation results. Its transverse direction and common hemisphere are
constructed from the original barycenter, with a positive time bound. -/
example : ∃ (μ ν : Perspective.ProbSphere 3) (p m : EucSpace 3),
    μ ≠ ν ∧ (μ : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 ∧
    (ν : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 ∧
    barycenter 3 μ = barycenter 3 ν ∧ ‖p‖ = 1 ∧ inner ℝ p m = 0 ∧
    (∀ᵐ x : SSphere 3 ∂(μ : Measure (SSphere 3)), 0 < inner ℝ (x : EucSpace 3) m) ∧
    (∀ᵐ x : SSphere 3 ∂(ν : Measure (SSphere 3)), 0 < inner ℝ (x : EucSpace 3) m) ∧
    (0 : ℝ) < 1 := by
  obtain ⟨μ, ν, hne, hμ, hν, hbary, -⟩ := perturbation_equal_barycenter_counterexample
  have hd := two_le_dim_of_distinct_positiveQuadrant μ ν hμ hν hne
  obtain ⟨p, hp, hpm, hμhem, hνhem⟩ :=
    exists_transverse_hemisphere_of_positiveQuadrants hd μ ν hμ hν
  exact ⟨μ, ν, p, barycenter 3 ν, hne, hμ, hν, hbary,
    hp, hpm, hμhem, hνhem, zero_lt_one⟩

end Transformer.Interpolation
