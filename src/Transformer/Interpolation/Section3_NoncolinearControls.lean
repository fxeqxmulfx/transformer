module

public import Transformer.Interpolation.Section3_UniformNoncolinearity
public import Transformer.Interpolation.Section3_PositiveBarycenters
public import Transformer.Interpolation.Section1_TimeRescaling

/-!
# Constant controls separating the original collinear barycenters

ArXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2.

The original positive-quadrant hypotheses give a common open hemisphere
and an actual unit transverse direction. Weighted ReLU ridge separation
selects one neuron with different scaled transverse barycenter fluxes.
The uniform first-variation theorem separates every original weak
solution pair at one common small positive time.

Exact time rescaling places this separation at the prescribed horizon.
The pushed ridge flows supply actual global weak solutions. Inverse
rescaling applies to every weak solution of the final parameters.
The construction uses one constant pure-perceptron control, which is
stronger than the source's allowance of at most two switches.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

open Perspective

/-- One constant control fits the source's allowance of at most two
switches, using three consecutive intervals of equal length.
Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem piecewiseConstant_const_three {d : ℕ} (θ₀ : Params d) (T : ℝ) (hT : 0 ≤ T) :
    PiecewiseConstant d (fun _ => θ₀) T 3 := by
  refine ⟨fun i : Fin 4 => (i.val : ℝ) * T / 3, ?_, by norm_num, by norm_num,
    fun _ _ _ _ _ => rfl⟩
  intro i j hij
  exact div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hij) hT) (by norm_num)

/-- The original Part 2 hypotheses admit one constant actual neuron
control, with zero attention, actual global weak solutions, and final
non-collinearity for every solution pair and every real scale.
Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2.
The result strengthens the permitted switch count without changing
any of the original support, mean-ratio, or distinctness hypotheses. -/
theorem exists_constant_noncolinear_controls {d : ℕ} (T γ : ℝ)
    (μ₀ ν₀ : Perspective.ProbSphere d) (hT : 0 < T) (hne : μ₀ ≠ ν₀)
    (hμ₀ : (μ₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hν₀ : (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hγ₀ : 0 < γ) (hγ₁ : γ ≤ 1) (hbary : barycenter d μ₀ = γ • barycenter d ν₀)
    (hγne : γ ≠ 1) :
    ∃ θ₀ : Params d, θ₀.V = 0 ∧ θ₀.B = 0 ∧
      (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ ∧ cauchyPB d (fun _ => θ₀) μ) ∧
      (∃ ν : ℝ → Perspective.ProbSphere d, ν 0 = ν₀ ∧ cauchyPB d (fun _ => θ₀) ν) ∧
      ∀ μ ν : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ → cauchyPB d (fun _ => θ₀) μ →
        ν 0 = ν₀ → cauchyPB d (fun _ => θ₀) ν →
        ∀ q : ℝ, barycenter d (μ T) ≠ q • barycenter d (ν T) := by
  have hd := two_le_dim_of_distinct_positiveQuadrant μ₀ ν₀ hμ₀ hν₀ hne
  let i : Fin d := ⟨0, by omega⟩
  obtain ⟨p, hp, hpm, hμhem, hνhem⟩ :=
    exists_transverse_hemisphere_of_positiveQuadrants hd μ₀ ν₀ hμ₀ hν₀
  obtain ⟨u, b, hvariation⟩ := exists_transverse_relu_variation μ₀ ν₀ γ hγ₀.le hγne p
    (barycenter d ν₀) hp hpm hμhem hνhem
  let θ : TimeParams d := fun _ => reluRidgeParams i p u b
  let F (x : EucSpace d) := max (b - inner ℝ x u) 0 • proj d x p
  let B := 4 * (|b| + ‖u‖) * ‖p‖
  have hF : Continuous F := by unfold F proj; fun_prop
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have hfield (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d) : fullVF d θ ρ s x = F x :=
    fullVF_reluRidgeParams i p u b ρ s x
  have hbound (x : SSphere d) : ‖F x‖ ≤ B := by
    rw [← hfield ν₀ 0 x]
    exact norm_fullVF_reluRidgeParams_le i p u b ν₀ 0 x
  have hm : barycenter d ν₀ ≠ 0 := by
    intro he
    have h := barycenter_coordinate_pos_of_positiveQuadrant ν₀ hν₀ i
    rw [he] at h
    exact (lt_irrefl 0) h
  have hflux : (∫ x : SSphere d, inner ℝ p (F x) ∂(μ₀ : Measure (SSphere d))) ≠
      γ * ∫ x : SSphere d, inner ℝ p (F x) ∂(ν₀ : Measure (SSphere d)) := by
    have he (x : EucSpace d) : inner ℝ p (F x) =
        (1 - inner ℝ x p ^ 2) * max (b - inner ℝ x u) 0 := by
      simpa only [fullVF_reluRidgeParams] using inner_fullVF_reluRidgeParams i p u b hp ν₀ 0 x
    simp_rw [he]
    exact hvariation
  obtain ⟨δ, hδ, hsep⟩ := exists_uniform_noncolinear_time_of_flux_variation θ F hF B hB hfield hbound
    μ₀ ν₀ γ (by rwa [abs_of_pos hγ₀]) hbary p hp hpm hm hflux
  let τ := δ / 2
  have hτ : 0 < τ := half_pos hδ
  have hτδ : τ < δ := half_lt_self hδ
  let κ := τ / T
  have hκ : κ ≠ 0 := (div_pos hτ hT).ne'
  let θ₀ := rescaleTimeParams κ θ 0
  have hθ : rescaleTimeParams κ θ = fun _ => θ₀ := rfl
  have hsol (ρ : Perspective.ProbSphere d) :
      ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ρ ∧ cauchyPB d (fun _ => θ₀) μ := by
    refine ⟨fun s => ridgeLaw p u b ρ (κ * s), ?_, ?_⟩
    · simp only [mul_zero, ridgeLaw_zero]
    · rw [← hθ]
      exact cauchyPB_rescaleTimeParams κ hκ θ _ (cauchyPB_ridgeLaw i p u b ρ)
  refine ⟨θ₀, rescaleTimeParams_V_zero κ θ (fun _ => rfl) 0,
    rescaleTimeParams_B_zero κ θ (fun _ => rfl) 0, hsol μ₀, hsol ν₀, ?_⟩
  intro μ ν hμ₀ hμ hν₀ hν q
  have hμ' : cauchyPB d θ (fun s => μ (κ⁻¹ * s)) :=
    cauchyPB_unscaleTimeParams κ hκ θ μ (by rw [hθ]; exact hμ)
  have hν' : cauchyPB d θ (fun s => ν (κ⁻¹ * s)) :=
    cauchyPB_unscaleTimeParams κ hκ θ ν (by rw [hθ]; exact hν)
  have hκτ : κ⁻¹ * τ = T := by dsimp [κ]; field_simp
  have h := hsep τ hτ hτδ (fun s => μ (κ⁻¹ * s)) (fun s => ν (κ⁻¹ * s))
    (by simpa only [mul_zero] using hμ₀) hμ' (by simpa only [mul_zero] using hν₀) hν' q
  simpa only [hκτ] using h

/-- The hypotheses of `exists_constant_noncolinear_controls` are satisfiable, with two different
measures: on `𝕊^1`, `μ_0 = ½δ_{(5,12)/13} + ½δ_{(12,5)/13}` and
`ν_0 = ½δ_{(3,4)/5} + ½δ_{(4,3)/5}`, supported in `ℚ_1^1`, whose barycenters
`(17/26)(1, 1)` and `(7/10)(1, 1)` are collinear with `γ_1 = 85/91 ∈ (0, 1)`.
In `d = 1` no witness exists, since `𝒫(ℚ_1^0) = {δ_{+1}}`. -/
example : ∃ (μ₀ ν₀ : ProbSphere 2) (γ₁ : ℝ),
    (0 : ℝ) < 1 ∧ μ₀ ≠ ν₀ ∧
    (μ₀ : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 ∧
    (ν₀ : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 ∧
    0 < γ₁ ∧ γ₁ ≤ 1 ∧ barycenter 2 μ₀ = γ₁ • barycenter 2 ν₀ ∧ γ₁ ≠ 1 := by
  have h1 : (5 / 13 : ℝ) ^ 2 + (12 / 13) ^ 2 = 1 := by norm_num
  have h2 : (12 / 13 : ℝ) ^ 2 + (5 / 13) ^ 2 = 1 := by norm_num
  have h3 : (3 / 5 : ℝ) ^ 2 + (4 / 5) ^ 2 = 1 := by norm_num
  have h4 : (4 / 5 : ℝ) ^ 2 + (3 / 5) ^ 2 = 1 := by norm_num
  have hne : ∀ {p q p' q' : ℝ} (h : p ^ 2 + q ^ 2 = 1) (h' : p' ^ 2 + q' ^ 2 = 1),
      p ≠ p' → circlePt p q h ≠ circlePt p' q' h' := fun h h' hp he =>
    hp (by simpa [circlePt] using congrArg (fun z : SSphere 2 => (z : EucSpace 2) 0) he)
  have hQ : ∀ {x y : SSphere 2}, x ∈ positiveQuadrant 2 → y ∈ positiveQuadrant 2 →
      (halfDirac 2 x y : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 :=
    fun hx hy z hz => by
      rcases mem_of_mem_support_halfDirac hz with rfl | rfl
      exacts [hx, hy]
  refine ⟨halfDirac 2 (circlePt _ _ h1) (circlePt _ _ h2), halfDirac 2 (circlePt _ _ h3) (circlePt _ _ h4),
    85 / 91, one_pos, fun h => ?_,
    hQ (circlePt_mem_positiveQuadrant h1 (by norm_num) (by norm_num))
      (circlePt_mem_positiveQuadrant h2 (by norm_num) (by norm_num)),
    hQ (circlePt_mem_positiveQuadrant h3 (by norm_num) (by norm_num))
      (circlePt_mem_positiveQuadrant h4 (by norm_num) (by norm_num)),
    by norm_num, by norm_num, ?_, by norm_num⟩
  · have := congrArg (fun m : ProbSphere 2 => (m : Measure (SSphere 2)) {circlePt _ _ h3}) h
    simp [Pi.single_eq_of_ne (hne h1 h3 (by norm_num)),
      Pi.single_eq_of_ne (hne h2 h3 (by norm_num))] at this
    exact absurd (add_eq_zero.mp this.symm).1 (by simp)
  · rw [barycenter_halfDirac, barycenter_halfDirac]
    ext i
    fin_cases i <;> simp [circlePt] <;> norm_num


end Transformer.Interpolation
