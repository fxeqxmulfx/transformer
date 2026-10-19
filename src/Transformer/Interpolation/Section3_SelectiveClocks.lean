module

public import Transformer.Interpolation.Section3_MeanNeuralPhases
public import Transformer.Interpolation.Section3_MeanAxisStability
public import Transformer.Interpolation.Section3_GatedCenterLaw

/-!
# Three-phase selective controls at a prescribed horizon

ArXiv:2411.04551v3, §3.2, Steps 1–3 of `lem: induction.barycenter`.
The actual attention, gated perceptron and reversed attention occupy
three equal physical slots. Their amplitudes retain independently chosen
attention and contraction clocks at any positive prescribed horizon.

Every original weak solution has the exact three-map endpoint.
A selected zero-mean input stays fixed during isolation and its final
radius grows by at most the proved reverse-attention factor. Inputs on
which the gate is inactive are restored exactly by the actual inverse law.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Actual three-phase network parameters with independently prescribed
phase clocks. Source: arXiv:2411.04551v3, §3.2, Steps 1–3. -/
noncomputable def selectiveControls (i j : Fin d) (p w : SSphere d) (c t₁ t₂ T : ℝ) : TimeParams d :=
  concatenate (T / 3 + T / 3)
    (concatenate (T / 3) (fun _ => meanAxisParams (w : EucSpace d) (t₁ / (T / 3)))
      (fun _ => twoRidgeParams i j ((t₂ / (T / 3)) • (p : EucSpace d))
        (-(p : EucSpace d)) (-c) 0 0 (-c)))
    (fun _ => meanAxisParams (w : EucSpace d) ((-t₁) / (T / 3)))

/-- The prescribed-horizon control has three constant pieces.
Source: arXiv:2411.04551v3, §3.2, switch count. -/
theorem piecewiseConstant_selectiveControls (i j : Fin d) (p w : SSphere d) (c t₁ t₂ T : ℝ)
    (hT : 0 < T) : PiecewiseConstant d (selectiveControls i j p w c t₁ t₂ T) T 3 := by
  have hh : 0 < T / 3 := by positivity
  have he : PiecewiseConstant d (selectiveControls i j p w c t₁ t₂ T) (T / 3 + T / 3 + T / 3) 3 :=
    piecewiseConstant_three_phases _ _ _ _ _ _ hh hh hh
  rwa [show T / 3 + T / 3 + T / 3 = T by ring] at he

/-- Every initial probability has an actual global weak solution for
these controls. Source: arXiv:2411.04551v3, §3.2. -/
theorem exists_cauchyPB_selectiveControls (i j : Fin d) (hij : i ≠ j) (p w : SSphere d)
    (c t₁ t₂ T : ℝ) (hT : 0 < T) (ν : Perspective.ProbSphere d) :
    ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧ cauchyPB d (selectiveControls i j p w c t₁ t₂ T) μ := by
  have hh : 0 < T / 3 := by positivity
  exact exists_cauchyPB_meanAxis_twoRidge_meanAxis w w (t₁ / (T / 3)) ((-t₁) / (T / 3))
    (T / 3) (T / 3) hh hh i j hij ((t₂ / (T / 3)) • (p : EucSpace d)) (-(p : EucSpace d)) (-c) 0 0 (-c) ν

/-- Every original weak endpoint equals the three actual laws at their
chosen clocks. Source: arXiv:2411.04551v3, §3.2, flow reversal. -/
theorem cauchyPB_selectiveControls_endpoint (i j : Fin d) (hij : i ≠ j) (p w : SSphere d)
    (c t₁ t₂ T : ℝ) (hT : 0 < T) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (selectiveControls i j p w c t₁ t₂ T) μ) :
    μ T = meanAxisLaw w (-1)
      (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) (meanAxisLaw w 1 (μ 0) t₁) t₂) t₁ := by
  let h := T / 3
  have hh : 0 < h := by dsimp only [h]; positivity
  have hfirst (ρ : Perspective.ProbSphere d) : meanAxisLaw w (t₁ / h) ρ h = meanAxisLaw w 1 ρ t₁ := by
    rw [meanAxisLaw_scale w (t₁ / h) ρ h, div_mul_cancel₀ _ hh.ne']
  have hmid (ρ : Perspective.ProbSphere d) : twoRidgeLaw ((t₂ / h) • (p : EucSpace d))
      (-(p : EucSpace d)) (-c) 0 0 (-c) ρ h = ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) ρ t₂ := by
    rw [twoRidgeLaw_scaled_ridge, div_mul_cancel₀ _ hh.ne']
  have hlast (ρ : Perspective.ProbSphere d) : meanAxisLaw w ((-t₁) / h) ρ h = meanAxisLaw w (-1) ρ t₁ := by
    calc
      _ = meanAxisLaw w 1 ρ (-t₁) := by rw [meanAxisLaw_scale w ((-t₁) / h) ρ h, div_mul_cancel₀ _ hh.ne']
      _ = _ := by rw [meanAxisLaw_scale w (-1) ρ t₁, neg_one_mul]
  have he := cauchyPB_meanAxis_twoRidge_meanAxis_endpoint w w (t₁ / h) ((-t₁) / h)
    h h h hh hh hh.le i j hij ((t₂ / h) • (p : EucSpace d)) (-(p : EucSpace d)) (-c) 0 0 (-c) μ hμ
  rw [show h + h + h = T by dsimp only [h]; ring, hfirst, hmid, hlast] at he
  exact he

/-- Selected inputs contract through the gate and obey a uniform radius
bound after reversal. Source: arXiv:2411.04551v3, §3.2, Step 3. -/
theorem selectiveControls_selected_radius (i j : Fin d) (hij : i ≠ j) (p w : SSphere d)
    (c t₁ t₂ T δ : ℝ) (hT : 0 < T) (ht₁ : 0 ≤ t₁)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (ν : Perspective.ProbSphere d) (hν : meanAxisMoment (w : EucSpace d) ν = 0)
    (hmid : (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) ν t₂ : Measure (SSphere d)).support ⊆ Metric.closedBall p δ)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (selectiveControls i j p w c t₁ t₂ T) μ)
    (h0 : μ 0 = ν) : (μ T : Measure (SSphere d)).support ⊆ Metric.closedBall p (δ * exp t₁) := by
  have he := cauchyPB_selectiveControls_endpoint i j hij p w c t₁ t₂ T hT μ hμ
  rw [h0, meanAxisLaw_of_moment_eq_zero w 1 ν hν] at he
  rw [he]
  simpa only [abs_neg, abs_one, one_mul] using
    support_meanAxisLaw_radius w p (-1) (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) ν t₂)
      δ hwp (fun x hx => hmid hx) t₁ ht₁

/-- Every original endpoint is restored when the middle law is fixed.
Source: arXiv:2411.04551v3, §3.2, `Φ_fin = Ψ₁⁻¹ ∘ Ψ₂ ∘ Ψ₁`. -/
theorem selectiveControls_restore_fixed (i j : Fin d) (hij : i ≠ j) (p w : SSphere d)
    (c t₁ t₂ T : ℝ) (hT : 0 < T) (ν : Perspective.ProbSphere d)
    (hmid : ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) (meanAxisLaw w 1 ν t₁) t₂ = meanAxisLaw w 1 ν t₁)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (selectiveControls i j p w c t₁ t₂ T) μ)
    (h0 : μ 0 = ν) : μ T = ν := by
  rw [cauchyPB_selectiveControls_endpoint i j hij p w c t₁ t₂ T hT μ hμ, h0, hmid, meanAxisLaw_inverse]

/-- Actual gate inactivity is sufficient for exact restoration.
Source: arXiv:2411.04551v3, §3.2, Steps 2 and 3. -/
theorem selectiveControls_restore_inactive (i j : Fin d) (hij : i ≠ j) (p w : SSphere d)
    (c t₁ t₂ T : ℝ) (hT : 0 < T) (ν : Perspective.ProbSphere d)
    (hgate : ∀ x ∈ (meanAxisLaw w 1 ν t₁ : Measure (SSphere d)).support,
      -c - inner ℝ (x : EucSpace d) (-(p : EucSpace d)) ≤ 0)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (selectiveControls i j p w c t₁ t₂ T) μ)
    (h0 : μ 0 = ν) : μ T = ν := by
  rw [cauchyPB_selectiveControls_endpoint i j hij p w c t₁ t₂ T hT μ hμ, h0,
    ridgeLaw_inactive_support _ _ _ _ hgate t₂, meanAxisLaw_inverse]

/-- A selected pole and an inactive transverse pole give genuine data
for all three phases, with actual original weak curves and a positive
middle radius, rather than assuming the desired endpoint conclusion. -/
example : ∃ (p w : SSphere 2) (ν₀ ν₁ : Perspective.ProbSphere 2)
    (μ₀ μ₁ : ℝ → Perspective.ProbSphere 2) (δ : ℝ),
    (0 : Fin 2) ≠ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ 0 < δ ∧
    inner ℝ (w : EucSpace 2) (p : EucSpace 2) = 0 ∧
    meanAxisMoment (w : EucSpace 2) ν₀ = 0 ∧
    (ridgeLaw (p : EucSpace 2) (-(p : EucSpace 2)) (-(1 / 4 : ℝ)) ν₀ 1 :
      Measure (SSphere 2)).support ⊆ Metric.closedBall p δ ∧
    (∀ x ∈ (meanAxisLaw w 1 ν₁ 1 : Measure (SSphere 2)).support,
      -(1 / 4 : ℝ) - inner ℝ (x : EucSpace 2) (-(p : EucSpace 2)) ≤ 0) ∧
    ridgeLaw (p : EucSpace 2) (-(p : EucSpace 2)) (-(1 / 4 : ℝ))
      (meanAxisLaw w 1 ν₁ 1) 1 = meanAxisLaw w 1 ν₁ 1 ∧
    cauchyPB 2 (selectiveControls 0 1 p w (1 / 4) 1 1 1) μ₀ ∧ μ₀ 0 = ν₀ ∧
    cauchyPB 2 (selectiveControls 0 1 p w (1 / 4) 1 1 1) μ₁ ∧ μ₁ 0 = ν₁ := by
  let p : SSphere 2 := basePoint 1
  let w : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let ν₀ := Perspective.diracProb 2 p
  let ν₁ := Perspective.diracProb 2 w
  let δ : ℝ := 2 * exp (-(((1 / 2 : ℝ) - 1 / 4) / 2 * 1))
  have hwp : inner ℝ (w : EucSpace 2) (p : EucSpace 2) = 0 := by
    simp [w, p, basePoint, EuclideanSpace.inner_single_left]
  have hm : meanAxisMoment (w : EucSpace 2) ν₀ = 0 := by
    simpa [meanAxisMoment, ν₀, barycenter, Perspective.diracProb] using hwp
  have hheight : ∀ x ∈ (ν₀ : Measure (SSphere 2)).support,
      (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (p : EucSpace 2) := by
    intro x hx
    rw [eq_of_mem_support_dirac hx, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  have hmid := support_ridgeLaw_center_radius p ν₀ (1 / 2) (1 / 4)
    (by norm_num) (by norm_num) hheight 1 zero_le_one
  have hgate : ∀ x ∈ (meanAxisLaw w 1 ν₁ 1 : Measure (SSphere 2)).support,
      -(1 / 4 : ℝ) - inner ℝ (x : EucSpace 2) (-(p : EucSpace 2)) ≤ 0 := by
    rw [meanAxisLaw, show ν₁ = Perspective.diracProb 2 w from rfl, projectionLaw_dirac_axis]
    intro x hx
    rw [eq_of_mem_support_dirac hx, inner_neg_right, hwp]
    norm_num
  obtain ⟨μ₀, h₀, hμ₀⟩ := exists_cauchyPB_selectiveControls (0 : Fin 2) 1
    (by norm_num) p w (1 / 4) 1 1 1 zero_lt_one ν₀
  obtain ⟨μ₁, h₁, hμ₁⟩ := exists_cauchyPB_selectiveControls (0 : Fin 2) 1
    (by norm_num) p w (1 / 4) 1 1 1 zero_lt_one ν₁
  exact ⟨p, w, ν₀, ν₁, μ₀, μ₁, δ, by norm_num, zero_lt_one, zero_le_one,
    mul_pos (by norm_num) (exp_pos _), hwp, hm, hmid, hgate,
    ridgeLaw_inactive_support _ _ _ _ hgate 1, hμ₀, h₀, hμ₁, h₁⟩

end Transformer.Interpolation
