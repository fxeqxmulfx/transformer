/-
# Bounded two-phase controls and their original weak solutions

ArXiv:2411.04551v3, §3.1, time rescaling in `lem: first.quadrant`.
Each intrinsic projection phase occupies half the requested horizon.
The actual original parameter curve has two constant pieces, the
source's zero components and all-ones bias, and a `1/T` norm bound.
Existence follows by concatenating constructed weak solutions.
Every original weak solution transports its support by the same two
characteristic maps, including at the switch and the final time.
-/

import Transformer.Interpolation.Section1_WeakConcatenation

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The source's two rank-one phases, each assigned half the horizon.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
noncomputable def twoProjectionControls (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ T : ℝ) : TimeParams d :=
  concatenate (T / 2)
    (fun _ => projectionControl i ((2 * τ₁ / T) • (w₁ : EucSpace d)))
    (fun _ => projectionControl i ((2 * τ₂ / T) • (w₂ : EucSpace d)))

/-- The first phase uses its rescaled original projection control.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem twoProjectionControls_before (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ T s : ℝ) (hs : s < T / 2) :
    twoProjectionControls i w₁ w₂ τ₁ τ₂ T s =
      projectionControl i ((2 * τ₁ / T) • (w₁ : EucSpace d)) :=
  concatenate_of_lt _ _ _ _ hs

/-- The second phase includes the actual switching time.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem twoProjectionControls_after (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ T s : ℝ) (hs : T / 2 ≤ s) :
    twoProjectionControls i w₁ w₂ τ₁ τ₂ T s =
      projectionControl i ((2 * τ₂ / T) • (w₂ : EucSpace d)) :=
  concatenate_of_ge _ _ _ _ hs

/-- The switching value is the second control, as in the original
right-continuous two-phase parameter curve.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem twoProjectionControls_switch (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ T : ℝ) :
    twoProjectionControls i w₁ w₂ τ₁ τ₂ T (T / 2) =
      projectionControl i ((2 * τ₂ / T) • (w₂ : EucSpace d)) :=
  twoProjectionControls_after i w₁ w₂ τ₁ τ₂ T (T / 2) le_rfl

/-- The actual control curve has the source's two constant pieces.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem twoProjectionControls_piecewise (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ T : ℝ) (hT : 0 < T) :
    PiecewiseConstant d (twoProjectionControls i w₁ w₂ τ₁ τ₂ T) T 2 :=
  piecewiseConstant_concatenate_half _ _ T hT

/-- All other original parameters have exactly the required values.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem twoProjectionControls_components (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ T s : ℝ) :
    let θ := twoProjectionControls i w₁ w₂ τ₁ τ₂ T
    (θ s).V = 0 ∧ (θ s).B = 0 ∧ (θ s).U = 0 ∧ (θ s).b = onesBias d := by
  dsimp only
  by_cases hs : s < T / 2
  · rw [twoProjectionControls_before i w₁ w₂ τ₁ τ₂ T s hs]
    exact projectionControl_components _ _
  · rw [twoProjectionControls_after i w₁ w₂ τ₁ τ₂ T s (le_of_not_gt hs)]
    exact projectionControl_components _ _

/-- One constant depending on the intrinsic phase lengths bounds both
original matrix norms at every time, before choosing a horizon.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem twoProjectionControls_norm (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ T : ℝ) (hτ₁ : 0 ≤ τ₁) (hτ₂ : 0 ≤ τ₂) (hT : 0 < T) (s : ℝ) :
    ‖(twoProjectionControls i w₁ w₂ τ₁ τ₂ T s).W‖ ≤ 2 * (τ₁ + τ₂) / T := by
  by_cases hs : s < T / 2
  · rw [twoProjectionControls_before i w₁ w₂ τ₁ τ₂ T s hs]
    exact (norm_projectionControl_half_time i w₁ τ₁ T hτ₁ hT).trans
      (div_le_div_of_nonneg_right (by linarith) hT.le)
  · rw [twoProjectionControls_after i w₁ w₂ τ₁ τ₂ T s (le_of_not_gt hs)]
    exact (norm_projectionControl_half_time i w₂ τ₂ T hτ₂ hT).trans
      (div_le_div_of_nonneg_right (by linarith) hT.le)

/-- The two phases admit an actual original weak solution for every
original initial probability law.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem exists_cauchyPB_twoProjectionControls (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ T : ℝ) (hT : 0 < T) (ν : Perspective.ProbSphere d) :
    ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧
      cauchyPB d (twoProjectionControls i w₁ w₂ τ₁ τ₂ T) μ := by
  apply exists_cauchyPB_concatenate (T / 2) (half_pos hT)
  · exact ⟨fun s => projectionLaw w₁ ν ((2 * τ₁ / T) * s),
      cauchyPB_projectionControl i w₁ ν (2 * τ₁ / T)⟩
  · intro ρ
    exact ⟨fun s => projectionLaw w₂ ρ ((2 * τ₂ / T) * s),
      cauchyPB_projectionControl i w₂ ρ (2 * τ₂ / T)⟩

/-- Every original weak solution follows the same two support maps,
with the original intrinsic times restored after time rescaling.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem twoProjectionControls_support (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ T : ℝ) (hT : 0 < T) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (twoProjectionControls i w₁ w₂ τ₁ τ₂ T) μ) :
    (μ T : Measure (SSphere d)).support ⊆ sphereProjectionFlow w₂ τ₂ ''
      (sphereProjectionFlow w₁ τ₁ '' (μ 0 : Measure (SSphere d)).support) := by
  let θ := twoProjectionControls i w₁ w₂ τ₁ τ₂ T
  have hf₁ : ∀ s ∈ Set.Ioo 0 (T / 2), ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) =
        (2 * τ₁ / T) • proj d (x : EucSpace d) (w₁ : EucSpace d) := by
    intro s hs x
    rw [fullVF_projectionControl_at θ (μ s) s (x : EucSpace d) i
      ((2 * τ₁ / T) • (w₁ : EucSpace d))
      (twoProjectionControls_before i w₁ w₂ τ₁ τ₂ T s hs.2), projection_field_smul]
  have hf₂ : ∀ s ∈ Set.Ioo (T / 2) T, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) =
        (2 * τ₂ / T) • proj d (x : EucSpace d) (w₂ : EucSpace d) := by
    intro s hs x
    rw [fullVF_projectionControl_at θ (μ s) s (x : EucSpace d) i
      ((2 * τ₂ / T) • (w₂ : EucSpace d))
      (twoProjectionControls_after i w₁ w₂ τ₁ τ₂ T s hs.1.le), projection_field_smul]
  have htime₁ : 2 * τ₁ / T * (T / 2) = τ₁ := by field_simp
  have htime₂ : 2 * τ₂ / T * (T - T / 2) = τ₂ := by field_simp; ring
  have h₁ := cauchyPB_scaled_projection_support_initial θ μ hμ w₁ (2 * τ₁ / T)
    (T / 2) (half_pos hT) hf₁
  have h₂ := cauchyPB_scaled_projection_support θ μ hμ w₂ (2 * τ₂ / T)
    (T / 2) T (by linarith) hf₂
  rw [htime₁] at h₁
  rw [htime₂] at h₂
  intro x hx
  obtain ⟨y, hy, rfl⟩ := h₂ hx
  exact ⟨y, h₁ hy, rfl⟩

/-- The phase-selection and norm hypotheses hold for two positive
intrinsic lengths at a genuine positive horizon. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) / 2 ≤ 1 ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 := by norm_num

/-- The weak-solution hypothesis is inhabited by a constructed original
law at these concrete circle controls. -/
example : ∃ μ : ℝ → Perspective.ProbSphere 2,
    μ 0 = Perspective.diracProb 2 (basePoint 1) ∧
      cauchyPB 2 (twoProjectionControls (0 : Fin 2) (basePoint 1) (basePoint 1) 1 1 1) μ :=
  exists_cauchyPB_twoProjectionControls 0 (basePoint 1) (basePoint 1) 1 1 1 one_pos
    (Perspective.diracProb 2 (basePoint 1))

end Transformer.Interpolation
