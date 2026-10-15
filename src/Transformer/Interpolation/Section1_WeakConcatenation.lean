/-
# Concatenating actual solutions of the original weak equation

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
the two constant-field phases of `lem: first.quadrant`.
Two original weak solutions with matching laws at the switch produce
an original weak solution for the joined parameter curve. Both the
measure and the controls use the second phase's elapsed time.
The proof joins their actual scalar test primitives and fluxes, so
it retains the original integrated equation at the control switch.
-/

import Transformer.Interpolation.Section1_ConcatenatedPrimitives
import Transformer.Interpolation.Section3_ProjectionControls
import Transformer.Interpolation.Clustering

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original vector field of joined controls and laws is the joined
original field with its proper second-phase clock.
Source: arXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`. -/
theorem fullVF_concatenate (S : ℝ) (θ₁ θ₂ : TimeParams d)
    (μ₁ μ₂ : ℝ → Perspective.ProbSphere d) (t : ℝ) (x : EucSpace d) :
    fullVF d (concatenate S θ₁ θ₂) (concatenate S μ₁ μ₂ t) t x =
      if t < S then fullVF d θ₁ (μ₁ t) t x
      else fullVF d θ₂ (μ₂ (t - S)) (t - S) x := by
  by_cases ht : t < S
  · simp only [fullVF, concatenate, ite_eq_left ht]
  · simp only [fullVF, concatenate, ite_eq_right ht]

/-- Original test moments join by the same actual selection of laws.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem integral_concatenate_law (S : ℝ) (μ₁ μ₂ : ℝ → Perspective.ProbSphere d)
    (φ : EucSpace d → ℝ) (t : ℝ) :
    (∫ x, φ (x : EucSpace d)
      ∂(concatenate (E := Perspective.ProbSphere d) S μ₁ μ₂ t : Measure (SSphere d))) =
      concatenate S
        (fun s => ∫ x, φ (x : EucSpace d) ∂(μ₁ s : Measure (SSphere d)))
        (fun s => ∫ x, φ (x : EucSpace d) ∂(μ₂ s : Measure (SSphere d))) t := by
  by_cases ht : t < S
  · simp only [concatenate, ite_eq_left ht]
  · simp only [concatenate, ite_eq_right ht]

/-- Joining actual original weak solutions with matching endpoint laws
preserves the full original integrated equation for every `C¹` test.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
the control switch in `lem: first.quadrant`. -/
theorem cauchyPB_concatenate (S : ℝ) (hS : 0 < S) (θ₁ θ₂ : TimeParams d)
    (μ₁ μ₂ : ℝ → Perspective.ProbSphere d)
    (hμ₁ : cauchyPB d θ₁ μ₁) (hμ₂ : cauchyPB d θ₂ μ₂)
    (hmatch : μ₂ 0 = μ₁ S) :
    cauchyPB d (concatenate S θ₁ θ₂) (concatenate S μ₁ μ₂) := by
  intro φ hφ t
  let F₁ (s : ℝ) := ∫ x, φ (x : EucSpace d) ∂(μ₁ s : Measure (SSphere d))
  let F₂ (s : ℝ) := ∫ x, φ (x : EucSpace d) ∂(μ₂ s : Measure (SSphere d))
  let g₁ (s : ℝ) := ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
    (fullVF d θ₁ (μ₁ s) s (x : EucSpace d)) ∂(μ₁ s : Measure (SSphere d))
  let g₂ (s : ℝ) := ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
    (fullVF d θ₂ (μ₂ s) s (x : EucSpace d)) ∂(μ₂ s : Measure (SSphere d))
  have hI₁ (a b : ℝ) : IntervalIntegrable g₁ volume a b :=
    ((hμ₁ φ hφ a).1.symm).trans (hμ₁ φ hφ b).1
  have hI₂ (a b : ℝ) : IntervalIntegrable g₂ volume a b :=
    ((hμ₂ φ hφ a).1.symm).trans (hμ₂ φ hφ b).1
  have hFmatch : F₂ 0 = F₁ S := by
    dsimp only [F₁, F₂]
    rw [hmatch]
  have hflux : (fun s => ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
      (fullVF d (concatenate S θ₁ θ₂) (concatenate S μ₁ μ₂ s) s (x : EucSpace d))
        ∂(concatenate (E := Perspective.ProbSphere d) S μ₁ μ₂ s : Measure (SSphere d))) =
      concatenate S g₁ g₂ := by
    funext s
    simp_rw [fullVF_concatenate]
    by_cases hs : s < S
    · simp only [concatenate, ite_eq_left hs, g₁]
    · simp only [concatenate, ite_eq_right hs, g₂]
  rw [hflux, integral_concatenate_law S μ₁ μ₂ φ t, integral_concatenate_law S μ₁ μ₂ φ 0]
  exact concatenate_primitives S hS F₁ F₂ g₁ g₂ hI₁ hI₂
    (fun s => (hμ₁ φ hφ s).2) (fun s => (hμ₂ φ hφ s).2) hFmatch t

/-- The joined original weak solution keeps its original prescribed
initial law. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem cauchyPB_concatenate_initial (S : ℝ) (hS : 0 < S) (θ₁ θ₂ : TimeParams d)
    (μ₁ μ₂ : ℝ → Perspective.ProbSphere d) (ν : Perspective.ProbSphere d)
    (hinit : μ₁ 0 = ν) (hμ₁ : cauchyPB d θ₁ μ₁) (hμ₂ : cauchyPB d θ₂ μ₂)
    (hmatch : μ₂ 0 = μ₁ S) :
    concatenate S μ₁ μ₂ 0 = ν ∧
      cauchyPB d (concatenate S θ₁ θ₂) (concatenate S μ₁ μ₂) :=
  ⟨(concatenate_zero S hS μ₁ μ₂).trans hinit,
    cauchyPB_concatenate S hS θ₁ θ₂ μ₁ μ₂ hμ₁ hμ₂ hmatch⟩

/-- Existence for each original phase supplies existence for their actual
composition, retaining the first prescribed initial law.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem exists_cauchyPB_concatenate (S : ℝ) (hS : 0 < S) (θ₁ θ₂ : TimeParams d)
    (ν : Perspective.ProbSphere d)
    (hfirst : ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧ cauchyPB d θ₁ μ)
    (hsecond : ∀ ρ : Perspective.ProbSphere d,
      ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ρ ∧ cauchyPB d θ₂ μ) :
    ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧
      cauchyPB d (concatenate S θ₁ θ₂) μ := by
  obtain ⟨μ₁, hinit, hμ₁⟩ := hfirst
  obtain ⟨μ₂, hmatch, hμ₂⟩ := hsecond (μ₁ S)
  exact ⟨concatenate S μ₁ μ₂,
    cauchyPB_concatenate_initial S hS θ₁ θ₂ μ₁ μ₂ ν hinit hμ₁ hμ₂ hmatch⟩

/-- The source's two constant phases have exactly the required two-piece
description on a prescribed positive horizon.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem piecewiseConstant_concatenate_half (θ₁ θ₂ : Params d) (T : ℝ) (hT : 0 < T) :
    PiecewiseConstant d (concatenate (T / 2) (fun _ => θ₁) (fun _ => θ₂)) T 2 := by
  refine ⟨![0, T / 2, T], ?_, by simp, by simp, ?_⟩
  · intro a b hab
    fin_cases a <;> fin_cases b <;> simp at hab ⊢ <;> linarith
  · intro k s s' hs hs'
    fin_cases k
    · change s ∈ Set.Ico 0 (T / 2) at hs
      change s' ∈ Set.Ico 0 (T / 2) at hs'
      simp only [concatenate, ite_eq_left hs.2, ite_eq_left hs'.2]
    · change s ∈ Set.Ico (T / 2) T at hs
      change s' ∈ Set.Ico (T / 2) T at hs'
      simp only [concatenate, ite_eq_right (not_lt_of_ge hs.1),
        ite_eq_right (not_lt_of_ge hs'.1)]

/-- All joining hypotheses are realized by constructed original weak
solutions for every initial spherical probability, with two genuinely
different control speeds. Their laws match at the actual switch.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
example (ν : Perspective.ProbSphere 2) :
    concatenate 1 (projectionLaw (basePoint 1) ν)
      (fun s => projectionLaw (basePoint 1) (projectionLaw (basePoint 1) ν 1) ((2 : ℝ) * s)) 0 = ν ∧
    cauchyPB 2
      (concatenate 1 (fun _ => projectionControl (0 : Fin 2) (basePoint 1 : EucSpace 2))
        (fun _ => projectionControl (0 : Fin 2) ((2 : ℝ) • (basePoint 1 : EucSpace 2))))
      (concatenate 1 (projectionLaw (basePoint 1) ν)
        (fun s => projectionLaw (basePoint 1) (projectionLaw (basePoint 1) ν 1) (2 * s))) := by
  have h₁ := cauchyPB_projectionControl (0 : Fin 2) (basePoint 1 : SSphere 2) ν (1 : ℝ)
  have h₂ := cauchyPB_projectionControl (0 : Fin 2) (basePoint 1 : SSphere 2)
    (projectionLaw (basePoint 1) ν 1) (2 : ℝ)
  apply cauchyPB_concatenate_initial 1 one_pos
  · exact projectionLaw_zero (basePoint 1) ν
  · simpa only [one_mul, one_smul] using h₁.2
  · exact h₂.2
  · simp only [mul_zero, projectionLaw_zero]

/-- Both generic existence inputs hold for original concrete controls,
at every original initial spherical probability. -/
example (ν : Perspective.ProbSphere 2) : (0 : ℝ) < 1 ∧
    (∃ μ : ℝ → Perspective.ProbSphere 2, μ 0 = ν ∧
      cauchyPB 2 (fun _ => projectionControl (0 : Fin 2) (basePoint 1 : EucSpace 2)) μ) ∧
    (∀ ρ : Perspective.ProbSphere 2, ∃ μ : ℝ → Perspective.ProbSphere 2, μ 0 = ρ ∧
      cauchyPB 2 (fun _ => projectionControl (0 : Fin 2) ((2 : ℝ) • (basePoint 1 : EucSpace 2))) μ) := by
  refine ⟨one_pos, ?_, ?_⟩
  · exact ⟨fun s => projectionLaw (basePoint 1) ν ((1 : ℝ) * s), by
      simpa only [one_smul] using cauchyPB_projectionControl (0 : Fin 2) (basePoint 1 : SSphere 2) ν 1⟩
  · intro ρ
    exact ⟨fun s => projectionLaw (basePoint 1) ρ ((2 : ℝ) * s),
      cauchyPB_projectionControl (0 : Fin 2) (basePoint 1 : SSphere 2) ρ 2⟩

/-- The positive-horizon hypothesis for the two-piece construction is
realized at one, with any original constant control values. -/
example : (0 : ℝ) < 1 := one_pos

end Transformer.Interpolation
