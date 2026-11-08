module

public import Transformer.Grokking.GradientEvidence.ContrastInteraction

/-!
# Joint parameter and feature movement of an actual linear score

Source: `review/flat_spectrum.md`, *Joint difficult-query control*, JC1.
Features include final normalization. The increment p is the intended update;
rho retains floating-point parameter rounding. Signed first-order terms stay
together, and only their mixed term, rounding and execution change are bounded.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset FlatSpectrum.FeatureCodes

variable {K : Type*} [Fintype K]

/-- Intended parameter increment plus explicit rounding, JC1. -/
def scoreStep (w p ρ : K → ℝ) (k : K) : ℝ := w k + p k + ρ k

/-- Signed first-order score movement, JC1. Cancellation is retained. -/
def jointScoreDrift (w p x dx : K → ℝ) : ℝ :=
  linearScore p x + linearScore w dx

/-- Mixed parameter/feature term, parameter rounding and execution allowance,
JC1. It contains no desired new score, margin or loss. -/
def scoreStepRadius (p ρ x dx : K → ℝ) (e e' : ℝ) : ℝ :=
  pairBudget p dx + pairBudget ρ (fun k => x k + dx k) + |e' - e|

/-- The full finite-step identity, JC1. The mixed term is not discarded. -/
theorem score_step_identity (w p ρ x dx : K → ℝ) :
    linearScore (scoreStep w p ρ) (fun k => x k + dx k) - linearScore w x =
      jointScoreDrift w p x dx + linearScore p dx +
        linearScore ρ (fun k => x k + dx k) := by
  unfold scoreStep jointScoreDrift linearScore
  simp only [add_mul, mul_add, sum_add_distrib]
  ring

/-- A remainder bound around the signed joint contribution, JC1.
Both old and new executed scores may differ from real dot products. -/
theorem score_step_error_bound (w p ρ x dx : K → ℝ) (e e' : ℝ) :
    |((linearScore (scoreStep w p ρ) (fun k => x k + dx k) + e') -
        (linearScore w x + e)) - jointScoreDrift w p x dx| ≤
      scoreStepRadius p ρ x dx e e' := by
  have hi := score_step_identity w p ρ x dx
  have he : ((linearScore (scoreStep w p ρ) (fun k => x k + dx k) + e') -
      (linearScore w x + e)) - jointScoreDrift w p x dx =
      linearScore p dx + linearScore ρ (fun k => x k + dx k) + (e' - e) := by
    linarith
  rw [he]
  have h0 := abs_add_le (linearScore p dx)
    (linearScore ρ (fun k => x k + dx k))
  have h1 := abs_add_le (linearScore p dx +
    linearScore ρ (fun k => x k + dx k)) (e' - e)
  have hp := interaction_abs_le p dx
  have hr := interaction_abs_le ρ (fun k => x k + dx k)
  unfold scoreStepRadius
  linarith

/-- Both sides of the executed-score interval, JC1. -/
theorem score_step_interval (w p ρ x dx : K → ℝ) (e e' : ℝ) :
    (linearScore w x + e) + jointScoreDrift w p x dx - scoreStepRadius p ρ x dx e e' ≤
        linearScore (scoreStep w p ρ) (fun k => x k + dx k) + e' ∧
      linearScore (scoreStep w p ρ) (fun k => x k + dx k) + e' ≤
        (linearScore w x + e) + jointScoreDrift w p x dx + scoreStepRadius p ρ x dx e e' := by
  obtain ⟨hl, hu⟩ := abs_le.mp (score_step_error_bound w p ρ x dx e e')
  constructor <;> linarith

/-- The joint upper allowance is no worse than discarding both first-order
signs separately, JC1. This comparison does not assert either is small. -/
theorem joint_drift_abs_le (w p x dx : K → ℝ) :
    |jointScoreDrift w p x dx| ≤ pairBudget p x + pairBudget w dx := by
  have h := abs_add_le (linearScore p x) (linearScore w dx)
  have hp := interaction_abs_le p x
  have hw := interaction_abs_le w dx
  unfold jointScoreDrift
  linarith

/-- The separate norm estimate is also a valid two-sided bound, JC1. -/
theorem score_step_separate_bound (w p ρ x dx : K → ℝ) (e e' : ℝ) :
    |(linearScore (scoreStep w p ρ) (fun k => x k + dx k) + e') -
      (linearScore w x + e)| ≤
        pairBudget p x + pairBudget w dx + scoreStepRadius p ρ x dx e e' := by
  have h := score_step_error_bound w p ρ x dx e e'
  have hj := joint_drift_abs_le w p x dx
  have ht := abs_add_le
    (((linearScore (scoreStep w p ρ) (fun k => x k + dx k) + e') -
      (linearScore w x + e)) - jointScoreDrift w p x dx) (jointScoreDrift w p x dx)
  simp only [sub_add_cancel] at ht
  linarith

/-- Retaining the joint sign cannot increase the upper allowance, JC1. -/
theorem joint_upper_le_separate (w p x dx : K → ℝ) (R : ℝ) :
    jointScoreDrift w p x dx + R ≤ pairBudget p x + pairBudget w dx + R := by
  have hp := (le_abs_self (linearScore p x)).trans (interaction_abs_le p x)
  have hw := (le_abs_self (linearScore w dx)).trans (interaction_abs_le w dx)
  unfold jointScoreDrift
  linarith

/-- Nonzero parameter and feature increments cancel at first order, but the
actual score changes by -1 through the mixed term, JC1. The joint allowance
is 1, while the separate norm allowance is 5. -/
theorem joint_score_cancellation_witness :
    let w : Fin 1 → ℝ := fun _ => 2
    let p : Fin 1 → ℝ := fun _ => 1
    let x : Fin 1 → ℝ := fun _ => 2
    let dx : Fin 1 → ℝ := fun _ => -1
    jointScoreDrift w p x dx = 0 ∧
      linearScore (scoreStep w p (fun _ => 0)) (fun k => x k + dx k) - linearScore w x = -1 ∧
      scoreStepRadius p (fun _ => 0) x dx 0 0 = 1 ∧
      pairBudget p x + pairBudget w dx + scoreStepRadius p (fun _ => 0) x dx 0 0 = 5 := by
  norm_num [jointScoreDrift, scoreStep, scoreStepRadius, pairBudget, linearScore]

end Transformer.Grokking.GradientEvidence
