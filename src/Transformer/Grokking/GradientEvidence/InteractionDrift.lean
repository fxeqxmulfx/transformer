module

public import Transformer.Grokking.GradientEvidence.AdaptiveEnergy

/-!
# Signed interaction changes retain cancellation

Source: `review/flat_spectrum.md`, *Adaptive interaction control*, AI4.
Two separately large residuals need not have a large interaction. Finite
changes retain readout, feature and mixed terms before bounding their sizes.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset FlatSpectrum.FeatureCodes

variable {K : Type*} [Fintype K]

/-- Cauchy allowance for an interaction of explicit vectors, AI4. -/
def pairBudget (u r : K → ℝ) : ℝ :=
  Real.sqrt ((∑ k, u k ^ 2) * ∑ k, r k ^ 2)

/-- The norm-product allowance is a bound, not the interaction itself, AI4. -/
theorem interaction_abs_le (u r : K → ℝ) : |linearScore u r| ≤ pairBudget u r := by
  apply Real.abs_le_sqrt
  exact sum_mul_sq_le_sq_mul_sq univ u r

/-- Exact finite interaction change, including its mixed term, AI4.
For saved states, u and r include the changing training residual map. -/
theorem interaction_change (u r u' r' : K → ℝ) :
    linearScore u' r' - linearScore u r =
      linearScore (fun k => u' k - u k) r +
      linearScore u (fun k => r' k - r k) +
      linearScore (fun k => u' k - u k) (fun k => r' k - r k) := by
  simp only [linearScore, sub_mul, mul_sub, sum_sub_distrib]
  ring

/-- Explicit allowance for the three signed finite-change contributions,
AI4; no monotonicity of either residual is required. -/
def interactionChangeBudget (u r u' r' : K → ℝ) : ℝ :=
  pairBudget (fun k => u' k - u k) r +
    pairBudget u (fun k => r' k - r k) +
    pairBudget (fun k => u' k - u k) (fun k => r' k - r k)

/-- A finite interval's change is controlled by its actual vector changes,
AI4. This is not a prediction from training CE alone. -/
theorem interaction_change_abs_le (u r u' r' : K → ℝ) :
    |linearScore u' r' - linearScore u r| ≤ interactionChangeBudget u r u' r' := by
  rw [interaction_change]
  calc
    _ ≤ |linearScore (fun k => u' k - u k) r| +
        |linearScore u (fun k => r' k - r k)| +
        |linearScore (fun k => u' k - u k) (fun k => r' k - r k)| :=
      (abs_add_le _ _).trans (add_le_add_left (abs_add_le _ _) _)
    _ ≤ _ := add_le_add (add_le_add (interaction_abs_le _ _) (interaction_abs_le _ _))
      (interaction_abs_le _ _)

/-- An old interaction bound persists with an explicit drift allowance,
AI4. This hypothesis is a prior-state bound, not the desired new conclusion. -/
theorem interaction_bound_transport (u r u' r' : K → ℝ) (B : ℝ)
    (h : |linearScore u r| ≤ B) :
    |linearScore u' r'| ≤ B + interactionChangeBudget u r u' r' := by
  have hc := interaction_change_abs_le u r u' r'
  have ht := abs_add_le (linearScore u r) (linearScore u' r' - linearScore u r)
  have he : linearScore u r + (linearScore u' r' - linearScore u r) = linearScore u' r' := by ring
  rw [he] at ht
  linarith

/-- A larger norm product can accompany zero interaction, AI4. The three
signed changes are -1, 2 and -2, so their cancellation must be retained. -/
theorem interaction_cancellation_witness :
    let u : Bool → ℝ := fun k => if k then 0 else 1
    let r : Bool → ℝ := fun k => if k then 0 else 1
    let u' : Bool → ℝ := fun k => if k then 2 else 0
    let r' : Bool → ℝ := fun k => if k then 0 else 3
    linearScore u r = 1 ∧ linearScore u' r' = 0 ∧
      pairBudget u r = 1 ∧ pairBudget u' r' = 6 ∧
      linearScore (fun k => u' k - u k) r = -1 ∧
      linearScore u (fun k => r' k - r k) = 2 ∧
      linearScore (fun k => u' k - u k) (fun k => r' k - r k) = -2 := by
  norm_num [linearScore, pairBudget]

example := interaction_bound_transport (fun (_ : Fin 1) => (2 : ℝ))
  (fun _ => 3) (fun _ => 1) (fun _ => 4) 6 (by norm_num [linearScore])

end Transformer.Grokking.GradientEvidence
