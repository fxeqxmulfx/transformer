module

public import Transformer.Grokking.Geometry.ProbabilityMotion
public import Transformer.Grokking.GradientEvidence.InteractionDrift

/-!
# Readout and feature motion give prospective probability ceilings

Source: `review/flat_spectrum.md`, *Prospective probability control*, PM3.
The old vectors and explicit displacement radii bound the finite bilinear
change, including the mixed term. Executed-score residuals remain explicit.
Feature radii must cover the moving tied encoder and final normalization.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset Geometry FlatSpectrum.FeatureCodes
variable {C K : Type*} [Fintype K]

/-- PM3: squared-energy caps give a norm-product allowance. -/
theorem pair_budget_le_caps (w x : K → ℝ) (A X : ℝ)
    (hA : 0 ≤ A) (hX : 0 ≤ X)
    (hw : ∑ k, w k^2 ≤ A^2) (hx : ∑ k, x k^2 ≤ X^2) :
    pairBudget w x ≤ A*X := by
  apply Real.sqrt_le_iff.mpr
  refine ⟨mul_nonneg hA hX, ?_⟩
  have h := mul_le_mul hw hx (sum_nonneg fun k _ => sq_nonneg (x k)) (sq_nonneg A)
  nlinarith

/-- PM3: a parameter radius and a feature radius bound their joint finite
motion. The mixed product R*S is essential even when first-order terms cancel. -/
theorem interaction_change_le_caps (w x w' x' : K → ℝ) (A X R S : ℝ)
    (hA : 0 ≤ A) (hX : 0 ≤ X) (hR : 0 ≤ R) (hS : 0 ≤ S)
    (hw : ∑ k, w k^2 ≤ A^2) (hx : ∑ k, x k^2 ≤ X^2)
    (hdw : ∑ k, (w' k-w k)^2 ≤ R^2) (hdx : ∑ k, (x' k-x k)^2 ≤ S^2) :
    |linearScore w' x'-linearScore w x| ≤ R*X+A*S+R*S := by
  have h := interaction_change_abs_le w x w' x'
  have h1 := pair_budget_le_caps (fun k => w' k-w k) x R X hR hX hdw hx
  have h2 := pair_budget_le_caps w (fun k => x' k-x k) A S hA hS hw hdx
  have h3 := pair_budget_le_caps (fun k => w' k-w k) (fun k => x' k-x k)
    R S hR hS hdw hdx
  unfold interactionChangeBudget at h
  linarith

/-- PM3: differences of actual class scores are scores of their readout
contrast, without assuming a shared adaptive scale. -/
theorem linear_score_contrast (E : C → K → ℝ) (x : K → ℝ) (u c : C) :
    linearScore (E u) x-linearScore (E c) x =
      linearScore (fun k => E u k-E c k) x := by
  simp only [linearScore, sub_mul, sum_sub_distrib]

/-- PM3: finite motion of a genuine executed logit contrast. The error
budget controls the change of the execution contrast, not each logit alone. -/
theorem executed_contrast_motion_le (E E' : C → K → ℝ) (x x' : K → ℝ)
    (e e' : C → ℝ) (u c : C) (A X R S ε : ℝ)
    (hA : 0 ≤ A) (hX : 0 ≤ X) (hR : 0 ≤ R) (hS : 0 ≤ S)
    (hw : ∑ k, (E u k-E c k)^2 ≤ A^2) (hx : ∑ k, x k^2 ≤ X^2)
    (hdw : ∑ k, ((E' u k-E' c k)-(E u k-E c k))^2 ≤ R^2)
    (hdx : ∑ k, (x' k-x k)^2 ≤ S^2)
    (he : |(e' u-e' c)-(e u-e c)| ≤ ε) :
    |((linearScore (E' u) x'+e' u)-(linearScore (E' c) x'+e' c))-
      ((linearScore (E u) x+e u)-(linearScore (E c) x+e c))| ≤ R*X+A*S+R*S+ε := by
  have h := interaction_change_le_caps (fun k => E u k-E c k) x
    (fun k => E' u k-E' c k) x' A X R S hA hX hR hS hw hx hdw hdx
  rw [← linear_score_contrast, ← linear_score_contrast] at h
  have hi : ((linearScore (E' u) x'+e' u)-(linearScore (E' c) x'+e' c))-
      ((linearScore (E u) x+e u)-(linearScore (E c) x+e c)) =
      ((linearScore (E' u) x'-linearScore (E' c) x')-
        (linearScore (E u) x-linearScore (E c) x)) + ((e' u-e' c)-(e u-e c)) := by ring
  rw [hi]
  exact (abs_add_le _ _).trans (add_le_add h he)

variable [Fintype C] [DecidableEq C]

/-- PM3: old full logits plus parameter/feature motion bounds supply a
future probability ceiling. Every competing class remains in the sum. -/
theorem probability_le_parameter_feature_motion (E E' : C → K → ℝ)
    (x x' : K → ℝ) (e e' : C → ℝ) (u : C) (A R ε : C → ℝ) (X S : ℝ)
    (hA : ∀ c, 0 ≤ A c) (hX : 0 ≤ X) (hR : ∀ c, 0 ≤ R c) (hS : 0 ≤ S)
    (hw : ∀ c, c ≠ u → ∑ k, (E u k-E c k)^2 ≤ (A c)^2)
    (hx : ∑ k, x k^2 ≤ X^2)
    (hdw : ∀ c, c ≠ u → ∑ k, ((E' u k-E' c k)-(E u k-E c k))^2 ≤ (R c)^2)
    (hdx : ∑ k, (x' k-x k)^2 ≤ S^2)
    (he : ∀ c, c ≠ u → |(e' u-e' c)-(e u-e c)| ≤ ε c) :
    probability (fun c => linearScore (E' c) x'+e' c) u ≤
      1 / (1 + oddsEnvelope (fun c => (linearScore (E c) x+e c)-
        (linearScore (E u) x+e u)-(R c*X+A c*S+R c*S+ε c)) u) := by
  apply probability_le_motion_envelope
  intro c hc
  exact (le_abs_self _).trans (executed_contrast_motion_le E E' x x' e e' u c
    (A c) X (R c) S (ε c) (hA c) hX (hR c) hS (hw c hc) hx (hdw c hc) hdx (he c hc))

example := pair_budget_le_caps (fun _ : Fin 1 => (2 : ℝ)) (fun _ => 3) 2 3
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := interaction_change_le_caps (fun _ : Fin 1 => (2 : ℝ)) (fun _ => 1)
  (fun _ => 3) (fun _ => 2) 2 1 1 1
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := executed_contrast_motion_le
  (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 1)
  (fun (c : Bool) (_ : Fin 1) => if c then (3 : ℝ) else 1)
  (fun _ => 1) (fun _ => 2) (fun _ => 0) (fun _ => 0) true false 1 1 1 1 0
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := probability_le_parameter_feature_motion
  (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 1)
  (fun (c : Bool) (_ : Fin 1) => if c then (3 : ℝ) else 1)
  (fun _ => 1) (fun _ => 2) (fun _ => 0) (fun _ => 0) true
  (fun _ => 1) (fun _ => 1) (fun _ => 0) 1 1
  (by intro c; norm_num) (by norm_num) (by intro c; norm_num) (by norm_num)
  (by intro c hc; cases c <;> norm_num at *) (by norm_num)
  (by intro c hc; cases c <;> norm_num at *) (by norm_num) (by intro c hc; norm_num)

end Transformer.Grokking.GradientEvidence
