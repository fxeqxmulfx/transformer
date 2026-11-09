module

public import Transformer.Grokking.GradientEvidence.MainAdamMotion
public import Transformer.Grokking.GradientEvidence.ForecastTransport

/-!
# Preserve a discovered margin and combine independent certificates

Source: `review/flat_spectrum.md`, *Main-model AdamW motion*, AD4–AD5.
Parameter-motion bounds still need feature-motion control and a positive
starting margin. Both are explicit here. The certificate envelope combines
valid lower bounds for the same target; it pays for all constituent bases.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset FlatSpectrum.FeatureCodes
variable {K C : Type*} [Fintype K]

/-- AD4: coordinate budgets for a moving readout and feature retain the
mixed product. The right side contains old values and motion budgets. -/
theorem linear_score_motion_budget (w x w' x' r s : K → ℝ)
    (hw : ∀ k, |w' k - w k| ≤ r k) (hx : ∀ k, |x' k - x k| ≤ s k) :
    |linearScore w' x' - linearScore w x| ≤
      ∑ k, (|w k| * s k + |x k| * r k + r k * s k) := by
  unfold linearScore
  rw [← sum_sub_distrib]
  apply (abs_sum_le_sum_abs _ _).trans
  apply sum_le_sum
  intro k hk
  have h := product_forecast_bound (w k) (x k) 0 0 (w' k) (x' k) (r k) (s k)
    (by simpa only [add_zero] using hw k) (by simpa only [add_zero] using hx k)
  simpa only [mul_zero, zero_mul, add_zero, zero_add, abs_zero] using h

/-- AD4: a certified old contrast remains positive when its finite motion
budget is strictly smaller than the old margin lower bound. -/
theorem positive_contrast_persists (w x w' x' r s : K → ℝ) (margin : ℝ)
    (hw : ∀ k, |w' k - w k| ≤ r k) (hx : ∀ k, |x' k - x k| ≤ s k)
    (hold : margin ≤ linearScore w x)
    (hbudget : (∑ k, (|w k| * s k + |x k| * r k + r k * s k)) < margin) :
    0 < linearScore w' x' := by
  have h := (abs_le.mp (linear_score_motion_budget w x w' x' r s hw hx)).1
  linarith

omit [Fintype K] in
/-- AD4: independently bounded readout rows give a contrast budget. -/
theorem readout_contrast_motion (E E' : C → K → ℝ) (r : C → K → ℝ)
    (h : ∀ c k, |E' c k - E c k| ≤ r c k) (t c : C) (k : K) :
    |(E' t k-E' c k)-(E t k-E c k)| ≤ r t k+r c k := by
  have he : (E' t k-E' c k)-(E t k-E c k) =
      (E' t k-E t k) + -(E' c k-E c k) := by ring
  rw [he]
  have h1 := abs_add_le (E' t k-E t k) (-(E' c k-E c k))
  rw [abs_neg] at h1
  linarith [h t k, h c k]

/-- AD4: all rival classes stay in the persistence statement. A readout
budget alone never substitutes for control of the changing query feature. -/
theorem prediction_persists (E E' : C → K → ℝ) (x x' : K → ℝ)
    (r : C → K → ℝ) (s : K → ℝ) (margin : C → ℝ) (t : C)
    (hE : ∀ c k, |E' c k-E c k| ≤ r c k)
    (hx : ∀ k, |x' k-x k| ≤ s k)
    (hold : ∀ c, c ≠ t → margin c ≤ linearScore (fun k => E t k-E c k) x)
    (hbudget : ∀ c, c ≠ t →
      (∑ k, (|E t k-E c k| * s k + |x k| * (r t k+r c k) + (r t k+r c k)*s k))
        < margin c) :
    ∀ c, c ≠ t → linearScore (E' c) x' < linearScore (E' t) x' := by
  intro c hc
  have hp := positive_contrast_persists (fun k => E t k-E c k) x
    (fun k => E' t k-E' c k) x' (fun k => r t k+r c k) s (margin c)
    (fun k => readout_contrast_motion E E' r hE t c k) hx (hold c hc) (hbudget c hc)
  simp only [linearScore, sub_mul, sum_sub_distrib] at hp ⊢
  linarith

omit [Fintype K] in
/-- AD5: the maximum of three valid lower bounds remains valid for the same
margin. The constituent constructions can have different bases. -/
theorem certificate_envelope_valid (a b c margin : ℝ)
    (ha : a ≤ margin) (hb : b ≤ margin) (hc : c ≤ margin) :
    max a (max b c) ≤ margin := by
  apply max_le ha
  exact max_le hb hc

omit [Fintype K] in
/-- AD5: an envelope cannot discard a positive constituent certificate. -/
theorem certificate_envelope_retains (a b c : ℝ) (h : 0 < a ∨ 0 < b ∨ 0 < c) :
    0 < max a (max b c) := by
  rcases h with h | h | h
  · exact h.trans_le (le_max_left _ _)
  · exact h.trans_le ((le_max_left _ _).trans (le_max_right _ _))
  · exact h.trans_le ((le_max_right _ _).trans (le_max_right _ _))

omit [Fintype K] in
/-- AD5: a positive envelope certifies the true margin under the unchanged
constituent bounds; selecting by the bounds does not require query outcomes. -/
theorem certificate_envelope_sound (a b c margin : ℝ)
    (ha : a ≤ margin) (hb : b ≤ margin) (hc : c ≤ margin)
    (h : 0 < max a (max b c)) : 0 < margin := by
  exact h.trans_le (certificate_envelope_valid a b c margin ha hb hc)

example := linear_score_motion_budget (K := Fin 1) (fun _ => 2) (fun _ => 3)
  (fun _ => 21/10) (fun _ => 31/10) (fun _ => 1/10) (fun _ => 1/10)
  (by intro k; norm_num) (by intro k; norm_num)
example := positive_contrast_persists (K := Fin 1) (fun _ => 2) (fun _ => 3)
  (fun _ => 21/10) (fun _ => 31/10) (fun _ => 1/10) (fun _ => 1/10) 5
  (by intro k; norm_num) (by intro k; norm_num)
  (by norm_num [linearScore]) (by norm_num)
example := readout_contrast_motion (C := Bool) (K := Fin 1)
  (fun c _ => if c then 2 else 1) (fun c _ => if c then 21/10 else 11/10)
  (fun _ _ => 1/10) (by intro c k; cases c <;> norm_num) true false 0
example := prediction_persists (C := Bool) (K := Fin 1)
  (fun c _ => if c then 2 else 1) (fun c _ => if c then 21/10 else 11/10)
  (fun _ => 3) (fun _ => 31/10) (fun _ _ => 1/10) (fun _ => 1/10) (fun _ => 2) true
  (by intro c k; cases c <;> norm_num) (by intro k; norm_num)
  (by intro c hc; cases c <;> norm_num [linearScore] at *)
  (by intro c hc; cases c <;> norm_num at *)
example := certificate_envelope_valid (-1) 2 1 3 (by norm_num) (by norm_num) (by norm_num)
example := certificate_envelope_retains (-1) 2 1 (by norm_num)
example := certificate_envelope_sound (-1) 2 1 3
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.GradientEvidence
