/-
# Corrected main sign rate and refutation of the published rate

arXiv:2402.19449v2, Theorem 3 and Appendix I, Lemma 7. The main theorem's
exp(-ct) rate is also false for c=3, not just the exact loss formula.
-/

import Transformer.Imbalance.Section3_SignFlow
import Transformer.Imbalance.Section3_GradientFlow

open Filter Asymptotics

namespace Transformer.Imbalance

variable {c : ℕ}

/-- Corrected sign-descent conclusion of Theorem 3. Ordinary sign descent
has rate exp(-2t), since diagonal and off-diagonal logits move at +1 and -1.
At least two classes and positive frequencies are necessary. -/
theorem sign_flow_rate (hc : 2 ≤ c) (π : Fin c → ℝ) (hπ : ∀ k, 0 < π k)
    (W : ℝ → Parameters c c) (hW : IsSignFlow π W) (k : Fin c) :
    (fun t => sampleLoss (W t) (Pi.single k 1) k) =Θ[atTop]
      (fun t => Real.exp (-(2 * t))) := by
  rw [signFlow_unique hc π hπ W hW]
  simp_rw [signParameters_loss]
  exact sign_margin_rate _ (otherClasses_pos c hc)

/-- Nonvacuity of the corrected main sign theorem. -/
example : 2 ≤ 3 ∧ (∀ k : Fin 3, (0 : ℝ) < (fun _ : Fin 3 => (1 / 3 : ℝ)) k) ∧
    IsSignFlow (fun _ : Fin 3 => (1 / 3 : ℝ)) (signParameters 3) :=
  ⟨by decide, fun _ => by norm_num, signParameters_flow (by decide) _ (fun _ => by norm_num)⟩

/-- Counterexample to Theorem 3's sign-descent rate: for three classes,
the actual loss is not even O(exp(-3t)), hence cannot be Theta(exp(-3t)). -/
theorem paper_sign_rate_false :
    ¬ ((fun t => sampleLoss (signParameters 3 t) (Pi.single (0 : Fin 3) 1) 0)
      =Θ[atTop] (fun t => Real.exp (-(3 * t)))) := by
  intro h
  obtain ⟨C, hC⟩ := isBigO_iff.1 h.1
  have he : ∀ᶠ t : ℝ in atTop, 3 * C / 2 < Real.exp t :=
    Real.tendsto_exp_atTop.eventually (eventually_gt_atTop (3 * C / 2))
  obtain ⟨t, ht0, htC, hte⟩ := ((eventually_ge_atTop (0 : ℝ)).and (hC.and he)).exists
  rw [signParameters_loss] at htC
  have hb := (sign_loss_bounds 2 (by norm_num) t ht0).1
  have hn := marginLoss_nonneg 2 (2 * t) (by norm_num)
  norm_num only [Nat.cast_ofNat, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _),
    show (3 : ℝ) - 1 = 2 by norm_num, abs_of_nonneg hn] at htC
  have hl : (2 / 3 : ℝ) * Real.exp (-(2 * t)) ≤ C * Real.exp (-(3 * t)) := by
    exact le_trans (by norm_num at hb ⊢; exact hb) htC
  have hexp : Real.exp (-(2 * t)) = Real.exp t * Real.exp (-(3 * t)) := by
    rw [← Real.exp_add]
    congr 1
    ring
  rw [hexp] at hl
  have hpos := Real.exp_pos (-(3 * t))
  have hbound : (2 / 3 : ℝ) * Real.exp t ≤ C := by
    by_contra h
    have hlt : C < (2 / 3 : ℝ) * Real.exp t := lt_of_not_ge h
    nlinarith
  linarith

/-- Combined corrected Theorem 3: the actual zero-initialized solutions
have different rates, with frequency dependence only for gradient flow. -/
theorem convergence_separation (hc : 2 ≤ c) (π : Fin c → ℝ) (hπ : ∀ k, 0 < π k)
    (k : Fin c) :
    ((fun t => sampleLoss (gdParameters c hc π t) (Pi.single k 1) k)
      =Θ[atTop] (fun t => 1 / (π k * t))) ∧
    ((fun t => sampleLoss (signParameters c t) (Pi.single k 1) k)
      =Θ[atTop] (fun t => Real.exp (-(2 * t)))) := by
  exact ⟨gradient_flow_rate hc π _ (gdParameters_flow c hc π) k (hπ k),
    sign_flow_rate hc π hπ _ (signParameters_flow hc π hπ) k⟩

/-- Nonvacuity of the combined corrected Theorem 3. -/
example : 2 ≤ 3 ∧ ∀ k : Fin 3, (0 : ℝ) < (fun _ : Fin 3 => (1 / 3 : ℝ)) k :=
  ⟨by decide, fun _ => by norm_num⟩

end Transformer.Imbalance
