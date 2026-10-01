/-
# Ordinary continuous-time sign descent

arXiv:2402.19449v2, Section 3.3, Theorem 3, Appendix I, Lemma 7.
The source correctly gives a'=1 and b'=-1, but substitutes exp(-ct)
into the loss. The actual margin is a-b=2t, so the corrected exponential
rate is exp(-2t), independently of both frequencies and class count.
-/

import Transformer.Imbalance.Section3_SimpleModel
import Mathlib.Analysis.Calculus.MeanValue

open Filter Asymptotics

noncomputable section

namespace Transformer.Imbalance

variable {c : ℕ}

/-- Finite softmax probabilities are strictly between zero and one
when at least two classes exist; Appendix I, Lemma 7's sign calculation. -/
theorem softmax_strict_bounds (hc : 2 ≤ c) (z : Fin c → ℝ) (i : Fin c) :
    0 < Perspective.softmaxWeight z i ∧ Perspective.softmaxWeight z i < 1 := by
  let : Nontrivial (Fin c) := Fin.nontrivial_iff_two_le.2 hc
  have hc0 : 0 < c := by omega
  have hZ := Perspective.softmaxPartition_pos hc0 z
  unfold Perspective.softmaxWeight
  constructor
  · exact div_pos (Real.exp_pos _) hZ
  · apply (div_lt_one hZ).2
    obtain ⟨j, hji⟩ := exists_ne i
    exact Finset.single_lt_sum hji (Finset.mem_univ i) (Finset.mem_univ j)
      (Real.exp_pos _) (fun k _ _ => (Real.exp_pos _).le)

/-- Nonvacuity of the nontrivial-softmax hypothesis; Appendix I, Lemma 7. -/
example : 2 ≤ 3 := by decide

/-- Positive class weights cancel from ordinary sign descent;
Appendix I, Lemma 7. The field is constant even away from symmetric logits. -/
theorem signField_eq (hc : 2 ≤ c) (π : Fin c → ℝ) (hπ : ∀ k, 0 < π k)
    (W : Parameters c c) (i k : Fin c) :
    signField π W i k = if i = k then 1 else -1 := by
  have hp := softmax_strict_bounds hc (fun j => W j k) i
  unfold signField gradientField
  by_cases hi : i = k
  · subst i
    simp only [ite_true]
    exact Real.sign_of_pos (mul_pos (hπ k) (by linarith))
  · simp only [hi, ite_false]
    exact Real.sign_of_neg (mul_neg_of_pos_of_neg (hπ k) (by linarith))

/-- The assumptions used to cancel class weights are satisfiable; Lemma 7. -/
example : 2 ≤ 3 ∧ ∀ k : Fin 3, (0 : ℝ) < (fun _ : Fin 3 => (1 / 3 : ℝ)) k := by
  exact ⟨by decide, fun _ => by norm_num⟩

/-- Exact parameter trajectory for ordinary sign descent; Lemma 7. -/
def signParameters (c : ℕ) (t : ℝ) : Parameters c c :=
  fun i k => if i = k then t else -t

/-- Verification of the exact trajectory against the ordinary sign field;
Appendix I, Lemma 7. -/
theorem signParameters_flow (hc : 2 ≤ c) (π : Fin c → ℝ) (hπ : ∀ k, 0 < π k) :
    IsSignFlow π (signParameters c) := by
  constructor
  · ext i k
    simp [signParameters]
  · intro t i k
    rw [signField_eq hc π hπ]
    by_cases hi : i = k
    · simpa [signParameters, hi] using hasDerivAt_id' t
    · simpa [signParameters, hi] using hasDerivAt_neg t

/-- Nonvacuity of the sign-trajectory hypotheses; Lemma 7. -/
example : 2 ≤ 3 ∧ ∀ k : Fin 3, (0 : ℝ) < (fun _ : Fin 3 => (1 / 3 : ℝ)) k :=
  ⟨by decide, fun _ => by norm_num⟩

/-- Uniqueness of ordinary sign descent from zero; Appendix I, Lemma 7. -/
theorem signFlow_unique (hc : 2 ≤ c) (π : Fin c → ℝ) (hπ : ∀ k, 0 < π k)
    (W : ℝ → Parameters c c) (hW : IsSignFlow π W) : W = signParameters c := by
  funext t i k
  have hg := signParameters_flow hc π hπ
  have hd : ∀ s, HasDerivAt (fun r => W r i k - signParameters c r i k) 0 s := by
    intro s
    have h1 := hW.2 s i k
    have h2 := hg.2 s i k
    rw [signField_eq hc π hπ] at h1 h2
    simpa using h1.fun_sub h2
  have he := is_const_of_deriv_eq_zero (fun s => (hd s).differentiableAt)
    (fun s => (hd s).deriv) t 0
  have h0 : W 0 i k = 0 := by rw [hW.1]; rfl
  have he0 : W t i k - signParameters c t i k = 0 := by
    simpa [h0, signParameters] using he
  exact sub_eq_zero.mp he0

/-- Every assumption of sign-flow uniqueness has a witness; Lemma 7. -/
example : 2 ≤ 3 ∧ (∀ k : Fin 3, (0 : ℝ) < (fun _ : Fin 3 => (1 / 3 : ℝ)) k) ∧
    IsSignFlow (fun _ : Fin 3 => (1 / 3 : ℝ)) (signParameters 3) :=
  ⟨by decide, fun _ => by norm_num, signParameters_flow (by decide) _ (fun _ => by norm_num)⟩

/-- Corrected exact loss formula in Appendix I, Lemma 7:
`log(1+(c-1)exp(-2t))`. The source states `exp(-ct)`. -/
theorem signParameters_loss (k : Fin c) (t : ℝ) :
    sampleLoss (signParameters c t) (Pi.single k 1) k =
      marginLoss ((c : ℝ) - 1) (2 * t) := by
  rw [sampleLoss_basis]
  have he : (fun i => signParameters c t i k) = twoLevel (2 * t + -t) (-t) k := by
    funext i
    simp [signParameters, twoLevel, show 2 * t + -t = t by ring]
  rw [he, crossEntropy_twoLevel]

/-- Refutation of Appendix I, Lemma 7 at three classes and time one.
The actual loss is strictly greater than the paper's claimed value. -/
theorem paper_sign_loss_false :
    Real.log (1 + (3 - 1 : ℝ) * Real.exp (-(3 : ℝ))) <
      sampleLoss (signParameters 3 1) (Pi.single (0 : Fin 3) 1) 0 := by
  rw [signParameters_loss]
  unfold marginLoss
  norm_num only [Nat.cast_ofNat, sub_self, mul_one]
  apply Real.log_lt_log (by positivity)
  have he : Real.exp (-3 : ℝ) < Real.exp (-2 : ℝ) := Real.exp_lt_exp.2 (by norm_num)
  linarith

/-- Finite-time bounds behind the corrected sign-descent rate; Theorem 3
and Appendix I, Lemma 7. -/
theorem sign_loss_bounds (z : ℝ) (hz : 0 < z) (t : ℝ) (ht : 0 ≤ t) :
    (z / (z + 1)) * Real.exp (-(2 * t)) ≤ marginLoss z (2 * t) ∧
      marginLoss z (2 * t) ≤ z * Real.exp (-(2 * t)) := by
  have he := Real.exp_pos (-(2 * t))
  have he1 : Real.exp (-(2 * t)) ≤ 1 := Real.exp_le_one_iff.2 (by linarith)
  have hl := log_one_add_bounds (z * Real.exp (-(2 * t))) (by positivity)
  refine ⟨le_trans ?_ hl.1, hl.2⟩
  calc
    (z / (z + 1)) * Real.exp (-(2 * t)) = (z * Real.exp (-(2 * t))) / (z + 1) := by ring
    _ ≤ (z * Real.exp (-(2 * t))) / (1 + z * Real.exp (-(2 * t))) :=
      div_le_div_of_nonneg_left (by positivity) (by positivity) (by nlinarith)

/-- Nonvacuity of the finite-time sign-loss bound. -/
example : (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- Corrected exponential rate for the exact sign trajectory;
Theorem 3 and Appendix I, Lemma 7. -/
theorem sign_margin_rate (z : ℝ) (hz : 0 < z) :
    (fun t => marginLoss z (2 * t)) =Θ[atTop] (fun t => Real.exp (-(2 * t))) := by
  constructor
  · apply isBigO_iff.2
    refine ⟨z, ?_⟩
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    simpa only [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _),
      abs_of_nonneg (marginLoss_nonneg z _ hz.le)] using (sign_loss_bounds z hz t ht).2
  · apply isBigO_iff.2
    refine ⟨(z + 1) / z, ?_⟩
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _),
      abs_of_nonneg (marginLoss_nonneg z _ hz.le)]
    have ha : 0 < z / (z + 1) := by positivity
    calc
      Real.exp (-(2 * t)) ≤ marginLoss z (2 * t) / (z / (z + 1)) :=
        (le_div_iff₀ ha).2 (by simpa only [mul_comm] using (sign_loss_bounds z hz t ht).1)
      _ = (z + 1) / z * marginLoss z (2 * t) := by field_simp

/-- Nonvacuity of the positive class count in the corrected rate. -/
example : (0 : ℝ) < 2 := by norm_num

end Transformer.Imbalance
