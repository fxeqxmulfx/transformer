/-
# The simple imbalanced setting

arXiv:2402.19449v2, Section 3.3 and Appendix I.
Every sample of class k is the basis vector e_k. Frequencies weight an
actual cross-entropy objective. Rows are classes and columns are inputs.
-/

import Transformer.Imbalance.Section3_Derivatives
import Transformer.Imbalance.Section3_LossBounds
import Mathlib.Basic.Real.Sign

open scoped BigOperators

noncomputable section

namespace Transformer.Imbalance

variable {c : ℕ}

/-- Weighted cross-entropy on the class basis vectors; Section 3.3. -/
def simpleLoss (π : Fin c → ℝ) (W : Parameters c c) : ℝ :=
  ∑ k, π k * sampleLoss W (Pi.single k 1) k

/-- Gradient-flow vector field in the simple imbalanced setting;
Appendix I, Lemma 4. This uses the corrected cross-entropy gradient. -/
def gradientField (π : Fin c → ℝ) (W : Parameters c c) : Parameters c c :=
  fun i k => π k * ((if i = k then 1 else 0) - Perspective.softmaxWeight (fun j => W j k) i)

/-- Ordinary componentwise sign-descent field; Appendix A.6's update rule
and Appendix I, Lemma 7. No dimension-dependent normalization is applied. -/
def signField (π : Fin c → ℝ) (W : Parameters c c) : Parameters c c :=
  fun i k => Real.sign (gradientField π W i k)

/-- A global gradient-flow trajectory initialized at zero; Theorem 3.
The explicit solution below exists on all real times. -/
def IsGradientFlow (π : Fin c → ℝ) (W : ℝ → Parameters c c) : Prop :=
  W 0 = 0 ∧ ∀ t i k, HasDerivAt (fun s => W s i k) (gradientField π (W t) i k) t

/-- A global sign-descent trajectory initialized at zero; Theorem 3. -/
def IsSignFlow (π : Fin c → ℝ) (W : ℝ → Parameters c c) : Prop :=
  W 0 = 0 ∧ ∀ t i k, HasDerivAt (fun s => W s i k) (signField π (W t) i k) t

/-- Basis-vector inputs select matrix columns; Appendix I, Lemma 4. -/
theorem scores_basis (W : Parameters c c) (k i : Fin c) :
    scores W (Pi.single k 1) i = W i k := by
  simp [scores, Pi.single_apply, mul_ite]

/-- Class-k loss in the simple setting is the cross-entropy of column k;
Section 3.3, Theorem 3. -/
theorem sampleLoss_basis (W : Parameters c c) (k : Fin c) :
    sampleLoss W (Pi.single k 1) k = crossEntropy (fun j => W j k) k := by
  unfold sampleLoss
  congr 1
  funext i
  exact scores_basis W k i

/-- Column logits with one correct value and equal incorrect values;
Appendix I, Lemma 4. -/
def twoLevel (a b : ℝ) (k : Fin c) : Fin c → ℝ := fun i => if i = k then a else b

/-- Partition function in the two-dimensional reduction of Lemma 4. -/
theorem sum_exp_twoLevel (a b : ℝ) (k : Fin c) :
    (∑ i, Real.exp (twoLevel a b k i)) = Real.exp a + ((c : ℝ) - 1) * Real.exp b := by
  have he : ∀ i, Real.exp (twoLevel a b k i) =
      Real.exp b + if i = k then Real.exp a - Real.exp b else 0 := by
    intro i
    by_cases hi : i = k <;> simp [twoLevel, hi]
  simp_rw [he]
  simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  ring

/-- Softmax in terms of the margin `u=a-b`; Appendix I, Lemma 4. -/
theorem softmax_twoLevel (u b : ℝ) (k i : Fin c) :
    Perspective.softmaxWeight (twoLevel (u + b) b k) i =
      (if i = k then Real.exp u else 1) / (Real.exp u + ((c : ℝ) - 1)) := by
  have hc : (1 : ℝ) ≤ c := by
    exact_mod_cast (Nat.succ_le_of_lt (lt_of_le_of_lt (Nat.zero_le k.val) k.isLt))
  have hz : 0 ≤ (c : ℝ) - 1 := by linarith
  have hden : Real.exp u + ((c : ℝ) - 1) ≠ 0 := by positivity
  unfold Perspective.softmaxWeight
  rw [sum_exp_twoLevel]
  by_cases hi : i = k <;> simp only [twoLevel, hi, ite_true, ite_false, Real.exp_add]
  all_goals field_simp [Real.exp_ne_zero, hden]

/-- Exact loss of the reduced column; Appendix I, Lemma 6. -/
theorem crossEntropy_twoLevel (u b : ℝ) (k : Fin c) :
    crossEntropy (twoLevel (u + b) b k) k = marginLoss ((c : ℝ) - 1) u := by
  have hc : (1 : ℝ) ≤ c := by
    exact_mod_cast (Nat.succ_le_of_lt (lt_of_le_of_lt (Nat.zero_le k.val) k.isLt))
  have hz : 0 ≤ (c : ℝ) - 1 := by linarith
  have he : Real.exp u + ((c : ℝ) - 1) =
      Real.exp u * (1 + ((c : ℝ) - 1) * Real.exp (-u)) := by
    rw [Real.exp_neg]
    field_simp
  unfold crossEntropy
  rw [softmax_twoLevel, ite_eq_left rfl, he]
  rw [mul_comm, div_mul_cancel_right₀ (Real.exp_ne_zero u), Real.log_inv]
  simp [marginLoss]

end Transformer.Imbalance
