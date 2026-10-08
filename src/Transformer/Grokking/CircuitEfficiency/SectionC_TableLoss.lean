import Transformer.Grokking.NaiveLoss.Section4_LogitScaling
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# Actual finite-class CE of the fixed Gen/Mem tables

Source: Varma et al., arXiv:2309.02390v1, appendix C, equations
sim-gen-logits, sim-mem-logits, sim-overall-logits and the train/test
loss formulas. There are remaining+2 classes. Class zero is correct;
the successor of zero is the memorizing table's wrong test label.
The other remaining classes have zero logits. Circuit weights here
are the products of the source's two subweights, not learned tables.

Derive the stated losses from ordinary finite-class cross-entropy,
including the source's 113-class configuration. The penalty is kept
outside CE. Strict correctness compares all classes; this avoids
silently replacing the source's multiclass problem by a binary margin.
-/

namespace Transformer.Grokking.CircuitEfficiency

open scoped BigOperators

/-- Both fixed tables agree on every training label. Source:
arXiv:2309.02390v1, appendix C, sim-overall-logits on the training set. -/
def trainTableLogits (remaining : ℕ) (x y : ℝ) : Fin (remaining + 2) → ℝ :=
  fun k => if k = 0 then x + y else 0

/-- On test data Gen selects the correct class and Mem a distinct wrong
class. Source: arXiv:2309.02390v1, appendix C, sim-gen/mem-logits. -/
def heldoutTableLogits (remaining : ℕ) (x y : ℝ) : Fin (remaining + 2) → ℝ :=
  fun k => if k = 0 then x else if k = (0 : Fin (remaining + 1)).succ then y else 0

/-- Actual finite-class training CE as a function of total circuit score.
Source: arXiv:2309.02390v1, appendix C, train-loss formula without its penalty. -/
noncomputable def tableTrainCE (remaining : ℕ) (score : ℝ) : ℝ :=
  Transformer.Grokking.NaiveLoss.crossEntropy (trainTableLogits remaining score 0) 0

/-- Summing exponentials of the actual training table gives one active
class and q-1 zero classes. Source: arXiv:2309.02390v1, appendix C. -/
theorem train_table_exp_sum (remaining : ℕ) (x y : ℝ) :
    (∑ k, Real.exp (trainTableLogits remaining x y k)) =
      Real.exp (x + y) + (remaining : ℝ) + 1 := by
  rw [Fin.sum_univ_succ]
  simp [trainTableLogits, Fin.succ_ne_zero, add_assoc]

/-- Count all three kinds of test logits, including the remaining q-2
classes. Source: arXiv:2309.02390v1, appendix C, test-loss denominator. -/
theorem heldout_table_exp_sum (remaining : ℕ) (x y : ℝ) :
    (∑ k, Real.exp (heldoutTableLogits remaining x y k)) =
      Real.exp x + Real.exp y + (remaining : ℝ) := by
  rw [Fin.sum_univ_succ]
  simp only [heldoutTableLogits, ite_true, Fin.succ_ne_zero, ite_false, Fin.succ_inj]
  rw [Fin.sum_univ_succ]
  simp [Fin.succ_ne_zero, add_assoc]

/-- The source's training loss is exactly standard multiclass CE, rather
than an assigned surrogate. Source: arXiv:2309.02390v1, appendix C. -/
theorem table_train_ce_formula (remaining : ℕ) (score : ℝ) :
    tableTrainCE remaining score =
      Real.log (Real.exp score + (remaining : ℝ) + 1) - score := by
  unfold tableTrainCE
  rw [Transformer.Grokking.NaiveLoss.crossEntropy_eq_standard, train_table_exp_sum]
  simp [trainTableLogits]

/-- Actual held-out CE contains both competing weights and all other
classes. Source: arXiv:2309.02390v1, appendix C, test-loss formula. -/
theorem table_heldout_ce_formula (remaining : ℕ) (x y : ℝ) :
    Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining x y) 0 =
      Real.log (Real.exp x + Real.exp y + (remaining : ℝ)) - x := by
  rw [Transformer.Grokking.NaiveLoss.crossEntropy_eq_standard, heldout_table_exp_sum]
  simp [heldoutTableLogits]

/-- Differentiate actual multiclass training CE for every finite score.
Source: arXiv:2309.02390v1, appendix C, the train-loss formula;
the q-1 competitors remain in the derivative. -/
theorem table_train_ce_deriv (remaining : ℕ) (score : ℝ) :
    HasDerivAt (tableTrainCE remaining)
      (-((remaining : ℝ) + 1) / (Real.exp score + (remaining : ℝ) + 1)) score := by
  have hp : 0 < Real.exp score + (remaining : ℝ) + 1 := by positivity
  have hd := (((Real.hasDerivAt_exp score).add_const (remaining : ℝ)).add_const 1).log hp.ne'
  have hs := hd.sub (hasDerivAt_id score)
  convert hs using 1
  · funext t
    exact table_train_ce_formula remaining t
  · field_simp
    ring

/-- There is no finite stationary total score for unpenalized training CE.
Source: arXiv:2309.02390v1, appendix C, derived from the actual derivative. -/
theorem table_train_ce_slope_negative (remaining : ℕ) (score : ℝ) :
    -((remaining : ℝ) + 1) / (Real.exp score + (remaining : ℝ) + 1) < 0 := by
  have hn : 0 < (remaining : ℝ) + 1 := by positivity
  have hd : 0 < Real.exp score + (remaining : ℝ) + 1 := by positivity
  exact div_neg_of_neg_of_pos (neg_neg_of_pos hn) hd

/-- The source's q=113 case has initial CE derivative -112/113. Source:
arXiv:2309.02390v1, appendix C, Table simulation hyperparameters. -/
theorem source_113_class_initial_derivative :
    HasDerivAt (tableTrainCE 111) (-112 / 113) 0 := by
  have hd := table_train_ce_deriv 111 0
  norm_num at hd
  convert hd using 1
  ring

/-- At the all-tied initialization the actual loss is log q, including
the 113-class simulation. Source: arXiv:2309.02390v1, appendix C,
the two zero initial circuit products in Table simulation hyperparameters. -/
theorem table_train_ce_initial (remaining : ℕ) :
    tableTrainCE remaining 0 = Real.log ((remaining : ℝ) + 2) := by
  rw [table_train_ce_formula, Real.exp_zero, sub_zero]
  congr 1
  ring

example : tableTrainCE 111 0 = Real.log 113 := by
  have he := table_train_ce_initial 111
  norm_num at he
  exact he

/-- Training correctness is a positive total score, with a real competing
class even when q=2. Source: arXiv:2309.02390v1, appendix C, Gen/Mem tables. -/
theorem train_table_strict_correct_iff (remaining : ℕ) (x y : ℝ) :
    Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits remaining x y) 0 ↔
      0 < x + y := by
  constructor
  · intro h
    have hs := h (0 : Fin (remaining + 1)).succ (Fin.succ_ne_zero _)
    simpa [trainTableLogits, Fin.succ_ne_zero] using hs
  · intro hs k hk
    simpa [trainTableLogits, hk] using hs

/-- A positive Gen weight beating Mem beats every held-out class.
Source: arXiv:2309.02390v1, appendix C; positivity also handles q-2
zero logits. This sufficient criterion does not claim necessity for q=2. -/
theorem heldout_table_strict_correct (remaining : ℕ) (x y : ℝ)
    (hx : 0 < x) (hxy : y < x) :
    Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining x y) 0 := by
  intro k hk
  change (if k = 0 then x else if k = (0 : Fin (remaining + 1)).succ then y else 0) < x
  rw [ite_eq_right hk]
  by_cases hd : k = (0 : Fin (remaining + 1)).succ
  · rw [ite_eq_left hd]
    exact hxy
  · rw [ite_eq_right hd]
    exact hx

example : 0 < (2 : ℝ) ∧ (1 : ℝ) < 2 := by norm_num

/-- When Mem beats Gen, the actual wrong test class prevents correctness.
Source: arXiv:2309.02390v1, appendix C, the memorizing table's test label. -/
theorem heldout_table_incorrect (remaining : ℕ) (x y : ℝ) (hxy : x < y) :
    ¬ Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining x y) 0 := by
  intro h
  have hd := h (0 : Fin (remaining + 1)).succ (Fin.succ_ne_zero _)
  simp [heldoutTableLogits] at hd
  linarith

example : (1 : ℝ) < 2 := by norm_num

end Transformer.Grokking.CircuitEfficiency
