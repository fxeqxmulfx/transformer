import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic

/-!
# Confidence-only loss reduction and unchanged decisions

Source: Prieto et al., arXiv:2501.04697v1, section 4.2, the definition of
naive loss minimization (NLM). These are finite-class facts about actual
logit vectors, not an assumption that GPTMini is positively homogeneous.
The equivalence with standard cross-entropy is proved before any loss claim.

The source describes loss reduction after reaching 100% training accuracy.
Strict reduction under scaling requires a nonconstant target-margin vector;
ties at the maximum are allowed, but an all-tied vector does not improve.
The precise sufficient conditions are stated rather than hidden in a loss
definition. Parameter updates realizing the scaling remain a separate task.
-/

namespace Transformer.Grokking.NaiveLoss

open scoped BigOperators

variable {C : Type*} [Fintype C]

/-- Unique correct argmax, expressed through every competing class.
Source: the decision-invariance part of NLM, arXiv:2501.04697v1, section 4.2. -/
def StrictCorrect (z : C → ℝ) (y : C) : Prop :=
  ∀ k, k ≠ y → z k < z y

/-- Finite-class cross-entropy in target-relative coordinates. Its equality
with `log (sum exp z) - z y` is proved below. Source: the cross-entropy used
in the NLM definition, arXiv:2501.04697v1, section 4.2. -/
noncomputable def crossEntropy (z : C → ℝ) (y : C) : ℝ :=
  Real.log (∑ k, Real.exp (z k - z y))

/-- Target centering is exactly standard cross-entropy; it does not insert
correctness into the loss. Source: arXiv:2501.04697v1, section 4.2. -/
theorem crossEntropy_eq_standard (z : C → ℝ) (y : C) :
    crossEntropy z y = Real.log (∑ k, Real.exp (z k)) - z y := by
  have hsum : 0 < ∑ k, Real.exp (z k) := by
    apply Finset.sum_pos
    · intro k hk
      exact Real.exp_pos (z k)
    · exact ⟨y, Finset.mem_univ y⟩
  unfold crossEntropy
  simp_rw [Real.exp_sub]
  rw [← Finset.sum_div, Real.log_div (ne_of_gt hsum) (ne_of_gt (Real.exp_pos (z y))),
    Real.log_exp]

example : crossEntropy (fun b : Bool => if b then 1 else 0) true =
    Real.log (∑ b : Bool, Real.exp (if b then 1 else 0)) - 1 := by
  exact crossEntropy_eq_standard _ true

omit [Fintype C] in
/-- A positive rescaling preserves the complete strict decision property.
Source: the unchanged-decision requirement in NLM, arXiv:2501.04697v1,
section 4.2. No architecture homogeneity is assumed. -/
theorem strictCorrect_scale_iff (z : C → ℝ) (y : C) (c : ℝ) (hc : 0 < c) :
    StrictCorrect (fun k => c * z k) y ↔ StrictCorrect z y := by
  constructor
  · intro h k hne
    exact lt_of_mul_lt_mul_left (h k hne) hc.le
  · intro h k hne
    exact mul_lt_mul_of_pos_left (h k hne) hc

example : (0 : ℝ) < 2 ∧ StrictCorrect (fun b : Bool => if b then 1 else 0) true := by
  refine ⟨by norm_num, ?_⟩
  intro k hne
  cases k
  · norm_num
  · exact False.elim (hne rfl)

/-- Common logit shifts leave finite-class cross-entropy unchanged. Source:
the softmax cross-entropy in arXiv:2501.04697v1, sections 3 and 4.2;
this justifies removing row shifts in the experimental endpoint probe. -/
theorem crossEntropy_shift (z : C → ℝ) (y : C) (b : ℝ) :
    crossEntropy (fun k => z k + b) y = crossEntropy z y := by
  unfold crossEntropy
  congr 1
  apply Finset.sum_congr rfl
  intro k hk
  congr 1
  ring

/-- Correct maximum margins and one strictly worse class make scale-up
strictly lower CE. Source: arXiv:2501.04697v1, section 4.2, NLM.
Correction to the informal 100%-accuracy wording: the target must be a
maximum and at least one target-relative margin must be strictly negative.
All-tied logits would otherwise be a counterexample to strict reduction. -/
theorem crossEntropy_scale_lt (z : C → ℝ) (y : C) (c : ℝ) (hc : 1 < c)
    (hmax : ∀ k, z k ≤ z y) (hgap : ∃ k, z k < z y) :
    crossEntropy (fun k => c * z k) y < crossEntropy z y := by
  have hle : ∀ k, Real.exp (c * z k - c * z y) ≤ Real.exp (z k - z y) := by
    intro k
    apply Real.exp_le_exp.mpr
    have h := mul_le_mul_of_nonpos_right hc.le (sub_nonpos.mpr (hmax k))
    nlinarith
  obtain ⟨j, hj⟩ := hgap
  have hlt : Real.exp (c * z j - c * z y) < Real.exp (z j - z y) := by
    apply Real.exp_lt_exp.mpr
    have h := mul_lt_mul_of_neg_right hc (sub_neg.mpr hj)
    nlinarith
  have hsum : (∑ k, Real.exp (c * z k - c * z y)) < ∑ k, Real.exp (z k - z y) := by
    apply Finset.sum_lt_sum
    · intro k hk
      exact hle k
    · exact ⟨j, Finset.mem_univ j, hlt⟩
  have hpos : 0 < ∑ k, Real.exp (c * z k - c * z y) := by
    apply Finset.sum_pos
    · intro k hk
      exact Real.exp_pos _
    · exact ⟨y, Finset.mem_univ y⟩
  exact Real.log_lt_log hpos hsum

example : (1 : ℝ) < 2 ∧
    (∀ b : Bool, (if b then (1 : ℝ) else 0) ≤ 1) ∧
    (∃ b : Bool, (if b then (1 : ℝ) else 0) < 1) := by
  refine ⟨by norm_num, ?_, false, by norm_num⟩
  intro b
  cases b <;> norm_num

/-- CE can decrease while the complete correct-decision property is
identical. Source: arXiv:2501.04697v1, section 4.2. A loss-only progress
signal cannot distinguish this mechanism from changed predictions. -/
theorem confidence_reduction_without_new_decisions
    (z : C → ℝ) (y : C) (c : ℝ) (hc : 1 < c)
    (hmax : ∀ k, z k ≤ z y) (hgap : ∃ k, z k < z y) :
    crossEntropy (fun k => c * z k) y < crossEntropy z y ∧
      (StrictCorrect (fun k => c * z k) y ↔ StrictCorrect z y) := by
  refine ⟨crossEntropy_scale_lt z y c hc hmax hgap, ?_⟩
  exact strictCorrect_scale_iff z y c (by linarith)

example : (1 : ℝ) < 3 ∧
    (∀ b : Bool, (if b then (1 : ℝ) else 0) ≤ 1) ∧
    (∃ b : Bool, (if b then (1 : ℝ) else 0) < 1) := by
  refine ⟨by norm_num, ?_, false, by norm_num⟩
  intro b
  cases b <;> norm_num

/-- All-tied logits give no CE improvement under scaling, even when a
fixed tie rule predicts a dataset's constant label correctly. Source:
arXiv:2501.04697v1, section 4.2, counterexample to strict reduction based
on 100% accuracy alone without a nonconstant margin vector. -/
theorem crossEntropy_scale_eq_of_flat (z : C → ℝ) (y : C) (c : ℝ)
    (hflat : ∀ k, z k = z y) :
    crossEntropy (fun k => c * z k) y = crossEntropy z y := by
  unfold crossEntropy
  congr 1
  apply Finset.sum_congr rfl
  intro k hk
  simp only [hflat, sub_self]

example : ∀ b : Bool, (if b then (1 : ℝ) else 1) = 1 := by
  intro b
  cases b <;> norm_num

end Transformer.Grokking.NaiveLoss
