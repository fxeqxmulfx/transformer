module

public import Transformer.ScheduleFree.SectionC_Bregman

/-!
# Pathwise online-to-batch conversion

arXiv:2405.15682v4, §2.1, Theorem 2, and Appendices A and C.
This is the deterministic part of the proof. The oracle can be any
supporting-subgradient selection. Random oracles are handled separately
by taking expectations and using their unbiased pairings.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix C: all four discarded Bregman terms at time `t`. -/
def defect (F : E → ℝ) (grad : E → E) (w α : ℕ → ℝ) (z : ℕ → E)
    (u : E) (t : ℕ) : ℝ :=
  w t * mixGap F grad (α t) (average w z (t + 1)) (z t) +
    weightSum w t * bregman F grad (average w z t) (average w z (t + 1)) +
    w t * bregman F grad u (query w α z t)

/-- Appendix C, Theorem 5: the exact telescoped identity, before expectation. -/
theorem conversion_identity (F : E → ℝ) (grad : E → E) (w α : ℕ → ℝ)
    (z : ℕ → E) (u : E) (T : ℕ)
    (hW : ∀ t < T, weightSum w (t + 1) ≠ 0) :
    weightSum w T * (F (average w z T) - F u) =
      ∑ t ∈ Finset.range T,
        (w t * inner ℝ (grad (query w α z t)) (z t - u) -
          defect F grad w α z u t) := by
  have hstep (t : ℕ) (ht : t ∈ Finset.range T) :
      weightSum w (t + 1) * (F (average w z (t + 1)) - F u) -
        weightSum w t * (F (average w z t) - F u) =
      w t * inner ℝ (grad (query w α z t)) (z t - u) - defect F grad w α z u t := by
    have hp : t = 0 ∨ weightSum w t ≠ 0 := by
      cases t with
      | zero => exact Or.inl rfl
      | succ t => exact Or.inr (hW t (by have := Finset.mem_range.mp ht; omega))
    have hb := average_balance w z t (hW t (Finset.mem_range.mp ht)) hp
    rw [weightSum_succ] at hb ⊢
    exact bregman_step F grad (α t) (weightSum w t) (w t)
      (average w z t) (average w z (t + 1)) (z t) u hb
  have hs := Finset.sum_congr rfl hstep
  rw [Finset.sum_range_sub (fun t => weightSum w t * (F (average w z t) - F u)) T] at hs
  simpa [weightSum_zero] using hs

example (F : E → ℝ) (grad : E → E) (α : ℕ → ℝ) (z : ℕ → E) (u : E) (T : ℕ) :
    weightSum (fun _ => 1) T * (F (average (fun _ => 1) z T) - F u) =
      ∑ t ∈ Finset.range T,
        (1 * inner ℝ (grad (query (fun _ => 1) α z t)) (z t - u) -
          defect F grad (fun _ => 1) α z u t) :=
  conversion_identity F grad _ α z u T
    (fun t _ => (weightSum_pos_succ (fun _ => zero_le_one) zero_lt_one t).ne')

/-- The signs needed for Theorem 2 follow from supporting subgradients. -/
theorem defect_nonneg {F : E → ℝ} {grad : E → E} {w α : ℕ → ℝ}
    (hgrad : ∀ x, IsSubgradient F x (grad x)) (hw : ∀ t, 0 ≤ w t)
    (hα : ∀ t, 0 ≤ α t ∧ α t ≤ 1) (z : ℕ → E) (u : E) (t : ℕ) :
    0 ≤ defect F grad w α z u t := by
  unfold defect
  exact add_nonneg
    (add_nonneg (mul_nonneg (hw t) (mixGap_nonneg (hα t).1 (hα t).2
      (hgrad _) (hgrad _)))
      (mul_nonneg (weightSum_nonneg hw t) (bregman_nonneg (hgrad _))))
    (mul_nonneg (hw t) (bregman_nonneg (hgrad _)))

example (z : ℕ → ℝ) (u : ℝ) (t : ℕ) :
    0 ≤ defect (fun _ => 0) (fun _ => 0) (fun _ => 1) (fun _ => 0) z u t :=
  defect_nonneg (by intro x v; simp) (fun _ => zero_le_one)
    (fun _ => ⟨le_rfl, zero_le_one⟩) z u t

/-- Theorem 2 / Appendix A: weighted error is bounded by linearized regret.
The endpoints `α=0` and `α=1` are included without a limiting argument. -/
theorem online_to_batch {F : E → ℝ} {grad : E → E} {w α : ℕ → ℝ}
    (hgrad : ∀ x, IsSubgradient F x (grad x)) (hw : ∀ t, 0 ≤ w t)
    (hfirst : 0 < w 0) (hα : ∀ t, 0 ≤ α t ∧ α t ≤ 1)
    (z : ℕ → E) (u : E) (T : ℕ) :
    weightSum w T * (F (average w z T) - F u) ≤
      ∑ t ∈ Finset.range T, w t * inner ℝ (grad (query w α z t)) (z t - u) := by
  rw [conversion_identity F grad w α z u T
    (fun t _ => (weightSum_pos_succ hw hfirst t).ne')]
  exact Finset.sum_le_sum fun t _ => sub_le_self _ (defect_nonneg hgrad hw hα z u t)

example (z : ℕ → ℝ) (u : ℝ) (T : ℕ) :
    weightSum (fun _ => 1) T * ((0 : ℝ) - 0) ≤
      ∑ t ∈ Finset.range T, (1 : ℝ) * inner ℝ (0 : ℝ) (z t - u) := by
  exact online_to_batch (F := fun _ => 0) (grad := fun _ => 0)
    (by intro x v; simp) (fun _ => zero_le_one) zero_lt_one
    (fun _ => ⟨le_rfl, zero_le_one⟩) z u T

end Transformer.ScheduleFree
