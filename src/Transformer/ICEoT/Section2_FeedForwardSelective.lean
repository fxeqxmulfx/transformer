/-
# IC-EoT: state-only passthrough restrictions for autoregressive ICNNs

arXiv:2603.22095v2, §2.2.1, the Bünning et al. autoregressive variant.
Only state columns need non-negative passthrough weights for monotonicity
in the recursively substituted variables; fixed control columns may have
arbitrary signs. Single-step convexity is already proved in `fnnRun_convex`.
-/

import Transformer.ICEoT.Section2_FeedForward

noncomputable section

namespace Transformer.ICEoT

/-- The targeted state-monotonicity condition of §2.2.1, last paragraph.
The predicate identifies state columns; every other input column is held
fixed in the comparison. Control passthrough weights remain unrestricted. -/
theorem fnnRun_state_monotone {din : ℕ} (width : ℕ → ℕ)
    (p : (k : ℕ) → FCLayer din (width k) (width (k + 1)))
    (stateColumn : Fin din → Prop) (L : ℕ)
    (hp : ∀ k < L, Nonnegative (p k).hidden ∧
      (∀ r i, stateColumn i → 0 ≤ (p k).passthrough r i) ∧
      Monotone (p k).activation) :
    ∀ x y : Fin din → ℝ, (∀ i, stateColumn i → x i ≤ y i) →
      (∀ i, ¬ stateColumn i → x i = y i) → fnnRun width p L x ≤ fnnRun width p L y := by
  classical
  induction L with
  | zero => intro x y hx hc; exact le_rfl
  | succ L ih =>
    intro x y hx hc r
    have hprev := ih (fun k hk => hp k (by omega)) x y hx hc
    apply (hp L (by omega)).2.2
    apply add_le_add
    · exact affine_monotone (p L).hidden (fun _ => 0) id
        (hp L (by omega)).1 monotone_id hprev r
    · apply add_le_add _ le_rfl
      apply Finset.sum_le_sum
      intro i hi
      by_cases hs : stateColumn i
      · exact mul_le_mul_of_nonneg_left (hx i hs) ((hp L (by omega)).2.1 r i hs)
      · exact le_of_eq (congrArg (fun v => (p L).passthrough r i * v) (hc i hs))

/-- A real two-input layer with positive state weight and negative control
weight, witnessing the targeted restriction of §2.2.1. -/
def selectiveFCLayer : FCLayer 2 1 1 where
  hidden := fun _ _ => 1
  passthrough := fun _ i => if i = 0 then 1 else -1
  bias := fun _ => 0
  activation := relu

/-- The state-only conditions allow a strictly negative control coefficient;
§2.2.1, the autoregressive variant. -/
example : (∀ k : ℕ, k < 3 → Nonnegative ((fun _ : ℕ => selectiveFCLayer) k).hidden ∧
    (∀ r i, i = (0 : Fin 2) → 0 ≤ ((fun _ : ℕ => selectiveFCLayer) k).passthrough r i) ∧
    Monotone ((fun _ : ℕ => selectiveFCLayer) k).activation) ∧
    (∀ i : Fin 2, i = 0 → (0 : ℝ) ≤ 1) ∧
    (∀ i : Fin 2, i ≠ 0 → (0 : ℝ) = 0) ∧
    selectiveFCLayer.passthrough 0 1 = -1 := by
  refine ⟨?_, fun _ _ => zero_le_one, fun _ _ => rfl, ?_⟩
  · intro k hk
    refine ⟨fun _ _ => zero_le_one, ?_, relu_conditions.2.1⟩
    intro r i hi
    simp [selectiveFCLayer, hi]
  · norm_num [selectiveFCLayer]

end Transformer.ICEoT
