module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Complete

/-!
# Merge learned neurons with identical activation constraints

Source: `review/convex/moments.md`, CM76. On the entire declared input region,
neurons with the same sign pattern add exactly. Every original vector enters
the sum. This is a quotient preserving predictions, not recovery of the old
individual factors or preservation of an arbitrary parameter regularizer.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction

open scoped BigOperators

variable {E X J C : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable [Fintype X] [DecidableEq X] [Fintype J]

/-- One scalar component pattern on every admitted input, CM76. -/
abbrev ScalarPattern (X : Type*) := X → Bool

/-- Retain all old slot values by summing those with the same pattern, CM76.
The class and output sign coordinates are not merged. -/
noncomputable def collapse (mask : J → ScalarPattern X) (p : Parameters J C E) :
    Parameters (ScalarPattern X) C E := by
  classical
  exact fun m c s => ∑ j, if mask j = m then p j c s else 0

/-- The two sign orientations commute with a finite sum, CM76. -/
theorem orient_sum {K : Type*} [Fintype K] (active : Bool) (v : K → ℝ) :
    orient active (∑ k, v k) = ∑ k, orient active (v k) := by
  cases active <;> simp [orient]

/-- The two constrained branches commute with a finite sum, CM76. -/
theorem gated_sum {K : Type*} [Fintype K] (active : Bool) (v : K → ℝ) :
    gated active (∑ k, v k) = ∑ k, gated active (v k) := by
  cases active <;> simp [gated]

omit [DecidableEq X] in
/-- Every merged coordinate satisfies the same input-only constraints,
CM76. This uses all old coordinates, including zero and negative ones. -/
theorem collapse_mem (input : X → E →ₗ[ℝ] ℝ) (region : Set X)
    (mask : J → ScalarPattern X) {p : Parameters J C E}
    (hp : p ∈ domain input region mask) :
    collapse mask p ∈ domain input region (fun m x => m x) := by
  classical
  intro m c s x hx
  change 0 ≤ orient (m x) (input x (collapse mask p m c s))
  rw [collapse, map_sum, orient_sum]
  apply Finset.sum_nonneg
  intro j hj
  by_cases h : mask j = m
  · simpa [h] using hp j c s x hx
  · simp [h, orient]

/-- Regrouping keeps the complete constrained linear expression, CM76.
This algebraic identity is not an assumption about an unconstrained ReLU. -/
theorem gated_collapse (mask : J → ScalarPattern X) (p : Parameters J C E)
    (input : E →ₗ[ℝ] ℝ) (x : X) (c : C) (s : Bool) :
    (∑ m, gated (m x) (input (collapse mask p m c s))) =
      ∑ j, gated (mask j x) (input (p j c s)) := by
  classical
  calc
    _ = ∑ m, ∑ j, if mask j = m then gated (mask j x) (input (p j c s)) else 0 := by
      apply Finset.sum_congr rfl
      intro m hm
      rw [collapse, map_sum, gated_sum]
      apply Finset.sum_congr rfl
      intro j hj
      by_cases h : mask j = m
      · simp [h]
      · simp [h, gated]
    _ = ∑ j, ∑ m, if mask j = m then gated (mask j x) (input (p j c s)) else 0 :=
      Finset.sum_comm
    _ = _ := by simp

/-- Actual learned-factor inference is unchanged at every constrained input,
CM76. Equality outside the declared region is deliberately not asserted. -/
theorem forward_collapse (input : X → E →ₗ[ℝ] ℝ) (region : Set X)
    (mask : J → ScalarPattern X) {p : Parameters J C E}
    (hp : p ∈ domain input region mask) {x : X} (hx : x ∈ region) :
    forward (collapse mask p) (input x) = forward p (input x) := by
  rw [forward_eq input region (fun m x => m x) (collapse_mem input region mask hp) hx,
    forward_eq input region mask hp hx]
  funext c
  simp only [gatedForward, Finset.sum_sub_distrib, gated_collapse]

/-- The remaining pattern index has one Boolean choice per input, CM76. -/
theorem scalar_pattern_count (X : Type*) [Fintype X] [DecidableEq X] :
    Fintype.card (ScalarPattern X) = 2 ^ Fintype.card X := by
  simp [ScalarPattern]

example := collapse_mem (E := ℝ) (X := Bool) (J := Bool) (C := Bool)
  (fun _ => LinearMap.id) Set.univ (fun _ _ => true)
  (p := fun j _ _ => if j then 2 else 1)
  (by intro j c s x hx; cases j <;> norm_num [orient])
example := forward_collapse (E := ℝ) (X := Bool) (J := Bool) (C := Bool)
  (fun _ => LinearMap.id) Set.univ (fun _ _ => true)
  (p := fun j _ _ => if j then 2 else 1)
  (by intro j c s x hx; cases j <;> norm_num [orient]) (Set.mem_univ false)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction
