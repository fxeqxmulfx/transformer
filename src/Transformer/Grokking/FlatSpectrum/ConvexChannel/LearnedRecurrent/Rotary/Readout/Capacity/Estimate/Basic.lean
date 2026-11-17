module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Gap

/-!
# Complete coefficient-error geometry under shared capacity

Source: `review/flat_spectrum.md`, CM105. Full complex support controls a
coefficient error on every component/class, including zero-weight components.
This is a bound on the actual discrepancy, not an assumed stochastic certificate.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate

open Pair Optimality
open scoped BigOperators
variable {J O : Type*} [Fintype J] [Fintype O]

/-- The complete contraction is linear in the coefficient table, CM105. -/
theorem cost_right_sub (u a h : J → O → State) :
    linearCost u (a - h) = linearCost u a - linearCost u h := by
  simp only [linearCost, Pi.sub_apply, map_sub, Finset.sum_sub_distrib]

/-- Negating every coefficient reverses its full contraction, CM105. -/
theorem cost_right_neg (u a : J → O → State) : linearCost u (-a) = -linearCost u a := by
  simp only [linearCost, Pi.neg_apply, map_neg, Finset.sum_neg_distrib]

/-- Complete component support satisfies the norm triangle inequality, CM105. -/
theorem block_add_le (a h : O → State) : blockSupport (a + h) ≤ blockSupport a + blockSupport h := by
  have ht := Finset.sum_le_sum (s := Finset.univ) (fun (c : O) _ => norm_add_le (a c) (h c))
  simpa only [blockSupport, Pi.add_apply, Finset.sum_add_distrib] using ht

variable [Nonempty J]

/-- The full shared support is symmetric, CM105. -/
theorem peak_neg (a : J → O → State) : peak (-a) = peak a := by
  simp only [peak, blockSupport, Pi.neg_apply, norm_neg]

/-- All components participate in the shared support triangle bound, CM105. -/
theorem peak_add_le (a h : J → O → State) : peak (a + h) ≤ peak a + peak h := by
  apply Finset.sup'_le
  intro j hj
  exact (block_add_le (a j) (h j)).trans
    (add_le_add (support_le_peak a j) (support_le_peak h j))

/-- The complete discrepancy has the same support in either orientation, CM105. -/
theorem peak_sub_symm (a h : J → O → State) : peak (a - h) = peak (h - a) := by
  simp only [peak, blockSupport, Pi.sub_apply, norm_sub_rev]

/-- Actual support is bounded by an estimated support plus full discrepancy, CM105. -/
theorem peak_le_add_error (a h : J → O → State) : peak a ≤ peak h + peak (a - h) := by
  have he : a = h + (a - h) := by abel
  calc
    peak a = peak (h + (a - h)) := congrArg peak he
    _ ≤ peak h + peak (a - h) := peak_add_le _ _

/-- Complete support perturbation is bounded without dropping a class or component, CM105. -/
theorem peak_difference (a h : J → O → State) : |peak a - peak h| ≤ peak (a - h) := by
  have h₁ := peak_le_add_error a h
  have h₂ := peak_le_add_error h a
  rw [peak_sub_symm h a] at h₂
  exact abs_le.mpr ⟨by linarith, by linarith⟩

variable [Nonempty O]

/-- The upper complete contraction bound follows from CM104 at the negative coefficient, CM105. -/
theorem cost_upper (B : ℝ) (u a : J → O → State) (hu : u ∈ jointDomain B) :
    linearCost u a ≤ B * peak a := by
  have h := full_cost_lower B u (-a) hu
  rw [cost_right_neg, peak_neg] at h
  linarith

/-- Shared capacity controls the absolute complete contraction, CM105. -/
theorem cost_abs (B : ℝ) (u a : J → O → State) (hu : u ∈ jointDomain B) :
    |linearCost u a| ≤ B * peak a :=
  abs_le.mpr ⟨full_cost_lower B u a hu, cost_upper B u a hu⟩

/-- A feasible displacement has twice the complete support radius, CM105. -/
theorem displacement_abs (B : ℝ) (u v a : J → O → State)
    (hu : u ∈ jointDomain B) (hv : v ∈ jointDomain B) :
    |linearCost (v - u) a| ≤ 2 * B * peak a := by
  have h₁ := full_cost_lower B u a hu
  have h₂ := cost_upper B u a hu
  have h₃ := full_cost_lower B v a hv
  have h₄ := cost_upper B v a hv
  rw [linearCost_sub]
  exact abs_le.mpr ⟨by linarith, by linarith⟩

/-- Error of a current complete contraction uses the actual coefficient difference, CM105. -/
theorem coefficient_error (B : ℝ) (u a h : J → O → State) (hu : u ∈ jointDomain B) :
    |linearCost u a - linearCost u h| ≤ B * peak (a - h) := by
  rw [← cost_right_sub]
  exact cost_abs B u (a - h) hu

/-- Actual and estimated directional contractions differ by at most the full
shared-capacity diameter times coefficient discrepancy, CM105. -/
theorem direction_error (B : ℝ) (u v a h : J → O → State)
    (hu : u ∈ jointDomain B) (hv : v ∈ jointDomain B) :
    |linearCost (v - u) a - linearCost (v - u) h| ≤ 2 * B * peak (a - h) := by
  rw [← cost_right_sub]
  exact displacement_abs B u v (a - h) hu hv

example : linearCost (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0))
    (fun _ _ => (Complex.I, 2)) ≤ 16 * peak (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 2)) :=
  cost_upper _ _ _ (by change size _ ≤ 16; simp [size, block_const, rowSize])

example : |linearCost (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0))
    (fun _ _ => (Complex.I, 2))| ≤ 16 * peak (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 2)) :=
  cost_abs _ _ _ (by change size _ ≤ 16; simp [size, block_const, rowSize])

example : |linearCost ((fun (_ : Fin 1) (_ : Bool) => ((0, Complex.I) : State)) -
    (fun _ _ => (Complex.I, 0))) (fun _ _ => (Complex.I, 2))| ≤
      2 * 16 * peak (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 2)) :=
  displacement_abs _ _ _ _ (by change size _ ≤ 16; simp [size, block_const, rowSize])
    (by change size _ ≤ 16; simp [size, block_const, rowSize])

example : |linearCost (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0))
    (fun _ _ => (Complex.I, 2)) - linearCost (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0))
      (fun _ _ => (0, Complex.I))| ≤
        16 * peak ((fun (_ : Fin 1) (_ : Bool) => ((Complex.I, 2) : State)) -
          (fun _ _ => (0, Complex.I))) :=
  coefficient_error _ _ _ _ (by change size _ ≤ 16; simp [size, block_const, rowSize])

example : |linearCost ((fun (_ : Fin 1) (_ : Bool) => ((0, Complex.I) : State)) -
    (fun _ _ => (Complex.I, 0))) (fun _ _ => (Complex.I, 2)) -
      linearCost ((fun (_ : Fin 1) (_ : Bool) => ((0, Complex.I) : State)) -
        (fun _ _ => (Complex.I, 0))) (fun _ _ => (0, Complex.I))| ≤
    2 * 16 * peak ((fun (_ : Fin 1) (_ : Bool) => ((Complex.I, 2) : State)) -
      (fun _ _ => (0, Complex.I))) :=
  direction_error _ _ _ _ _ (by change size _ ≤ 16; simp [size, block_const, rowSize])
    (by change size _ ≤ 16; simp [size, block_const, rowSize])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate
