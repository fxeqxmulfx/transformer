import Transformer.Modes.Section3_MixedMoments
import Transformer.Modes.Section3_LogDeriv

/-!
# Homogeneous expressions in the moments of a law

The proof of `lem:eta` in arXiv:2412.09080v3, §5.3, bounds each cumulant by
the Euclidean moment of the same order. A mixed derivative of the logarithm
of the moment generating function is a finite sum of products of moments,
divided by powers of that generating function. This module records those
expressions, their ordinary product and inverse derivatives, and their
homogeneous degree in the sample coordinates.

The expressions do not define cumulants: `cumulantOf` remains the actual
mixed derivative from §3.1. The connection to that derivative is proved in
`Section3_CumulantCalculus` and `Section3_Edgeworth`. At the origin of a
probability law the denominator is one, and the degree controls the product
of moment bounds.
-/

open Real MeasureTheory Filter
open scoped Topology

namespace Transformer.Modes

/-- Finite algebraic expressions in mixed moments and the inverse MGF.
Source: arXiv:2412.09080v3, §3.1 and the proof of `lem:eta` in §5.3. -/
inductive MomentExpression where
  | moment (i j : ℕ)
  | inverseMGF
  | add (e f : MomentExpression)
  | mul (e f : MomentExpression)
  | scale (c : ℝ) (e : MomentExpression)

namespace MomentExpression

/-- Evaluate an expression at an actual array of mixed moments. -/
noncomputable def eval (M : ℕ → ℕ → ℝ) : MomentExpression → ℝ
  | moment i j => M i j
  | inverseMGF => (M 0 0)⁻¹
  | add e f => eval M e + eval M f
  | mul e f => eval M e * eval M f
  | scale c e => c * eval M e

/-- Sum and product bounds for the absolute coefficients, independent of
the law. Exponents affect the homogeneous degree instead. -/
noncomputable def size : MomentExpression → ℝ
  | moment _ _ => 1
  | inverseMGF => 1
  | add e f => size e + size f
  | mul e f => size e * size f
  | scale c e => |c| * size e

/-- Formal product and inverse differentiation in the chosen coordinate. -/
def diff (first : Bool) : MomentExpression → MomentExpression
  | moment i j => if first then moment (i + 1) j else moment i (j + 1)
  | inverseMGF => scale (-1)
      (mul (if first then moment 1 0 else moment 0 1) (mul inverseMGF inverseMGF))
  | add e f => add (diff first e) (diff first f)
  | mul e f => add (mul (diff first e) f) (mul e (diff first f))
  | scale c e => scale c (diff first e)

/-- Homogeneity in the sample coordinates; the inverse MGF has degree zero.
Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
inductive Degree : MomentExpression → ℕ → Prop where
  | moment (i j : ℕ) : Degree (moment i j) (i + j)
  | inverseMGF : Degree inverseMGF 0
  | add {e f : MomentExpression} {n : ℕ} : Degree e n → Degree f n → Degree (add e f) n
  | mul {e f : MomentExpression} {n m : ℕ} : Degree e n → Degree f m → Degree (mul e f) (n + m)
  | scale (c : ℝ) {e : MomentExpression} {n : ℕ} : Degree e n → Degree (scale c e) n

/-- The coefficient bound is nonnegative.
Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
theorem size_nonneg (e : MomentExpression) : 0 ≤ e.size := by
  induction e with
  | moment => exact zero_le_one
  | inverseMGF => exact zero_le_one
  | add e f he hf => exact add_nonneg he hf
  | mul e f he hf => exact mul_nonneg he hf
  | scale c e he => exact mul_nonneg (abs_nonneg c) he

/-- A derivative raises the total moment order by one.
Source: arXiv:2412.09080v3, §3.1 and `lem:eta`. -/
theorem Degree.diff {e : MomentExpression} {n : ℕ} (h : Degree e n) (first : Bool) :
    Degree (e.diff first) (n + 1) := by
  induction h with
  | moment i j =>
      cases first with
      | false => simpa [MomentExpression.diff, Nat.add_assoc] using Degree.moment i (j + 1)
      | true => simpa [MomentExpression.diff, Nat.add_assoc, Nat.add_comm,
          Nat.add_left_comm] using Degree.moment (i + 1) j
  | inverseMGF =>
      cases first with
      | false =>
          simpa [MomentExpression.diff] using Degree.scale (-1)
            (Degree.mul (Degree.moment 0 1) (Degree.mul Degree.inverseMGF Degree.inverseMGF))
      | true =>
          simpa [MomentExpression.diff] using Degree.scale (-1)
            (Degree.mul (Degree.moment 1 0) (Degree.mul Degree.inverseMGF Degree.inverseMGF))
  | add he hf ihe ihf => exact Degree.add ihe ihf
  | mul he hf ihe ihf =>
      exact Degree.add
        (by simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using Degree.mul ihe hf)
        (by simpa [Nat.add_assoc] using Degree.mul he ihf)
  | scale c he ihe => exact Degree.scale c ihe

example : Degree (moment 1 1) 2 := Degree.moment 1 1

/-- Products of moment bounds respect their total homogeneous order.
Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
theorem eval_bound {e : MomentExpression} {n s : ℕ} (hd : Degree e n)
    (hns : n ≤ s) (M : ℕ → ℕ → ℝ) (R : ℝ) (hR : 0 ≤ R) (hM0 : M 0 0 = 1)
    (hM : ∀ i j, i + j ≤ s → |M i j| ≤ R ^ (i + j)) :
    |e.eval M| ≤ e.size * R ^ n := by
  induction hd with
  | moment i j => simpa [eval, size] using hM i j hns
  | inverseMGF => simp [eval, size, hM0]
  | @add e f n he hf ihe ihf =>
      calc |(add e f).eval M| ≤ |e.eval M| + |f.eval M| := abs_add_le _ _
        _ ≤ e.size * R ^ n + f.size * R ^ n := add_le_add (ihe hns) (ihf hns)
        _ = (add e f).size * R ^ n := by simp [size]; ring
  | @mul e f n m he hf ihe ihf =>
      rw [eval, abs_mul]
      calc |e.eval M| * |f.eval M| ≤ (e.size * R ^ n) * (f.size * R ^ m) :=
          mul_le_mul (ihe (by omega)) (ihf (by omega)) (abs_nonneg _)
            (mul_nonneg e.size_nonneg (pow_nonneg hR _))
        _ = (mul e f).size * R ^ (n + m) := by rw [size, pow_add]; ring
  | @scale c e n he ihe =>
      rw [eval, abs_mul]
      calc |c| * |e.eval M| ≤ |c| * (e.size * R ^ n) :=
          mul_le_mul_of_nonneg_left (ihe hns) (abs_nonneg c)
        _ = (scale c e).size * R ^ n := by rw [size]; ring

example : Degree (mul (moment 1 0) (moment 0 1)) 2 ∧ (2 : ℕ) ≤ 2 ∧
    (0 : ℝ) ≤ 1 ∧ (fun _ _ : ℕ => (1 : ℝ)) 0 0 = 1 ∧
    (∀ i j : ℕ, i + j ≤ 2 → |(fun _ _ : ℕ => (1 : ℝ)) i j| ≤ 1 ^ (i + j)) := by
  exact ⟨Degree.mul (Degree.moment 1 0) (Degree.moment 0 1), le_rfl,
    zero_le_one, rfl, by simp⟩

/-- Formal differentiation computes the actual derivative whenever the
moment array has the ordinary moment derivatives and a nonzero MGF.
Source: arXiv:2412.09080v3, §3.1; product and inverse rules only. -/
theorem hasDerivAt_eval (e : MomentExpression) (first : Bool)
    (M : ℕ → ℕ → ℝ → ℝ) (x : ℝ)
    (hM : ∀ i j, HasDerivAt (M i j)
      (if first then M (i + 1) j x else M i (j + 1) x) x)
    (hne : M 0 0 x ≠ 0) :
    HasDerivAt (fun t => e.eval (fun i j => M i j t))
      ((e.diff first).eval (fun i j => M i j x)) x := by
  induction e with
  | moment i j =>
      cases first <;> simpa [eval, diff] using hM i j
  | inverseMGF =>
      convert (hM 0 0).inv hne using 1
      · rfl
      · cases first <;> simp [eval, diff, div_eq_mul_inv, pow_two]
  | add e f he hf => exact he.add hf
  | mul e f he hf => exact he.fun_mul hf
  | scale c e he => exact he.const_mul c

example : (∀ i j : ℕ, HasDerivAt (fun t : ℝ => exp t)
    (if true then (fun _ _ : ℕ => exp 0) (i + 1) j
      else (fun _ _ : ℕ => exp 0) i (j + 1)) 0) ∧ exp 0 ≠ 0 := by
  exact ⟨fun _ _ => by simpa using Real.hasDerivAt_exp (0 : ℝ), by simp⟩

end MomentExpression
end Transformer.Modes
