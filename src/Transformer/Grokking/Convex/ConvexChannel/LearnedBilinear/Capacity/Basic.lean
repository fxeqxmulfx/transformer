module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Forms

/-!
# Finite capacity pays for a nonzero identification residual

Source: `review/convex/moments.md`, CM91. Observed lower margins and a complete
squared coordinate budget yield a sufficient unseen-margin certificate even
when the query is not a nonnegative combination of observed forms. The budget
is an additional model constraint, not a consequence of flatness or fitting.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Capacity

open scoped BigOperators

variable {Z I : Type*} [Fintype Z] [Fintype I]

/-- Complete squared real-coordinate size, CM91. No coordinate is removed. -/
def energy (p : Z → ℝ) : ℝ := ∑ z, p z ^ 2

/-- Complete finite linear reading, CM91. -/
def reading (v p : Z → ℝ) : ℝ := ∑ z, v z * p z

/-- The part of a query form not supplied by observed forms, CM91. -/
def residual (A : I → Z → ℝ) (q : Z → ℝ) (w : I → ℝ) : Z → ℝ :=
  fun z => q z - ∑ i, w i * A i z

/-- Squared capacity is nonnegative, CM91. -/
theorem energy_nonneg (p : Z → ℝ) : 0 ≤ energy p :=
  Finset.sum_nonneg (fun z _ => sq_nonneg (p z))

/-- A complete coordinate ball is convex, CM91. This constrains magnitude,
not rank or an unobserved-answer table. Negative budgets give the empty set. -/
theorem energyBall_convex (B : ℝ) : Convex ℝ {p : Z → ℝ | energy p ≤ B} := by
  intro x hx y hy a b ha hb hab
  change energy (a • x + b • y) ≤ B
  have hs : energy (a • x + b • y) ≤ a * energy x + b * energy y := by
    unfold energy
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_le_sum
    intro z hz
    change (a * x z + b * y z) ^ 2 ≤ a * x z ^ 2 + b * y z ^ 2
    have hid : a * x z ^ 2 + b * y z ^ 2 - (a * x z + b * y z) ^ 2 =
        a * b * (x z - y z) ^ 2 := by
      nlinarith [congrArg (fun t : ℝ => t * (a * x z ^ 2 + b * y z ^ 2)) hab]
    have hn := mul_nonneg (mul_nonneg ha hb) (sq_nonneg (x z - y z))
    linarith
  have hax := mul_le_mul_of_nonneg_left hx ha
  have hby := mul_le_mul_of_nonneg_left hy hb
  have he : a * B + b * B = B := by rw [← add_mul, hab, one_mul]
  linarith

/-- Cauchy--Schwarz uses every coordinate, CM91. Squared arithmetic avoids
an unchecked square root in an exact rational certificate. -/
theorem reading_sq_le (v p : Z → ℝ) :
    reading v p ^ 2 ≤ energy v * energy p := by
  simpa [reading, energy] using Finset.sum_mul_sq_le_sq_mul_sq Finset.univ v p

/-- The residual identity retains every indexed observation, CM91. -/
theorem residual_identity (A : I → Z → ℝ) (q p : Z → ℝ) (w : I → ℝ) :
    reading q p = reading (residual A q w) p + ∑ i, w i * reading (A i) p := by
  simp only [reading, residual, sub_mul, Finset.sum_sub_distrib, Finset.sum_mul,
    Finset.mul_sum, mul_assoc]
  rw [Finset.sum_comm]
  ring

/-- Bounded-capacity transfer, CM91. The nonnegative observed combination
need not equal the query. Its full residual is charged against the complete
capacity budget. This is sufficient certification, not optimizer convergence
or a converse duality theorem. Lower margins may include explicit errors. -/
theorem bounded_transfer (A : I → Z → ℝ) (q p : Z → ℝ) (w lower : I → ℝ)
    (B beta : ℝ) (hw : ∀ i, 0 ≤ w i)
    (hfit : ∀ i, lower i ≤ reading (A i) p) (hcap : energy p ≤ B)
    (hslack : 0 ≤ (∑ i, w i * lower i) - beta)
    (hres : B * energy (residual A q w) ≤ ((∑ i, w i * lower i) - beta) ^ 2) :
    beta ≤ reading q p := by
  have hs : ∑ i, w i * lower i ≤ ∑ i, w i * reading (A i) p :=
    Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (hfit i) (hw i))
  have hc := reading_sq_le (residual A q w) p
  have he := mul_le_mul_of_nonneg_left hcap (energy_nonneg (residual A q w))
  have hr : -((∑ i, w i * lower i) - beta) ≤ reading (residual A q w) p := by
    nlinarith
  rw [residual_identity A q p w]
  linarith

example : Convex ℝ {p : Fin 2 → ℝ | energy p ≤ 1} := energyBall_convex 1

example : (1 : ℝ) ≤ reading (fun _ : Fin 1 => 1) (fun _ => 1) := by
  apply bounded_transfer (fun (_ : Fin 1) (_ : Fin 1) => 1) (fun _ => 1)
    (fun _ => 1) (fun _ => 2) (fun _ => 1) 1 1
  all_goals norm_num [reading, energy, residual]

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Capacity
