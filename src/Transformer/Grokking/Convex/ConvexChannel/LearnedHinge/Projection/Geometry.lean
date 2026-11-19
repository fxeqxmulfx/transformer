module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.Basic

/-!
# Certified detection of universally forced activation equalities

Source: `review/convex/moments.md`, CM80. Project each original constraint normal
onto the complete cone. A zero result certifies that its row vanishes at every
feasible point; a nonzero result supplies positive slack. All conclusions use
the finite CM79 certificate, with no assumed projection oracle or row deletion.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection

open scoped BigOperators

variable {D R : Type*} [Fintype D] [Fintype R]

/-- Zero Euclidean energy means every original coordinate is zero, CM80. -/
theorem squareDistance_zero_iff (y : D → ℝ) : squareDistance 0 y = 0 ↔ y = 0 := by
  constructor
  · intro h
    have he := (Finset.sum_eq_zero_iff_of_nonneg
      (fun d (_ : d ∈ Finset.univ) => sq_nonneg ((0 : D → ℝ) d - y d))).mp h
    funext d
    have hd := sq_eq_zero_iff.mp (he d (Finset.mem_univ d))
    simpa using hd
  · intro h
    subst y
    simp [squareDistance]

/-- Complementarity makes the projection residual orthogonal to the accepted
point itself, CM80. The statement also covers degenerate cones. -/
theorem certificate_orthogonal {normal : R → D → ℝ} {v y : D → ℝ} {dual : R → ℝ}
    (h : Certificate normal v y dual) : dot (y - v) y = 0 := by
  have he : dot (y - v) y = ∑ r, dual r * dot (normal r) y := by
    simp only [dot, Pi.sub_apply]
    simp_rw [h.2.2.2, add_sub_cancel_left]
    simp only [Finset.sum_mul]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro r hr
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro d hd
    ring
  rw [he]
  exact Finset.sum_eq_zero (fun r _ => h.2.2.1 r)

/-- The response to the proposal is the complete squared projection norm,
CM80. Applied to an original normal, it measures a feasible row's slack. -/
theorem certificate_energy {normal : R → D → ℝ} {v y : D → ℝ} {dual : R → ℝ}
    (h : Certificate normal v y dual) : dot v y = squareDistance 0 y := by
  have ho := certificate_orthogonal h
  have he : dot (y - v) y = squareDistance 0 y - dot v y := by
    simp only [dot, squareDistance, Pi.sub_apply, Pi.zero_apply,
      ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro d hd
    ring
  rw [he] at ho
  linarith

/-- Projecting an original normal to zero is equivalent to its constraint
being an equality throughout the cone, CM80. This is a global certificate,
not a conclusion from sampled feasible points or coordinate probes. -/
theorem normal_projection_zero_iff {normal : R → D → ℝ} (r : R)
    {y : D → ℝ} {dual : R → ℝ} (h : Certificate normal (normal r) y dual) :
    y = 0 ↔ ∀ z, Feasible normal z → dot (normal r) z = 0 := by
  constructor
  · intro hy z hz
    have hv := certificate_variational h hz
    rw [hy] at hv
    have hn : 0 ≤ -dot (normal r) z := by
      simpa [dot, Pi.sub_apply, Finset.sum_neg_distrib] using hv
    have hp := hz r
    linarith
  · intro hz
    apply (squareDistance_zero_iff y).mp
    rw [← certificate_energy h]
    exact hz y h.2.1

/-- Nonzero opposed original normals project to zero in the degenerate
cone containing only zero, CM80. Both rows and their multipliers are kept. -/
theorem opposed_normal_certificate :
    Certificate (fun b : Bool => fun _ : Unit => if b then 1 else -1)
      (fun _ => 1) 0 (fun b => if b then 0 else 1) := by
  norm_num [Certificate, Feasible, dot, Fintype.sum_bool]

example := certificate_orthogonal halfline_certificate
example := certificate_energy halfline_certificate
example := normal_projection_zero_iff true opposed_normal_certificate

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection
