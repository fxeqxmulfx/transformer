module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming

/-!
# Finite exact certificates for projection onto the learned activation cones

Source: `review/flat_spectrum.md`, CM79. Feasibility, nonnegative multipliers,
complementarity and stationarity certify a Euclidean projection. These are
finite arithmetic checks suitable for an exact rational native checker. No
optimizer oracle, omitted inequality or approximately feasible point is assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection

open scoped BigOperators

variable {D R : Type*} [Fintype D] [Fintype R]

/-- The complete real coordinate inner product used by the update, CM79. -/
def dot (a b : D → ℝ) : ℝ := ∑ d, a d * b d

/-- Squared Euclidean distance of all coordinates, CM79. This update metric
is not the representation-recovery norm or a new factor regularizer. -/
def squareDistance (a b : D → ℝ) : ℝ := ∑ d, (a d - b d) ^ 2

/-- Every original oriented activation inequality is retained, CM79. -/
def Feasible (normal : R → D → ℝ) (point : D → ℝ) : Prop :=
  ∀ r, 0 ≤ dot (normal r) point

/-- Checkable projection conditions, CM79. This predicate describes only
the supplied arrays; it does not contain a presumed optimality conclusion. -/
def Certificate (normal : R → D → ℝ) (proposal point : D → ℝ) (dual : R → ℝ) : Prop :=
  (∀ r, 0 ≤ dual r) ∧ Feasible normal point ∧
  (∀ r, dual r * dot (normal r) point = 0) ∧
  ∀ d, point d = proposal d + ∑ r, dual r * normal r d

/-- The update distance is nonnegative without dropping any coordinate, CM79. -/
theorem squareDistance_nonneg (a b : D → ℝ) : 0 ≤ squareDistance a b :=
  Finset.sum_nonneg (fun d _ => sq_nonneg (a d - b d))

/-- The exact distance decomposition used to check projection optimality,
CM79. No convexity or stationarity premise is hidden in this identity. -/
theorem distance_identity (v y z : D → ℝ) :
    squareDistance v z = squareDistance v y + squareDistance y z +
      2 * dot (y - v) (z - y) := by
  simp only [squareDistance, dot, Pi.sub_apply, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro d hd
  ring

/-- The certificate makes the residual point into the cone at every other
feasible point, CM79. All constraints, including redundant ones, are used. -/
theorem certificate_variational {normal : R → D → ℝ} {v y : D → ℝ} {dual : R → ℝ}
    (h : Certificate normal v y dual) {z : D → ℝ} (hz : Feasible normal z) :
    0 ≤ dot (y - v) (z - y) := by
  have he : dot (y - v) (z - y) =
      ∑ r, dual r * (dot (normal r) z - dot (normal r) y) := by
    simp only [dot, Pi.sub_apply]
    simp_rw [h.2.2.2, add_sub_cancel_left]
    simp only [Finset.sum_mul]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro r hr
    rw [← Finset.sum_sub_distrib, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro d hd
    ring
  rw [he]
  apply Finset.sum_nonneg
  intro r hr
  rw [mul_sub, h.2.2.1 r, sub_zero]
  exact mul_nonneg (h.1 r) (hz r)

/-- A checked certificate proves the strong projection inequality, CM79.
The retained distance term rules out a different equally close feasible point. -/
theorem certificate_distance {normal : R → D → ℝ} {v y : D → ℝ} {dual : R → ℝ}
    (h : Certificate normal v y dual) {z : D → ℝ} (hz : Feasible normal z) :
    squareDistance v y + squareDistance y z ≤ squareDistance v z := by
  have hv := certificate_variational h hz
  rw [distance_identity v y z]
  linarith

/-- The finite certificate proves global nearest-point optimality over the
whole cone, CM79, rather than checking only enumerated competitors. -/
theorem certificate_nearest {normal : R → D → ℝ} {v y : D → ℝ} {dual : R → ℝ}
    (h : Certificate normal v y dual) {z : D → ℝ} (hz : Feasible normal z) :
    squareDistance v y ≤ squareDistance v z := by
  have hb := certificate_distance h hz
  have hn := squareDistance_nonneg y z
  linarith

/-- A feasible point projects to itself with zero multipliers, CM79. -/
theorem identity_certificate (normal : R → D → ℝ) (v : D → ℝ) (hv : Feasible normal v) :
    Certificate normal v v 0 := by
  exact ⟨by simp, hv, by simp, by simp⟩

/-- A nonidentity one-dimensional projection inhabits all certificate
premises, CM79: proposal -1 projects to 0 in the nonnegative half-line. -/
theorem halfline_certificate :
    Certificate (fun _ : Unit => fun _ : Unit => 1) (fun _ => -1) (fun _ => 0) (fun _ => 1) := by
  norm_num [Certificate, Feasible, dot]

example := certificate_variational halfline_certificate
  (z := fun _ => 3) (by norm_num [Feasible, dot])
example := certificate_distance halfline_certificate
  (z := fun _ => 3) (by norm_num [Feasible, dot])
example := certificate_nearest halfline_certificate
  (z := fun _ => 3) (by norm_num [Feasible, dot])
example := identity_certificate (fun _ : Unit => fun _ : Unit => (1 : ℝ)) (fun _ => 2)
  (by norm_num [Feasible, dot])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection
