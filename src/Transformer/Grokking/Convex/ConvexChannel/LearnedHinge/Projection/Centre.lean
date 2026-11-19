module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.Geometry

/-!
# A centre with slack on every non-forced activation constraint

Source: `review/convex/moments.md`, CM80. Sum the certified projections of all
original normals. Its zero-slack rows are exactly the universally forced
equalities. This supplies geometry for feasible forward-only queries without
discarding degenerate patterns or replacing the actual model forward.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection

open scoped BigOperators

variable {D R : Type*} [Fintype D] [Fintype R]

/-- The centre retains the projection of every row, including zero ones,
CM80. Feasibility and strictness are proved separately from this definition. -/
def centre (projected : R → D → ℝ) : D → ℝ := fun d => ∑ r, projected r d

/-- A row's centre response is the sum of all its projected responses, CM80. -/
theorem dot_centre (a : D → ℝ) (projected : R → D → ℝ) :
    dot a (centre projected) = ∑ r, dot a (projected r) := by
  simp only [dot, centre, Finset.mul_sum]
  exact Finset.sum_comm

/-- Summing every feasible row projection stays feasible, CM80. -/
theorem centre_feasible {normal projected : R → D → ℝ}
    (h : ∀ r, Feasible normal (projected r)) : Feasible normal (centre projected) := by
  intro r
  rw [dot_centre]
  exact Finset.sum_nonneg (fun s _ => h s r)

/-- The constructed centre detects exactly the rows forced to zero on the
whole cone, CM80. Checking only its own projected normal suffices in the
forward implication because every term in the centre response is nonnegative. -/
theorem centre_zero_iff {normal projected : R → D → ℝ} {dual : R → R → ℝ}
    (h : ∀ r, Certificate normal (normal r) (projected r) (dual r)) (r : R) :
    dot (normal r) (centre projected) = 0 ↔
      ∀ z, Feasible normal z → dot (normal r) z = 0 := by
  constructor
  · intro hz
    rw [dot_centre] at hz
    have he := (Finset.sum_eq_zero_iff_of_nonneg
      (fun s (_ : s ∈ Finset.univ) => (h s).2.1 r)).mp hz
    have hy : projected r = 0 := by
      apply (squareDistance_zero_iff (projected r)).mp
      rw [← certificate_energy (h r)]
      exact he r (Finset.mem_univ r)
    exact (normal_projection_zero_iff r (h r)).mp hy
  · intro hz
    exact hz _ (centre_feasible (fun s => (h s).2.1))

/-- The centre has positive slack exactly when that row can have positive
slack anywhere in the cone, CM80. No strict-interior assumption is made. -/
theorem centre_positive_iff {normal projected : R → D → ℝ} {dual : R → R → ℝ}
    (h : ∀ r, Certificate normal (normal r) (projected r) (dual r)) (r : R) :
    0 < dot (normal r) (centre projected) ↔
      ∃ z, Feasible normal z ∧ 0 < dot (normal r) z := by
  have hc := centre_feasible (fun s => (h s).2.1)
  constructor
  · intro hp
    exact ⟨centre projected, hc, hp⟩
  · rintro ⟨z, hz, hp⟩
    by_contra hn
    have he : dot (normal r) (centre projected) = 0 := by
      have hnonneg := hc r
      linarith
    have hz0 := (centre_zero_iff h r).mp he z hz
    linarith

example := centre_feasible (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (projected := fun _ => fun _ => 1) (by norm_num [Feasible, dot])
example := centre_zero_iff (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (projected := fun _ => fun _ => 1) (dual := fun _ => 0)
  (by norm_num [Certificate, Feasible, dot]) ()
example := centre_positive_iff (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (projected := fun _ => fun _ => 1) (dual := fun _ => 0)
  (by norm_num [Certificate, Feasible, dot]) ()

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection
