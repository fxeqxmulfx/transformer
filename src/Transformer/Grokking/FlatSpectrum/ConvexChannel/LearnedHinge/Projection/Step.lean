module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection.Basic

/-!
# Certified projected steps and their explicit descent condition

Source: `review/convex/moments.md`, CM79. A checked cone projection of a
gradient proposal gives a quantitative work inequality. If the actual loss
has the stated quadratic upper bound on that step, a small enough rate
guarantees descent. The upper bound is an explicit premise, not an asserted
smoothness or convergence property of an unimplemented optimizer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection

open scoped BigOperators

variable {D R : Type*} [Fintype D] [Fintype R]

/-- The proposal uses every supplied direction coordinate, CM79. -/
def proposal (x g : D → ℝ) (rate : ℝ) : D → ℝ := fun d => x d - rate * g d

/-- The exact residual identity for a projected proposal, CM79. -/
theorem step_identity (x y g : D → ℝ) (rate : ℝ) :
    dot (y - proposal x g rate) (x - y) =
      -squareDistance x y - rate * dot g (y - x) := by
  simp only [dot, squareDistance, proposal, Pi.sub_apply, Finset.mul_sum,
    ← Finset.sum_neg_distrib, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro d hd
  ring

/-- Certified projection makes the proposed direction pay for the complete
step distance, CM79. A positive rate is needed only for subsequent division. -/
theorem projected_work {normal : R → D → ℝ} {x y g : D → ℝ} {dual : R → ℝ} (rate : ℝ)
    (hx : Feasible normal x) (h : Certificate normal (proposal x g rate) y dual) :
    rate * dot g (y - x) ≤ -squareDistance x y := by
  have hv := certificate_variational h hx
  rw [step_identity] at hv
  linarith

/-- With the stated actual-loss quadratic upper bound, the certified step
decreases loss quantitatively, CM79. This does not infer the bound from CE
convexity or flatness alone. -/
theorem projected_descent {normal : R → D → ℝ} {x y g : D → ℝ} {dual : R → ℝ}
    (F : (D → ℝ) → ℝ) (rate L : ℝ) (hr : 0 < rate) (hL : rate * L ≤ 1)
    (hx : Feasible normal x) (h : Certificate normal (proposal x g rate) y dual)
    (hupper : F y ≤ F x + dot g (y - x) + L / 2 * squareDistance x y) :
    rate * F y ≤ rate * F x - squareDistance x y / 2 := by
  have hw := projected_work rate hx h
  have hd := squareDistance_nonneg x y
  have hu := mul_le_mul_of_nonneg_left hupper hr.le
  have hl := mul_le_mul_of_nonneg_right hL hd
  nlinarith

example := projected_work (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (x := fun _ => 1) (y := fun _ => 0) (g := fun _ => 2) (dual := fun _ => 1) 1
  (by norm_num [Feasible, dot]) (by norm_num [Certificate, Feasible, proposal, dot])
example := projected_descent (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (x := fun _ => 1) (y := fun _ => 0) (g := fun _ => 2) (dual := fun _ => 1)
  (fun z => 2 * z ()) 1 0 (by norm_num) (by norm_num)
  (by norm_num [Feasible, dot]) (by norm_num [Certificate, Feasible, proposal, dot])
  (by norm_num [dot])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection
