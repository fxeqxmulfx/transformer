module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.Queries

/-!
# Moving a feasible iterate to a point admitting complete tangent queries

Source: `review/convex/moments.md`, CM80. A positive mixture with the certified
centre opens every non-forced row while preserving all forced equalities.
The displacement and convex-loss cost are explicit: mixing is not claimed to
leave the iterate, its objective or the optimizer trajectory unchanged.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection

open scoped BigOperators

variable {D R : Type*} [Fintype D] [Fintype R]

/-- Mix complete feasible arrays before making forward queries, CM80. -/
def mixture (point middle : D → ℝ) (amount : ℝ) : D → ℝ :=
  fun d => (1 - amount) * point d + amount * middle d

/-- Every original row responds linearly to the complete mixture, CM80. -/
theorem dot_mixture (a point middle : D → ℝ) (amount : ℝ) :
    dot a (mixture point middle amount) =
      (1 - amount) * dot a point + amount * dot a middle := by
  simp only [dot, mixture, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro d hd
  ring

omit [Fintype R] in
/-- Mixtures with coefficients in the full unit interval remain feasible,
CM80. This uses all constraints without requiring strict feasibility. -/
theorem mixture_feasible {normal : R → D → ℝ} {point middle : D → ℝ}
    (hp : Feasible normal point) (hc : Feasible normal middle) {amount : ℝ}
    (ha : 0 ≤ amount) (hb : amount ≤ 1) : Feasible normal (mixture point middle amount) := by
  intro r
  rw [dot_mixture]
  exact add_nonneg (mul_nonneg (sub_nonneg.mpr hb) (hp r)) (mul_nonneg ha (hc r))

omit [Fintype R] in
/-- A positive centre mixture opens every row that can be strict anywhere,
CM80. The original iterate may lie on any boundary of the cone. -/
theorem mixture_positive {normal : R → D → ℝ} {point middle : D → ℝ}
    (hp : Feasible normal point) {amount : ℝ} (ha : 0 < amount) (hb : amount ≤ 1)
    (r : R) (hr : 0 < dot (normal r) middle) :
    0 < dot (normal r) (mixture point middle amount) := by
  rw [dot_mixture]
  have hn := mul_nonneg (sub_nonneg.mpr hb) (hp r)
  have ht := mul_pos ha hr
  linarith

/-- The exact complete-coordinate displacement caused by mixing, CM80. -/
theorem mixture_distance (point middle : D → ℝ) (amount : ℝ) :
    squareDistance point (mixture point middle amount) =
      amount ^ 2 * squareDistance point middle := by
  simp only [squareDistance, mixture, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro d hd
  ring

omit [Fintype R] in
/-- The actual convex objective can increase when an iterate is mixed with
the centre; this explicit bound records that cost, CM80. Convexity is a
premise on the chosen loss, not a claim about an arbitrary forward estimator. -/
theorem mixture_loss {normal : R → D → ℝ} {point middle : D → ℝ}
    (F : (D → ℝ) → ℝ) (hF : ConvexOn ℝ {x | Feasible normal x} F)
    (hp : Feasible normal point) (hc : Feasible normal middle) {amount : ℝ}
    (ha : 0 ≤ amount) (hb : amount ≤ 1) :
    F (mixture point middle amount) ≤ (1 - amount) * F point + amount * F middle := by
  exact hF.2 hp hc (sub_nonneg.mpr hb) ha (by ring)

/-- After any positive centre mixture, every direction respecting the
universally forced equalities admits two feasible queries, CM80. No parameter
coordinate or pattern is frozen merely because the old point was on a boundary. -/
theorem mixture_queries {normal projected : R → D → ℝ} {dual : R → R → ℝ}
    (h : ∀ r, Certificate normal (normal r) (projected r) (dual r))
    {point : D → ℝ} (hp : Feasible normal point) (direction : D → ℝ)
    (hz : ∀ r, (∀ z, Feasible normal z → dot (normal r) z = 0) →
      dot (normal r) direction = 0) {amount : ℝ} (ha : 0 < amount) (hb : amount ≤ 1) :
    ∃ step : ℝ, 0 < step ∧
      Feasible normal (shift (mixture point (centre projected) amount) direction step) ∧
      Feasible normal (shift (mixture point (centre projected) amount) direction (-step)) := by
  have hc := centre_feasible (fun r => (h r).2.1)
  apply exists_symmetric_queries direction (mixture_feasible hp hc ha.le hb)
  intro r hr
  have he : dot (normal r) (centre projected) = 0 := by
    have hn := hc r
    by_contra hne
    have hpos : 0 < dot (normal r) (centre projected) := lt_of_le_of_ne hn (Ne.symm hne)
    have hmix := mixture_positive hp ha hb r hpos
    linarith
  exact hz r ((centre_zero_iff h r).mp he)

example := mixture_feasible (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (point := 0) (middle := fun _ => 2) (amount := 1 / 2)
  (by norm_num [Feasible, dot]) (by norm_num [Feasible, dot]) (by norm_num) (by norm_num)
example := mixture_positive (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (point := 0) (middle := fun _ => 2) (amount := 1 / 2)
  (by norm_num [Feasible, dot]) (by norm_num) (by norm_num) () (by norm_num [dot])
example := mixture_loss (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (point := 0) (middle := fun _ => 2) (amount := 1 / 2) (fun x => x ())
  (by
    constructor
    · intro x hx y hy a b ha hb hab r
      have hx0 : 0 ≤ x () := by simpa [dot] using hx ()
      have hy0 : 0 ≤ y () := by simpa [dot] using hy ()
      simpa [dot] using add_nonneg (mul_nonneg ha hx0) (mul_nonneg hb hy0)
    · intro x hx y hy a b ha hb hab
      exact le_rfl)
  (by norm_num [Feasible, dot]) (by norm_num [Feasible, dot]) (by norm_num) (by norm_num)
example := mixture_queries (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (projected := fun _ => fun _ => 1) (dual := fun _ => 0) (point := 0) (amount := 1 / 2)
  (by norm_num [Certificate, Feasible, dot]) (by norm_num [Feasible, dot]) (fun _ => -3)
  (by
    intro r hz
    have he := hz (fun _ => 1) (by norm_num [Feasible, dot])
    norm_num [dot] at he)
  (by norm_num) (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection
