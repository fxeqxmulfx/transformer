module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection.Centre

/-!
# Feasible two-sided forward queries, including degenerate cones

Source: `review/convex/moments.md`, CM80. All original row bounds determine a
positive query radius. At the certified centre the admissible directions are
exactly the kernel of the universally forced equalities. This proves domain
safety, not finite-difference accuracy, a gradient identity or convergence.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection

open scoped BigOperators

variable {D R : Type*} [Fintype D] [Fintype R]

/-- Every finite collection of nonnegative slacks admits a positive common
step when each zero slack has zero directional response, CM80. The proof
takes a minimum over all rows, including the zero-slack ones. -/
theorem finite_query_radius (slack slope : R → ℝ) (hn : ∀ r, 0 ≤ slack r)
    (hz : ∀ r, slack r = 0 → slope r = 0) :
    ∃ step : ℝ, 0 < step ∧ ∀ r, step * |slope r| ≤ slack r := by
  classical
  have hf (s : Finset R) :
      ∃ step : ℝ, 0 < step ∧ ∀ r ∈ s, step * |slope r| ≤ slack r := by
    induction s using Finset.induction_on with
    | empty => exact ⟨1, by norm_num, by simp⟩
    | @insert r s _ ih =>
      rcases ih with ⟨step, hp, hb⟩
      by_cases he : slack r = 0
      · refine ⟨step, hp, ?_⟩
        intro i hi
        rcases Finset.mem_insert.mp hi with hi | hi
        · subst i
          rw [hz r he, abs_zero, mul_zero, he]
        · exact hb i hi
      · have hs : 0 < slack r := lt_of_le_of_ne (hn r) (Ne.symm he)
        have hd : 0 < |slope r| + 1 := by have := abs_nonneg (slope r); linarith
        let bound := slack r / (|slope r| + 1)
        have hbound : 0 < bound := div_pos hs hd
        have hmin : 0 < min step bound := lt_min hp hbound
        refine ⟨min step bound, hmin, ?_⟩
        intro i hi
        rcases Finset.mem_insert.mp hi with hi | hi
        · subst i
          have hm : min step bound * (|slope r| + 1) ≤ slack r :=
            (le_div_iff₀ hd).mp (min_le_right step bound)
          nlinarith
        · exact (mul_le_mul_of_nonneg_right (min_le_left step bound)
            (abs_nonneg (slope i))).trans (hb i hi)
  rcases hf Finset.univ with ⟨step, hp, hb⟩
  exact ⟨step, hp, fun r => hb r (Finset.mem_univ r)⟩

/-- A query changes all supplied coordinates of the direction, CM80. -/
def shift (point direction : D → ℝ) (step : ℝ) : D → ℝ :=
  fun d => point d + step * direction d

/-- Exact row response of a forward query, CM80. -/
theorem dot_shift (a point direction : D → ℝ) (step : ℝ) :
    dot a (shift point direction step) = dot a point + step * dot a direction := by
  simp only [dot, shift, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro d hd
  ring

omit [Fintype R] in
/-- Checking the complete absolute directional bound certifies both query
points, CM80. It does not substitute a gated forward outside the domain. -/
theorem symmetric_queries {normal : R → D → ℝ} (point direction : D → ℝ) (step : ℝ)
    (hp : 0 ≤ step)
    (hb : ∀ r, step * |dot (normal r) direction| ≤ dot (normal r) point) :
    Feasible normal (shift point direction step) ∧
      Feasible normal (shift point direction (-step)) := by
  constructor
  · intro r
    rw [dot_shift]
    have hm := mul_le_mul_of_nonneg_left (neg_abs_le (dot (normal r) direction)) hp
    have hr := hb r
    nlinarith
  · intro r
    rw [dot_shift]
    have hm := mul_le_mul_of_nonneg_left (le_abs_self (dot (normal r) direction)) hp
    have hr := hb r
    nlinarith

/-- A direction tangent to every currently tight row admits a positive
two-sided query radius, CM80. No full-dimensional-cone premise is needed. -/
theorem exists_symmetric_queries {normal : R → D → ℝ} {point : D → ℝ}
    (direction : D → ℝ) (hp : Feasible normal point)
    (hz : ∀ r, dot (normal r) point = 0 → dot (normal r) direction = 0) :
    ∃ step : ℝ, 0 < step ∧ Feasible normal (shift point direction step) ∧
      Feasible normal (shift point direction (-step)) := by
  rcases finite_query_radius (fun r => dot (normal r) point)
    (fun r => dot (normal r) direction) hp hz with ⟨step, hs, hb⟩
  exact ⟨step, hs, symmetric_queries point direction step hs.le hb⟩

/-- At the certified centre, two-sided query directions are exactly those
annihilated by every forced row, CM80. Necessity uses global row equality;
sufficiency retains every original inequality in the radius bound. -/
theorem centre_query_iff {normal projected : R → D → ℝ} {dual : R → R → ℝ}
    (h : ∀ r, Certificate normal (normal r) (projected r) (dual r))
    (direction : D → ℝ) :
    (∀ r, (∀ z, Feasible normal z → dot (normal r) z = 0) →
      dot (normal r) direction = 0) ↔
    ∃ step : ℝ, 0 < step ∧ Feasible normal (shift (centre projected) direction step) ∧
      Feasible normal (shift (centre projected) direction (-step)) := by
  constructor
  · intro hz
    apply exists_symmetric_queries direction (centre_feasible (fun r => (h r).2.1))
    intro r hr
    exact hz r ((centre_zero_iff h r).mp hr)
  · rintro ⟨step, hs, hp, _⟩ r hz
    have hc := hz (centre projected) (centre_feasible (fun s => (h s).2.1))
    have hq := hz _ hp
    rw [dot_shift] at hq
    nlinarith

/-- Opposed diagonal rows leave a line of feasible directions, CM80. -/
def diagonalNormal (row coordinate : Bool) : ℝ := if row = coordinate then 1 else -1

/-- Every normal of the diagonal equality cone projects to zero, CM80.
The nonzero multiplier sits on the opposite original row. -/
theorem diagonal_normal_certificates (r : Bool) :
    Certificate diagonalNormal (diagonalNormal r) 0 (fun s => if s = r then 0 else 1) := by
  cases r <;> norm_num [Certificate, Feasible, dot, diagonalNormal, Fintype.sum_bool]

/-- Both original coordinates can move together in a degenerate cone even
though neither coordinate can move alone, CM80. The query direction is nonzero
and the centre is zero; no cone or parameter coordinate is discarded. -/
theorem diagonal_centre_queries :
    ∃ step : ℝ, 0 < step ∧
      Feasible diagonalNormal (shift (centre (fun _ : Bool => 0)) (fun _ => 1) step) ∧
      Feasible diagonalNormal (shift (centre (fun _ : Bool => 0)) (fun _ => 1) (-step)) := by
  apply (centre_query_iff diagonal_normal_certificates (fun _ => 1)).mp
  intro r hz
  cases r <;> norm_num [dot, diagonalNormal, Fintype.sum_bool]

/-- In the same degenerate example, either nonzero isolated coordinate
query is infeasible, CM80. Joint feasible movement is therefore not detected
by accepting only individual-coordinate perturbations. -/
theorem diagonal_coordinate_refused (coordinate : Bool) (step : ℝ) (hs : step ≠ 0) :
    ¬Feasible diagonalNormal (fun d => if d = coordinate then step else 0) := by
  intro h
  have ht := h true
  have hf := h false
  apply hs
  cases coordinate <;> norm_num [dot, diagonalNormal, Fintype.sum_bool] at ht hf <;> linarith

example := finite_query_radius (R := Bool) (fun r => if r then 0 else 2)
  (fun r => if r then 0 else -3) (by intro r; cases r <;> norm_num)
  (by intro r; cases r <;> norm_num)
example := symmetric_queries (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (fun _ => 2) (fun _ => -3) (1 / 2) (by norm_num) (by norm_num [dot])
example := exists_symmetric_queries (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (point := fun _ => 2) (fun _ => -3) (by norm_num [Feasible, dot]) (by norm_num [dot])
example := centre_query_iff (normal := fun _ : Unit => fun _ : Unit => (1 : ℝ))
  (projected := fun _ => fun _ => 1) (dual := fun _ => 0)
  (by norm_num [Certificate, Feasible, dot]) (fun _ => -3)
example := diagonal_coordinate_refused true 1 (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection
