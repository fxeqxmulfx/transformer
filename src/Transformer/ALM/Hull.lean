/-
# The hull answers the lookup: joining the two halves of ALM

`Transformer.ALM.Query` proves that `lower_bound` returns a maximizer of
`lineEval` — a statement about lines in the plane.  `Transformer.ALM.Basic`
proves that `score` is maximized at the matching key — a statement about
`EucSpace m`.  Nothing connected the two, so the geometry of
`transformer_vm/attention/hull2d_cht.h` and the lookup it implements were
still separate theories.  This file joins them in dimension one, which is the
dimension the published construction runs in
(`Transformer.ALM.Defs`, `sScore`).

Two facts do the work.

* `score_eq_lineEval` — under the paraboloid lift the scalar score *is* a
  line value, evaluated at the query itself.
* `interX_liftKey` — two lifted keys cross exactly at their **midpoint**.

The midpoint formula has a consequence the implementation does not state:
the breakpoints of a sorted key list are automatically sorted, so the
invariant `hull2d_cht.h` maintains by an erase loop (lines 95-98) holds for
free, and `liftKey_not_dominated` shows the loop can never fire.  Lifted keys
are points of a strictly convex parabola, and every point of a strictly
convex curve is extreme.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 11-17 (the reduction)
and 95-98, 203-215 (the invariant and the query).
-/

import Transformer.ALM.Query
import Transformer.ALM.Basic

namespace Transformer
namespace ALM

/-! ### The scalar score is a line value -/

/-- **The bridge.**  In dimension one the paraboloid score of a key against a
query is the value at `q` of the line the hull stores for that key.  No
division occurs, because a lifted query has second coordinate `1`
(`liftQuery_snd`). -/
theorem score_eq_lineEval (q k : EucSpace 1) :
    score q k = lineEval (liftKey (k 0)) (q 0) := by
  have hinner : inner (𝕜 := ℝ) k q = k 0 * q 0 := by
    simp [PiLp.inner_apply, RCLike.inner_apply, mul_comm]
  have hnorm : ‖k‖ ^ 2 = (k 0) ^ 2 := by
    rw [norm_sq_eq_sum, Fin.sum_univ_one]
  unfold score lineEval liftKey
  rw [hinner, hnorm]
  ring

/-- The same statement for the integer scalar keys the `LookUp` primitive
compiles to: `sScore` is the lifted line value. -/
theorem sScore_eq_lineEval (q k : ℤ) :
    sScore q k = lineEval (liftKey (k : ℝ)) (q : ℝ) := by
  rw [← dot_lift_int, dot_eq_mul_lineEval _ _ (by rw [liftQuery_snd]; norm_num)]
  rw [liftQuery_snd, div_one]
  simp [liftQuery]

/-! ### Lifted keys cross at their midpoint -/

/-- **The breakpoint of two lifted keys is their midpoint.**  This is the
`isect` of `hull2d_cht.h` specialized to the paraboloid: the parabola
`-k²` makes the crossing abscissa `(k + k') / 2`, independent of anything
else in the instance. -/
theorem interX_liftKey {k k' : ℝ} (h : k ≠ k') :
    interX (liftKey k) (liftKey k') = (k + k') / 2 := by
  have hne : (2 : ℝ) * k - 2 * k' ≠ 0 := by
    intro hz
    exact h (by linarith)
  unfold interX liftKey
  simp only []
  rw [div_eq_iff hne]
  ring

/-- **The erase loop never fires.**  For three lifted keys in increasing
order the middle line is *somewhere* strictly above the upper envelope of the
outer two, so `add_line` never discards it: on the paraboloid every key stays
on the hull, and the binary search of `argmax` runs over all `n` of them. -/
theorem liftKey_not_dominated {k₁ k₂ k₃ : ℝ} (h₁₂ : k₁ < k₂) (h₂₃ : k₂ < k₃) :
    ¬ ∀ x, lineEval (liftKey k₂) x
        ≤ max (lineEval (liftKey k₁) x) (lineEval (liftKey k₃) x) := by
  have s₁₂ : (liftKey k₁).1 < (liftKey k₂).1 := by unfold liftKey; simp; linarith
  have s₂₃ : (liftKey k₂).1 < (liftKey k₃).1 := by unfold liftKey; simp; linarith
  rw [dominated_iff_le_at_interX s₁₂ s₂₃, ← interX_le_interX_iff s₁₂ s₂₃,
    interX_liftKey (ne_of_lt h₁₂), interX_liftKey (ne_of_lt h₂₃)]
  intro hle
  linarith

/-! ### The invariants of the query hold for a sorted key list -/

/-- A strictly increasing step relation is a monotone sequence on its range.
Needed to compare breakpoints at distant indices. -/
lemma le_of_step_lt {n : ℕ} {K : ℕ → ℝ} (hstep : ∀ j, j < n → K j < K (j + 1))
    {a b : ℕ} (hab : a ≤ b) (hb : b ≤ n) : K a ≤ K b := by
  induction b, hab using Nat.le_induction with
  | base => exact le_rfl
  | succ b hab ih => exact le_trans (ih (by omega)) (hstep b (by omega)).le

/-- **Slopes increase.**  Sorting the keys sorts the lines, because the lift
doubles the key. -/
lemma hslope_liftKey {n : ℕ} {K : ℕ → ℝ} (hstep : ∀ j, j < n → K j < K (j + 1)) :
    ∀ j, j < n → (liftKey (K j)).1 < (liftKey (K (j + 1))).1 := by
  intro j hj
  have := hstep j hj
  unfold liftKey
  simp only []
  linarith

/-- **Breakpoints increase, for free.**  Since the breakpoint is a midpoint,
sorting the keys already sorts the breakpoints: the invariant that
`hull2d_cht.h` maintains by erasing lines is automatic here. -/
lemma hbp_liftKey {n : ℕ} {K : ℕ → ℝ} (hstep : ∀ j, j < n → K j < K (j + 1)) :
    ∀ a b, a ≤ b → b < n →
      interX (liftKey (K a)) (liftKey (K (a + 1)))
        ≤ interX (liftKey (K b)) (liftKey (K (b + 1))) := by
  intro a b hab hb
  have ha : a < n := by omega
  rw [interX_liftKey (ne_of_lt (hstep a ha)), interX_liftKey (ne_of_lt (hstep b hb))]
  have h1 : K a ≤ K b := le_of_step_lt hstep hab (by omega)
  have h2 : K (a + 1) ≤ K (b + 1) := le_of_step_lt hstep (by omega) (by omega)
  linarith

/-! ### The query returns a true argmax of the score -/

/-- **What the machine computes is what the lookup asks for.**  Let the keys
be sorted, and let `i` be the index `lower_bound(q)` yields — every earlier
midpoint is at or left of `q`, and `i`'s own midpoint is at or right of it,
with `i = n` the `it == end` fallback.  Then key `i` maximizes the scalar
paraboloid score over the whole list.  A query sitting exactly on a midpoint
is allowed: there the two keys score the same, so either branch is right.

This is the statement `hull2d_cht.h` assumes and never proves. -/
theorem hull_isGreatest {n : ℕ} (K : ℕ → ℝ) (q : ℝ)
    (hstep : ∀ j, j < n → K j < K (j + 1))
    (i : ℕ) (hi : i ≤ n)
    (hlt : ∀ j, j < i → (K j + K (j + 1)) / 2 ≤ q)
    (hge : i < n → q ≤ (K i + K (i + 1)) / 2) :
    ∀ j ≤ n, lineEval (liftKey (K j)) q ≤ lineEval (liftKey (K i)) q := by
  refine lowerBound_isGreatest (n := n) (fun j => liftKey (K j)) q
    (hslope_liftKey hstep) (hbp_liftKey hstep) i hi (fun j hj => ?_) (fun hin => ?_)
  · rw [interX_liftKey (ne_of_lt (hstep j (by omega)))]
    exact hlt j hj
  · rw [interX_liftKey (ne_of_lt (hstep i hin))]
    exact hge hin

/-- **The lookup, end to end.**  The same conclusion stated for `score` on
`EucSpace 1`: the key the hull's binary search returns is an exact argmax of
the attention score, which is what `NNIndex.ans_isGreatest` demands. -/
theorem hull_isGreatest_score {n : ℕ} (Kv : ℕ → EucSpace 1) (q : EucSpace 1)
    (hstep : ∀ j, j < n → Kv j 0 < Kv (j + 1) 0)
    (i : ℕ) (hi : i ≤ n)
    (hlt : ∀ j, j < i → (Kv j 0 + Kv (j + 1) 0) / 2 ≤ q 0)
    (hge : i < n → q 0 ≤ (Kv i 0 + Kv (i + 1) 0) / 2) :
    ∀ j ≤ n, score q (Kv j) ≤ score q (Kv i) := by
  intro j hj
  rw [score_eq_lineEval, score_eq_lineEval]
  exact hull_isGreatest (fun j => Kv j 0) (q 0) hstep i hi hlt hge j hj

/-- The hypotheses are satisfiable and the conclusion has content: three keys
`0 < 1 < 2`, queried at `1.9`, where `lower_bound` lands on the last one and
the first key is strictly worse. -/
example :
    let K : ℕ → ℝ := fun j => (j : ℝ)
    (∀ j, j < 2 → K j < K (j + 1)) ∧
      (∀ j, j < 2 → (K j + K (j + 1)) / 2 ≤ (1.9 : ℝ)) ∧
      lineEval (liftKey (K 0)) 1.9 < lineEval (liftKey (K 2)) 1.9 := by
  refine ⟨fun j hj => ?_, fun j hj => ?_, ?_⟩
  · simp only []
    exact_mod_cast Nat.lt_succ_self j
  · interval_cases j <;> norm_num
  · norm_num [lineEval, liftKey]

end ALM
end Transformer
