/-
# The 1D reduction behind the hull

The machine does not answer `argmax_k ⟪q, k⟫` by scanning the keys.  It keeps
two dynamic one-dimensional convex envelopes and queries them at a single
point.  The reduction that licenses this is stated, without proof, in
`transformer_vm/attention/hull2d_cht.h` (lines 11-17):

> For `q = (qx, qy)` with `qy ≠ 0`, maximizing `q · (kx, ky)` is equivalent to
>   maximizing `(kx * m + ky)` for `m = qx / qy`  if `qy > 0`
>   minimizing `(kx * m + ky)`                    if `qy < 0`

This file proves it, together with the degenerate branch `qy = 0` that the
same file handles separately by querying at `±∞`, and the consequence for the
lookup machine: under the paraboloid embedding the query's second coordinate
is always `1`, so only the upper envelope is ever consulted.

The scores themselves are in `Transformer.ALM.Defs`.
-/

import Transformer.ALM.Defs

namespace Transformer
namespace ALM

/-- A planar key `(kx, ky)`, read by the hull as the line `m ↦ kx · m + ky`:
the first coordinate is the slope, the second the intercept. -/
def lineEval (k : ℝ × ℝ) (m : ℝ) : ℝ := k.1 * m + k.2

/-- The hard-attention score `q · k` of the planar head. -/
def dot (q k : ℝ × ℝ) : ℝ := q.1 * k.1 + q.2 * k.2

/-- **The reduction.**  For `q.2 ≠ 0` the planar score is the line value at
`m = q.1 / q.2`, rescaled by `q.2`. -/
theorem dot_eq_mul_lineEval (q k : ℝ × ℝ) (hq : q.2 ≠ 0) :
    dot q k = q.2 * lineEval k (q.1 / q.2) := by
  unfold dot lineEval
  field_simp

/-- For `q.2 > 0` the rescaling is by a positive factor, so the planar order
and the line order agree: the hull maximizes. -/
theorem dot_le_dot_iff_of_pos {q : ℝ × ℝ} (hq : 0 < q.2) (k k' : ℝ × ℝ) :
    dot q k ≤ dot q k' ↔ lineEval k (q.1 / q.2) ≤ lineEval k' (q.1 / q.2) := by
  rw [dot_eq_mul_lineEval q k hq.ne', dot_eq_mul_lineEval q k' hq.ne']
  exact mul_le_mul_iff_right₀ hq

/-- For `q.2 < 0` the factor is negative and the order reverses: the hull
minimizes. -/
theorem dot_le_dot_iff_of_neg {q : ℝ × ℝ} (hq : q.2 < 0) (k k' : ℝ × ℝ) :
    dot q k ≤ dot q k' ↔ lineEval k' (q.1 / q.2) ≤ lineEval k (q.1 / q.2) := by
  rw [dot_eq_mul_lineEval q k hq.ne, dot_eq_mul_lineEval q k' hq.ne]
  exact mul_le_mul_left_of_neg hq

/-- The degenerate branch.  When `q.2 = 0` the intercept drops out and only the
slope matters, which is why querying the envelope at `±∞` is the right
answer there. -/
theorem dot_of_snd_eq_zero (q k : ℝ × ℝ) (hq : q.2 = 0) :
    dot q k = q.1 * k.1 := by
  unfold dot
  rw [hq]
  ring

/-- With `q.2 = 0` and `q.1 > 0` the planar argmax is the largest slope. -/
theorem dot_le_dot_iff_of_snd_eq_zero {q : ℝ × ℝ} (hq : q.2 = 0) (hq1 : 0 < q.1)
    (k k' : ℝ × ℝ) :
    dot q k ≤ dot q k' ↔ k.1 ≤ k'.1 := by
  rw [dot_of_snd_eq_zero q k hq, dot_of_snd_eq_zero q k' hq]
  exact mul_le_mul_iff_right₀ hq1

/-- The reduction, transported to a finite family of keys: `i₀` maximizes the
planar score exactly when it maximizes the line value at `q.1 / q.2`. -/
theorem isGreatest_dot_iff_of_pos {n : ℕ} {q : ℝ × ℝ} (hq : 0 < q.2)
    (K : Fin n → ℝ × ℝ) (i₀ : Fin n) :
    (∀ j, dot q (K j) ≤ dot q (K i₀)) ↔
      ∀ j, lineEval (K j) (q.1 / q.2) ≤ lineEval (K i₀) (q.1 / q.2) :=
  forall_congr' fun j => dot_le_dot_iff_of_pos hq (K j) (K i₀)

/-- And for `q.2 < 0`, against the lower envelope. -/
theorem isGreatest_dot_iff_of_neg {n : ℕ} {q : ℝ × ℝ} (hq : q.2 < 0)
    (K : Fin n → ℝ × ℝ) (i₀ : Fin n) :
    (∀ j, dot q (K j) ≤ dot q (K i₀)) ↔
      ∀ j, lineEval (K i₀) (q.1 / q.2) ≤ lineEval (K j) (q.1 / q.2) :=
  forall_congr' fun j => dot_le_dot_iff_of_neg hq (K j) (K i₀)

/-! ## The lookup machine only needs the upper envelope -/

/-- The paraboloid lift of a scalar key, `k ↦ (2k, -k²)`. -/
def liftKey (k : ℝ) : ℝ × ℝ := (2 * k, -k ^ 2)

/-- The lift of a scalar query, `q ↦ (q, 1)`. -/
def liftQuery (q : ℝ) : ℝ × ℝ := (q, 1)

/-- The lifted planar score is the scalar paraboloid score. -/
theorem dot_lift (q k : ℝ) : dot (liftQuery q) (liftKey k) = 2 * k * q - k ^ 2 := by
  unfold dot liftQuery liftKey
  ring

/-- On integer keys the lifted score is `sScore`, the score the `LookUp`
primitive compiles to. -/
theorem dot_lift_int (q k : ℤ) :
    dot (liftQuery (q : ℝ)) (liftKey (k : ℝ)) = sScore q k := by
  rw [dot_lift]
  rfl

/-- A lifted query always has second coordinate `1`. -/
theorem liftQuery_snd (q : ℝ) : (liftQuery q).2 = 1 := rfl

/-- Consequently the degenerate branch is unreachable for the lookup machine,
and its query point is the key itself, not a ratio. -/
theorem liftQuery_snd_pos (q : ℝ) : 0 < (liftQuery q).2 := by
  rw [liftQuery_snd]; norm_num

/-- **What the machine actually relies on.**  Under the paraboloid embedding
the planar argmax over keys is the argmax of the one-dimensional upper
envelope evaluated at `q`.  The lower envelope is never consulted, and no
division occurs: `q.1 / q.2 = q`. -/
theorem lookup_reduces_to_upper_envelope {n : ℕ} (q : ℝ) (K : Fin n → ℝ)
    (i₀ : Fin n) :
    (∀ j, dot (liftQuery q) (liftKey (K j)) ≤ dot (liftQuery q) (liftKey (K i₀))) ↔
      ∀ j, lineEval (liftKey (K j)) q ≤ lineEval (liftKey (K i₀)) q := by
  have h := isGreatest_dot_iff_of_pos (liftQuery_snd_pos q) (fun j => liftKey (K j)) i₀
  rwa [liftQuery_snd, div_one] at h

/-- The hypotheses of the reduction are satisfiable: a query with positive
second coordinate, two distinct keys, and a genuine winner. -/
example : (0 : ℝ) < (liftQuery 3).2 ∧
    dot (liftQuery 3) (liftKey 0) < dot (liftQuery 3) (liftKey 3) := by
  constructor
  · exact liftQuery_snd_pos 3
  · rw [dot_lift, dot_lift]; norm_num

end ALM
end Transformer
