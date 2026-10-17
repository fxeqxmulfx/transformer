/-
# Saddle avoidance — the cone estimate

Let `A` be a linear map with an expanding splitting `P` (`IsExpandingSplitting
A P c`, `c > 1`), and let `g` differ from `A` on the closed ball `B̄(z, r)` by a
map of Lipschitz constant `(c - 1)/4`.  Call a difference `v` *steep* when
`‖v - P v‖ ≤ ‖P v‖`.  If `x - y` is steep and `x, y ∈ B̄(z, r)`, then
`g x - g y` is steep and `‖P (g x - g y)‖ ≥ (c + 1)/2 · ‖P (x - y)‖`
(`cone_step`).  Two starts whose orbits both stay in the ball therefore never
differ by a nonzero steep vector (`eq_of_mem_stayNear_of_steep`): the steep
part of their difference would grow like `((c + 1)/2)ᵏ` while the ball bounds
it.  In particular, `stayNear g z r` meets each line in a direction `e` with
`P e = e` at most once (`line_meets_stayNear_once`).

This is the graph property of the local center-stable set: `stayNear g z r`
is the graph of a Lipschitz map over the range of `1 - P`.  The center-stable
manifold theorem (Shub, Theorem III.7, cited as `thm:stable-center-manifold`
in arXiv:1710.07406) gives more, a `C¹` disk tangent to that range, but the
proof of saddle avoidance uses only that the set is null.
-/

import Transformer.SaddleAvoidance.Basic

open Metric

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **One step of the cone estimate.**  If `v` is steep, `‖v - P v‖ ≤ ‖P v‖`, and
`w` differs from `A v` by at most `(c - 1)/4 · ‖v‖`, then `w` is steep and its
steep part is at least `(c + 1)/2` times that of `v`. -/
theorem cone_step {A P : E →ₗ[ℝ] E} {c : ℝ} (hP : IsExpandingSplitting A P c) (hc : 1 < c)
    {v w : E} (hv : ‖v - P v‖ ≤ ‖P v‖) (hw : ‖w - A v‖ ≤ (c - 1) / 4 * ‖v‖) :
    ‖w - P w‖ ≤ ‖P w‖ ∧ (c + 1) / 2 * ‖P v‖ ≤ ‖P w‖ := by
  obtain ⟨R, hR⟩ : ∃ R, R = w - A v := ⟨_, rfl⟩
  rw [← hR] at hw
  have hRv : ‖R‖ ≤ (c - 1) / 2 * ‖P v‖ := by
    have hsplit : ‖v‖ ≤ ‖P v‖ + ‖v - P v‖ := by
      calc ‖v‖ = ‖P v + (v - P v)‖ := by rw [add_sub_cancel]
        _ ≤ ‖P v‖ + ‖v - P v‖ := norm_add_le _ _
    calc ‖R‖ ≤ (c - 1) / 4 * ‖v‖ := hw
      _ ≤ (c - 1) / 4 * (2 * ‖P v‖) :=
          mul_le_mul_of_nonneg_left (by linarith) (by linarith)
      _ = (c - 1) / 2 * ‖P v‖ := by ring
  have hw' : w = A v + R := by rw [hR, add_sub_cancel]
  have hPw : P w = A (P v) + P R := by rw [hw', map_add, hP.comm]
  have hQw : w - P w = A (v - P v) + (R - P R) := by
    rw [hPw, hw', map_sub]
    abel
  have hlow : (c + 1) / 2 * ‖P v‖ ≤ ‖P w‖ := by
    have h1 : ‖A (P v)‖ ≤ ‖P w‖ + ‖P R‖ := by
      calc ‖A (P v)‖ = ‖P w - P R‖ := by rw [hPw, add_sub_cancel_right]
        _ ≤ ‖P w‖ + ‖P R‖ := norm_sub_le _ _
    linarith [hP.expand v, hP.norm_le R]
  refine ⟨?_, hlow⟩
  calc ‖w - P w‖ ≤ ‖A (v - P v)‖ + ‖R - P R‖ := by rw [hQw]; exact norm_add_le _ _
    _ ≤ ‖P v‖ + (c - 1) / 2 * ‖P v‖ :=
        add_le_add ((hP.contract v).trans hv) ((hP.norm_sub_le R).trans hRv)
    _ = (c + 1) / 2 * ‖P v‖ := by ring
    _ ≤ ‖P w‖ := hlow

/-- **Two orbits staying near `z` never differ by a nonzero steep vector.**
Along the two orbits the difference stays steep and its steep part grows at
least like `((c + 1)/2)ᵏ` (`cone_step`), while both orbits stay in the ball of
radius `r`, which bounds that part by `2r`; so it vanishes, and with it the
difference. -/
theorem eq_of_mem_stayNear_of_steep {g : E → E} {z : E} {r : ℝ} {A P : E →ₗ[ℝ] E} {c : ℝ}
    (hP : IsExpandingSplitting A P c) (hc : 1 < c)
    (happrox : ∀ x ∈ closedBall z r, ∀ y ∈ closedBall z r,
      ‖g x - g y - A (x - y)‖ ≤ (c - 1) / 4 * ‖x - y‖)
    {x y : E} (hx : x ∈ stayNear g z r) (hy : y ∈ stayNear g z r)
    (hsteep : ‖x - y - P (x - y)‖ ≤ ‖P (x - y)‖) : x = y := by
  set v : ℕ → E := fun k => g^[k] x - g^[k] y with hv
  have hv0 : v 0 = x - y := by simp [hv]
  have hgrow : ∀ k, ‖v k - P (v k)‖ ≤ ‖P (v k)‖ ∧
      ((c + 1) / 2) ^ k * ‖P (v 0)‖ ≤ ‖P (v k)‖ := by
    intro k
    induction k with
    | zero => exact ⟨by rw [hv0]; exact hsteep, by simp⟩
    | succ k ih =>
      have hvk : v (k + 1) = g (g^[k] x) - g (g^[k] y) := by
        simp only [hv, Function.iterate_succ_apply']
      obtain ⟨h1, h2⟩ := cone_step hP hc ih.1 (w := v (k + 1))
        (by rw [hvk]; exact happrox _ (hx k) _ (hy k))
      refine ⟨h1, ?_⟩
      calc ((c + 1) / 2) ^ (k + 1) * ‖P (v 0)‖
          = (c + 1) / 2 * (((c + 1) / 2) ^ k * ‖P (v 0)‖) := by ring
        _ ≤ (c + 1) / 2 * ‖P (v k)‖ := mul_le_mul_of_nonneg_left ih.2 (by linarith)
        _ ≤ ‖P (v (k + 1))‖ := h2
  have hbound : ∀ k, ‖P (v k)‖ ≤ 2 * r := fun k => (hP.norm_le _).trans (by
    calc ‖v k‖ = ‖(g^[k] x - z) - (g^[k] y - z)‖ := by simp only [hv, sub_sub_sub_cancel_right]
      _ ≤ ‖g^[k] x - z‖ + ‖g^[k] y - z‖ := norm_sub_le _ _
      _ ≤ r + r := add_le_add (mem_closedBall_iff_norm.1 (hx k))
          (mem_closedBall_iff_norm.1 (hy k))
      _ = 2 * r := by ring)
  have hP0 : P (v 0) = 0 := by
    by_contra hne
    have hpos : 0 < ‖P (v 0)‖ := norm_pos_iff.2 hne
    obtain ⟨k, hk⟩ := pow_unbounded_of_one_lt (2 * r / ‖P (v 0)‖)
      (by linarith : (1 : ℝ) < (c + 1) / 2)
    rw [div_lt_iff₀ hpos] at hk
    linarith [(hgrow k).2, hbound k]
  rw [hv0] at hP0
  rw [hP0, norm_zero, norm_le_zero_iff, sub_zero, sub_eq_zero] at hsteep
  exact hsteep

/-- **`stayNear g z r` meets each line of a direction fixed by `P` at most
once.**  The difference of two of its points on such a line is a multiple of
`e`, which is steep since `P e = e`. -/
theorem line_meets_stayNear_once {g : E → E} {z : E} {r : ℝ} {A P : E →ₗ[ℝ] E} {c : ℝ}
    (hP : IsExpandingSplitting A P c) (hc : 1 < c)
    (happrox : ∀ x ∈ closedBall z r, ∀ y ∈ closedBall z r,
      ‖g x - g y - A (x - y)‖ ≤ (c - 1) / 4 * ‖x - y‖)
    {e : E} (he : P e = e) (he0 : e ≠ 0) :
    ∀ x ∈ stayNear g z r, ∀ t : ℝ, x + t • e ∈ stayNear g z r → t = 0 := by
  intro x hx t hxt
  have hxx := eq_of_mem_stayNear_of_steep hP hc happrox hxt hx (by
    rw [add_sub_cancel_left, map_smul, he, sub_self, norm_zero]
    exact norm_nonneg _)
  rw [add_eq_left] at hxx
  exact (smul_eq_zero.1 hxx).resolve_right he0

/-- The hypotheses of the three theorems above hold for the doubling map `A`
of the line with `P = 1` and `c = 2`: the steep vector `1` and its image `2`,
the ball `B̄(0, 1)`, on which `g = A`, the fixed point `0`, which stays in it,
and the direction `e = 1`. -/
example : let A : ℝ →ₗ[ℝ] ℝ := (2 : ℝ) • LinearMap.id
    let P : ℝ →ₗ[ℝ] ℝ := LinearMap.id
    IsExpandingSplitting A P 2 ∧ (1 : ℝ) < 2 ∧ ‖(1 : ℝ) - P 1‖ ≤ ‖P 1‖ ∧
      ‖(2 : ℝ) - A 1‖ ≤ (2 - 1) / 4 * ‖(1 : ℝ)‖ ∧
      (∀ x ∈ closedBall (0 : ℝ) 1, ∀ y ∈ closedBall (0 : ℝ) 1,
        ‖A x - A y - A (x - y)‖ ≤ (2 - 1) / 4 * ‖x - y‖) ∧
      (0 : ℝ) ∈ stayNear A 0 1 ∧ ‖(0 : ℝ) - 0 - P (0 - 0)‖ ≤ ‖P (0 - 0)‖ ∧
      P 1 = 1 ∧ (1 : ℝ) ≠ 0 := by
  intro A P
  refine ⟨⟨fun v => by simp [A, P], fun v => le_rfl, fun v => by simp [P],
    fun v => by simp [A, P], fun v => by simp [P]⟩, by norm_num, by simp [P], by norm_num [A],
    fun x _ y _ => ?_, fun k => ?_, by simp, rfl, one_ne_zero⟩
  · rw [map_sub, sub_self, norm_zero]
    positivity
  · rw [Function.iterate_fixed (map_zero A)]
    exact mem_closedBall_self zero_le_one

end Transformer.SaddleAvoidance
