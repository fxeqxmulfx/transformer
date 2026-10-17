/-
# Appendix A — near the saddle the cone form of a difference grows

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, `l:nosaddleconv`.

Let `x`, `y` be configurations within `r` of the saddle `Z = signedTuple e {i₀}`.
The error `coneErr` of the linearization of `e:gradfl` at `Z` is at most
`12 r Σ_k ‖P(x_k - y_k)‖` (`norm_coneErr_le`): the coefficients of the field
differ from those of `Z` by `O(n r)`, and the difference `x - y` is almost
tangent.  For `n ≥ 3` and `60 n r ≤ 1` the error is absorbed by the growth of
the linearization, so the derivative of the cone form `q(x - y)` along the flow
is at least `2 q(x - y)` (`two_mul_coneForm_le`).
-/

import Transformer.Perspective.AppendixA_ConeError

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- `⟨Z_j, Σ_k Z_k⟩ = σ_j (n - 2)` at `Z = signedTuple e {i₀}`. -/
theorem inner_signedTuple_sum (e : SSphere d) (i₀ j : Idx n) :
    inner (𝕜 := ℝ) (signedTuple e {i₀} j : EucSpace d)
        (∑ k : Idx n, (signedTuple e {i₀} k : EucSpace d))
      = (if j = i₀ then -1 else 1) * ((n : ℝ) - 2) := by
  have he : inner (𝕜 := ℝ) (e : EucSpace d) e = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp e.2]; norm_num
  rw [sum_signedTuple, Finset.card_singleton, coe_signedTuple]
  by_cases h : j = i₀
  · rw [ite_eq_left (Finset.mem_singleton.mpr h), ite_eq_left h, inner_neg_left,
      real_inner_smul_right, he]
    ring
  · rw [ite_eq_right (fun h' => h (Finset.mem_singleton.mp h')), ite_eq_right h,
      real_inner_smul_right, he]
    ring

/-- The norm of a sum of unit vectors is at most their number. -/
theorem norm_sum_sphere_le (x : SphereTuple d n) : ‖∑ k : Idx n, (x k : EucSpace d)‖ ≤ n := by
  refine (norm_sum_le _ _).trans (le_of_eq ?_)
  simp [mem_sphere_zero_iff_norm.mp (x _).2]

/-- `|σ_j (n - 2) - ⟨x_j, Σ x⟩| ≤ 2 n r` within `r` of `signedTuple e {i₀}`. -/
theorem abs_sigma_sub_inner_le (e : SSphere d) (i₀ : Idx n) {r : ℝ} (x : SphereTuple d n)
    (hx : ∀ k : Idx n, ‖(x k : EucSpace d) - signedTuple e {i₀} k‖ ≤ r) (j : Idx n) :
    |(if j = i₀ then -1 else 1) * ((n : ℝ) - 2)
        - inner (𝕜 := ℝ) (x j : EucSpace d) (∑ k : Idx n, (x k : EucSpace d))| ≤ 2 * n * r := by
  rw [← inner_signedTuple_sum e i₀ j]
  set Z : Idx n → EucSpace d := fun k => (signedTuple e {i₀} k : EucSpace d) with hZ
  have hr0 : 0 ≤ r := (norm_nonneg _).trans (hx j)
  have hsplit : inner (𝕜 := ℝ) (Z j) (∑ k : Idx n, Z k)
        - inner (𝕜 := ℝ) (x j : EucSpace d) (∑ k : Idx n, (x k : EucSpace d))
      = -(inner (𝕜 := ℝ) ((x j : EucSpace d) - Z j) (∑ k : Idx n, (x k : EucSpace d))
          + inner (𝕜 := ℝ) (Z j) (∑ k : Idx n, ((x k : EucSpace d) - Z k))) := by
    rw [Finset.sum_sub_distrib, inner_sub_left, inner_sub_right]; ring
  have hd : ‖∑ k : Idx n, ((x k : EucSpace d) - Z k)‖ ≤ n * r := by
    refine (norm_sum_le _ _).trans ?_
    calc ∑ k : Idx n, ‖(x k : EucSpace d) - Z k‖ ≤ ∑ _k : Idx n, r :=
          Finset.sum_le_sum fun k _ => hx k
      _ = n * r := by simp
  have hZj : ‖Z j‖ = 1 := mem_sphere_zero_iff_norm.mp (signedTuple e {i₀} j).2
  rw [hsplit, abs_neg]
  refine (abs_add_le _ _).trans ?_
  have h1 := (abs_real_inner_le_norm ((x j : EucSpace d) - Z j) _).trans
    (mul_le_mul (hx j) (norm_sum_sphere_le x) (norm_nonneg _) hr0)
  have h2 := abs_real_inner_le_norm (Z j) (∑ k : Idx n, ((x k : EucSpace d) - Z k))
  rw [hZj, one_mul] at h2
  linarith

/-- `|⟨x_j, Σ x⟩ - ⟨y_j, Σ y⟩| ≤ n ‖x_j - y_j‖ + Σ_k ‖x_k - y_k‖` for unit vectors. -/
theorem abs_inner_sum_sub_le (x y : SphereTuple d n) (j : Idx n) :
    |inner (𝕜 := ℝ) (x j : EucSpace d) (∑ k : Idx n, (x k : EucSpace d))
        - inner (𝕜 := ℝ) (y j : EucSpace d) (∑ k : Idx n, (y k : EucSpace d))|
      ≤ n * ‖(x j : EucSpace d) - y j‖ + ∑ k : Idx n, ‖(x k : EucSpace d) - y k‖ := by
  have hsplit : inner (𝕜 := ℝ) (x j : EucSpace d) (∑ k : Idx n, (x k : EucSpace d))
        - inner (𝕜 := ℝ) (y j : EucSpace d) (∑ k : Idx n, (y k : EucSpace d))
      = inner (𝕜 := ℝ) ((x j : EucSpace d) - y j) (∑ k : Idx n, (x k : EucSpace d))
          + inner (𝕜 := ℝ) (y j : EucSpace d) (∑ k : Idx n, ((x k : EucSpace d) - y k)) := by
    rw [Finset.sum_sub_distrib, inner_sub_left, inner_sub_right]; ring
  rw [hsplit]
  refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
  · refine (abs_real_inner_le_norm _ _).trans ?_
    rw [mul_comm]
    exact mul_le_mul_of_nonneg_right (norm_sum_sphere_le x) (norm_nonneg _)
  · refine (abs_real_inner_le_norm _ _).trans ?_
    rw [mem_sphere_zero_iff_norm.mp (y j).2, one_mul]
    exact norm_sum_le _ _

/-- `‖a v - b w‖ ≤ |a| ‖v‖ + |b| ‖w‖`. -/
theorem norm_smul_sub_smul_le (a b : ℝ) (v w : EucSpace d) :
    ‖a • v - b • w‖ ≤ |a| * ‖v‖ + |b| * ‖w‖ := by
  refine (norm_sub_le _ _).trans (le_of_eq ?_)
  rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs]

/-- **The error of the linearization is small:**
`‖coneErr_j‖ ≤ 12 r Σ_k ‖P(x_k - y_k)‖` within `r ≤ 1/2` of `signedTuple e {i₀}`. -/
theorem norm_coneErr_le (e : SSphere d) (i₀ : Idx n) {r : ℝ} (hr : r ≤ 1 / 2)
    (x y : SphereTuple d n)
    (hx : ∀ k : Idx n, ‖(x k : EucSpace d) - signedTuple e {i₀} k‖ ≤ r)
    (hy : ∀ k : Idx n, ‖(y k : EucSpace d) - signedTuple e {i₀} k‖ ≤ r) (j : Idx n) :
    ‖coneErr (e : EucSpace d) i₀ (fun k => (x k : EucSpace d)) (fun k => (y k : EucSpace d)) j‖
      ≤ 12 * r * ∑ k : Idx n, ‖proj d e ((x k : EucSpace d) - y k)‖ := by
  set U := ∑ k : Idx n, ‖proj d e ((x k : EucSpace d) - y k)‖ with hU
  have hN : (1 : ℝ) ≤ n := by exact_mod_cast j.pos
  have hr0 : 0 ≤ r := (norm_nonneg _).trans (hx j)
  have hU0 : 0 ≤ U := Finset.sum_nonneg fun k _ => norm_nonneg _
  have huj : ‖proj d e ((x j : EucSpace d) - y j)‖ ≤ U :=
    Finset.single_le_sum (f := fun k => ‖proj d e ((x k : EucSpace d) - y k)‖)
      (fun k _ => norm_nonneg _) (Finset.mem_univ j)
  have hδ : ∀ k : Idx n, ‖(x k : EucSpace d) - y k‖ ≤ 2 * ‖proj d e ((x k : EucSpace d) - y k)‖ :=
    fun k => norm_sub_le_two_mul_norm_proj e {i₀} hr x y k (hx k) (hy k)
  have hδsum : ∑ k : Idx n, ‖(x k : EucSpace d) - y k‖ ≤ 2 * U := by
    rw [hU, Finset.mul_sum]; exact Finset.sum_le_sum fun k _ => hδ k
  have hδj : ‖(x j : EucSpace d) - y j‖ ≤ 2 * U := (hδ j).trans (by linarith)
  have hβ := (abs_inner_sum_sub_le x y j).trans
    (add_le_add (mul_le_mul_of_nonneg_left hδj (by linarith)) hδsum)
  have he : ‖(e : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp e.2
  have hPy : ‖proj d e (y j : EucSpace d)‖ ≤ r := by
    have hZ : proj d e (signedTuple e {i₀} j : EucSpace d) = 0 := by
      rw [coe_signedTuple]
      split_ifs
      · simpa only [neg_one_smul] using proj_smul_self he (-1 : ℝ)
      · simpa only [one_smul] using proj_smul_self he (1 : ℝ)
    have hsub : proj d e (y j : EucSpace d)
        = proj d e ((y j : EucSpace d) - signedTuple e {i₀} j) := by
      have h := map_sub (projLin (e : EucSpace d)) (y j : EucSpace d)
        (signedTuple e {i₀} j : EucSpace d)
      simp only [projLin_apply] at h
      rw [h, hZ, sub_zero]
    rw [hsub]
    exact (norm_proj_le he _).trans (hy j)
  have hn0 : (0 : ℝ) < 2 / n := div_pos two_pos (by linarith)
  have hA := mul_le_mul (abs_sigma_sub_inner_le e i₀ x hx j) huj (norm_nonneg _)
    (mul_nonneg (mul_nonneg zero_le_two (Nat.cast_nonneg n)) hr0)
  have hB := mul_le_mul hβ hPy (norm_nonneg _) (by nlinarith)
  have hsum : 2 / (n : ℝ) * (2 * n * r * U + (n * (2 * U) + 2 * U) * r)
      = 12 * r * U - 4 * (r * U) + 4 * (r * U) / n := by
    field_simp; ring
  have hdiv : 4 * (r * U) / n ≤ 4 * (r * U) := div_le_self (mul_nonneg (by norm_num) (mul_nonneg hr0 hU0)) hN
  simp only [coneErr]
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos hn0]
  refine (mul_le_mul_of_nonneg_left (norm_smul_sub_smul_le _ _ _ _) hn0.le).trans ?_
  refine (mul_le_mul_of_nonneg_left (add_le_add hA hB) hn0.le).trans ?_
  linarith

/-- The hypotheses of `abs_sigma_sub_inner_le` and `norm_coneErr_le` hold at `r = 0`
for `x = y = signedTuple e {0}`. -/
example (e : SSphere 2) : (0 : ℝ) ≤ 1 / 2 ∧
    ∀ k : Idx 3, ‖(signedTuple e {0} k : EucSpace 2) - signedTuple e {0} k‖ ≤ 0 :=
  ⟨by norm_num, fun k => by simp⟩

/-- **Near the saddle the cone form of a difference grows along the flow.** For
`n ≥ 3` and configurations `x`, `y` within `r` of `signedTuple e {i₀}`, with
`60 n r ≤ 1`, the derivative `B(P(x - y), P(F x - F y))` of the cone form of
`x - y` along `e:gradfl` is at least `2 q(x - y)`. -/
theorem two_mul_coneForm_le (hn : 3 ≤ n) (e : SSphere d) (i₀ : Idx n) {r : ℝ}
    (hr : 60 * (n : ℝ) * r ≤ 1) (x y : SphereTuple d n)
    (hx : ∀ k : Idx n, ‖(x k : EucSpace d) - signedTuple e {i₀} k‖ ≤ r)
    (hy : ∀ k : Idx n, ‖(y k : EucSpace d) - signedTuple e {i₀} k‖ ≤ r) :
    2 * coneForm (e : EucSpace d) i₀ (fun k => (x k : EucSpace d) - y k)
      ≤ coneBil i₀ (fun k => proj d e ((x k : EucSpace d) - y k))
          (fun k => proj d e (e0Field (fun l => (x l : EucSpace d)) k
            - e0Field (fun l => (y l : EucSpace d)) k)) := by
  have hN : (3 : ℝ) ≤ n := by exact_mod_cast hn
  have hr0 : 0 ≤ r := (norm_nonneg _).trans (hx i₀)
  have hr2 : r ≤ 1 / 2 := by nlinarith [mul_nonneg (sub_nonneg.mpr hN) hr0]
  set u : Idx n → EucSpace d := fun k => proj d e ((x k : EucSpace d) - y k) with hu
  set U := ∑ k : Idx n, ‖u k‖ with hU
  set S₂ := ∑ k : Idx n, ‖u k‖ ^ 2 with hS₂
  have hsplit : (fun k => proj d e (e0Field (fun l => (x l : EucSpace d)) k
        - e0Field (fun l => (y l : EucSpace d)) k))
      = fun k => coneLin i₀ u k + coneErr (e : EucSpace d) i₀ (fun l => (x l : EucSpace d))
          (fun l => (y l : EucSpace d)) k :=
    funext fun k => proj_e0Field_sub _ i₀ _ _ k
  rw [hsplit, coneBil_add]
  have hlin := coneLin_growth (by omega) i₀ u
  have herr := abs_le.mp (abs_coneBil_le (by omega) i₀ u _
    (fun k => norm_coneErr_le e i₀ hr2 x y hx hy k))
  have hU2 : U ^ 2 ≤ n * S₂ := by
    simpa using sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := fun k => ‖u k‖)
  have hS0 : 0 ≤ S₂ := Finset.sum_nonneg fun k _ => sq_nonneg _
  have hsmall : 120 * r * U ^ 2 ≤ 2 * S₂ := by
    nlinarith [mul_le_mul_of_nonneg_left hU2 hr0, mul_le_mul_of_nonneg_right hr hS0]
  rw [show 2 * coneForm (e : EucSpace d) i₀ (fun k => (x k : EucSpace d) - y k)
      = 2 * (2 / ((n : ℝ) * ((n : ℝ) - 1))) * ‖coneW i₀ u‖ ^ 2 - 2 * S₂ by rw [coneForm]; ring]
  nlinarith [herr.1]

/-- The hypotheses of `two_mul_coneForm_le` hold for `n = 3`, `r = 0` and
`x = y = signedTuple e {0}`. -/
example (e : SSphere 2) : 3 ≤ 3 ∧ 60 * ((3 : ℕ) : ℝ) * 0 ≤ 1 ∧
    ∀ k : Idx 3, ‖(signedTuple e {0} k : EucSpace 2) - signedTuple e {0} k‖ ≤ 0 :=
  ⟨le_rfl, by norm_num, fun k => by simp⟩

end Perspective
end Transformer
