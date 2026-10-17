/-
# Appendix A — the linearization error at the saddle

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, `l:nosaddleconv`.

For two configurations `x`, `y`, the projected difference `P(F x - F y)` of the
vector field `F` of `e:gradfl` is the linearization `coneLin` of
`Perspective.AppendixA_ConeForm` applied to `u = P(x - y)`, plus an error
`coneErr` (`proj_e0Field_sub`).  Near `Z = signedTuple e S` the difference
`x - y` is almost tangent (`norm_sub_le_two_mul_norm_proj`), and the polarized
form is bounded by the size of its direction (`abs_coneBil_le`); the error is
estimated in `Perspective.AppendixA_ConeGrowth`.
-/

import Transformer.Perspective.AppendixA_ConeForm

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- The vector field of `e:gradfl`: `F(x)_j = Proj_{x_j}((2/n) Σ_k x_k)`. -/
noncomputable def e0Field (x : Idx n → EucSpace d) (j : Idx n) : EucSpace d :=
  proj d (x j) ((2 * (n : ℝ)⁻¹) • ∑ k : Idx n, x k)

/-- The error of the linearization along two configurations:
`(2/n) ((σ_j (n-2) - α_j) P(x_j - y_j) - β_j P y_j)` with `α_j = ⟨x_j, Σ x⟩` and
`β_j = ⟨x_j, Σ x⟩ - ⟨y_j, Σ y⟩`. -/
noncomputable def coneErr (e : EucSpace d) (i₀ : Idx n) (x y : Idx n → EucSpace d)
    (j : Idx n) : EucSpace d :=
  (2 / (n : ℝ)) •
    (((if j = i₀ then -1 else 1) * ((n : ℝ) - 2) - inner (𝕜 := ℝ) (x j) (∑ k : Idx n, x k))
        • proj d e (x j - y j)
      - (inner (𝕜 := ℝ) (x j) (∑ k : Idx n, x k) - inner (𝕜 := ℝ) (y j) (∑ k : Idx n, y k))
        • proj d e (y j))

/-- `Proj_e` as a linear map. -/
noncomputable def projLin (e : EucSpace d) : EucSpace d →ₗ[ℝ] EucSpace d where
  toFun := proj d e
  map_add' v w := by simp only [proj, inner_add_right, add_smul]; abel
  map_smul' a v := proj_smul d a e v

/-- `projLin e` is `Proj_e`. -/
theorem projLin_apply (e v : EucSpace d) : projLin e v = proj d e v := rfl

/-- **The projected field difference is linear part plus error:**
`P(F x - F y)_j = L(P(x - y))_j + coneErr_j`. -/
theorem proj_e0Field_sub (e : EucSpace d) (i₀ : Idx n) (x y : Idx n → EucSpace d) (j : Idx n) :
    proj d e (e0Field x j - e0Field y j)
      = coneLin i₀ (fun k => proj d e (x k - y k)) j + coneErr e i₀ x y j := by
  have hF : e0Field x j - e0Field y j
      = (2 / (n : ℝ)) • (∑ k : Idx n, (x k - y k)
          - inner (𝕜 := ℝ) (x j) (∑ k : Idx n, x k) • (x j - y j)
          - (inner (𝕜 := ℝ) (x j) (∑ k : Idx n, x k)
              - inner (𝕜 := ℝ) (y j) (∑ k : Idx n, y k)) • y j) := by
    simp only [e0Field, proj, real_inner_smul_right, Finset.sum_sub_distrib]
    module
  have hP : proj d e (e0Field x j - e0Field y j)
      = (2 / (n : ℝ)) • (∑ k : Idx n, proj d e (x k - y k)
          - inner (𝕜 := ℝ) (x j) (∑ k : Idx n, x k) • proj d e (x j - y j)
          - (inner (𝕜 := ℝ) (x j) (∑ k : Idx n, x k)
              - inner (𝕜 := ℝ) (y j) (∑ k : Idx n, y k)) • proj d e (y j)) := by
    rw [hF]
    simp only [← projLin_apply, map_sub, map_smul, map_sum]
  rw [hP, coneLin, coneErr]
  module

/-- Two unit vectors within `r` of a unit vector `z` differ almost orthogonally to
it: `|⟨x - y, z⟩| ≤ r ‖x - y‖`, because `x - y ⟂ x + y`. -/
theorem abs_inner_sub_le {x y z : EucSpace d} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) {r : ℝ}
    (hxz : ‖x - z‖ ≤ r) (hyz : ‖y - z‖ ≤ r) :
    |inner (𝕜 := ℝ) (x - y) z| ≤ r * ‖x - y‖ := by
  have h0 : inner (𝕜 := ℝ) (x - y) (x + y) = 0 := by
    rw [inner_sub_left, inner_add_right, inner_add_right, real_inner_self_eq_norm_sq,
      real_inner_self_eq_norm_sq, hx, hy, real_inner_comm x y]
    ring
  have h1 : inner (𝕜 := ℝ) (x - y) z
      = inner (𝕜 := ℝ) (x - y) ((1 / 2 : ℝ) • ((z - x) + (z - y))) := by
    have : (1 / 2 : ℝ) • ((z - x) + (z - y)) = z - (1 / 2 : ℝ) • (x + y) := by module
    rw [this, inner_sub_right, real_inner_smul_right, h0, mul_zero, sub_zero]
  have h2 : ‖(1 / 2 : ℝ) • ((z - x) + (z - y))‖ ≤ r := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (by norm_num : (0 : ℝ) < 1 / 2)]
    have := norm_add_le (z - x) (z - y)
    rw [norm_sub_rev z x, norm_sub_rev z y] at this
    linarith
  rw [h1]
  calc _ ≤ ‖x - y‖ * ‖(1 / 2 : ℝ) • ((z - x) + (z - y))‖ := abs_real_inner_le_norm _ _
    _ ≤ ‖x - y‖ * r := by gcongr
    _ = r * ‖x - y‖ := mul_comm _ _

/-- `‖v‖ ≤ ‖P v‖ + |⟨e, v⟩|` for a unit vector `e`. -/
theorem norm_le_norm_proj_add (e : SSphere d) (v : EucSpace d) :
    ‖v‖ ≤ ‖proj d e v‖ + |inner (𝕜 := ℝ) (e : EucSpace d) v| := by
  have he : ‖(e : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp e.2
  calc ‖v‖ = ‖proj d e v + inner (𝕜 := ℝ) (e : EucSpace d) v • (e : EucSpace d)‖ := by
        rw [proj, sub_add_cancel]
    _ ≤ _ := by
        refine (norm_add_le _ _).trans (le_of_eq ?_)
        rw [norm_smul, he, mul_one, Real.norm_eq_abs]

/-- `|⟨e, v⟩| = |⟨v, Z_j⟩|`: the particles of `signedTuple e S` are `± e`. -/
theorem abs_inner_signedTuple (e : SSphere d) (S : Finset (Idx n)) (j : Idx n)
    (v : EucSpace d) :
    |inner (𝕜 := ℝ) (e : EucSpace d) v|
      = |inner (𝕜 := ℝ) v (signedTuple e S j : EucSpace d)| := by
  rw [coe_signedTuple, real_inner_comm]
  split_ifs
  · rw [inner_neg_right, abs_neg]
  · rfl

/-- **Near the saddle the difference is mostly tangent:** `‖x_j - y_j‖ ≤ 2 ‖P(x_j - y_j)‖`
for configurations within `r ≤ 1/2` of `signedTuple e S`. -/
theorem norm_sub_le_two_mul_norm_proj (e : SSphere d) (S : Finset (Idx n)) {r : ℝ}
    (hr : r ≤ 1 / 2) (x y : SphereTuple d n) (j : Idx n)
    (hx : ‖(x j : EucSpace d) - signedTuple e S j‖ ≤ r)
    (hy : ‖(y j : EucSpace d) - signedTuple e S j‖ ≤ r) :
    ‖(x j : EucSpace d) - y j‖ ≤ 2 * ‖proj d e ((x j : EucSpace d) - y j)‖ := by
  have hi := abs_inner_sub_le (mem_sphere_zero_iff_norm.mp (x j).2)
    (mem_sphere_zero_iff_norm.mp (y j).2) hx hy
  rw [← abs_inner_signedTuple] at hi
  have h := norm_le_norm_proj_add e ((x j : EucSpace d) - y j)
  have : r * ‖(x j : EucSpace d) - y j‖ ≤ 1 / 2 * ‖(x j : EucSpace d) - y j‖ :=
    mul_le_mul_of_nonneg_right hr (norm_nonneg _)
  linarith

/-- The hypotheses of `abs_inner_sub_le` hold for `x = y = z = e` and `r = 0`, and with
`r = 0 ≤ 1/2` those of `norm_sub_le_two_mul_norm_proj` for `x = y = signedTuple e S`. -/
example (e : SSphere 2) (j : Idx 3) :
    ‖(e : EucSpace 2)‖ = 1 ∧ ‖(e : EucSpace 2) - e‖ ≤ 0 ∧ (0 : ℝ) ≤ 1 / 2 ∧
      ‖(signedTuple e {0} j : EucSpace 2) - signedTuple e {0} j‖ ≤ 0 :=
  ⟨mem_sphere_zero_iff_norm.mp e.2, by simp, by norm_num, by simp⟩

/-- `|B(u, a)| ≤ 10 ε Σ_j ‖u_j‖` when every `‖a_j‖ ≤ ε`, from `‖W u‖ ≤ (n-1) Σ_j ‖u_j‖`
and `‖W a‖ ≤ (2n-2) ε`. -/
theorem abs_coneBil_le (hn : 2 ≤ n) (i₀ : Idx n) (u a : Idx n → EucSpace d) {ε : ℝ}
    (ha : ∀ j : Idx n, ‖a j‖ ≤ ε) :
    |coneBil i₀ u a| ≤ 10 * ε * ∑ j : Idx n, ‖u j‖ := by
  have hN : (2 : ℝ) ≤ n := by exact_mod_cast hn
  set U := ∑ j : Idx n, ‖u j‖ with hU
  have hε : 0 ≤ ε := (norm_nonneg _).trans (ha i₀)
  have hU0 : 0 ≤ U := Finset.sum_nonneg fun j _ => norm_nonneg _
  have hui : ‖u i₀‖ ≤ U := Finset.single_le_sum (f := fun j => ‖u j‖)
    (fun j _ => norm_nonneg _) (Finset.mem_univ i₀)
  have hn2 : |(n : ℝ) - 2| = (n : ℝ) - 2 := abs_of_nonneg (by linarith)
  have hWu : ‖coneW i₀ u‖ ≤ ((n : ℝ) - 1) * U := by
    refine (norm_add_le _ _).trans ?_
    rw [norm_smul, Real.norm_eq_abs, hn2]
    have := norm_sum_le Finset.univ u
    nlinarith
  have hsa : ‖∑ j : Idx n, a j‖ ≤ n * ε := by
    refine (norm_sum_le _ _).trans ?_
    calc ∑ j : Idx n, ‖a j‖ ≤ ∑ _j : Idx n, ε := Finset.sum_le_sum fun j _ => ha j
      _ = n * ε := by simp
  have hWa : ‖coneW i₀ a‖ ≤ (2 * (n : ℝ) - 2) * ε := by
    refine (norm_add_le _ _).trans ?_
    rw [norm_smul, Real.norm_eq_abs, hn2]
    nlinarith [ha i₀]
  have hS : |∑ j : Idx n, inner (𝕜 := ℝ) (u j) (a j)| ≤ ε * U := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    rw [hU, Finset.mul_sum]
    exact Finset.sum_le_sum fun j _ =>
      (abs_real_inner_le_norm _ _).trans (by nlinarith [ha j, norm_nonneg (u j)])
  have hI : |inner (𝕜 := ℝ) (coneW i₀ u) (coneW i₀ a)|
      ≤ ((n : ℝ) - 1) * U * ((2 * (n : ℝ) - 2) * ε) :=
    (abs_real_inner_le_norm _ _).trans (mul_le_mul hWu hWa (norm_nonneg _) (by nlinarith))
  have hc : 2 / ((n : ℝ) * ((n : ℝ) - 1)) * (((n : ℝ) - 1) * U * ((2 * (n : ℝ) - 2) * ε))
      ≤ 4 * ε * U := by
    have h1 : (n : ℝ) - 1 ≠ 0 := by linarith
    have h0 : (n : ℝ) ≠ 0 := by linarith
    have heq : 2 / ((n : ℝ) * ((n : ℝ) - 1)) * (((n : ℝ) - 1) * U * ((2 * (n : ℝ) - 2) * ε))
        = 4 * ε * U - 4 / n * (ε * U) := by
      field_simp; ring
    have : 0 ≤ 4 / (n : ℝ) * (ε * U) := by positivity
    linarith
  have hcpos : 0 ≤ 2 / ((n : ℝ) * ((n : ℝ) - 1)) :=
    div_nonneg (by norm_num) (mul_nonneg (by linarith) (by linarith))
  have hcI := mul_le_mul_of_nonneg_left hI hcpos
  have e1 : |2 * (2 / ((n : ℝ) * ((n : ℝ) - 1))) * inner (𝕜 := ℝ) (coneW i₀ u) (coneW i₀ a)|
      = 2 * (2 / ((n : ℝ) * ((n : ℝ) - 1)))
          * |inner (𝕜 := ℝ) (coneW i₀ u) (coneW i₀ a)| := by
    rw [abs_mul, abs_of_nonneg (mul_nonneg zero_le_two hcpos)]
  have e2 : |2 * ∑ j : Idx n, inner (𝕜 := ℝ) (u j) (a j)|
      = 2 * |∑ j : Idx n, inner (𝕜 := ℝ) (u j) (a j)| := by rw [abs_mul, abs_two]
  rw [coneBil]
  refine (abs_sub _ _).trans ?_
  rw [e1, e2]
  nlinarith

/-- The hypotheses of `abs_coneBil_le` hold for `n = 2`, the zero direction and `ε = 0`. -/
example : 2 ≤ 2 ∧ ∀ j : Idx 2, ‖(0 : Idx 2 → EucSpace 1) j‖ ≤ 0 :=
  ⟨le_rfl, fun _ => by simp⟩

end Perspective
end Transformer
