/-
# Appendix A — the cone form at a saddle with one particle on the far side

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, `l:nosaddleconv`.

After `Perspective.AppendixA_Barycenter`, the saddles left in `l:nosaddleconv`
are `Z = signedTuple e {i₀}` with `n ≥ 3`: one particle at `-e`, the others at
`e`.  The source's center-stable manifold is replaced there by a cone.  For a
perturbation `v` of `Z` write `u = P v`, `P = Proj_e`, and

  `W(u) = Σ_j u_j + (n - 2) u_{i₀}`,  `q(v) = 2/(n(n-1)) ‖W(u)‖² - Σ_j ‖u_j‖²`.

The linearization `L` of `e:gradfl` at `Z` (`coneLin`) has `W(L u) = 2 W(u)`:
`W` reads off the unstable direction, and along `L` the form grows at least
like `2q` (`coneLin_growth`).  A perturbation of the particle `i₀` alone has
`q = (n-2)/n ‖P v_{i₀}‖² ≥ 0` (`coneForm_single`).
-/

import Transformer.Perspective.AppendixA_CriticalShape
import Mathlib.Algebra.Order.Chebyshev

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- The unstable coordinate at `signedTuple e {i₀}`:
`W(u) = Σ_j u_j + (n - 2) u_{i₀}`. -/
noncomputable def coneW (i₀ : Idx n) (u : Idx n → EucSpace d) : EucSpace d :=
  ∑ j : Idx n, u j + ((n : ℝ) - 2) • u i₀

/-- The polarized cone form on projected perturbations,
`B(u, a) = 2 · 2/(n(n-1)) ⟨W u, W a⟩ - 2 Σ_j ⟨u_j, a_j⟩`: the derivative of
`u ↦ 2/(n(n-1)) ‖W u‖² - Σ_j ‖u_j‖²` at `u` in the direction `a`. -/
noncomputable def coneBil (i₀ : Idx n) (u a : Idx n → EucSpace d) : ℝ :=
  2 * (2 / ((n : ℝ) * ((n : ℝ) - 1))) * inner (𝕜 := ℝ) (coneW i₀ u) (coneW i₀ a)
    - 2 * ∑ j : Idx n, inner (𝕜 := ℝ) (u j) (a j)

/-- **The cone form** at `signedTuple e {i₀}`:
`q(v) = 2/(n(n-1)) ‖W(P v)‖² - Σ_j ‖P v_j‖²` with `P = Proj_e`. -/
noncomputable def coneForm (e : EucSpace d) (i₀ : Idx n) (v : Idx n → EucSpace d) : ℝ :=
  2 / ((n : ℝ) * ((n : ℝ) - 1)) * ‖coneW i₀ (fun j => proj d e (v j))‖ ^ 2
    - ∑ j : Idx n, ‖proj d e (v j)‖ ^ 2

/-- The linearization of `e:gradfl` at `signedTuple e {i₀}` on tangent
perturbations: `L(u)_j = (2/n) (Σ_k u_k - σ_j (n - 2) u_j)` with `σ_{i₀} = -1`
and `σ_j = 1` otherwise. -/
noncomputable def coneLin (i₀ : Idx n) (u : Idx n → EucSpace d) (j : Idx n) : EucSpace d :=
  (2 / (n : ℝ)) • (∑ k : Idx n, u k - ((if j = i₀ then -1 else 1) * ((n : ℝ) - 2)) • u j)

/-- The polarized form on the diagonal is twice the cone form. -/
theorem coneBil_self (e : EucSpace d) (i₀ : Idx n) (v : Idx n → EucSpace d) :
    coneBil i₀ (fun j => proj d e (v j)) (fun j => proj d e (v j)) = 2 * coneForm e i₀ v := by
  simp only [coneBil, coneForm, real_inner_self_eq_norm_sq]
  ring

/-- `coneBil` is additive in the direction. -/
theorem coneBil_add (i₀ : Idx n) (u a b : Idx n → EucSpace d) :
    coneBil i₀ u (fun j => a j + b j) = coneBil i₀ u a + coneBil i₀ u b := by
  have hW : coneW i₀ (fun j => a j + b j) = coneW i₀ a + coneW i₀ b := by
    simp only [coneW, Finset.sum_add_distrib, smul_add]
    abel
  simp only [coneBil, hW, inner_add_right, Finset.sum_add_distrib]
  ring

/-- `L(u)_j = (2/n) (Σ_k u_k - (n - 2) u_j) + [j = i₀] (4(n - 2)/n) u_j`. -/
theorem coneLin_eq (i₀ : Idx n) (u : Idx n → EucSpace d) (j : Idx n) :
    coneLin i₀ u j = (2 / (n : ℝ)) • (∑ k : Idx n, u k - ((n : ℝ) - 2) • u j)
      + if j = i₀ then (4 * ((n : ℝ) - 2) / n) • u j else 0 := by
  unfold coneLin
  split_ifs
  · module
  · rw [one_mul, add_zero]

/-- `Σ_j L(u)_j = (4/n) W(u)`. -/
theorem sum_coneLin (hn : n ≠ 0) (i₀ : Idx n) (u : Idx n → EucSpace d) :
    ∑ j : Idx n, coneLin i₀ u j = (4 / (n : ℝ)) • coneW i₀ u := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn
  simp_rw [coneLin_eq]
  rw [Finset.sum_add_distrib, ← Finset.smul_sum, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, ← Finset.smul_sum, Finset.sum_ite_eq',
    ite_eq_left (Finset.mem_univ _), ← Nat.cast_smul_eq_nsmul ℝ, coneW]
  match_scalars <;> field_simp
  ring

/-- **`W` reads the unstable direction:** `W(L u) = 2 W(u)`. -/
theorem coneW_coneLin (hn : n ≠ 0) (i₀ : Idx n) (u : Idx n → EucSpace d) :
    coneW i₀ (coneLin i₀ u) = (2 : ℝ) • coneW i₀ u := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn
  have hi : coneLin i₀ u i₀ = (2 / (n : ℝ)) • coneW i₀ u := by
    rw [coneLin, ite_eq_left rfl, coneW]; module
  rw [coneW, sum_coneLin hn, hi, smul_smul, ← add_smul]
  congr 1
  field_simp; ring

/-- `Σ_j ⟨u_j, L(u)_j⟩ = (2/n)(‖Σ_k u_k‖² - (n - 2) Σ_j ‖u_j‖²) + (4(n - 2)/n) ‖u_{i₀}‖²`. -/
theorem sum_inner_coneLin (i₀ : Idx n) (u : Idx n → EucSpace d) :
    ∑ j : Idx n, inner (𝕜 := ℝ) (u j) (coneLin i₀ u j)
      = 2 / (n : ℝ) * (‖∑ k : Idx n, u k‖ ^ 2 - ((n : ℝ) - 2) * ∑ j : Idx n, ‖u j‖ ^ 2)
        + 4 * ((n : ℝ) - 2) / n * ‖u i₀‖ ^ 2 := by
  have h : ∀ j : Idx n, inner (𝕜 := ℝ) (u j) (coneLin i₀ u j)
      = 2 / (n : ℝ) * (inner (𝕜 := ℝ) (u j) (∑ k : Idx n, u k) - ((n : ℝ) - 2) * ‖u j‖ ^ 2)
        + if j = i₀ then 4 * ((n : ℝ) - 2) / n * ‖u j‖ ^ 2 else 0 := by
    intro j
    rw [coneLin_eq, inner_add_right, real_inner_smul_right, inner_sub_right,
      real_inner_smul_right, real_inner_self_eq_norm_sq]
    split_ifs
    exacts [by rw [real_inner_smul_right, real_inner_self_eq_norm_sq], by rw [inner_zero_right]]
  rw [Finset.sum_congr rfl fun j _ => h j, Finset.sum_add_distrib, ← Finset.mul_sum,
    Finset.sum_sub_distrib, ← sum_inner, real_inner_self_eq_norm_sq, ← Finset.mul_sum,
    Finset.sum_ite_eq', ite_eq_left (Finset.mem_univ _)]

/-- Cauchy–Schwarz over the particles other than `i₀`:
`‖Σ_{j ≠ i₀} u_j‖² ≤ (n - 1) Σ_{j ≠ i₀} ‖u_j‖²`. -/
theorem norm_sum_sub_sq_le (i₀ : Idx n) (u : Idx n → EucSpace d) :
    ‖∑ j : Idx n, u j - u i₀‖ ^ 2
      ≤ ((n : ℝ) - 1) * (∑ j : Idx n, ‖u j‖ ^ 2 - ‖u i₀‖ ^ 2) := by
  classical
  have hn : 1 ≤ n := i₀.pos
  rw [← Finset.sum_erase_eq_sub (Finset.mem_univ i₀),
    ← Finset.sum_erase_eq_sub (Finset.mem_univ i₀)]
  have hcard : (((Finset.univ : Finset (Idx n)).erase i₀).card : ℝ) = (n : ℝ) - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ, Fintype.card_fin,
      Nat.cast_sub hn, Nat.cast_one]
  calc ‖∑ j ∈ Finset.univ.erase i₀, u j‖ ^ 2
      ≤ (∑ j ∈ Finset.univ.erase i₀, ‖u j‖) ^ 2 := by gcongr; exact norm_sum_le _ _
    _ ≤ _ := by rw [← hcard]; exact sq_sum_le_card_mul_sum_sq

/-- **The cone form grows along the linearization:**
`B(u, L u) ≥ 2 · 2/(n(n-1)) ‖W u‖²`, i.e. `B(u, L u) ≥ 2 q + 2 Σ_j ‖u_j‖²` — the
unstable part grows at rate `2`, the rest does not grow. -/
theorem coneLin_growth (hn : 2 ≤ n) (i₀ : Idx n) (u : Idx n → EucSpace d) :
    2 * (2 / ((n : ℝ) * ((n : ℝ) - 1))) * ‖coneW i₀ u‖ ^ 2 ≤ coneBil i₀ u (coneLin i₀ u) := by
  have hN : (2 : ℝ) ≤ n := by exact_mod_cast hn
  set s := ∑ k : Idx n, u k with hs
  set b := u i₀ with hb
  set S₂ := ∑ j : Idx n, ‖u j‖ ^ 2 with hS₂
  have hW2 : ‖coneW i₀ u‖ ^ 2
      = ‖s‖ ^ 2 + 2 * ((n : ℝ) - 2) * inner (𝕜 := ℝ) s b + ((n : ℝ) - 2) ^ 2 * ‖b‖ ^ 2 := by
    rw [coneW, ← hs, ← hb, norm_add_sq_real, real_inner_smul_right, norm_smul,
      Real.norm_eq_abs, mul_pow, sq_abs]
    ring
  have hCS : ‖s‖ ^ 2 - 2 * inner (𝕜 := ℝ) s b + ‖b‖ ^ 2 ≤ ((n : ℝ) - 1) * (S₂ - ‖b‖ ^ 2) := by
    have := norm_sum_sub_sq_le i₀ u
    rwa [← hs, ← hb, ← hS₂, norm_sub_sq_real] at this
  have hval : coneBil i₀ u (coneLin i₀ u)
      = 2 * (2 / ((n : ℝ) * ((n : ℝ) - 1))) * (2 * ‖coneW i₀ u‖ ^ 2)
        - 2 * (2 / (n : ℝ) * (‖s‖ ^ 2 - ((n : ℝ) - 2) * S₂)
          + 4 * ((n : ℝ) - 2) / n * ‖b‖ ^ 2) := by
    rw [coneBil, coneW_coneLin (by omega), real_inner_smul_right, real_inner_self_eq_norm_sq,
      sum_inner_coneLin]
  have hdiff : coneBil i₀ u (coneLin i₀ u)
        - 2 * (2 / ((n : ℝ) * ((n : ℝ) - 1))) * ‖coneW i₀ u‖ ^ 2
      = 4 / ((n : ℝ) * ((n : ℝ) - 1)) * (((n : ℝ) - 2)
          * (((n : ℝ) - 1) * (S₂ - ‖b‖ ^ 2) - (‖s‖ ^ 2 - 2 * inner (𝕜 := ℝ) s b + ‖b‖ ^ 2))) := by
    have h1 : (n : ℝ) - 1 ≠ 0 := by linarith
    have h0 : (n : ℝ) ≠ 0 := by linarith
    rw [hval, hW2]
    field_simp
    ring
  have hpos : 0 ≤ 4 / ((n : ℝ) * ((n : ℝ) - 1)) :=
    div_nonneg (by norm_num) (mul_nonneg (by linarith) (by linarith))
  have := mul_nonneg hpos (mul_nonneg (by linarith : (0 : ℝ) ≤ (n : ℝ) - 2) (sub_nonneg.mpr hCS))
  linarith

/-- The hypotheses of `coneLin_growth`, `sum_coneLin` and `coneW_coneLin` hold for `n = 3`. -/
example : 2 ≤ 3 ∧ 3 ≠ 0 := by norm_num

/-- **A perturbation of the particle `i₀` alone lies in the unstable cone:**
`q(v) = (n - 2)/n ‖P v_{i₀}‖²` when `v_j = 0` for `j ≠ i₀`. -/
theorem coneForm_single (hn : 2 ≤ n) (e : EucSpace d) (i₀ : Idx n) (v : Idx n → EucSpace d)
    (hv : ∀ j : Idx n, j ≠ i₀ → v j = 0) :
    coneForm e i₀ v = ((n : ℝ) - 2) / n * ‖proj d e (v i₀)‖ ^ 2 := by
  have hN : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hzero : ∀ j : Idx n, j ≠ i₀ → proj d e (v j) = 0 := fun j hj => by
    rw [hv j hj, proj, inner_zero_right, zero_smul, sub_zero]
  have hsum : ∑ j : Idx n, proj d e (v j) = proj d e (v i₀) :=
    Finset.sum_eq_single i₀ (fun j _ hj => hzero j hj) (fun h => absurd (Finset.mem_univ _) h)
  have hsq : ∑ j : Idx n, ‖proj d e (v j)‖ ^ 2 = ‖proj d e (v i₀)‖ ^ 2 :=
    Finset.sum_eq_single i₀ (fun j _ hj => by rw [hzero j hj, norm_zero]; norm_num)
      (fun h => absurd (Finset.mem_univ _) h)
  have hW : coneW i₀ (fun j => proj d e (v j)) = ((n : ℝ) - 1) • proj d e (v i₀) := by
    rw [coneW, hsum]; module
  rw [coneForm, hW, hsq, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
  have h1 : (n : ℝ) - 1 ≠ 0 := by linarith
  have h0 : (n : ℝ) ≠ 0 := by linarith
  field_simp
  ring

/-- The hypotheses of `coneForm_single` hold for `n = 3`, `i₀ = 0` and the
perturbation `Pi.single 0 w`. -/
example (w : EucSpace 2) : 2 ≤ 3 ∧ ∀ j : Idx 3, j ≠ 0 → (Pi.single 0 w : Idx 3 → EucSpace 2) j = 0 :=
  ⟨by norm_num, fun _ hj => by simp [hj]⟩

end Perspective
end Transformer
