/-
# Critical points of `𝖤_β` on `(𝕊^{d-1})^n` for `d ≥ 3`

Criscitiello, Rebjock, McRae, Boumal — arXiv:2405.18273, Appendix C, the
direct proof of its Theorem `thm:spheres`, for the energy `𝖤_β` of
arXiv:2312.10794v5, which is the paper's `f` with `w_{ij} = 1` and
`φ = φ_β`, `φ_β(t) = e^{βt}/β`.

Every token moves along the projection of one vector `γ` on its tangent
space, `v_i = Proj_{x_i} γ` (`proj`).  Summed over an orthonormal basis
`γ = e_1, …, e_d`, the Hessian form `hessQuad` of
`Perspective.AppendixB_Intrinsic` is at least
`½ Σ_{i,j} e^{β⟨x_i,x_j⟩} ((1 - ⟨x_i,x_j⟩)² + (d - 3)(1 - ⟨x_i,x_j⟩))`
(`sum_hessQuad_proj_ge`), which is positive when `d ≥ 3` and two tokens
differ (`exists_hessQuad_proj_pos`).  The antipodal pair is such a
configuration, and a critical point of `𝖤_β` (`antipodalPair_isCriticalEBeta`).
-/

import Transformer.Perspective.AppendixA_Beta0
import Transformer.Perspective.AppendixB_Intrinsic

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **The average of the projected directions.**  For unit vectors `x`, `y`
and the standard basis `e_1, …, e_d`,

  `Σ_k (⟨Proj_x e_k, Proj_y e_k⟩ - ⟨x, y⟩ ‖Proj_x e_k‖²)
      = (1 - ⟨x, y⟩)² + (d - 3)(1 - ⟨x, y⟩)`.

This is the entry `(i, j)` of `E[Ẋᵀ Ẋ] = (d - 2) 𝟙𝟙ᵀ + (XᵀX)^{⊙2}` of
arXiv:2405.18273, App. C, minus `⟨x_i, x_j⟩` times its diagonal entry
`d - 1`. -/
theorem sum_inner_proj_basisFun {x y : EucSpace d} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    ∑ k : Fin d, (inner (𝕜 := ℝ) (proj d x (EuclideanSpace.basisFun (Fin d) ℝ k))
        (proj d y (EuclideanSpace.basisFun (Fin d) ℝ k))
      - inner (𝕜 := ℝ) x y * ‖proj d x (EuclideanSpace.basisFun (Fin d) ℝ k)‖ ^ 2)
      = (1 - inner (𝕜 := ℝ) x y) ^ 2 + ((d : ℝ) - 3) * (1 - inner (𝕜 := ℝ) x y) := by
  set b := EuclideanSpace.basisFun (Fin d) ℝ
  have hpair : ∀ (u w γ : EucSpace d), inner (𝕜 := ℝ) (proj d u γ) (proj d w γ)
      = inner (𝕜 := ℝ) γ γ - inner (𝕜 := ℝ) w γ * inner (𝕜 := ℝ) γ w
        - inner (𝕜 := ℝ) u γ * inner (𝕜 := ℝ) γ u
        + inner (𝕜 := ℝ) u w * (inner (𝕜 := ℝ) u γ * inner (𝕜 := ℝ) γ w) := by
    intro u w γ
    simp only [proj, inner_sub_left, inner_sub_right, real_inner_smul_left,
      real_inner_smul_right, real_inner_comm u γ, real_inner_comm w γ]
    ring
  have hb : ∀ k, inner (𝕜 := ℝ) (b k) (b k) = 1 := fun k =>
    inner_self_eq_one_of_norm_eq_one (b.orthonormal.1 k)
  have hsum : ∀ u w : EucSpace d, ‖u‖ = 1 → ‖w‖ = 1 →
      ∑ k : Fin d, inner (𝕜 := ℝ) (proj d u (b k)) (proj d w (b k))
        = (d : ℝ) - 2 + inner (𝕜 := ℝ) u w ^ 2 := by
    intro u w hu hw
    simp only [hpair, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
      b.sum_inner_mul_inner, inner_self_eq_one_of_norm_eq_one hu,
      inner_self_eq_one_of_norm_eq_one hw, hb, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, mul_one]
    ring
  have hnorm : ∀ k, ‖proj d x (b k)‖ ^ 2 = inner (𝕜 := ℝ) (proj d x (b k)) (proj d x (b k)) :=
    fun k => (real_inner_self_eq_norm_sq _).symm
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_congr rfl fun k _ => hnorm k,
    hsum x y hx hy, hsum x x hx hx, inner_self_eq_one_of_norm_eq_one hx]
  ring

/-- **At `β > 0`, `hessQuad` is at least half its first-order part**, written
with `λ_i = Σ_j e^{β⟨x_i,x_j⟩} ⟨x_i, x_j⟩`: the term in `β²` is a sum of
squares with positive weights. -/
theorem half_le_hessQuad (β : ℝ) (hβ : 0 < β) (X : SphereTuple d n)
    (v : Idx n → EucSpace d) :
    (∑ i : Idx n, ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
          * (inner (𝕜 := ℝ) (v i) (v j)
            - inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) * ‖v i‖ ^ 2)) / 2
      ≤ hessQuad d n β X v := by
  set K : Idx n → Idx n → ℝ := fun i j =>
    Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))) with hK
  have hlam : ∀ i, inner (𝕜 := ℝ) ((X i : EucSpace d))
      (∑ j : Idx n, K i j • ((X j : EucSpace d)))
        = ∑ j : Idx n, K i j * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) := by
    intro i
    simp only [inner_sum, real_inner_smul_right]
  set C := ∑ i : Idx n, ∑ j : Idx n, K i j * (β ^ 2 * (inner (𝕜 := ℝ) ((X i : EucSpace d)) (v j)
      + inner (𝕜 := ℝ) (v i) ((X j : EucSpace d))) ^ 2 / 2) with hC
  have hC0 : 0 ≤ C := Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ =>
    mul_nonneg (exp_pos _).le (by positivity)
  have hA : ∑ i : Idx n, ∑ j : Idx n, K i j * (inner (𝕜 := ℝ) (v i) (v j)
        - inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) * ‖v i‖ ^ 2)
      = (∑ i : Idx n, ∑ j : Idx n, K i j * inner (𝕜 := ℝ) (v i) (v j))
        - ∑ i : Idx n, inner (𝕜 := ℝ) ((X i : EucSpace d))
            (∑ j : Idx n, K i j • ((X j : EucSpace d))) * ‖v i‖ ^ 2 := by
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [hlam i, Finset.sum_mul, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have hQ : hessQuad d n β X v = (∑ i : Idx n, ∑ j : Idx n, K i j * (inner (𝕜 := ℝ) (v i) (v j)
        - inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) * ‖v i‖ ^ 2)) / 2
      + C / (2 * β) := by
    rw [hA, hessQuad, hC]
    field_simp
    ring
  rw [hQ]
  have : 0 ≤ C / (2 * β) := div_nonneg hC0 (by positivity)
  linarith

/-- **Appendix C of arXiv:2405.18273 for `𝖤_β`.**  Summed over the standard
basis `e_1, …, e_d`, the Hessian form along `v_i = Proj_{x_i} e_k` is at least

  `½ Σ_{i,j} e^{β⟨x_i,x_j⟩} ((1 - ⟨x_i,x_j⟩)² + (d - 3)(1 - ⟨x_i,x_j⟩))`.

The source averages over a Gaussian `γ` instead of the basis, which gives the
same second moments, and drops the `β²` term by `φ_β'' ≥ 0`, as here. -/
theorem sum_hessQuad_proj_ge (β : ℝ) (hβ : 0 < β) (X : SphereTuple d n) :
    (∑ i : Idx n, ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
          * ((1 - inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))) ^ 2
            + ((d : ℝ) - 3)
              * (1 - inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))))) / 2
      ≤ ∑ k : Fin d, hessQuad d n β X
          (fun i => proj d (X i) (EuclideanSpace.basisFun (Fin d) ℝ k)) := by
  refine le_trans (le_of_eq ?_) (Finset.sum_le_sum fun k _ => half_le_hessQuad d n β hβ X _)
  rw [← Finset.sum_div]
  congr 1
  have hX : ∀ i, ‖(X i : EucSpace d)‖ = 1 := fun i => mem_sphere_zero_iff_norm.mp (X i).2
  simp_rw [← sum_inner_proj_basisFun d (hX _) (hX _), Finset.mul_sum]
  rw [Finset.sum_congr rfl fun i _ => Finset.sum_comm, Finset.sum_comm]

/-- **Some basis direction has a positive Hessian form** at a configuration
with two distinct tokens, when `d ≥ 3` and `β > 0`: the bound of
`sum_hessQuad_proj_ge` is positive, since every term is non-negative and the
pair of distinct tokens has `⟨x_i, x_j⟩ < 1`. -/
theorem exists_hessQuad_proj_pos (hd : 3 ≤ d) (β : ℝ) (hβ : 0 < β) (X : SphereTuple d n)
    (hnt : NonTrivialTuple d n X) :
    ∃ k : Fin d, 0 < hessQuad d n β X
      (fun i => proj d (X i) (EuclideanSpace.basisFun (Fin d) ℝ k)) := by
  obtain ⟨i₀, j₀, hij⟩ := hnt
  have hX : ∀ i, ‖(X i : EucSpace d)‖ = 1 := fun i => mem_sphere_zero_iff_norm.mp (X i).2
  have hd' : (0 : ℝ) ≤ (d : ℝ) - 3 := by
    have : (3 : ℝ) ≤ d := by exact_mod_cast hd
    linarith
  have hg : ∀ i j, 0 ≤ 1 - inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) := by
    intro i j
    have := real_inner_le_norm ((X i : EucSpace d)) ((X j : EucSpace d))
    rw [hX, hX, one_mul] at this
    linarith
  have hterm : ∀ i j, 0 ≤ Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
      * ((1 - inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))) ^ 2
        + ((d : ℝ) - 3) * (1 - inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))) :=
    fun i j => mul_nonneg (exp_pos _).le (add_nonneg (sq_nonneg _) (mul_nonneg hd' (hg i j)))
  have hlt : 0 < 1 - inner (𝕜 := ℝ) ((X i₀ : EucSpace d)) ((X j₀ : EucSpace d)) :=
    sub_pos.2 ((inner_lt_one_iff_real_of_norm_eq_one (hX i₀) (hX j₀)).2 hij)
  have hpos := (sum_hessQuad_proj_ge d n β hβ X).trans_lt' (div_pos (Finset.sum_pos'
    (fun i _ => Finset.sum_nonneg fun j _ => hterm i j) ⟨i₀, Finset.mem_univ _,
      Finset.sum_pos' (fun j _ => hterm i₀ j) ⟨j₀, Finset.mem_univ _, mul_pos (exp_pos _)
        (add_pos_of_pos_of_nonneg (pow_pos hlt 2) (mul_nonneg hd' hlt.le))⟩⟩) two_pos)
  by_contra hneg
  push Not at hneg
  linarith [Finset.sum_nonpos fun k (_ : k ∈ Finset.univ) => hneg k]

/-- The antipodal pair is a critical point of `𝖤_β` for every `β`: the field
at each token is a multiple of it. -/
theorem antipodalPair_isCriticalEBeta (β : ℝ) (x : SSphere d) :
    IsCriticalEBeta d 2 β (antipodalPair d x) := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have h0 : ((antipodalPair d x 0 : SSphere d) : EucSpace d) = (x : EucSpace d) := rfl
  have h1 : ((antipodalPair d x 1 : SSphere d) : EucSpace d) = -(x : EucSpace d) := rfl
  intro i
  rw [Fin.sum_univ_two, h0, h1, smul_neg, ← sub_eq_add_neg, ← sub_smul]
  fin_cases i
  · exact proj_smul_self hx _
  · show proj d ((antipodalPair d x 1 : SSphere d) : EucSpace d) _ = 0
    rw [h1, ← neg_smul_neg]
    exact proj_smul_self (by rw [norm_neg, hx]) _

/-! ### The hypotheses are satisfiable -/

/-- The hypotheses of `sum_inner_proj_basisFun`, `half_le_hessQuad`,
`sum_hessQuad_proj_ge` and `exists_hessQuad_proj_pos` are satisfiable: in
`ℝ³`, at `β = 1`, the antipodal pair has two distinct tokens of norm `1`. -/
example : ‖((basePoint 2 : SSphere 3) : EucSpace 3)‖ = 1 ∧ 3 ≤ 3 ∧ (0 : ℝ) < 1 ∧
    NonTrivialTuple 3 2 (antipodalPair 3 (basePoint 2)) :=
  ⟨mem_sphere_zero_iff_norm.mp (basePoint 2).2, le_rfl, one_pos,
    (antipodalPair_critical_nonTrivial 3 (basePoint 2)).2⟩

end Perspective
end Transformer
