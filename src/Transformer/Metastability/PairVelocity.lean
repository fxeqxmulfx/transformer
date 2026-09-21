/-
# Metastability — the softmax velocity of a pair (behind `eq: ze.equation`)

The estimate that drives §2's differential inequality for

  `ρ_q(t) = min_{i, j ∈ I} ⟨x_i(t), x_j(t)⟩`,

stated — as in `Metastability.CapVelocity` — about a plain tuple of unit
vectors, with no dynamics in sight.  The two ingredients are

* `pair_sum_bound`, the arithmetic: one distinguished term carries the gain,
  the remaining `n - 1` cost at most `e` each;
* `inner_proj_softmax_pair`, the geometry: `⟨Proj_{x_i} v_i, x_j⟩` is bounded
  below by `(1/n)(1 - ρ²) e^{β(ρ-1)} - n e^{-(1-α)β}` when `ρ` is the
  within-`I` minimum attained at `(i, j)`;

and `inner_proj_softmax_pair_sum` (in `PairVelocitySum`) adds the two halves of
`d/dt ⟨x_i, x_j⟩`.
-/

import Transformer.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- The velocity field of `eq: SA` read on a single tuple `x` of unit
vectors:

  `v_k = Proj_{x_k} ( (Σ_l e^{β⟨x_k, x_l⟩})⁻¹ Σ_m e^{β⟨x_k, x_m⟩} x_m )`.

This is `Perspective.SA`'s right-hand side at the frozen configuration `x`;
a solution `X` of `SA` satisfies `HasDerivAt (X · i) (softmaxVel d n β (X t) i) t`
definitionally. -/
noncomputable def softmaxVel (β : ℝ) (x : Idx n → EucSpace d) (k : Idx n) :
    EucSpace d :=
  proj d (x k)
    ((∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l)))⁻¹ •
      ∑ m : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x m)) • x m)

/-- **The arithmetic behind `eq: ze.equation`.**

A sum `Σ_k a_k c_k` over `Idx n` in which one index `j` already carries `M`
and every other index costs at least `-E` is bounded below by
`M - n E`.  Stated over opaque reals so that the estimate is separated from
the softmax expressions it will be instantiated at.

Source: arXiv:2410.06833v1, §2, `eq: ze.equation` (the leakage bookkeeping). -/
theorem pair_sum_bound (M E : ℝ) (a c : Idx n → ℝ) (j : Idx n) (hE : 0 ≤ E)
    (hj : M ≤ a j * c j)
    (hrest : ∀ k : Idx n, k ≠ j → -E ≤ a k * c k) :
    M - (n : ℝ) * E ≤ ∑ k : Idx n, a k * c k := by
  have hsplit : ∑ k : Idx n, a k * c k
      = a j * c j + ∑ k ∈ Finset.univ.erase j, a k * c k :=
    (Finset.add_sum_erase _ _ (Finset.mem_univ j)).symm
  have hle : ∑ _k ∈ Finset.univ.erase j, (-E)
      ≤ ∑ k ∈ Finset.univ.erase j, a k * c k :=
    Finset.sum_le_sum fun k hk => hrest k (Finset.ne_of_mem_erase hk)
  rw [Finset.sum_const, nsmul_eq_mul] at hle
  have hcard : (((Finset.univ : Finset (Idx n)).erase j).card : ℝ) ≤ (n : ℝ) := by
    have h := Finset.card_erase_le (s := (Finset.univ : Finset (Idx n))) (a := j)
    rw [Finset.card_univ, Fintype.card_fin] at h
    exact_mod_cast h
  nlinarith [hcard, hE]

/-- The hypotheses of `pair_sum_bound` are satisfiable: `n = 1`, all data
equal to `1`, `M = E = 1`. -/
example : (1 : ℝ) - ((1 : ℕ) : ℝ) * 1 ≤ ∑ _k : Idx 1, (1 : ℝ) * 1 :=
  pair_sum_bound 1 1 1 (fun _ => 1) (fun _ => 1) 0 zero_le_one (by norm_num)
    (fun k hk => absurd (Subsingleton.elim k 0) hk)

/-- **The pair estimate.**

Let `x` be a tuple of unit vectors, `I` a set of indices, and let
`ρ = ⟨x_i, x_j⟩` be the minimum of `⟨x_k, x_l⟩` over `I × I`, attained at
`(i, j)` and nonnegative.  If every token outside `I` is `α`-separated from
`x_i`, then

  `⟨Proj_{x_i} v_i, x_j⟩ ≥ (1/n)(1 - ρ²) e^{β(ρ-1)} - n e^{-(1-α)β}`.

The `k = j` term of `⟨Proj_{x_i} v_i, x_j⟩ = Σ_k a_{ik}(⟨x_k,x_j⟩ - ⟨x_i,x_k⟩ρ)`
contributes `a_{ij}(1 - ρ²)` with `a_{ij} ≥ e^{βρ}/(n e^{β})`; a `k ∈ I`
contributes a nonnegative amount, which is where `0 ≤ ρ` is used; and a
`k ∉ I` has `a_{ik} ≤ e^{-(1-α)β}` against a bracket
`⟨x_k, x_j - ρ x_i⟩ ≥ -‖x_j - ρ x_i‖ = -√(1 - ρ²) ≥ -1`.

Source: arXiv:2410.06833v1, §2, `eq: ze.equation`. -/
theorem inner_proj_softmax_pair (β α ρ : ℝ) (hβ : 0 ≤ β)
    (x : Idx n → EucSpace d) (I : Finset (Idx n)) (i j : Idx n) (hj : j ∈ I)
    (hx : ∀ k : Idx n, ‖x k‖ = 1)
    (hρ0 : 0 ≤ ρ) (hρij : ρ = inner (𝕜 := ℝ) (x i) (x j))
    (hmin : ∀ k ∈ I, ∀ l ∈ I, ρ ≤ inner (𝕜 := ℝ) (x k) (x l))
    (hfar : ∀ k : Idx n, k ∉ I → inner (𝕜 := ℝ) (x i) (x k) ≤ α) :
    (1 / (n : ℝ)) * (1 - ρ ^ 2) * Real.exp (β * (ρ - 1))
        - (n : ℝ) * Real.exp (-((1 - α) * β))
      ≤ inner (𝕜 := ℝ) (softmaxVel d n β x i) (x j) := by
  have hle1 : ∀ k l : Idx n, inner (𝕜 := ℝ) (x k) (x l) ≤ 1 := by
    intro k l
    have h := abs_real_inner_le_norm (x k) (x l)
    rw [hx k, hx l, one_mul] at h
    exact (abs_le.mp h).2
  have hρ1 : ρ ≤ 1 := by rw [hρij]; exact hle1 i j
  have hn : (0 : ℝ) < (n : ℝ) := by exact_mod_cast Fin.pos j
  have hSpos : 0 < ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x l)) :=
    Finset.sum_pos (fun k _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩
  have hdiag : Real.exp (β * inner (𝕜 := ℝ) (x i) (x i)) = Real.exp β := by
    rw [real_inner_self_eq_norm_mul_norm, hx i]; ring_nf
  have hSge : Real.exp β ≤ ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x l)) := by
    rw [← hdiag]
    exact Finset.single_le_sum
      (f := fun l : Idx n => Real.exp (β * inner (𝕜 := ℝ) (x i) (x l)))
      (fun l _ => (Real.exp_pos _).le) (Finset.mem_univ i)
  have hSle : (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x l)))
      ≤ (n : ℝ) * Real.exp β := by
    calc (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x l)))
        ≤ ∑ _l : Idx n, Real.exp β :=
          Finset.sum_le_sum fun l _ => Real.exp_le_exp.mpr (by nlinarith [hle1 i l])
      _ = (n : ℝ) * Real.exp β := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  simp only [softmaxVel]
  set S := ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x l)) with hSdef
  clear_value S
  -- resolve the velocity against `x j`
  have hA : inner (𝕜 := ℝ)
        (proj d (x i)
          (S⁻¹ • ∑ m : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x m)) • x m)) (x j)
      = ∑ k : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (x i) (x k)) / S
            * (inner (𝕜 := ℝ) (x k) (x j) - inner (𝕜 := ℝ) (x i) (x k) * ρ) := by
    have h1 : ∑ k : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (x i) (x k)) / S
            * (inner (𝕜 := ℝ) (x k) (x j) - inner (𝕜 := ℝ) (x i) (x k) * ρ)
        = (∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x k)) / S
              * inner (𝕜 := ℝ) (x k) (x j))
          - (∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x k)) / S
              * inner (𝕜 := ℝ) (x i) (x k)) * ρ := by
      rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun k _ => by ring
    rw [h1, hρij]
    simp only [proj, inner_sub_left, real_inner_smul_left, real_inner_smul_right,
      sum_inner, inner_sum, Finset.mul_sum]
    rw [Finset.sum_mul, Finset.sum_mul]
    refine congrArg₂ (· - ·) (Finset.sum_congr rfl fun k _ => by ring)
      (Finset.sum_congr rfl fun k _ => by ring)
  rw [hA]
  -- the `k = j` term
  have hcj : inner (𝕜 := ℝ) (x j) (x j) - inner (𝕜 := ℝ) (x i) (x j) * ρ
      = 1 - ρ ^ 2 := by
    rw [real_inner_self_eq_norm_mul_norm, hx j, ← hρij]; ring
  have haj : (1 / (n : ℝ)) * Real.exp (β * (ρ - 1))
      ≤ Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) / S := by
    rw [← hρij, le_div_iff₀ hSpos]
    have hexp : Real.exp (β * (ρ - 1)) * Real.exp β = Real.exp (β * ρ) := by
      rw [← Real.exp_add]; congr 1; ring
    have hcancel : (1 / (n : ℝ)) * Real.exp (β * (ρ - 1)) * ((n : ℝ) * Real.exp β)
        = Real.exp (β * (ρ - 1)) * Real.exp β := by
      field_simp
    have hstep : (1 / (n : ℝ)) * Real.exp (β * (ρ - 1)) * S
        ≤ (1 / (n : ℝ)) * Real.exp (β * (ρ - 1)) * ((n : ℝ) * Real.exp β) :=
      mul_le_mul_of_nonneg_left hSle (by positivity)
    rw [hcancel, hexp] at hstep
    exact hstep
  have hjb : (1 / (n : ℝ)) * (1 - ρ ^ 2) * Real.exp (β * (ρ - 1))
      ≤ (Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) / S)
          * (inner (𝕜 := ℝ) (x j) (x j) - inner (𝕜 := ℝ) (x i) (x j) * ρ) := by
    rw [hcj]
    have h1ρ : (0 : ℝ) ≤ 1 - ρ ^ 2 := by nlinarith
    calc (1 / (n : ℝ)) * (1 - ρ ^ 2) * Real.exp (β * (ρ - 1))
        = ((1 / (n : ℝ)) * Real.exp (β * (ρ - 1))) * (1 - ρ ^ 2) := by ring
      _ ≤ (Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) / S) * (1 - ρ ^ 2) :=
          mul_le_mul_of_nonneg_right haj h1ρ
  refine pair_sum_bound n _ (Real.exp (-((1 - α) * β)))
    (fun k => Real.exp (β * inner (𝕜 := ℝ) (x i) (x k)) / S)
    (fun k => inner (𝕜 := ℝ) (x k) (x j) - inner (𝕜 := ℝ) (x i) (x k) * ρ)
    j (Real.exp_pos _).le hjb ?_
  intro k _
  have hanneg : (0 : ℝ) ≤ Real.exp (β * inner (𝕜 := ℝ) (x i) (x k)) / S :=
    div_nonneg (Real.exp_pos _).le hSpos.le
  have hEpos : (0 : ℝ) < Real.exp (-((1 - α) * β)) := Real.exp_pos _
  by_cases hk : k ∈ I
  · -- a token inside `I` moves `x_i` towards `x_j`
    have h1 : ρ ≤ inner (𝕜 := ℝ) (x k) (x j) := hmin k hk j hj
    have h3 : (0 : ℝ)
        ≤ inner (𝕜 := ℝ) (x k) (x j) - inner (𝕜 := ℝ) (x i) (x k) * ρ := by
      nlinarith [hle1 i k]
    nlinarith [mul_nonneg hanneg h3]
  · -- a token outside `I` leaks at most `e^{-(1-α)β}`
    have hak : Real.exp (β * inner (𝕜 := ℝ) (x i) (x k)) / S
        ≤ Real.exp (-((1 - α) * β)) := by
      rw [div_le_iff₀ hSpos]
      have hnum : Real.exp (β * inner (𝕜 := ℝ) (x i) (x k)) ≤ Real.exp (β * α) :=
        Real.exp_le_exp.mpr (by nlinarith [hfar k hk])
      have hprod : Real.exp (-((1 - α) * β)) * Real.exp β = Real.exp (β * α) := by
        rw [← Real.exp_add]; congr 1; ring
      have hmul : Real.exp (-((1 - α) * β)) * Real.exp β
          ≤ Real.exp (-((1 - α) * β)) * S :=
        mul_le_mul_of_nonneg_left hSge (Real.exp_pos _).le
      linarith
    -- the bracket is `⟨x_k, x_j - ρ x_i⟩`, and `‖x_j - ρ x_i‖² = 1 - ρ² ≤ 1`
    have hjρ : ‖x j - ρ • x i‖ ≤ 1 := by
      have hsq : ‖x j - ρ • x i‖ ^ 2 = 1 - ρ ^ 2 := by
        rw [norm_sub_sq_real, norm_smul, real_inner_smul_right, hx i, hx j,
          Real.norm_eq_abs, mul_pow, sq_abs, real_inner_comm (x i) (x j), ← hρij]
        ring
      nlinarith [norm_nonneg (x j - ρ • x i), sq_nonneg ρ]
    have hck : (-1 : ℝ)
        ≤ inner (𝕜 := ℝ) (x k) (x j) - inner (𝕜 := ℝ) (x i) (x k) * ρ := by
      have h := abs_real_inner_le_norm (x k) (x j - ρ • x i)
      rw [hx k, one_mul, inner_sub_right, real_inner_smul_right,
        real_inner_comm (x i) (x k)] at h
      linarith [(abs_le.mp (h.trans hjρ)).1]
    nlinarith [mul_nonneg hanneg (by linarith : (0 : ℝ)
      ≤ inner (𝕜 := ℝ) (x k) (x j) - inner (𝕜 := ℝ) (x i) (x k) * ρ + 1)]

/-- The hypotheses of `inner_proj_softmax_pair` are satisfiable: `d = n = 1`,
the single unit vector `v`, `I = univ`, `ρ = 1`. -/
example (v : EucSpace 1) (hv : ‖v‖ = 1) :
    (1 / ((1 : ℕ) : ℝ)) * (1 - (1 : ℝ) ^ 2) * Real.exp (1 * ((1 : ℝ) - 1))
        - ((1 : ℕ) : ℝ) * Real.exp (-((1 - (1 : ℝ)) * 1))
      ≤ inner (𝕜 := ℝ) (softmaxVel 1 1 1 (fun _ => v) 0) v := by
  have hvv : inner (𝕜 := ℝ) v v = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_mul_norm, hv]; ring
  exact inner_proj_softmax_pair 1 1 1 1 1 zero_le_one (fun _ => v) Finset.univ 0 0
    (Finset.mem_univ 0) (fun _ => hv) zero_le_one hvv.symm (fun _ _ _ _ => le_of_eq hvv.symm)
    (fun k hk => absurd (Finset.mem_univ k) hk)

end Metastability
end Transformer
