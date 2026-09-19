/-
# Appendix B — `eq: claim.yury`, proved

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`eq: dr1` holds for every skew direction `B`; summing it over the family of
`Perspective.russian_trick_pairs` — the family that computes both `Σ_k B_k²`
and `Σ_k ⟨B_k x, y⟩²` — collapses the two `B`-dependent terms into

  `β (1 - ⟨x_i, x_j⟩²) - (d-1) ⟨x_i, x_j⟩ = -g_β(θ_{ij})`,

with `θ_{ij} = arccos ⟨x_i, x_j⟩`, which is `eq: claim.yury`.
-/

import Transformer.Perspective.AppendixB_HighD
import Transformer.Perspective.RussianPairs

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (eq: claim.yury).** *Higher-dimensional generalization of
`eq: taylor3`.*

For a point `(x_1,…,x_n) ∈ (𝕊^{d-1})^n` of `𝖤_β` with non-positive Hessian,
and `θ_{ij} = arccos ⟨x_i, x_j⟩ ∈ [0, π]` the geodesic distance,

  `Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} g_β(θ_{ij}) ≥ 0`.

**What the source says and what is changed here.**  The same two deviations as
in `Perspective.dr1_skew_inequality`, which this is summed from: `β ≠ 0` is
added (at `β = 0` the hypothesis `EBetaHessianNonPos` is vacuous and the
conclusion reads `0 ≤ (d-1) Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} ⟨x_i, x_j⟩`, which
`Perspective.taylor_eq` refutes at a non-trivial critical point), and
`IsCriticalEBeta` is dropped, criticality entering the proof nowhere.

Source: arXiv:2312.10794v5, Appendix B, `eq: claim.yury`. -/
theorem claim_yury
    (β : ℝ) (hβ : β ≠ 0) (X : SphereTuple d n) (𝒮 : Finset (Idx n))
    (h_hess : EBetaHessianNonPos d n β X) :
    0 ≤ ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
      g_β_d d β
        (Real.arccos (inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))) := by
  classical
  obtain ⟨B, hskew, hsq, hcross⟩ := russian_trick_pairs d
  set F : Idx d × Idx d → Idx n → Idx n → ℝ := fun k i j =>
    Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
      * (β * (inner (𝕜 := ℝ) (B k (X i)) ((X j : EucSpace d))) ^ 2
          + inner (𝕜 := ℝ) (B k (B k (X i))) ((X j : EucSpace d))) with hF
  -- `eq: dr1` in each of the `d²` directions, summed.
  have hle : ∑ k : Idx d × Idx d, ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, F k i j ≤ 0 :=
    Finset.sum_nonpos fun k _ =>
      dr1_skew_inequality d n β hβ X 𝒮 (B k) (hskew k) h_hess
  have hswap : ∑ k : Idx d × Idx d, ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, F k i j
      = ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, ∑ k : Idx d × Idx d, F k i j := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_comm
  -- In each pair the family collapses the two `B`-dependent terms.
  have hkey : ∀ i j : Idx n, ∑ k : Idx d × Idx d, F k i j
      = -g_β_d d β
          (Real.arccos (inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))) := by
    intro i j
    have hxi : ‖(X i : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (X i).2
    have hxj : ‖(X j : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (X j).2
    set u : ℝ := inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) with hu
    have habs : |u| ≤ 1 := by
      have h := abs_real_inner_le_norm ((X i : EucSpace d)) ((X j : EucSpace d))
      rwa [hxi, hxj, one_mul] at h
    have hu1 : -1 ≤ u := neg_le_of_abs_le habs
    have hu2 : u ≤ 1 := le_of_abs_le habs
    have hcos : Real.cos (Real.arccos u) = u := Real.cos_arccos hu1 hu2
    have hsin : Real.sin (Real.arccos u) ^ 2 = 1 - u ^ 2 := by
      rw [Real.sin_arccos, Real.sq_sqrt]
      nlinarith [habs, abs_nonneg u, sq_abs u]
    -- `Σ_k ⟨B_k² x_i, x_j⟩ = -(d-1) ⟨x_i, x_j⟩`
    have hB2 : ∑ k : Idx d × Idx d,
        inner (𝕜 := ℝ) (B k (B k ((X i : EucSpace d)))) ((X j : EucSpace d))
          = (-((d : ℝ) - 1)) * u := by
      rw [← sum_inner, hsq ((X i : EucSpace d)), real_inner_smul_left]
    -- `Σ_k ⟨B_k x_i, x_j⟩² = 1 - ⟨x_i, x_j⟩²`
    have hB1 : ∑ k : Idx d × Idx d,
        (inner (𝕜 := ℝ) (B k ((X i : EucSpace d))) ((X j : EucSpace d))) ^ 2
          = 1 - u ^ 2 := by
      rw [hcross ((X i : EucSpace d)) ((X j : EucSpace d)), hxi, hxj]
      ring
    calc ∑ k : Idx d × Idx d, F k i j
        = Real.exp (β * u) * ∑ k : Idx d × Idx d,
            (β * (inner (𝕜 := ℝ) (B k ((X i : EucSpace d))) ((X j : EucSpace d))) ^ 2
              + inner (𝕜 := ℝ) (B k (B k ((X i : EucSpace d)))) ((X j : EucSpace d))) := by
          rw [Finset.mul_sum]
      _ = Real.exp (β * u) * (β * (1 - u ^ 2) + (-((d : ℝ) - 1)) * u) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, hB1, hB2]
      _ = -g_β_d d β (Real.arccos u) := by
          rw [g_β_d, hcos, hsin]
          ring
  rw [hswap, Finset.sum_congr rfl fun i (_ : i ∈ 𝒮) =>
      Finset.sum_congr rfl fun j (_ : j ∈ 𝒮ᶜ) => hkey i j] at hle
  have hneg : ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
      -g_β_d d β (inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))).arccos
        = -∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
            g_β_d d β (inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))).arccos := by
    simp only [Finset.sum_neg_distrib]
  rw [hneg, neg_nonpos] at hle
  exact hle

/-- The hypotheses of `claim_yury` are satisfiable: `β = 1` is non-zero, and
the single token has non-positive Hessian. -/
example : (1 : ℝ) ≠ 0 ∧ EBetaHessianNonPos 1 1 1 singleToken :=
  ⟨one_ne_zero, singleToken_isSkew_critical_hessianNonPos.2.2⟩

end Perspective
end Transformer
