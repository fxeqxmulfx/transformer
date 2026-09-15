/-
# Normalization — The radial velocity under Pre-LN (§3 of 2510.22026v2)

The corollary of `thm: convergence` for Pre-LN and Peri-LN rests on one
explicit lower bound for the radial velocity,

  `ṙ_j = ⟨θ_j, A_j(Θ)⟩ ≥ (1 / (n e^β)) (e^β - (n - 1)) ≥ 1 / (n e^β)`,

valid whenever `n ≤ e^β`.  This file proves it.  The passage from the bound to
unconditional synchronization is an asymptotic argument and is not formalized.
-/

import Transformer.Basic
import Transformer.Normalization.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Normalization

variable (d n : ℕ)

/-- **The radial velocity in closed form.**  For `Q = K = V = I_d`,

  `⟨θ_j, A_j(Θ)⟩ = Z_j⁻¹ Σ_k e^{β ⟨θ_j, θ_k⟩} ⟨θ_j, θ_k⟩`.

Source: arXiv:2510.22026v2, §3 (proof of the corollary to `thm:
convergence`). -/
theorem inner_attentionVec_self
    (β : ℝ) (Θ : Idx n → EucSpace d) (j : Idx n) :
    inner (𝕜 := ℝ) (Θ j)
        (attentionVec d n β (ContinuousLinearMap.id ℝ (EucSpace d))
          (ContinuousLinearMap.id ℝ (EucSpace d))
          (ContinuousLinearMap.id ℝ (EucSpace d)) Θ j)
      = (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Θ j) (Θ l)))⁻¹ *
          ∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Θ j) (Θ k)) *
            inner (𝕜 := ℝ) (Θ j) (Θ k) := by
  rw [attentionVec]
  simp only [ContinuousLinearMap.coe_id', id_eq]
  rw [real_inner_smul_right, inner_sum]
  congr 1
  exact Finset.sum_congr rfl fun k _ => real_inner_smul_right (Θ j) (Θ k) _

/-- **The bound behind the corollary.**

At a unit-norm configuration and for `β > 0`, each score `⟨θ_j, θ_k⟩` lies in
`[-1, 1]`, so the partition function is at most `n e^β`, the `k = j` term of
the numerator is `e^β`, and each of the other `n - 1` terms is at least `-1`.
With `n ≤ e^β` the numerator is positive, and the two estimates combine into

  `⟨θ_j, A_j(Θ)⟩ ≥ (n e^β)⁻¹ (e^β - (n - 1))`.

Source: arXiv:2510.22026v2, §3. -/
theorem inner_attentionVec_self_lower_bound
    (β : ℝ) (hβ : 0 < β) (Θ : Idx n → EucSpace d) (hΘ : ∀ l : Idx n, ‖Θ l‖ = 1)
    (hn : (n : ℝ) ≤ Real.exp β) (j : Idx n) :
    ((n : ℝ) * Real.exp β)⁻¹ * (Real.exp β - ((n : ℝ) - 1))
      ≤ inner (𝕜 := ℝ) (Θ j)
          (attentionVec d n β (ContinuousLinearMap.id ℝ (EucSpace d))
            (ContinuousLinearMap.id ℝ (EucSpace d))
            (ContinuousLinearMap.id ℝ (EucSpace d)) Θ j) := by
  set c : Idx n → ℝ := fun k => inner (𝕜 := ℝ) (Θ j) (Θ k) with hc
  have hc_le : ∀ k : Idx n, c k ≤ 1 := by
    intro k
    have := real_inner_le_norm (Θ j) (Θ k)
    rwa [hΘ j, hΘ k, one_mul] at this
  have hc_ge : ∀ k : Idx n, -1 ≤ c k := by
    intro k
    have := abs_real_inner_le_norm (Θ j) (Θ k)
    rw [hΘ j, hΘ k, one_mul] at this
    exact neg_le_of_abs_le this
  have hcj : c j = 1 := by
    have := real_inner_self_eq_norm_mul_norm (Θ j)
    rw [hΘ j, one_mul] at this
    exact this
  -- The partition function: positive, and at most `n e^β`.
  have hZpos : (0 : ℝ) < ∑ l : Idx n, Real.exp (β * c l) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) ⟨j, Finset.mem_univ j⟩
  have hZle : (∑ l : Idx n, Real.exp (β * c l)) ≤ (n : ℝ) * Real.exp β := by
    have h := Finset.sum_le_card_nsmul Finset.univ (fun l : Idx n => Real.exp (β * c l))
      (Real.exp β) (fun l _ => Real.exp_le_exp.mpr (by nlinarith [hc_le l]))
    simpa [nsmul_eq_mul] using h
  -- The numerator: the `k = j` term is `e^β`, every other term is `≥ -1`.
  have hterm : ∀ k : Idx n, -1 ≤ Real.exp (β * c k) * c k := by
    intro k
    rcases le_or_gt 0 (c k) with h | h
    · nlinarith [Real.exp_pos (β * c k)]
    · have hexp : Real.exp (β * c k) ≤ 1 :=
        Real.exp_le_one_iff.mpr (by nlinarith)
      nlinarith [Real.exp_pos (β * c k), hc_ge k]
  have hcard : ((Finset.univ.erase j).card : ℝ) = (n : ℝ) - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_univ j), Finset.card_univ,
      Fintype.card_fin, Nat.cast_sub (Fin.pos j)]
    norm_num
  have hNge : Real.exp β - ((n : ℝ) - 1)
      ≤ ∑ k : Idx n, Real.exp (β * c k) * c k := by
    rw [← Finset.add_sum_erase Finset.univ (fun k : Idx n => Real.exp (β * c k) * c k)
      (Finset.mem_univ j)]
    have h := Finset.card_nsmul_le_sum (Finset.univ.erase j)
      (fun k : Idx n => Real.exp (β * c k) * c k) (-1) (fun k _ => hterm k)
    rw [nsmul_eq_mul, hcard] at h
    simp only [hcj, mul_one]
    linarith
  -- Combine: both factors are monotone in the right direction.
  have hNpos : (0 : ℝ) ≤ Real.exp β - ((n : ℝ) - 1) := by linarith
  rw [inner_attentionVec_self d n β Θ j]
  exact mul_le_mul (inv_anti₀ hZpos hZle) hNge hNpos (le_of_lt (inv_pos.mpr hZpos))

/-- **The Pre-LN radial velocity never stalls.**  Under `n ≤ e^β` the bound of
`inner_attentionVec_self_lower_bound` applies to `ṙ_j` of the Pre-LN scheme
directly, and is at least `1 / (n e^β) > 0`: the radius grows at a rate bounded
away from zero, which is what makes the synchronization of the corollary
unconditional.  Source: arXiv:2510.22026v2, §3. -/
theorem radialDerivative_pre_lower_bound
    (β : ℝ) (hβ : 0 < β) (θ : ℝ → Idx n → EucSpace d) (t τ : ℝ)
    (hθ : ∀ l : Idx n, ‖θ t l‖ = 1) (hn : (n : ℝ) ≤ Real.exp β) (j : Idx n) :
    ((n : ℝ) * Real.exp β)⁻¹
      ≤ radialDerivative d n β (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
          (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
          (fun _ => ContinuousLinearMap.id ℝ (EucSpace d)) θ τ .pre t j := by
  have h := inner_attentionVec_self_lower_bound d n β hβ (θ t) hθ hn j
  have hpos : (0 : ℝ) < (n : ℝ) * Real.exp β := by
    have : (0 : ℝ) < (n : ℝ) := by
      exact_mod_cast Fin.pos j
    positivity
  have hone : ((n : ℝ) * Real.exp β)⁻¹ * 1
      ≤ ((n : ℝ) * Real.exp β)⁻¹ * (Real.exp β - ((n : ℝ) - 1)) := by
    have : (1 : ℝ) ≤ Real.exp β - ((n : ℝ) - 1) := by linarith
    exact mul_le_mul_of_nonneg_left this (le_of_lt (inv_pos.mpr hpos))
  rw [radialDerivative]
  rw [mul_one] at hone
  linarith

/-- The hypotheses are satisfiable: one token on the circle, at inverse
temperature `β = 1`, has unit norm and satisfies `1 ≤ e^1`. -/
example : (0 : ℝ) < 1 ∧ (∀ l : Idx 1, ‖(fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 2)
      (1 : ℝ)) l‖ = 1) ∧ ((1 : ℕ) : ℝ) ≤ Real.exp 1 := by
  refine ⟨one_pos, fun l => ?_, ?_⟩
  · simp [PiLp.norm_single]
  · simp

end Normalization
end Transformer
