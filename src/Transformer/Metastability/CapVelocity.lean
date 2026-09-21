/-
# Metastability — the softmax velocity, tested against a direction
  (behind `rem: variance` of 2410.06833v1)

The computation the remark of §2 rests on.  Tested against a fixed direction
`w`, the `SA` velocity of a token splits into the attention-weighted gap to
the other tokens and the attention-weighted variance around it
(`inner_proj_softmax_eq`); bounding the first sum below by the leakage from
the `α`-separated tokens and the second by its restriction to a subset gives
the remark's inequality (`inner_proj_softmax_ge`, in `CapVelocityBound`).

Both are statements about a plain tuple of unit vectors: no dynamics, no cap,
no time.  `Metastability.variance_inequality` is what is left once the
envelope argument has identified `η̇_q(t)` with this velocity.
-/

import Transformer.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- **The `SA` velocity of a token, resolved against a fixed direction.**

For unit vectors `x_1,…,x_n` and any `w`,

  `⟨Proj_{x_k} (Σ_j a_{kj} x_j), w⟩
     = Σ_j a_{kj} (⟨x_j, w⟩ - ⟨x_k, w⟩)
       + ⟨x_k, w⟩ Σ_j a_{kj} ‖x_j - x_k‖² / 2`,

with `a_{kj} = e^{β⟨x_k,x_j⟩} / Σ_l e^{β⟨x_k,x_l⟩}`.  The only input is
`⟨x_k, x_j⟩ = 1 - ‖x_j - x_k‖²/2`, which holds because the `x_j` are unit
vectors; the weights need not even sum to one for the identity to hold, the
term `Σ_j a_{kj}` cancelling between the two sums.

This is the whole computation behind `rem: variance`: the remark's inequality
is what is left of it after the first sum is bounded below by the leakage and
the second is restricted to the cap. -/
theorem inner_proj_softmax_eq (β : ℝ) (x : Idx n → EucSpace d)
    (w : EucSpace d) (k : Idx n) (hx : ∀ j : Idx n, ‖x j‖ = 1) :
    inner (𝕜 := ℝ)
        (proj d (x k)
          ((∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l)))⁻¹ •
            ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) • x j)) w
      = (∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
                (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
              (inner (𝕜 := ℝ) (x j) w - inner (𝕜 := ℝ) (x k) w))
        + inner (𝕜 := ℝ) (x k) w *
            ∑ j : Idx n,
              Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
                  (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
                (‖x j - x k‖ ^ 2 / 2) := by
  have hnormsq : ∀ j : Idx n,
      ‖x j - x k‖ ^ 2 / 2 = 1 - inner (𝕜 := ℝ) (x k) (x j) := by
    intro j
    rw [norm_sub_sq_real, hx j, hx k, real_inner_comm]
    ring
  have h1 : (∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
            (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
          (inner (𝕜 := ℝ) (x j) w - inner (𝕜 := ℝ) (x k) w))
      = (∑ j : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
              (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
            inner (𝕜 := ℝ) (x j) w)
        - inner (𝕜 := ℝ) (x k) w *
            ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
              (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have h2 : (∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
            (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
          (‖x j - x k‖ ^ 2 / 2))
      = (∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
            (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))))
        - ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
                (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
              inner (𝕜 := ℝ) (x k) (x j) := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by rw [hnormsq j]; ring
  have h3 : inner (𝕜 := ℝ)
        (proj d (x k)
          ((∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l)))⁻¹ •
            ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) • x j)) w
      = (∑ j : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
              (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
            inner (𝕜 := ℝ) (x j) w)
        - (∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
                (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
              inner (𝕜 := ℝ) (x k) (x j)) * inner (𝕜 := ℝ) (x k) w := by
    simp only [proj, inner_sub_left, real_inner_smul_left, real_inner_smul_right,
      sum_inner, inner_sum, Finset.mul_sum]
    rw [Finset.sum_mul, Finset.sum_mul]
    refine congrArg₂ (· - ·) (Finset.sum_congr rfl fun j _ => by ring)
      (Finset.sum_congr rfl fun j _ => by ring)
  rw [h1, h2, h3]
  ring

end Metastability
end Transformer
