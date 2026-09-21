/-
# Metastability — both halves of `d/dt ⟨x_i, x_j⟩` (behind `eq: ze.equation`)

`inner_proj_softmax_pair` bounds `⟨Proj_{x_i} v_i, x_j⟩`; adding it to itself
with `i` and `j` exchanged gives the right-hand side of `eq: ze.equation`,
with the paper's leakage constant `2 n e^{-(1-α)β}`.
-/

import Transformer.Metastability.PairVelocity

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- **The two halves of `d/dt ⟨x_i, x_j⟩`.**

Adding `inner_proj_softmax_pair` to itself with `i` and `j` exchanged, and
weakening `1 - ρ²` to `ρ(1 - ρ)`:

  `⟨Proj_{x_i} v_i, x_j⟩ + ⟨x_i, Proj_{x_j} v_j⟩
     ≥ (2/n) ρ (1 - ρ) e^{β(ρ-1)} - 2 n e^{-(1-α)β}`.

Source: arXiv:2410.06833v1, §2, `eq: ze.equation`. -/
theorem inner_proj_softmax_pair_sum (β α ρ : ℝ) (hβ : 0 ≤ β)
    (x : Idx n → EucSpace d) (I : Finset (Idx n)) (i j : Idx n)
    (hi : i ∈ I) (hj : j ∈ I)
    (hx : ∀ k : Idx n, ‖x k‖ = 1)
    (hρ0 : 0 ≤ ρ) (hρij : ρ = inner (𝕜 := ℝ) (x i) (x j))
    (hmin : ∀ k ∈ I, ∀ l ∈ I, ρ ≤ inner (𝕜 := ℝ) (x k) (x l))
    (hfar : ∀ k ∈ I, ∀ l : Idx n, l ∉ I → inner (𝕜 := ℝ) (x k) (x l) ≤ α) :
    (2 / (n : ℝ)) * ρ * (1 - ρ) * Real.exp (β * (ρ - 1))
        - 2 * (n : ℝ) * Real.exp (-((1 - α) * β))
      ≤ inner (𝕜 := ℝ) (softmaxVel d n β x i) (x j)
        + inner (𝕜 := ℝ) (x i) (softmaxVel d n β x j) := by
  have hρ1 : ρ ≤ 1 := by
    rw [hρij]
    have h := abs_real_inner_le_norm (x i) (x j)
    rw [hx i, hx j, one_mul] at h
    exact (abs_le.mp h).2
  have h1 := inner_proj_softmax_pair d n β α ρ hβ x I i j hj hx hρ0 hρij hmin
    (fun k hk => hfar i hi k hk)
  have hρji : ρ = inner (𝕜 := ℝ) (x j) (x i) := by
    rw [hρij, real_inner_comm]
  have h2 := inner_proj_softmax_pair d n β α ρ hβ x I j i hi hx hρ0 hρji hmin
    (fun k hk => hfar j hj k hk)
  have hcomm : inner (𝕜 := ℝ) (x i) (softmaxVel d n β x j)
      = inner (𝕜 := ℝ) (softmaxVel d n β x j) (x i) :=
    (real_inner_comm (x i) (softmaxVel d n β x j)).symm
  have hgain : (2 / (n : ℝ)) * ρ * (1 - ρ) * Real.exp (β * (ρ - 1))
      ≤ 2 * ((1 / (n : ℝ)) * (1 - ρ ^ 2) * Real.exp (β * (ρ - 1))) := by
    have hc : (0 : ℝ) ≤ 1 / (n : ℝ) := by positivity
    have hE : (0 : ℝ) < Real.exp (β * (ρ - 1)) := Real.exp_pos _
    have hkey : ρ * (1 - ρ) ≤ 1 - ρ ^ 2 := by nlinarith
    have h2n : (2 : ℝ) / (n : ℝ) = 2 * (1 / (n : ℝ)) := by ring
    rw [h2n]
    nlinarith [mul_le_mul_of_nonneg_left hkey (mul_nonneg hc hE.le)]
  rw [hcomm]
  linarith

/-- The hypotheses of `inner_proj_softmax_pair_sum` are satisfiable: `d = n = 1`,
the single unit vector `v`, `I = univ`, `ρ = 1`. -/
example (v : EucSpace 1) (hv : ‖v‖ = 1) :
    (2 / ((1 : ℕ) : ℝ)) * 1 * (1 - (1 : ℝ)) * Real.exp (1 * ((1 : ℝ) - 1))
        - 2 * ((1 : ℕ) : ℝ) * Real.exp (-((1 - (1 : ℝ)) * 1))
      ≤ inner (𝕜 := ℝ) (softmaxVel 1 1 1 (fun _ => v) 0) v
        + inner (𝕜 := ℝ) v (softmaxVel 1 1 1 (fun _ => v) 0) := by
  have hvv : inner (𝕜 := ℝ) v v = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_mul_norm, hv]; ring
  exact inner_proj_softmax_pair_sum 1 1 1 1 1 zero_le_one (fun _ => v) Finset.univ 0 0
    (Finset.mem_univ 0) (Finset.mem_univ 0) (fun _ => hv) zero_le_one hvv.symm
    (fun _ _ _ _ => le_of_eq hvv.symm) (fun _ _ k hk => absurd (Finset.mem_univ k) hk)

end Metastability
end Transformer
