/-
# Normalization — Initial and terminal token velocities (§4.2–§4.3 of 2510.22026v2)

* `Theorem thm: initial-velocity`  — uniform bound on `‖A_j(0)‖` for random
                                      directional init (the deterministic
                                      part, `‖A_j‖ ≤ 1`, is proved),
* `Theorem thm: preln-slow`        — radial growth `r_k(t) ≥ (1 - δ) t`
                                      (proved, as a velocity bound) and
                                      `d/dt Var(t)` rates for each scheme
                                      (asymptotic, not formalized).
-/

import Transformer.Basic
import Transformer.Normalization.Basic
import Transformer.Normalization.Radial

open scoped BigOperators
open Real

namespace Transformer
namespace Normalization

open Normalization

variable (d n : ℕ)

/-- **The attention vector is a convex combination.**

`A_j(Θ) = Z_j⁻¹ Σ_k e^{β ⟨Q θ_j, K θ_k⟩} V θ_k` averages the vectors `V θ_k`
against positive weights summing to `Z_j`, so with unit-norm tokens and
`‖V‖_op ≤ 1`,

  `‖A_j(Θ)‖ ≤ 1`,

whatever `Q`, `K` and `β` are.  This is the deterministic bound under the
norm hypotheses of `thm: initial-velocity`; the theorem's point is the much
sharper `C (√(log n / n) + log n / d)`, which holds only with high probability
over an i.i.d. uniform directional initialization, and that is not formalized.
Source: arXiv:2510.22026v2, §4.2. -/
theorem norm_attentionVec_le_one
    (β : ℝ) (Q K V : ParamMatrix d) (hV : ‖V‖ ≤ 1)
    (Θ : Idx n → EucSpace d) (hΘ : ∀ l : Idx n, ‖Θ l‖ = 1) (j : Idx n) :
    ‖attentionVec d n β Q K V Θ j‖ ≤ 1 := by
  have hZpos : (0 : ℝ) < ∑ l : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ l))) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) ⟨j, Finset.mem_univ j⟩
  have hnum : ‖∑ k : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ k))) • V (Θ k)‖
      ≤ ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ l))) := by
    refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun k _ => ?_)
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    have hVk : ‖V (Θ k)‖ ≤ 1 := by
      have := V.le_opNorm (Θ k)
      rw [hΘ k, mul_one] at this
      linarith
    nlinarith [Real.exp_pos (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ k))), norm_nonneg (V (Θ k))]
  rw [attentionVec, norm_smul, norm_inv, Real.norm_eq_abs, abs_of_pos hZpos]
  calc (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ l))))⁻¹ *
        ‖∑ k : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ k))) • V (Θ k)‖
      ≤ (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ l))))⁻¹ *
          ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ l))) :=
        mul_le_mul_of_nonneg_left hnum (le_of_lt (inv_pos.mpr hZpos))
    _ = 1 := inv_mul_cancel₀ (ne_of_gt hZpos)

/-- The hypotheses are satisfiable: the zero map has operator norm `0 ≤ 1`,
and the first standard basis vector has norm `1`. -/
example (n : ℕ) : ‖(0 : ParamMatrix 1)‖ ≤ 1 ∧
    ∀ l : Idx n, ‖(fun _ : Idx n => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) l‖ = 1 := by
  refine ⟨by simp, fun l => ?_⟩
  simp

/-- The empirical *intra-cluster variance* used in `thm: preln-slow`:

  `Var(t) = (1/n) Σ_k ‖θ_k(t) - θ̄(t)‖²`,
  with `θ̄ = (1/n) Σ_j θ_j`. -/
noncomputable def intraClusterVar
    (n : ℕ) (θ : ℝ → Idx n → EucSpace d) (t : ℝ) : ℝ :=
  let θbar : EucSpace d := ((n : ℝ)⁻¹) • ∑ j : Idx n, θ t j
  ((n : ℝ)⁻¹) * ∑ k : Idx n, ‖θ t k - θbar‖^2

/-- **Theorem (thm: preln-slow), radial growth.**

In the local-cone initialization `⟨θ_j, θ_k⟩ ≥ 1 - δ` the radial velocity of
the Pre-LN scheme is at least `1 - δ`:

  `ṙ_k = ⟨θ_k, A_k(Θ)⟩ ≥ 1 - δ`,

because `⟨θ_k, A_k⟩` is the average of the scores `⟨θ_k, θ_j⟩` against the
positive attention weights, and every score is at least `1 - δ`.  Integrating
this is the paper's `r_k(t) ≥ (1 - δ) t`.

The rest of `thm: preln-slow` -- the per-scheme rates `d/dt Var(t)` of
`intraClusterVar` -- is asymptotic (`-Θ(·)`) and is not formalized.
Source: arXiv:2510.22026v2, §4.3. -/
theorem radialDerivative_pre_ge_of_localCone
    (β δ : ℝ) (θ : ℝ → Idx n → EucSpace d) (t τ : ℝ)
    (hcone : ∀ j k : Idx n, 1 - δ ≤ inner (𝕜 := ℝ) (θ t j) (θ t k)) (k : Idx n) :
    1 - δ ≤ radialDerivative d n β (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace d)) θ τ .pre t k := by
  have hZpos : (0 : ℝ) < ∑ l : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) (θ t k) (θ t l)) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) ⟨k, Finset.mem_univ k⟩
  have hsum : (1 - δ) * ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (θ t k) (θ t l))
      ≤ ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (θ t k) (θ t l)) *
          inner (𝕜 := ℝ) (θ t k) (θ t l) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun l _ => by
      nlinarith [Real.exp_pos (β * inner (𝕜 := ℝ) (θ t k) (θ t l)), hcone k l]
  rw [radialDerivative, inner_attentionVec_self d n β (θ t) k]
  calc 1 - δ
      = (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (θ t k) (θ t l)))⁻¹ *
          ((1 - δ) * ∑ l : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (θ t k) (θ t l))) := by
        field_simp
    _ ≤ _ := mul_le_mul_of_nonneg_left hsum (le_of_lt (inv_pos.mpr hZpos))

/-- The local-cone hypothesis is satisfiable, at `δ = 0`: a configuration of
`n` copies of a single unit vector has all scores equal to `1`. -/
example (n : ℕ) : ∀ j k : Idx n,
    1 - (0 : ℝ) ≤ inner (𝕜 := ℝ) ((fun _ : Idx n => EuclideanSpace.single (0 : Fin 1)
      (1 : ℝ)) j) ((fun _ : Idx n => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) k) := by
  intro j k
  simp

end Normalization
end Transformer
