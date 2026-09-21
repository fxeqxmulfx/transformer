/-
# Metastability — leaving the cap, mean-field (§5 of 2410.06833v1)

Step 2 of the proof of `thm: metastability MF`: `claim: de sortie de cap`,
the time `T_*(q, c)` at which `η_q V_q e^{-(1-η_q)β}` first drops below
`2e^{-cβ}`, and `eq: v.small`, the variance bound read off at that time.
The vocabulary is `Metastability.MeanField`; the hypotheses are witnessed by
the static solution of `Metastability.MeanFieldStatic`.
-/

import Transformer.Metastability.MeanFieldStatic

open Real MeasureTheory

namespace Transformer
namespace Metastability

open Perspective

variable (d : ℕ)

/-- **Claim (claim: de sortie de cap).** In the setting of
`thm: metastability MF`, for every cap `q` and every `c > 0`, the set

  `{t ∈ [0, T_esc] : η_q(t) V_q(t) e^{-(1-η_q(t))β} ≤ 2e^{-cβ}}`

is non-empty and its infimum `T_*(q, c)` satisfies
`T_*(q, c) < (4ε/k) e^{(c-8ε)β}`.

`η_q` is `capMin`, `V_q` is `capVariance` along a minimising selection `x`
(`IsCapArgmin`), and the window `[0, T_esc]` is `escapeWindow`.

Not proved here.

Source: arXiv:2410.06833v1, §5, proof of `thm: metastability MF`, Step 2,
`claim: de sortie de cap`. -/
theorem cap_exit
    (β ε c : ℝ) (k : ℕ) (w : Idx k → SSphere d) (ν : Idx k → ProbSphere d)
    (μ₀ : ProbSphere d) (μ : ℝ → ProbSphere d) (Φ : ℝ → SSphere d → SSphere d)
    (hβ : 1 < β) (hε : 0 < ε) (hε16 : ε < 1 / 16) (hk : 0 < k)
    (hν : ∀ q : Idx k, (ν q : Measure (SSphere d)).support ⊆ sphericalCap d (w q) ε)
    (hμ₀ : (μ₀ : Measure (SSphere d))
      = ((k : ℝ)⁻¹).toNNReal • ∑ q : Idx k, (ν q : Measure (SSphere d)))
    (hsep : 8 * ε < γβ k β (αDist d k w ε) ε)
    (hμ0 : μ 0 = μ₀) (hμ : meanFieldPDE d β μ) (hΦ : IsMFFlow d β μ Φ)
    (q : Idx k) (x : ℝ → SSphere d) (hx : IsCapArgmin d Φ (w q) ε x) (hc : 0 < c) :
    (capExitSet d β c ε k w μ₀ Φ q x).Nonempty ∧
      sInf (capExitSet d β c ε k w μ₀ Φ q x) < 4 * ε / k * Real.exp ((c - 8 * ε) * β) := by
  sorry

/-- **Equation (eq: v.small).** In the setting of `thm: metastability MF`,
for `λ ∈ (8ε, γ)`,

  `V_q(T_*(q, λ)) ≤ e^{-λβ}`.

**A gap in the source.**  The source derives this from
`claim: de sortie de cap` and `η_q(T_*) ≥ 1 - 8ε > 1/2`.  At `T_*` those give
`V_q ≤ 2e^{-λβ} / (η_q e^{-(1-η_q)β}) ≤ 4 e^{-(λ-8ε)β}`, which is weaker than
the displayed bound by the factor `4e^{8εβ}`; the argument written does not
reach the claim.  The statement here is the source's, unchanged, and is not
known to follow from what precedes it.

Not proved here.

Source: arXiv:2410.06833v1, §5, proof of `thm: metastability MF`, Step 2,
`eq: v.small`. -/
theorem variance_small
    (β ε lam : ℝ) (k : ℕ) (w : Idx k → SSphere d) (ν : Idx k → ProbSphere d)
    (μ₀ : ProbSphere d) (μ : ℝ → ProbSphere d) (Φ : ℝ → SSphere d → SSphere d)
    (hβ : 1 < β) (hε : 0 < ε) (hε16 : ε < 1 / 16) (hk : 0 < k)
    (hν : ∀ q : Idx k, (ν q : Measure (SSphere d)).support ⊆ sphericalCap d (w q) ε)
    (hμ₀ : (μ₀ : Measure (SSphere d))
      = ((k : ℝ)⁻¹).toNNReal • ∑ q : Idx k, (ν q : Measure (SSphere d)))
    (hμ0 : μ 0 = μ₀) (hμ : meanFieldPDE d β μ) (hΦ : IsMFFlow d β μ Φ)
    (q : Idx k) (x : ℝ → SSphere d) (hx : IsCapArgmin d Φ (w q) ε x)
    (hlam : 8 * ε < lam) (hlamγ : lam < γβ k β (αDist d k w ε) ε) :
    capVariance d μ₀ Φ (w q) ε x (sInf (capExitSet d β lam ε k w μ₀ Φ q x))
      ≤ Real.exp (-lam * β) := by
  sorry

/-- The hypotheses of `cap_exit` and `variance_small` are satisfiable: on
`𝕊^0`, one cap around `p`, `ν = μ_0 = δ_p` at rest, the identity flow with
`x ≡ p` its minimising selection, `β = 1000`, `ε = 1/32`, and `c = λ` the
midpoint of `(8ε, γ)` (`Metastability.MeanFieldStatic`). -/
example : let p := basePoint 0
    let γ := γβ 1 1000 (αDist 1 1 (fun _ => p) (1 / 32)) (1 / 32)
    (1 : ℝ) < 1000 ∧ (0 : ℝ) < 1 / 32 ∧ (1 / 32 : ℝ) < 1 / 16 ∧ 0 < 1 ∧
      (∀ q : Idx 1, ((fun _ => diracProb 1 p) q : Measure (SSphere 1)).support
        ⊆ sphericalCap 1 ((fun _ => p) q) (1 / 32)) ∧
      ((diracProb 1 p : ProbSphere 1) : Measure (SSphere 1))
        = (((1 : ℕ) : ℝ)⁻¹).toNNReal •
            ∑ q : Idx 1, ((fun _ => diracProb 1 p) q : Measure (SSphere 1)) ∧
      8 * (1 / 32 : ℝ) < γ ∧
      meanFieldPDE 1 1000 (fun _ => diracProb 1 p) ∧
      IsMFFlow 1 1000 (fun _ => diracProb 1 p) (fun _ x => x) ∧
      IsCapArgmin 1 (fun _ x => x) p (1 / 32) (fun _ => p) ∧
      0 < (8 * (1 / 32) + γ) / 2 ∧
      8 * (1 / 32 : ℝ) < (8 * (1 / 32) + γ) / 2 ∧ (8 * (1 / 32) + γ) / 2 < γ := by
  intro p γ
  have hsep : 8 * (1 / 32 : ℝ) < γ := separated_one_cap p
  refine ⟨by norm_num, by norm_num, by norm_num, one_pos, fun _ y hy => ?_, by simp, hsep,
    meanFieldPDE_const_diracProb 1000 p, isMFFlow_id_diracProb 1000 p,
    isCapArgmin_id_dim_one p _ (by norm_num) (by norm_num), by linarith, by linarith,
    by linarith⟩
  rw [Interpolation.eq_of_mem_support_dirac hy]
  simp only [sphericalCap, Set.mem_ofPred_eq, real_inner_self_eq_norm_mul_norm,
    mem_sphere_zero_iff_norm.mp p.2]
  norm_num

end Metastability
end Transformer
