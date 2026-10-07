/-
# Metastability — leaving the cap, mean-field (§5 of 2410.06833v1)

Step 2 of the proof of `thm: metastability MF`: `claim: de sortie de cap`,
the time `T_*(q, c)` at which `η_q V_q e^{-(1-η_q)β}` first drops below
`2e^{-cβ}`, and `eq: v.small`, refuted here by a stationary Dirac solution on the circle.
The vocabulary is `Metastability.MeanField`; the hypotheses are witnessed by
the static solution of `Metastability.MeanFieldStatic`.
-/

import Transformer.Metastability.MeanFieldStatic
import Transformer.Metastability.Section5_VarianceCounterexample

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

/-- **Counterexample to equation (eq: v.small).** The source claims
`V_q(T_*(q,λ)) ≤ exp(-λβ)` for `8ε < λ < γ`.

On the circle take one cap around `(1,0)`, `ν = μ₀ = δ_(1,0)`,
`β = 1000`, `ε = 1/10000` and `λ = log(15000)/1000`.
The constant measure is an actual solution of `eq: mean.field.pde`.
The explicit `diracFlow` solves `eq: flow.map` from every sphere point;
its push-forward at the mass agrees with this solution. A boundary
trajectory is a minimizer of the whole transported cap at every time.
The measure curve is continuous. Its push-forward representation and
membership of time zero in the stopping set are recorded explicitly.

Every original hypothesis is witnessed below. At time zero,
`η = 1-ε`, `V = ε` and the stopping threshold is already satisfied,
so `T_* = 0`. But `V(T_*) = 1/10000 > 1/15000 = exp(-λβ)`.
The stopping set is nonempty, so this does not use an empty-set infimum.
The one-cap convention is `alphaDist_one`, with `α = 0`; moreover
`variance_counter_gamma_lower_bound` proves the source's asymptotic
nondegeneracy of `γ` for these caps.

This replaces the false statement with a counterexample. The stopping
threshold does imply `variance_le_of_mem_capExitSet`, which retains
the factor `2` and the positive exponential denominator. Neither the
cap-exit claim nor the main mean-field theorem is assumed in this proof.

Source: arXiv:2410.06833v1, §5, proof of `thm: metastability MF`,
Step 2, `eq: v.small`. -/
theorem variance_small_counterexample :
    ∃ (β ε lam : ℝ) (k : ℕ) (w : Idx k → SSphere 2) (ν : Idx k → ProbSphere 2)
      (μ₀ : ProbSphere 2) (μ : ℝ → ProbSphere 2) (Φ : ℝ → SSphere 2 → SSphere 2)
      (q : Idx k) (x : ℝ → SSphere 2),
      1 < β ∧ 0 < ε ∧ ε < 1 / 16 ∧ 0 < k ∧
      (∀ q : Idx k, (ν q : Measure (SSphere 2)).support ⊆ sphericalCap 2 (w q) ε) ∧
      (μ₀ : Measure (SSphere 2))
        = ((k : ℝ)⁻¹).toNNReal • ∑ q : Idx k, (ν q : Measure (SSphere 2)) ∧
      μ 0 = μ₀ ∧ Continuous μ ∧
      meanFieldPDE 2 β μ ∧ IsMFFlow 2 β μ Φ ∧
      (∀ t : ℝ,
        (μ₀ : Measure (SSphere 2)).map (Φ t) = (μ t : Measure (SSphere 2))) ∧
      IsCapArgmin 2 Φ (w q) ε x ∧ 8 * ε < lam ∧
      lam < γβ k β (αDist 2 k w ε) ε ∧
      8 * ε < γβ k β (αDist 2 k w ε) ε ∧
      (0 : ℝ) ∈ capExitSet 2 β lam ε k w μ₀ Φ q x ∧
      sInf (capExitSet 2 β lam ε k w μ₀ Φ q x) = 0 ∧
      Real.exp (-lam * β) < capVariance 2 μ₀ Φ (w q) ε x
        (sInf (capExitSet 2 β lam ε k w μ₀ Φ q x)) := by
  refine ⟨1000, 1 / 10000, Real.log 15000 / 1000, 1,
    fun _ => varianceCounterCentre, fun _ => diracProb 2 varianceCounterCentre,
    diracProb 2 varianceCounterCentre, fun _ => diracProb 2 varianceCounterCentre,
    diracFlow 2 varianceCounterCentre, 0,
    fun t => diracFlow 2 varianceCounterCentre t varianceCounterBoundary,
    by norm_num, by norm_num, by norm_num, one_pos, ?_, by simp, rfl, continuous_const,
    meanFieldPDE_dirac 2 1000 _, isMFFlow_diracFlow 2 1000 _,
    fun t => map_diracFlow_dirac 2 _ t,
    isCapArgmin_diracFlow 2 _ _ _ varianceCounterBoundary_coordinate,
    (varianceCounterLambda_mem 2 varianceCounterCentre).1,
    (varianceCounterLambda_mem 2 varianceCounterCentre).2,
    (varianceCounterLambda_mem 2 varianceCounterCentre).1.trans
      (varianceCounterLambda_mem 2 varianceCounterCentre).2,
    zero_mem_varianceCounterExitSet, varianceCounterExitTime_zero,
    varianceCounter_strict_failure⟩
  intro q y hy
  rw [Interpolation.eq_of_mem_support_dirac hy]
  change 1 - (1 / 10000 : ℝ) ≤ inner (𝕜 := ℝ)
    (varianceCounterCentre : EucSpace 2) (varianceCounterCentre : EucSpace 2)
  rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp varianceCounterCentre.2]
  norm_num

/-- The hypotheses of `cap_exit` and the refuted equation `eq: v.small` are satisfiable: on
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
