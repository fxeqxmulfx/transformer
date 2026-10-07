/-
# Metastability — leaving the cap, mean-field (§5 of 2410.06833v1)

Step 2 of the proof of `thm: metastability MF`: `claim: de sortie de cap`,
the time `T_*(q, c)` at which `η_q V_q e^{-(1-η_q)β}` first drops below
`2e^{-cβ}`, and `eq: v.small`, both refuted by stationary Dirac solutions on the circle.
The vocabulary is `Metastability.MeanField`; the hypotheses are witnessed by
the genuine global flows of `Section5_DiracDynamics`.
-/

import Transformer.Metastability.MeanFieldStatic
import Transformer.Metastability.Section5_CapExitStopping

open Real MeasureTheory

namespace Transformer
namespace Metastability

open Perspective

/-- **Counterexample to Claim (claim: de sortie de cap).** The source asserts
that the stopping set is nonempty and that
`T_*(q,c) < (4ε/k) exp((c-8ε)β)` for every `c > 0`.

On the circle take one cap around `(1,0)`, `ν = μ₀ = δ_(1,0)`,
`β = 1000000`, `ε = 1/10000` and `c = 801/1000000`.
The constant measure solves `eq: mean.field.pde`, and `diracFlow`
is its global characteristic flow from every sphere point, with the
correct push-forward representation. A boundary trajectory minimizes
the whole transported cap at every time; its variance is `V = 1-η`.
The whole cap stays inside the escape window at every nonnegative time.

All original hypotheses are witnessed, including `γ = Ω(1)`;
`c` even lies in `(8ε,γ)`. The one-cap convention is `alphaDist_one`,
with `α = 0`. Time `801` is in the stopping set and no time in
`[0,1]` is, so `1 ≤ T_* ≤ 801`. The claimed bound is less than
`3/2500 = 0.0012`, giving a strict contradiction.
The stopping set is nonempty and bounded below, so this counterexample
uses its actual infimum. No unproved derivative estimate is assumed.

This replaces the false universal claim with a proved counterexample.
Source: arXiv:2410.06833v1, §5, proof of `thm: metastability MF`,
Step 2, `claim: de sortie de cap` and `ineq: derivative`. -/
theorem cap_exit_counterexample :
    ∃ (β ε c : ℝ) (k : ℕ) (w : Idx k → SSphere 2) (ν : Idx k → ProbSphere 2)
      (μ₀ : ProbSphere 2) (μ : ℝ → ProbSphere 2) (Φ : ℝ → SSphere 2 → SSphere 2)
      (q : Idx k) (x : ℝ → SSphere 2),
      1 < β ∧ 0 < ε ∧ ε < 1 / 16 ∧ 0 < k ∧
      (∀ q : Idx k, (ν q : Measure (SSphere 2)).support ⊆ sphericalCap 2 (w q) ε) ∧
      (μ₀ : Measure (SSphere 2))
        = ((k : ℝ)⁻¹).toNNReal • ∑ q : Idx k, (ν q : Measure (SSphere 2)) ∧
      8 * ε < γβ k β (αDist 2 k w ε) ε ∧
      μ 0 = μ₀ ∧ Continuous μ ∧ meanFieldPDE 2 β μ ∧ IsMFFlow 2 β μ Φ ∧
      (∀ t : ℝ, (μ₀ : Measure (SSphere 2)).map (Φ t) = (μ t : Measure (SSphere 2))) ∧
      IsCapArgmin 2 Φ (w q) ε x ∧ 0 < c ∧ 8 * ε < c ∧
      c < γβ k β (αDist 2 k w ε) ε ∧
      (∀ β' : ℝ, 1000 ≤ β' → (1 / 2 : ℝ) < γβ k β' (αDist 2 k w ε) ε) ∧
      (∀ t : ℝ, 0 ≤ t → t ∈ escapeWindow 2 k w Φ ε) ∧
      (capExitSet 2 β c ε k w μ₀ Φ q x).Nonempty ∧
      (1 : ℝ) ≤ sInf (capExitSet 2 β c ε k w μ₀ Φ q x) ∧
      sInf (capExitSet 2 β c ε k w μ₀ Φ q x) ≤ (801 : ℝ) ∧
      4 * ε / k * Real.exp ((c - 8 * ε) * β) <
        sInf (capExitSet 2 β c ε k w μ₀ Φ q x) := by
  have hc := capExitCounter_parameter_mem 2 varianceCounterCentre
  refine ⟨1000000, 1 / 10000, 801 / 1000000, 1,
    fun _ => varianceCounterCentre, fun _ => diracProb 2 varianceCounterCentre,
    diracProb 2 varianceCounterCentre, fun _ => diracProb 2 varianceCounterCentre,
    diracFlow 2 varianceCounterCentre, 0,
    fun t => diracFlow 2 varianceCounterCentre t varianceCounterBoundary,
    by norm_num, by norm_num, by norm_num, one_pos, ?_, by simp, hc.1.trans hc.2,
    rfl, continuous_const, meanFieldPDE_dirac 2 1000000 _, isMFFlow_diracFlow 2 1000000 _,
    fun t => map_diracFlow_dirac 2 _ t,
    isCapArgmin_diracFlow 2 _ _ _ varianceCounterBoundary_coordinate,
    by norm_num, hc.1, hc.2, fun β' hβ' => variance_counter_gamma_lower_bound 2 _ β' hβ',
    fun t ht => mem_escapeWindow_diracFlow 2 1 _ _ t (by norm_num) ht,
    capExitCounterSet_nonempty, capExitCounter_time_lower_bound, capExitCounter_time_upper_bound,
    by simpa only [Nat.cast_one, capExitCounterSet] using capExitCounter_strict_failure⟩
  intro q y hy
  rw [Interpolation.eq_of_mem_support_dirac hy]
  change 1 - (1 / 10000 : ℝ) ≤ inner (𝕜 := ℝ)
    (varianceCounterCentre : EucSpace 2) (varianceCounterCentre : EucSpace 2)
  rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp varianceCounterCentre.2]
  norm_num

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

end Metastability
end Transformer
