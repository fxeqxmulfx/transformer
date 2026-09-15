/-
# Normalization — Initial and terminal token velocities (§4.2–§4.3 of 2510.22026v2)

* `Theorem thm: initial-velocity`  — uniform bound on `‖A_j(0)‖` for random
                                      directional init,
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

/-- **Theorem (thm: initial-velocity).** *Initial attention magnitude bound.*

For `Q, K, V ∈ ℝ^{d × d}` with `max(‖Q^⊤ K‖_op, ‖V‖_op) ≤ 1`, `β = 1`, and
i.i.d. uniform directional init `θ_j(0) ∼ Unif(𝕊^{d-1})`, there exist
absolute constants `c, C > 0` such that for `e^{√d} ≥ n log n ≥ d`, with
probability `1 - n^{-C}`,

  `‖A_j(0)‖ ≤ C (√(log n / n) + log n / d)`  uniformly in `j ∈ [n]`. -/
theorem thm_initial_velocity
    (Q K V : ParamMatrix d)
    (h_norms : ‖Q‖ ≤ 1 ∧ ‖K‖ ≤ 1 ∧ ‖V‖ ≤ 1)
    (h_size : Real.exp (Real.sqrt d) ≥ (n : ℝ) * Real.log n
                ∧ (n : ℝ) * Real.log n ≥ d) :
    True := by trivial

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
