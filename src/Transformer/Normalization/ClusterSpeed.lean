/-
# Normalization — the speed of cluster collapse (§4.3 of 2510.22026v2)

* `Theorem thm: preln-slow (ii)` — from a local-cone initialization the
  intra-cluster variance decays at the per-scheme rate
  `d/dt Var(t) = -Θ(Var(t) / scale(t))`: `clustering_rate`, corrected;
* `not_clustering_rate` — the theorem with the source's hypotheses is false.

**What the source says and what is changed here.**  The cone is
`⟨θ_j(0), θ_k(0)⟩ ≥ 1 - δ` with `δ < 1/(100 n² β²)`, which for small `β` puts
no constraint at all: at `β = 1/100`, `n = 2` it allows `δ = 5/2`, and the
antipodal pair `u, -u` is then an admissible start.  It is stationary — every
attention vector is parallel to its token — so `Var ≡ 1` and `Var' ≡ 0`, and no
`-Θ(Var)` rate holds (`not_clustering_rate`).  The proof in the supplement uses
`2nδ Var ≥ 2n Var²` and `-2 + 2nδ + √2/(3√n) < 0`, which need `δ` small
independently of `β`; `δ < 1/(100 n²)` is added.  The nGPT row reads
`-Θ(Var/α_t)`, but Table 2 has `s_j = α_t⁻¹ ‖A_j‖`, the proof's own bounds give
`s_j ≍ α_t⁻¹` once the table is used, and the Remark after it gives
`Var = e^{-2∫α}`: the rate is `-Θ(α_t Var)`, and `varScale` reads it so.

Source: arXiv:2510.22026v2, §4.3, `thm: preln-slow` (ii); supplement, proof of
Theorem 4.3; Table 2.
-/

import Transformer.Basic
import Transformer.Normalization.Basic
import Transformer.Normalization.Velocities
import Transformer.Normalization.ClusterCounterexample
import Transformer.Normalization.ClusterWeights
import Transformer.Normalization.ClusterSpeedBounds
import Transformer.Normalization.ClusterVariance

open scoped BigOperators
open Real
open Set

namespace Transformer
namespace Normalization

variable (d n : ℕ)

/-- **Theorem (thm: preln-slow), (ii).** *Speed of cluster collapse.*

For `V = I_d` and arbitrary `Q, K` with `‖Q^⊤ K‖_op ≤ 1`, started on the
sphere in a local cone `⟨θ_j(0), θ_k(0)⟩ ≥ 1 - δ`, the intra-cluster variance
obeys `d/dt Var(t) = -Θ(Var(t) / scale(t))`, with `scale` the per-scheme time
scale of `varScale`.

`Θ` is asymptotic in `t`, as in the source's Remark ("as `t → ∞`"): two
constants `0 < c ≤ C`, fixed with the parameters, sandwich the derivative from
some time on. That time depends on the solution: the invariant initial cap
gives, for Pre-LN, `s_j(t) = r_j(t) ∈ [r_j(0) + (1 - 4δ)t, r_j(0) + t]`, which is `≍ t` only once
`t ≳ r_j(0)`.  The radii start positive, being norms, and `α_t > 0`.

What is changed: `δ < 1/(100 n²)` is added and the nGPT scale is `α_t⁻¹` (see
the module docstring); without the first the statement is false
(`not_clustering_rate`).  Part (i) is `radialDerivative_pre_ge_of_localCone`.

The proof establishes the uniform constants `c = 1/9` and `C = 4`.

Source: arXiv:2510.22026v2, §4.3, `thm: preln-slow` (ii). -/
theorem clustering_rate (β δ : ℝ) (α : ℝ → ℝ) (τ : ℝ) (scheme : Scheme)
    (hδ : δ < 1 / (100 * (n : ℝ) ^ 2 * β ^ 2)) (hδ₁ : δ < 1 / (100 * (n : ℝ) ^ 2))
    (hα : ∀ t, 0 < α t) :
    ∀ Q K : ℝ → ParamMatrix d,
      (∀ t : ℝ, ∀ x y : EucSpace d,
        |inner (𝕜 := ℝ) (Q t x) (K t y)| ≤ ‖x‖ * ‖y‖) →
      ∃ c C : ℝ, 0 < c ∧ c ≤ C ∧
        ∀ (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ),
          (∀ j : Idx n, ‖θ 0 j‖ = 1) → (∀ j : Idx n, 0 < r 0 j) →
          (∀ j k : Idx n, 1 - δ ≤ inner (𝕜 := ℝ) (θ 0 j) (θ 0 k)) →
          SchemeDynamics d n β Q K (idParams d) α τ scheme θ r →
            ∃ T₀ : ℝ, ∀ t : ℝ, T₀ ≤ t → ∃ v : ℝ,
              HasDerivAt (intraClusterVar d n θ) v t ∧
              -(C * intraClusterVar d n θ t / varScale α scheme t) ≤ v ∧
              v ≤ -(c * intraClusterVar d n θ t / varScale α scheme t) := by
  intro Q K hQK
  refine ⟨1 / 9, 4, by norm_num, by norm_num, ?_⟩
  intro θ r hunit hr hcone hdyn
  by_cases hn : 2 ≤ n
  · have hn0 : 0 < n := by omega
    let j₀ : Idx n := ⟨0, hn0⟩
    have hδ0 : 0 ≤ δ := by
      have h := hcone j₀ j₀
      rw [real_inner_self_eq_norm_sq, hunit j₀, one_pow] at h
      linarith
    have hn' : (2 : ℝ) ≤ n := by exact_mod_cast hn
    have hden : (100 : ℝ) ≤ 100 * (n : ℝ) ^ 2 := by nlinarith
    have hδsmall : δ < 1 / 100 :=
      hδ₁.trans_le (one_div_le_one_div_of_le (by norm_num) hden)
    have hinv := scheme_cluster_invariant hn0 β δ τ hδ0 (by linarith) Q K α θ r scheme
      hα hunit hr hcone hdyn
    have hu : ∀ t, 0 ≤ t → ∀ j, ‖θ t j‖ = 1 := fun t ht => (hinv t ht).1
    have hp : ∀ t, 0 ≤ t → ∀ j k, (7 / 8 : ℝ) ≤ inner (𝕜 := ℝ) (θ t j) (θ t k) := by
      intro t ht j k
      have h := (hinv t ht).2.1 j k
      linarith
    obtain ⟨T, hT, hs⟩ := eventual_inverse_speed_bounds β τ Q K α θ r scheme hα hr hu hp
      (fun t ht => (hinv t ht).2.2) hdyn
    refine ⟨T, ?_⟩
    intro t ht
    have htpos : 0 < t := by linarith
    have ht0 : 0 ≤ t := htpos.le
    obtain ⟨hscale, hspeed⟩ := hs t ht
    let v : Idx n → EucSpace d := fun j =>
      (speedFactor d n β Q K (idParams d) α θ r τ scheme t j)⁻¹ •
        proj d (θ t j) (attentionVec d n β (Q t) (K t) (idParams d t) (θ t) j)
    have hv := (hasDerivWithinAt_intraClusterVar hn0 θ v (Set.Ici 0) t
      (fun j => (hdyn t ht0 j).1)).hasDerivAt (Ici_mem_nhds htpos)
    refine ⟨_, hv, ?_⟩
    have hpoint := variance_decay_of_uniform_weights hn0 β (varScale α scheme t)⁻¹
      (inv_nonneg.mpr hscale.le) (Q t) (K t) (θ t)
      (fun j => (speedFactor d n β Q K (idParams d) α θ r τ scheme t j)⁻¹)
      (hu t ht0) (fun j k => (by norm_num : (3 / 4 : ℝ) ≤ 7 / 8).trans (hp t ht0 j k))
      hspeed (attentionWeight_close_to_uniform hn β δ hδ0 hδ (Q t) (K t) (θ t)
        (hQK t) (hu t ht0) (hinv t ht0).2.1)
    change -(4 * (varScale α scheme t)⁻¹ * intraClusterVar d n θ t) ≤ _ ∧
      _ ≤ -((1 / 9 : ℝ) * (varScale α scheme t)⁻¹ * intraClusterVar d n θ t) at hpoint
    simpa only [v, idParams, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hpoint
  · have hzero : intraClusterVar d n θ = fun _ => 0 := by
      have hnle : n ≤ 1 := by omega
      interval_cases n <;> funext t <;> simp [intraClusterVar]
    refine ⟨1, fun t _ => ⟨0, ?_, ?_⟩⟩
    · rw [hzero]
      exact hasDerivAt_const t 0
    · simp [hzero]

/-- The hypotheses of `clustering_rate` are satisfiable: one token, `β = 1`,
`δ = 0` and `α ≡ 1`; `Q = K = I_d` meet the operator bound by Cauchy–Schwarz. -/
example :
    (0 : ℝ) < 1 / (100 * ((1 : ℕ) : ℝ) ^ 2 * (1 : ℝ) ^ 2) ∧
      (0 : ℝ) < 1 / (100 * ((1 : ℕ) : ℝ) ^ 2) ∧ (∀ _t : ℝ, (0 : ℝ) < 1) ∧
      ∀ _t : ℝ, ∀ x y : EucSpace 1,
        |inner (𝕜 := ℝ) (idParams 1 _t x) (idParams 1 _t y)| ≤ ‖x‖ * ‖y‖ :=
  ⟨by norm_num, by norm_num, fun _ => one_pos, fun _ x y => abs_real_inner_le_norm x y⟩

end Normalization
end Transformer
