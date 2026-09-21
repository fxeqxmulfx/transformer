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

open scoped BigOperators
open Real

namespace Transformer
namespace Normalization

variable (d n : ℕ)

/-- The time scale by which `thm: preln-slow (ii)` divides the intra-cluster
variance, one row per scheme: `1` for Post-LN, `t` for Pre-LN, Mix-LN and
Peri-LN, `√t` for CoD, and for nGPT `α_t⁻¹` — the source prints `α_t`, against
its Table 2 and its Remark (see the module docstring).

Source: arXiv:2510.22026v2, §4.3, the display of `thm: preln-slow (ii)`. -/
noncomputable def varScale (α : ℝ → ℝ) (scheme : Scheme) (t : ℝ) : ℝ :=
  match scheme with
  | .post => 1
  | .pre  => t
  | .mix  => t
  | .peri => t
  | .nGPT => (α t)⁻¹
  | .CoD  => Real.sqrt t

/-- **Theorem (thm: preln-slow), (ii).** *Speed of cluster collapse.*

For `V = I_d` and arbitrary `Q, K` with `‖Q^⊤ K‖_op ≤ 1`, started on the
sphere in a local cone `⟨θ_j(0), θ_k(0)⟩ ≥ 1 - δ`, the intra-cluster variance
obeys `d/dt Var(t) = -Θ(Var(t) / scale(t))`, with `scale` the per-scheme time
scale of `varScale`.

`Θ` is asymptotic in `t`, as in the source's Remark ("as `t → ∞`"): two
constants `0 < c ≤ C`, fixed with the parameters, sandwich the derivative from
some time on.  That time depends on the solution: for Pre-LN
`s_j(t) = r_j(t) ∈ [r_j(0) + (1 - δ)t, r_j(0) + t]`, which is `≍ t` only once
`t ≳ r_j(0)`.  The radii start positive, being norms, and `α_t > 0`.

What is changed: `δ < 1/(100 n²)` is added and the nGPT scale is `α_t⁻¹` (see
the module docstring); without the first the statement is false
(`not_clustering_rate`).  Part (i) is `radialDerivative_pre_ge_of_localCone`.

Not proved here.

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
  sorry

/-- The hypotheses of `clustering_rate` are satisfiable: one token, `β = 1`,
`δ = 0` and `α ≡ 1`; `Q = K = I_d` meet the operator bound by Cauchy–Schwarz. -/
example :
    (0 : ℝ) < 1 / (100 * ((1 : ℕ) : ℝ) ^ 2 * (1 : ℝ) ^ 2) ∧
      (0 : ℝ) < 1 / (100 * ((1 : ℕ) : ℝ) ^ 2) ∧ (∀ _t : ℝ, (0 : ℝ) < 1) ∧
      ∀ _t : ℝ, ∀ x y : EucSpace 1,
        |inner (𝕜 := ℝ) (idParams 1 _t x) (idParams 1 _t y)| ≤ ‖x‖ * ‖y‖ :=
  ⟨by norm_num, by norm_num, fun _ => one_pos, fun _ x y => abs_real_inner_le_norm x y⟩

/-- With `Q = K = V = I_d`, tokens on one line `ℝ u` have attention vectors on
that line. -/
theorem attentionVec_id_smul (β : ℝ) (u : EucSpace d) (s : Idx n → ℝ) (j : Idx n) :
    ∃ c : ℝ, attentionVec d n β (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (fun k => s k • u) j = c • u := by
  simp only [attentionVec, ContinuousLinearMap.id_apply]
  simp_rw [smul_smul]
  rw [← Finset.sum_smul, smul_smul]
  exact ⟨_, rfl⟩

/-- The tangent projection at `±u` kills the line `ℝ u`. -/
theorem proj_smul_self (u : EucSpace d) (hu : ‖u‖ = 1) (s c : ℝ) (hs : s ^ 2 = 1) :
    proj d (s • u) (c • u) = 0 := by
  unfold proj
  rw [inner_smul_left, inner_smul_right, real_inner_self_eq_norm_sq, hu, smul_smul]
  simp only [conj_trivial, one_pow, mul_one]
  rw [show s * c * s = c * s ^ 2 by ring, hs, mul_one, sub_self]

/-- The signs `±1` of the antipodal pair. -/
def pairSign : Idx 2 → ℝ := ![1, -1]

theorem pairSign_sq (j : Idx 2) : pairSign j ^ 2 = 1 := by
  fin_cases j <;> simp [pairSign]

/-- **`thm: preln-slow (ii)` is false under the source's hypotheses.**

At `β = 1/100`, `n = 2` the condition `δ < 1/(100 n² β²)` admits `δ = 5/2`,
which every pair of unit vectors meets.  The antipodal pair `u, -u` with unit
radii is then a stationary Post-LN solution, `Var ≡ 1`, and no `c > 0` gives
`Var' ≤ -c Var` at any time.  In every dimension `d ≥ 1`, for every `α, τ`.

Source: arXiv:2510.22026v2, §4.3, `thm: preln-slow` (ii). -/
theorem not_clustering_rate (hd : 0 < d) (α : ℝ → ℝ) (τ : ℝ) :
    ¬ ∀ β δ : ℝ, δ < 1 / (100 * ((2 : ℕ) : ℝ) ^ 2 * β ^ 2) →
      ∀ Q K : ℝ → ParamMatrix d,
        (∀ t : ℝ, ∀ x y : EucSpace d,
          |inner (𝕜 := ℝ) (Q t x) (K t y)| ≤ ‖x‖ * ‖y‖) →
        ∃ c C : ℝ, 0 < c ∧ c ≤ C ∧
          ∀ (θ : ℝ → Idx 2 → EucSpace d) (r : ℝ → Idx 2 → ℝ),
            (∀ j : Idx 2, ‖θ 0 j‖ = 1) → (∀ j : Idx 2, 0 < r 0 j) →
            (∀ j k : Idx 2, 1 - δ ≤ inner (𝕜 := ℝ) (θ 0 j) (θ 0 k)) →
            SchemeDynamics d 2 β Q K (idParams d) α τ .post θ r →
              ∃ T₀ : ℝ, ∀ t : ℝ, T₀ ≤ t → ∃ v : ℝ,
                HasDerivAt (intraClusterVar d 2 θ) v t ∧
                -(C * intraClusterVar d 2 θ t / varScale α .post t) ≤ v ∧
                v ≤ -(c * intraClusterVar d 2 θ t / varScale α .post t) := by
  intro h
  set u : EucSpace d := EuclideanSpace.single (⟨0, hd⟩ : Fin d) (1 : ℝ) with hu_def
  have hu : ‖u‖ = 1 := by simp [hu_def]
  obtain ⟨c, C, hc, -, hall⟩ := h (1 / 100) (5 / 2) (by norm_num) (idParams d) (idParams d)
    (fun _ x y => abs_real_inner_le_norm x y)
  have hvar : intraClusterVar d 2 (fun _ k => pairSign k • u) = fun _ => 1 := by
    funext t
    simp [intraClusterVar, Fin.sum_univ_two, pairSign, hu]
    norm_num
  have hdyn : SchemeDynamics d 2 (1 / 100) (idParams d) (idParams d) (idParams d) α τ .post
      (fun _ k => pairSign k • u) (fun _ _ => 1) := by
    refine ⟨fun t j => ?_, fun t j => hasDerivAt_const t (1 : ℝ)⟩
    obtain ⟨a, ha⟩ := attentionVec_id_smul d 2 (1 / 100) u pairSign j
    have hzero : proj d (pairSign j • u) (attentionVec d 2 (1 / 100) (idParams d t)
        (idParams d t) (idParams d t) (fun k => pairSign k • u) j) = 0 := by
      rw [show idParams d t = ContinuousLinearMap.id ℝ (EucSpace d) from rfl, ha]
      exact proj_smul_self d u hu _ a (pairSign_sq j)
    rw [hzero, smul_zero]
    exact hasDerivAt_const t _
  obtain ⟨T₀, hT⟩ := hall (fun _ k => pairSign k • u) (fun _ _ => 1)
    (fun j => by fin_cases j <;> simp [pairSign, hu])
    (fun _ => one_pos)
    (fun j k => by
      fin_cases j <;> fin_cases k <;>
        simp [pairSign, hu] <;>
        norm_num)
    hdyn
  obtain ⟨v, hv, -, hle⟩ := hT T₀ le_rfl
  rw [hvar] at hv hle
  have hv0 : v = 0 := hv.unique (hasDerivAt_const T₀ (1 : ℝ))
  simp only [varScale, hv0] at hle
  linarith

end Normalization
end Transformer
