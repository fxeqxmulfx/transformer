/-
# Normalization — Symmetric (equiangular) initialization (§4.1 of 2510.22026v2)

* `Theorem thm: symmetric` — initial velocity of the cosine similarity
  `γ(t) = ⟨θ_j(t), θ_k(t)⟩` under the symmetric (orthonormal) initialization,
  for each normalization scheme;
* the per-scheme speed factor `s_j` is the only thing that distinguishes the
  schemes at `t = 0`, which is what the proof below isolates.
-/

import Transformer.Basic
import Transformer.Normalization.Basic
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real

namespace Transformer
namespace Normalization

open Normalization

/-- **The symmetric initialization of §4.1.**  The `n` tokens start mutually
orthogonal and of unit length, `⟨θ_j, θ_k⟩ = δ_{jk}` — the equiangular
configuration the theorem's table is computed at. -/
def SymmetricInit {d n : ℕ} (Θ : Idx n → EucSpace d) : Prop :=
  ∀ j k : Idx n, inner (𝕜 := ℝ) (Θ j) (Θ k) = if j = k then (1 : ℝ) else 0

/-! ### The attention vector at an equiangular configuration -/

/-- **The partition function is the same at every token.**  Of the `n` scores
one is `β` and the other `n - 1` are `0`, so `Z_j = e^β + n - 1`. -/
theorem partition_symmetricInit {d n : ℕ} (β : ℝ) {Θ : Idx n → EucSpace d}
    (hΘ : SymmetricInit Θ) (j : Idx n) :
    ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Θ j) (Θ l))
      = Real.exp β + ((n : ℝ) - 1) := by
  have hterm : ∀ l : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) (Θ j) (Θ l))
        = 1 + (if j = l then Real.exp β - 1 else 0) := by
    intro l
    rw [hΘ j l]
    by_cases h : j = l <;> simp [h]
  rw [Finset.sum_congr rfl fun l _ => hterm l, Finset.sum_add_distrib,
    Finset.sum_ite_eq Finset.univ j (fun _ => Real.exp β - 1)]
  simp
  ring

/-- **And the attention vector sees only one token at a time.**  Because the
configuration is orthonormal, pairing `A_j` with `θ_k` keeps exactly the `k`-th
term of the sum: `⟨A_j, θ_k⟩ = Z⁻¹ e^{β δ_{jk}}`. -/
theorem inner_attentionVec_symmetricInit {d n : ℕ} (β : ℝ) {Θ : Idx n → EucSpace d}
    (hΘ : SymmetricInit Θ) (j k : Idx n) :
    inner (𝕜 := ℝ)
        (attentionVec d n β (ContinuousLinearMap.id ℝ (EucSpace d))
          (ContinuousLinearMap.id ℝ (EucSpace d))
          (ContinuousLinearMap.id ℝ (EucSpace d)) Θ j) (Θ k)
      = (Real.exp β + ((n : ℝ) - 1))⁻¹ * (if j = k then Real.exp β else 1) := by
  rw [attentionVec]
  simp only [ContinuousLinearMap.coe_id', id_eq]
  rw [real_inner_smul_left, partition_symmetricInit β hΘ j, sum_inner]
  congr 1
  rw [Finset.sum_congr rfl fun l _ => real_inner_smul_left (Θ l) (Θ k) _,
    Finset.sum_congr rfl fun l _ => by rw [hΘ l k],
    Finset.sum_eq_single k (fun l _ hl => by simp [hl]) (by simp)]
  rw [hΘ j k]
  by_cases h : j = k <;> simp [h]

/-! ### The velocity of the cosine similarity -/

/-- **Theorem (thm: symmetric), initial velocity.**

At the symmetric initialization, for `Q = K = V = I_d` and any pair of
distinct tokens `j ≠ k`, the cosine similarity `γ(t) = ⟨θ_j(t), θ_k(t)⟩`
leaves with velocity

  `γ̇(t₀) = (s_j⁻¹ + s_k⁻¹) / (e^β + n - 1)`.

The tangential correction `Proj_{θ_j}` contributes nothing: it subtracts a
multiple of `θ_j`, and `⟨θ_j, θ_k⟩ = 0`.  So the six rows of the paper's table
differ only in the speed factor `s`, which is what the corollaries below read
off.  Source: arXiv:2510.22026v2, §4.1. -/
theorem hasDerivAt_similarity_symmetricInit {d n : ℕ} (β : ℝ)
    (s : ℝ → Idx n → ℝ) (θ : ℝ → Idx n → EucSpace d) (t₀ : ℝ)
    (hNA : NA d n β (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d)) s θ)
    (hΘ : SymmetricInit (θ t₀)) {j k : Idx n} (hjk : j ≠ k) :
    HasDerivAt (fun t => inner (𝕜 := ℝ) (θ t j) (θ t k))
      (((s t₀ j)⁻¹ + (s t₀ k)⁻¹) * (Real.exp β + ((n : ℝ) - 1))⁻¹) t₀ := by
  have hd := (hNA t₀ j).inner ℝ (hNA t₀ k)
  convert hd using 1
  simp only [proj, inner_sub_right, inner_sub_left, real_inner_smul_right,
    real_inner_smul_left]
  rw [hΘ j k, ite_eq_right hjk,
    real_inner_comm (attentionVec d n β _ _ _ (θ t₀) k) (θ t₀ j),
    inner_attentionVec_symmetricInit β hΘ k j,
    inner_attentionVec_symmetricInit β hΘ j k,
    ite_eq_right hjk, ite_eq_right (Ne.symm hjk)]
  ring

/-- **Post-LN.**  `s_j ≡ 1`, so `γ̇(0) = 2 / (e^β + n - 1)` — the first row of
the table in §4.1. -/
theorem thm_symmetric_post {d n : ℕ} (β : ℝ) (Q K V : ℝ → ParamMatrix d)
    (α : ℝ → ℝ) (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ) (τ t₀ : ℝ)
    (hNA : NA d n β (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun t j => speedFactor d n β Q K V α θ r τ .post t j) θ)
    (hΘ : SymmetricInit (θ t₀)) {j k : Idx n} (hjk : j ≠ k) :
    HasDerivAt (fun t => inner (𝕜 := ℝ) (θ t j) (θ t k))
      (2 * (Real.exp β + ((n : ℝ) - 1))⁻¹) t₀ := by
  have h := hasDerivAt_similarity_symmetricInit β _ θ t₀ hNA hΘ hjk
  simpa [speedFactor, show (1 : ℝ) + 1 = 2 by norm_num] using h

/-- **Pre-LN.**  `s_j = r_j`, so the same velocity divided by the common radius:
`γ̇(0) = 2 / (r₀ (e^β + n - 1))` — the second row. -/
theorem thm_symmetric_pre {d n : ℕ} (β : ℝ) (Q K V : ℝ → ParamMatrix d)
    (α : ℝ → ℝ) (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ) (τ t₀ r₀ : ℝ)
    (hNA : NA d n β (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun t j => speedFactor d n β Q K V α θ r τ .pre t j) θ)
    (hΘ : SymmetricInit (θ t₀)) (hr : ∀ i : Idx n, r t₀ i = r₀)
    {j k : Idx n} (hjk : j ≠ k) :
    HasDerivAt (fun t => inner (𝕜 := ℝ) (θ t j) (θ t k))
      (2 * (r₀ * (Real.exp β + ((n : ℝ) - 1)))⁻¹) t₀ := by
  have h := hasDerivAt_similarity_symmetricInit β _ θ t₀ hNA hΘ hjk
  rw [show ((2 : ℝ) * (r₀ * (Real.exp β + ((n : ℝ) - 1)))⁻¹)
      = ((speedFactor d n β Q K V α θ r τ .pre t₀ j)⁻¹
        + (speedFactor d n β Q K V α θ r τ .pre t₀ k)⁻¹)
        * (Real.exp β + ((n : ℝ) - 1))⁻¹ by
    simp [speedFactor, hr]; ring]
  exact h

/-! ### The hypotheses are satisfiable -/

/-- `SymmetricInit` is satisfiable: the standard basis of `ℝ^n` is exactly the
equiangular configuration of `n` tokens the theorem starts from. -/
example (n : ℕ) : SymmetricInit (fun j : Idx n => (EuclideanSpace.single j (1 : ℝ))) := by
  intro j k
  simp [EuclideanSpace.inner_single_right, eq_comm]

/-- **And all of them at once.**  Give the speed factor the value `0`: in Lean
`(0 : ℝ)⁻¹ = 0`, so `NA` asks for a stationary configuration, a constant
orthonormal frame is one, the two tokens are distinct, and the velocity the
theorem returns is `0` -- which is what the dynamics does.  The hypotheses are
therefore jointly satisfiable. -/
example : ∃ (s : ℝ → Idx 2 → ℝ) (θ : ℝ → Idx 2 → EucSpace 2),
    NA 2 2 1 (fun _ => ContinuousLinearMap.id ℝ (EucSpace 2))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace 2))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace 2)) s θ ∧
    SymmetricInit (θ 0) ∧ (0 : Idx 2) ≠ 1 := by
  refine ⟨fun _ _ => 0, fun _ j => EuclideanSpace.single j (1 : ℝ),
    fun t j => ?_, ?_, by decide⟩
  · simp only [inv_zero, zero_smul]
    exact hasDerivAt_const t _
  · intro j k
    simp [EuclideanSpace.inner_single_right, eq_comm]

end Normalization
end Transformer
