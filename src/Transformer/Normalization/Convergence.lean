/-
# Normalization — Asymptotic clustering (§3 of 2510.22026v2)

The gradient-flow energy is

  `E(Θ) = -Σ_{j,k} e^{β ⟨Q θ_k, K θ_j⟩}`,

and the dynamics is `θ̇_j = -(s_j(t) Z_j(t))⁻¹ Proj_{θ_j} ∇_{θ_j} E(Θ)`.  This
file proves that identification for `Q = K = V = I_d`:

* `energyGrad` and `hasDerivAt_energy` — the gradient of `E` in `θ_j`,
  `∇_{θ_j} E(Θ) = -2 β Σ_k e^{β ⟨θ_j, θ_k⟩} θ_k`, as a directional
  derivative;
* `na_velocity_eq_energyGrad` — the velocity prescribed by `eq: NA` is the
  projected gradient of `E`.

The clustering statement of `Theorem thm: convergence` itself is almost-sure
with respect to a random initialization and is not formalized here; the
explicit bound behind its corollary is in `Normalization.Radial`.
-/

import Transformer.Basic
import Transformer.Normalization.Basic
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators
open Real

namespace Transformer
namespace Normalization

open Normalization

variable (d n : ℕ)

/-- The gradient-flow energy for Post-LN with `K Q^⊤ = Q K^⊤ = V`:

  `E(Θ) = -Σ_{j,k} e^{β ⟨Q θ_k, K θ_j⟩}`. -/
noncomputable def Energy
    (β : ℝ) (Q K : ParamMatrix d) (Θ : Idx n → EucSpace d) : ℝ :=
  -∑ j : Idx n, ∑ k : Idx n,
    Real.exp (β * inner (𝕜 := ℝ) (Q (Θ k)) (K (Θ j)))

/-- **The energy gradient at token `j`.**

Differentiating `Energy` in `θ_j` touches the `k = j` row and the `j = k`
column of the double sum; the kernel is symmetric, so the two halves agree and

  `∇_{θ_j} E(Θ) = -2 β Σ_k e^{β ⟨θ_j, θ_k⟩} θ_k`.

Source: arXiv:2510.22026v2, §3 (the displayed gradient flow). -/
noncomputable def energyGrad
    (β : ℝ) (Θ : Idx n → EucSpace d) (j : Idx n) : EucSpace d :=
  (-(2 * β)) • ∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Θ j) (Θ k)) • Θ k

/-- **`energyGrad` is the gradient of `Energy`.**

Along the line `ε ↦ Θ + ε v e_j` the energy has derivative
`⟨∇_{θ_j} E(Θ), v⟩` at `ε = 0`, for every direction `v`.  This is the
computation announced in §3 of arXiv:2510.22026v2, for `Q = K = I_d`. -/
theorem hasDerivAt_energy
    (β : ℝ) (Θ : Idx n → EucSpace d) (j : Idx n) (v : EucSpace d) :
    HasDerivAt
      (fun ε : ℝ => Energy d n β (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d))
        (Function.update Θ j (Θ j + ε • v)))
      (inner (𝕜 := ℝ) (energyGrad d n β Θ j) v) 0 := by
  have hupd : ∀ c : Idx n,
      HasDerivAt (fun ε : ℝ => Function.update Θ j (Θ j + ε • v) c)
        (if c = j then v else 0) 0 := by
    intro c
    by_cases h : c = j
    · subst h
      simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add (Θ c)
    · simpa [Function.update_of_ne h, h] using
        hasDerivAt_const (0 : ℝ) (Θ c)
  have hterm : ∀ a b : Idx n, HasDerivAt
      (fun ε : ℝ => Real.exp (β * inner (𝕜 := ℝ)
        (Function.update Θ j (Θ j + ε • v) b)
        (Function.update Θ j (Θ j + ε • v) a)))
      (Real.exp (β * inner (𝕜 := ℝ) (Θ b) (Θ a)) *
        (β * (inner (𝕜 := ℝ) (if b = j then v else 0) (Θ a)
          + inner (𝕜 := ℝ) (Θ b) (if a = j then v else 0)))) 0 := by
    intro a b
    have h := ((((hupd b).inner ℝ (hupd a))).const_mul β).exp
    simpa [add_comm] using h
  have hsum := (HasDerivAt.sum fun a (_ : a ∈ Finset.univ) =>
    HasDerivAt.sum fun b (_ : b ∈ Finset.univ) => hterm a b).neg
  simp only [Energy, ContinuousLinearMap.coe_id', id_eq]
  convert hsum using 1
  · funext ε
    simp [Finset.sum_apply]
  -- The two halves of the derivative are equal, by symmetry of the kernel.
  have hrow : ∀ a : Idx n,
      ∑ b : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Θ b) (Θ a)) *
        (β * (inner (𝕜 := ℝ) (if b = j then v else 0) (Θ a)
          + inner (𝕜 := ℝ) (Θ b) (if a = j then v else 0)))
        = β * (Real.exp (β * inner (𝕜 := ℝ) (Θ j) (Θ a)) *
              inner (𝕜 := ℝ) (Θ a) v)
          + (if a = j then
              β * ∑ b : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Θ b) (Θ a)) *
                inner (𝕜 := ℝ) (Θ b) v
            else 0) := by
    intro a
    rw [Finset.sum_congr rfl fun b _ => show
        Real.exp (β * inner (𝕜 := ℝ) (Θ b) (Θ a)) *
          (β * (inner (𝕜 := ℝ) (if b = j then v else 0) (Θ a)
            + inner (𝕜 := ℝ) (Θ b) (if a = j then v else 0)))
        = (if b = j then β * (Real.exp (β * inner (𝕜 := ℝ) (Θ j) (Θ a)) *
              inner (𝕜 := ℝ) (Θ a) v) else 0)
          + (if a = j then β * (Real.exp (β * inner (𝕜 := ℝ) (Θ b) (Θ a)) *
              inner (𝕜 := ℝ) (Θ b) v) else 0) from by
      by_cases hb : b = j <;> by_cases ha : a = j <;>
        simp [hb, ha, real_inner_comm v] <;> ring]
    rw [Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ j]
    simp only [Finset.mem_univ, ite_true]
    by_cases ha : a = j
    · simp [ha, Finset.mul_sum]
    · simp [ha]
  rw [Finset.sum_congr rfl fun a _ => hrow a, Finset.sum_add_distrib,
    Finset.sum_ite_eq' Finset.univ j]
  simp only [Finset.mem_univ, ite_true]
  rw [energyGrad, real_inner_smul_left, sum_inner,
    Finset.sum_congr rfl fun k _ => real_inner_smul_left (Θ k) v _]
  rw [Finset.sum_congr rfl fun b _ =>
    show Real.exp (β * inner (𝕜 := ℝ) (Θ b) (Θ j)) * inner (𝕜 := ℝ) (Θ b) v
      = Real.exp (β * inner (𝕜 := ℝ) (Θ j) (Θ b)) * inner (𝕜 := ℝ) (Θ b) v from by
      rw [real_inner_comm (Θ b) (Θ j)]]
  rw [← Finset.mul_sum]
  ring

/-- `proj` is linear in the vector it projects. -/
theorem proj_smul (x y : EucSpace d) (c : ℝ) :
    proj d x (c • y) = c • proj d x y := by
  simp [proj, real_inner_smul_right, smul_sub, smul_smul]

/-- **Equation (NA) is the projected gradient flow of `E`.**

For `Q = K = V = I_d` the attention vector is `Z_j⁻¹ Σ_k e^{β ⟨θ_j, θ_k⟩} θ_k`
and the energy gradient is `-2 β Z_j A_j(Θ)`, so the velocity prescribed by
`NA`,

  `θ̇_j = s_j⁻¹ Proj_{θ_j} A_j(Θ)`,

is exactly `-(2 β s_j Z_j)⁻¹ Proj_{θ_j} ∇_{θ_j} E(Θ)` — the header's
gradient-flow form of the dynamics (arXiv:2510.22026v2, §3).  No hypothesis on
`s_j` is needed: at `s_j = 0` both sides are `0`, the convention under which
`NA` freezes. -/
theorem na_velocity_eq_energyGrad
    (β : ℝ) (hβ : β ≠ 0) (Θ : Idx n → EucSpace d) (j : Idx n) (s : ℝ) :
    s⁻¹ • proj d (Θ j) (attentionVec d n β (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d)) Θ j)
      = (-(2 * β * s * ∑ l : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (Θ j) (Θ l))))⁻¹ •
          proj d (Θ j) (energyGrad d n β Θ j) := by
  have hZ : (0 : ℝ) < ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Θ j) (Θ l)) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) ⟨j, Finset.mem_univ j⟩
  rw [attentionVec, energyGrad]
  simp only [ContinuousLinearMap.coe_id', id_eq]
  rw [proj_smul, proj_smul, smul_smul, smul_smul]
  congr 1
  rcases eq_or_ne s 0 with hs | hs
  · simp [hs]
  · field_simp

/-- The inverse temperature of `na_velocity_eq_energyGrad` is nonzero in the
regime the paper studies (`β > 0`), so its hypothesis is satisfiable. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

end Normalization
end Transformer
