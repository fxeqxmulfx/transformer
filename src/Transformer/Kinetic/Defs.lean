/-
# Kinetic theory for Transformers — the model

Formalization of Duerinckx, Geshkovski, Rossi — arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*, §1.

The minimal decoder on the torus: `USA` dynamics made causal, with the ALiBi
positional bias, its limiting directed graphon, and the Fourier coefficients of
the interaction kernel.
-/

import Transformer.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Kinetic

/-- `eq:wbeta`: the interaction kernel `w_β(θ) = e^{β cos θ}` on `𝕋 = ℝ/2πℤ`,
carried on `ℝ` — it is `2π`-periodic, as `wBeta_periodic` records, so it
descends to the torus.

Source: arXiv:2605.09213v1, §1, `eq:wbeta`. -/
noncomputable def wBeta (β θ : ℝ) : ℝ := Real.exp (β * Real.cos θ)

/-- `w_β' (θ) = -β sin θ · e^{β cos θ}`. -/
noncomputable def wBetaDeriv (β θ : ℝ) : ℝ := -β * Real.sin θ * Real.exp (β * Real.cos θ)

theorem hasDerivAt_wBeta (β θ : ℝ) : HasDerivAt (wBeta β) (wBetaDeriv β θ) θ := by
  have h : HasDerivAt (fun x : ℝ => β * Real.cos x) (β * -Real.sin θ) θ :=
    (Real.hasDerivAt_cos θ).const_mul β
  have h2 := h.exp
  rw [show wBeta β = fun x : ℝ => Real.exp (β * Real.cos x) from rfl]
  convert h2 using 1
  simp [wBetaDeriv]
  ring

theorem wBeta_periodic (β : ℝ) : Function.Periodic (wBeta β) (2 * π) := by
  intro θ
  simp [wBeta, Real.cos_add_two_pi]

theorem wBetaDeriv_periodic (β : ℝ) : Function.Periodic (wBetaDeriv β) (2 * π) := by
  intro θ
  simp [wBetaDeriv, Real.cos_add_two_pi, Real.sin_add_two_pi]

/-- `eq:partition-function`: `𝒵_{N,j} = Σ_{k<j} e^{-λ(j-k)/N}`, the ALiBi
normalization.

Positions `1 ≤ j ≤ N` of the source are the elements of `Idx N = Fin N`, so
`k < j` is the source's `1 ≤ k ≤ j-1`; the difference `j - k` is the same
either way.

Source: arXiv:2605.09213v1, §1, `eq:partition-function`. -/
noncomputable def alibiZ (lam : ℝ) (N : ℕ) (j : Idx N) : ℝ :=
  ∑ k ∈ Finset.univ.filter (fun k : Idx N => k < j),
    Real.exp (-(lam / N) * ((j : ℝ) - (k : ℝ)))

/-- The right-hand side of `eq:gpt-dyn-alibi`. -/
noncomputable def gptField (lam β : ℝ) (N : ℕ) (θ : Idx N → ℝ) (j : Idx N) : ℝ :=
  (alibiZ lam N j)⁻¹ *
    ∑ k ∈ Finset.univ.filter (fun k : Idx N => k < j),
      Real.exp (-(lam / N) * ((j : ℝ) - (k : ℝ))) * wBetaDeriv β (θ j - θ k)

/-- **Equation (eq:gpt-dyn-alibi).**  The causal `USA` dynamics with ALiBi:

  `θ_j'(t) = 𝒵_{N,j}⁻¹ Σ_{k<j} e^{-λ(j-k)/N} w_β'(θ_j(t) - θ_k(t))`.

The first token has an empty interaction sum and `𝒵_{N,1} = 0`; the field is
then `0⁻¹ · 0 = 0`, so `θ_1` is constant, which is what the source's dynamics
says.

Source: arXiv:2605.09213v1, §1, `eq:gpt-dyn-alibi`. -/
def IsGPTFlow (lam β : ℝ) (N : ℕ) (θ : ℝ → Idx N → ℝ) : Prop :=
  ∀ (t : ℝ) (j : Idx N), HasDerivAt (fun s => θ s j) (gptField lam β N (θ t) j) t

/-- The hypotheses of `IsGPTFlow` are satisfiable: one token, which never
moves. -/
example (lam β : ℝ) : IsGPTFlow lam β 1 (fun _ _ => 0) := by
  intro t j
  have : gptField lam β 1 (fun _ => (0 : ℝ)) j = 0 := by
    fin_cases j
    simp [gptField, alibiZ]
  rw [this]
  exact hasDerivAt_const t 0

/-- `eq:kernel-Klambda`: the limiting directed graphon

  `k_λ(σ, σ') = λ e^{-λ(σ-σ')} / (1 - e^{-λσ}) · 1_{σ' < σ}`,

understood by continuity as `σ⁻¹ 1_{σ' < σ}` at `λ = 0`, which is the value
the definition takes on that branch.

Source: arXiv:2605.09213v1, §1, `eq:kernel-Klambda`. -/
noncomputable def graphon (lam σ σ' : ℝ) : ℝ :=
  if σ' < σ then
    (if lam = 0 then σ⁻¹
      else lam * Real.exp (-(lam * (σ - σ'))) / (1 - Real.exp (-(lam * σ))))
  else 0

@[simp]
theorem graphon_of_le {lam σ σ' : ℝ} (h : σ ≤ σ') : graphon lam σ σ' = 0 := by
  simp [graphon, not_lt.mpr h]

/-- The `n`-th Fourier coefficient of `w_β`, `ŵ_β(n) = (2π)⁻¹ ∫_𝕋 e^{-inθ} w_β(θ) dθ`.

**What the source says and what is changed here.**  The source defines
`f̂(n) = ∫_𝕋 e^{-inθ} f(θ) dθ` but then states `ŵ_β(n) = I_n(β)`, the modified
Bessel function, which is the *normalized* coefficient `(2π)⁻¹ ∫`.  The
normalized one is used here, because it is what `a_n` and the smallness
condition of `thm:U-shape` are calibrated against.

The imaginary part of the integral vanishes — `w_β` is even — so only the
`cos(nθ)` part is written.

Source: arXiv:2605.09213v1, §1. -/
noncomputable def wHat (β : ℝ) (n : ℕ) : ℝ :=
  (2 * π)⁻¹ * ∫ θ in (0 : ℝ)..(2 * π), Real.cos (n * θ) * wBeta β θ

/-- `a_n := n² ŵ_β(n)`, the eigenvalue of the linearized dynamics at
frequency `n`.

Source: arXiv:2605.09213v1, §1, above `eq:Hardy-hom`. -/
noncomputable def aCoeff (β : ℝ) (n : ℕ) : ℝ := (n : ℝ) ^ 2 * wHat β n

end Kinetic
end Transformer
