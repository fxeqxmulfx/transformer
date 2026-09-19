/-
# Mean-Field Dynamics — Equiangular model & normalization (§6 of 2512.01868v4)

* The equiangular ODEs for `ρ(t) = ⟨x_i(t), x_j(t)⟩` under `eq: SA` and
  `eq: USA`,
* `Theorem thm: long-context-phase-transition` — phase transition for
  `β_n = γ log n` (Chen et al. 2025).

The local clustering rate of the `eq: SA` ODE is
`Transformer.MeanField.EquiangularRate`.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS

open scoped BigOperators
open Real

namespace Transformer
namespace MeanField

variable (n : ℕ)

/-- **Equiangular ODE for `eq: SA`:**

  `ρ̇(t) = 2 e^{β ρ(t)} (1 - ρ(t)) ((n - 1) ρ(t) + 1)
             / (e^β + (n - 1) e^{β ρ(t)})`. -/
def equiangularSA
    (β : ℝ) (ρ : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, HasDerivAt ρ
    (2 * Real.exp (β * ρ t) * (1 - ρ t) * ((n - 1 : ℝ) * ρ t + 1)
      / (Real.exp β + (n - 1 : ℝ) * Real.exp (β * ρ t))) t

/-- **Equiangular ODE for `eq: USA`:**

  `ρ̇(t) = (2/n) e^{β ρ(t)} (1 - ρ(t)) ((n - 1) ρ(t) + 1)`. -/
def equiangularUSA
    (β : ℝ) (ρ : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, HasDerivAt ρ
    ((2 / (n : ℝ)) * Real.exp (β * ρ t) * (1 - ρ t)
        * ((n - 1 : ℝ) * ρ t + 1)) t

/-- The equiangular simplex `ρ ≡ -1/(n-1)` is a stationary solution of the
equiangular `eq: SA` ODE: the factor `(n - 1) ρ + 1` vanishes.  For `n ≥ 2` it
is a configuration of `n` unit vectors — the vertices of a regular simplex —
and it is the obstruction to a global convergence rate below. -/
theorem equiangularSA_const_simplex (hn : 2 ≤ n) (β : ℝ) :
    equiangularSA n β (fun _ => -1 / ((n : ℝ) - 1)) := by
  have hn1 : ((n : ℝ) - 1) ≠ 0 := by
    have : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    intro h; linarith [h]
  intro t
  have hzero : 2 * Real.exp (β * (-1 / ((n : ℝ) - 1))) * (1 - -1 / ((n : ℝ) - 1))
      * ((n - 1 : ℝ) * (-1 / ((n : ℝ) - 1)) + 1)
      / (Real.exp β + (n - 1 : ℝ) * Real.exp (β * (-1 / ((n : ℝ) - 1)))) = 0 := by
    have : ((n : ℝ) - 1) * (-1 / ((n : ℝ) - 1)) + 1 = 0 := by
      field_simp
      norm_num
    rw [this]
    ring
  rw [hzero]
  exact hasDerivAt_const t _

/-- **The clustering rate is not global.**

At the equiangular simplex the right-hand side of the `eq: SA` ODE vanishes,
so `ρ` stays at `-1/(n-1) < 1` forever and `1 - ρ` stays at `n/(n-1) ≥ 1`,
which no decaying exponential dominates.  This is why
`Transformer.MeanField.equiangular_local_rate` — in
`Transformer.MeanField.EquiangularRate`, which proves it — carries the initial
condition `(n-1) ρ(0) + 1 > 0`: the rate is local to the basin of `ρ = 1`,
exactly as the word *linearization* says. -/
theorem not_exists_rate_at_simplex (hn : 2 ≤ n) :
    ¬ ∃ (C lam : ℝ), 0 < C ∧ 0 < lam ∧
      ∀ t : ℝ, 0 ≤ t →
        1 - (fun _ : ℝ => -1 / ((n : ℝ) - 1)) t ≤ C * Real.exp (-(lam * t)) := by
  rintro ⟨C, lam, hC, hlam, h⟩
  have hn2 : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hone : (1 : ℝ) ≤ 1 - -1 / ((n : ℝ) - 1) := by
    have h1 : (0 : ℝ) < (n : ℝ) - 1 := by linarith
    have : (0 : ℝ) < 1 / ((n : ℝ) - 1) := by positivity
    simp only [neg_div, sub_neg_eq_add]
    linarith
  set t : ℝ := max 0 ((Real.log C + 1) / lam) with ht
  have ht0 : 0 ≤ t := le_max_left _ _
  have hdiv : Real.log C + 1 ≤ lam * t := by
    have hle := mul_le_mul_of_nonneg_left (le_max_right (0 : ℝ) ((Real.log C + 1) / lam)) hlam.le
    have hcancel : lam * ((Real.log C + 1) / lam) = Real.log C + 1 := by
      field_simp
    rw [hcancel] at hle
    exact hle
  have hstep : C * Real.exp (-(lam * t)) ≤ C * Real.exp (-(Real.log C + 1)) :=
    mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by linarith)) hC.le
  have hval : C * Real.exp (-(Real.log C + 1)) = Real.exp (-1) := by
    rw [show -(Real.log C + 1) = -Real.log C + -1 by ring, Real.exp_add, Real.exp_neg,
      Real.exp_log hC, ← mul_assoc, mul_inv_cancel₀ (ne_of_gt hC), one_mul]
  have hlt : Real.exp (-1 : ℝ) < 1 := Real.exp_lt_one_iff.mpr (by norm_num)
  have := h t ht0
  simp only at this
  linarith

/-- The clustered state `ρ ≡ 1` is a stationary solution of the equiangular
`eq: SA` ODE: the factor `1 - ρ` vanishes. -/
theorem equiangularSA_const_one (β : ℝ) :
    equiangularSA n β (fun _ => 1) := by
  intro t
  simpa using hasDerivAt_const t (1 : ℝ)

/-! ### One attention layer on an equiangular configuration -/

/-- The Gram matrix of the equiangular configuration: `⟨x_i, x_j⟩ = ρ` off the
diagonal and `1` on it.  The configuration exists in `ℝ^d` for `d ≥ n - 1` and
`ρ ≥ -1/(n-1)`; only its Gram matrix enters the statements below. -/
def equiGram (n : ℕ) (ρ : ℝ) (i j : Idx n) : ℝ := if i = j then 1 else ρ

/-- The softmax weight token `i` puts on token `j` in the equiangular
configuration at inverse temperature `β`: the normalizer is the same for
every `i`, namely `e^β + (n - 1) e^{β ρ}`. -/
noncomputable def equiWeight (n : ℕ) (β ρ : ℝ) (i j : Idx n) : ℝ :=
  Real.exp (β * equiGram n ρ i j) / (Real.exp β + (n - 1 : ℝ) * Real.exp (β * ρ))

/-- `⟨y_i, y_j⟩` after one attention layer with `V = I_d`, where
`y_i = Σ_k w_{ik} x_k`: read off the Gram matrix alone. -/
noncomputable def equiOutInner (n : ℕ) (β ρ : ℝ) (i j : Idx n) : ℝ :=
  ∑ k : Idx n, ∑ l : Idx n,
    equiWeight n β ρ i k * equiWeight n β ρ j l * equiGram n ρ k l

/-- The cosine `⟨θ_i, θ_j⟩` of the output *directions* after one attention
layer, `θ_i = y_i / ‖y_i‖`. -/
noncomputable def equiOutCos (n : ℕ) (β ρ : ℝ) (i j : Idx n) : ℝ :=
  equiOutInner n β ρ i j
    / Real.sqrt (equiOutInner n β ρ i i * equiOutInner n β ρ j j)

/-- **Theorem (thm: long-context-phase-transition).**  *Phase transition at
`β_n = γ log n`.*

In the equiangular initialization with `⟨x_i, x_j⟩ = ρ`, after a single
attention layer, the output directions satisfy

  `lim_{n → ∞} ⟨θ_i, θ_j⟩ =
      1,                          if γ < 1/(1 - ρ),
      4ρ / (1 + 3 ρ),             if γ = 1/(1 - ρ),
      ρ,                          if γ > 1/(1 - ρ).`

Not proved here: none of the three limits is.  The configuration is entered
through its Gram matrix (`equiGram`), and the sequence is indexed so that
`n = m + 2` always admits the two distinct tokens `0` and `1` the statement
compares.

Source: arXiv:2512.01868v4, §6, `thm: long-context-phase-transition`
(Chen et al. 2025). -/
theorem long_context_phase_transition (γ ρ : ℝ) (hρ₀ : 0 < ρ) (hρ₁ : ρ < 1) :
    (γ < 1 / (1 - ρ) →
      Filter.Tendsto
        (fun m : ℕ => equiOutCos (m + 2) (γ * Real.log (m + 2 : ℕ)) ρ 0 1)
        Filter.atTop (nhds 1))
    ∧ (γ = 1 / (1 - ρ) →
      Filter.Tendsto
        (fun m : ℕ => equiOutCos (m + 2) (γ * Real.log (m + 2 : ℕ)) ρ 0 1)
        Filter.atTop (nhds (4 * ρ / (1 + 3 * ρ))))
    ∧ (1 / (1 - ρ) < γ →
      Filter.Tendsto
        (fun m : ℕ => equiOutCos (m + 2) (γ * Real.log (m + 2 : ℕ)) ρ 0 1)
        Filter.atTop (nhds ρ)) := by
  sorry

/-- The hypotheses of `long_context_phase_transition` are satisfiable: `ρ = 1/2`
lies strictly between `0` and `1`. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) / 2 < 1 := by norm_num

end MeanField
end Transformer
