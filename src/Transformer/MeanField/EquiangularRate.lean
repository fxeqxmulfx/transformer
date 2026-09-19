/-
# The local clustering rate of the equiangular `eq: SA` ODE

arXiv:2512.01868v4, §6.  Near the clustered state `ρ = 1` the equiangular ODE

  `ρ̇ = 2 e^{βρ} (1 - ρ) ((n - 1) ρ + 1) / (e^β + (n - 1) e^{βρ})`

contracts `1 - ρ` exponentially, and this module proves it: for every
`ρ(0)` in the basin `(n - 1) ρ(0) + 1 > 0` there are `C, λ > 0` with
`1 - ρ(t) ≤ C e^{-λ t}` on `t ≥ 0`.

The proof is a single barrier (`image_le_of_deriv_right_lt_deriv_boundary`):
the curve `t ↦ C e^{-λ t}` starts above `1 - ρ`, and wherever the two meet the
ODE pushes `1 - ρ` down faster than the curve.  At such a meeting point
`1 - ρ(x) ≤ C`, hence `ρ(x) ≥ 1 - C`, which bounds both the numerator
`(n - 1) ρ + 1` from below and the ratio `e^β / e^{βρ} ≤ e^{βC}` from above;
that is where `C = max (1 - ρ(0)) 1` and `λ = q₁ / (e^{βC} + n - 1)` with
`q₁ = min ((n - 1) ρ(0) + 1) 1` come from.  No integrating factor and no
Grönwall estimate are needed.

Two hypotheses are not decoration.  `(n - 1) ρ(0) + 1 > 0` is the basin
condition: `Transformer.MeanField.not_exists_rate_at_simplex` refutes the
statement at the equiangular simplex, where it fails with equality.  And
`1 ≤ n`, which the paper leaves implicit, is what keeps the denominator
positive: `not_equiangular_local_rate_zero` below refutes the statement at
`n = 0, β = 0`, where `e^β + (n - 1) e^{βρ}` vanishes at `ρ = 0` and Lean's
`x / 0 = 0` makes every constant a solution.
-/

import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Transformer.MeanField.Equiangular

namespace Transformer
namespace MeanField

variable (n : ℕ)

/-- **Local exponential clustering rate** for the equiangular `eq: SA` ODE:
past the basin condition `(n - 1) ρ(0) + 1 > 0`, the gap `1 - ρ(t)` is
dominated by a decaying exponential on `t ≥ 0`.

Source: arXiv:2512.01868v4, §6. -/
theorem equiangular_local_rate (hn : 1 ≤ n)
    (β : ℝ) (hβ : 0 ≤ β) (ρ : ℝ → ℝ)
    (hρ_sa : equiangularSA n β ρ)
    (hρ0 : 0 < ((n : ℝ) - 1) * ρ 0 + 1) :
    ∃ (C lam : ℝ), 0 < C ∧ 0 < lam ∧
      ∀ t : ℝ, 0 ≤ t →
        1 - ρ t ≤ C * Real.exp (-(lam * t)) := by
  have hn1 : (0 : ℝ) ≤ (n : ℝ) - 1 := by
    have : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    linarith
  -- the floor `q₁` of the numerator `(n - 1) ρ + 1` on the basin
  obtain ⟨q₁, hq₁pos, hq₁one, hq₁ρ⟩ :
      ∃ q₁ : ℝ, 0 < q₁ ∧ q₁ ≤ 1 ∧ q₁ ≤ ((n : ℝ) - 1) * ρ 0 + 1 :=
    ⟨min (((n : ℝ) - 1) * ρ 0 + 1) 1, lt_min hρ0 one_pos, min_le_right _ _, min_le_left _ _⟩
  -- the ceiling `C` of the gap `1 - ρ`, at least `1` so that the floor survives
  obtain ⟨C, hC1, hCρ, hCq⟩ :
      ∃ C : ℝ, 1 ≤ C ∧ 1 - ρ 0 ≤ C ∧ q₁ ≤ ((n : ℝ) - 1) * (1 - C) + 1 := by
    rcases le_or_gt (1 - ρ 0) 1 with h | h
    · exact ⟨1, le_rfl, h, by simpa using hq₁one⟩
    · refine ⟨1 - ρ 0, h.le, le_rfl, ?_⟩
      have hrw : ((n : ℝ) - 1) * (1 - (1 - ρ 0)) + 1 = ((n : ℝ) - 1) * ρ 0 + 1 := by ring
      rw [hrw]
      exact hq₁ρ
  have hC0 : (0 : ℝ) < C := by linarith
  have hden : 0 < Real.exp (β * C) + ((n : ℝ) - 1) := by
    have := Real.exp_pos (β * C)
    linarith
  set lam : ℝ := q₁ / (Real.exp (β * C) + ((n : ℝ) - 1)) with hlamdef
  have hlam : 0 < lam := div_pos hq₁pos hden
  refine ⟨C, lam, hC0, hlam, ?_⟩
  intro t ht
  -- the gap `1 - ρ` and its derivative along the ODE
  have hf' : ∀ x : ℝ, HasDerivAt (fun y : ℝ => 1 - ρ y)
      (-(2 * Real.exp (β * ρ x) * (1 - ρ x) * (((n : ℝ) - 1) * ρ x + 1)
        / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ x)))) x :=
    fun x => (hρ_sa x).const_sub 1
  -- the barrier `t ↦ C e^{-λ t}`
  have hB : ∀ x : ℝ, HasDerivAt (fun y : ℝ => C * Real.exp (-(lam * y)))
      (-(lam * (C * Real.exp (-(lam * x))))) x := by
    intro x
    have h1 : HasDerivAt (fun y : ℝ => -(lam * y)) (-lam) x := by
      simpa using (hasDerivAt_id x).const_mul (-lam)
    have h2 := HasDerivAt.const_mul C (HasDerivAt.exp h1)
    convert h2 using 1
    ring
  refine image_le_of_deriv_right_lt_deriv_boundary
    (f := fun y : ℝ => 1 - ρ y) (a := 0) (b := t)
    (fun x _ => (hf' x).continuousAt.continuousWithinAt)
    (fun x _ => (hf' x).hasDerivWithinAt)
    (by simpa using hCρ) hB ?_ (Set.mem_Icc.mpr ⟨ht, le_rfl⟩)
  rintro x ⟨hx0, -⟩ hx
  -- at a contact point the gap is positive and below its ceiling
  have hE : 0 < Real.exp (β * ρ x) := Real.exp_pos _
  have hu0 : 0 < 1 - ρ x := hx ▸ mul_pos hC0 (Real.exp_pos _)
  have huC : 1 - ρ x ≤ C := by
    rw [hx]
    exact mul_le_of_le_one_right hC0.le
      (Real.exp_le_one_iff.mpr (by nlinarith [mul_nonneg hlam.le hx0]))
  -- hence the numerator stays above `q₁` and the denominator below `e^{βC}`
  have hmono : ((n : ℝ) - 1) * (1 - C) ≤ ((n : ℝ) - 1) * ρ x :=
    mul_le_mul_of_nonneg_left (by linarith) hn1
  have hq : q₁ ≤ ((n : ℝ) - 1) * ρ x + 1 := by linarith
  have hexp : Real.exp β ≤ Real.exp (β * ρ x) * Real.exp (β * C) := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by nlinarith [mul_nonneg hβ (show (0 : ℝ) ≤ ρ x + C - 1 by linarith)])
  have hD : 0 < Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ x) := by
    have h1 := Real.exp_pos β
    have h2 : 0 ≤ ((n : ℝ) - 1) * Real.exp (β * ρ x) := mul_nonneg hn1 hE.le
    linarith
  -- so the ODE drives the gap down at a rate above `λ`
  have hcross : 2 * q₁ * (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ x))
      ≤ 2 * Real.exp (β * ρ x) * (((n : ℝ) - 1) * ρ x + 1)
          * (Real.exp (β * C) + ((n : ℝ) - 1)) := by
    have ha := mul_le_mul_of_nonneg_left hexp (show (0 : ℝ) ≤ 2 * q₁ by linarith)
    have hb := mul_le_mul_of_nonneg_left hq
      (show (0 : ℝ) ≤ 2 * Real.exp (β * ρ x) * Real.exp (β * C) by positivity)
    have hc := mul_le_mul_of_nonneg_left hq
      (show (0 : ℝ) ≤ 2 * Real.exp (β * ρ x) * ((n : ℝ) - 1) by positivity)
    nlinarith [ha, hb, hc]
  have hstep : 2 * lam ≤ 2 * Real.exp (β * ρ x) * (((n : ℝ) - 1) * ρ x + 1)
      / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ x)) := by
    have hrw : 2 * lam = 2 * q₁ / (Real.exp (β * C) + ((n : ℝ) - 1)) := by
      rw [hlamdef]; ring
    rw [hrw, div_le_div_iff₀ hden hD]
    linarith [hcross]
  have hkey : lam < 2 * Real.exp (β * ρ x) * (((n : ℝ) - 1) * ρ x + 1)
      / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ x)) := by linarith
  rw [← hx]
  refine neg_lt_neg_iff.mpr ?_
  have hmul := mul_lt_mul_of_pos_right hkey hu0
  have heq : 2 * Real.exp (β * ρ x) * (((n : ℝ) - 1) * ρ x + 1)
        / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ x)) * (1 - ρ x)
      = 2 * Real.exp (β * ρ x) * (1 - ρ x) * (((n : ℝ) - 1) * ρ x + 1)
        / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ x)) := by ring
  rw [heq] at hmul
  exact hmul

/-- The hypotheses of `equiangular_local_rate` are satisfiable: `ρ ≡ 1` solves
the equiangular `eq: SA` ODE at every `β ≥ 0`, and at `n = 2` it sits in the
basin, `(n - 1) ρ(0) + 1 = 2 > 0`. -/
example (β : ℝ) (hβ : 0 ≤ β) :
    ∃ (C lam : ℝ), 0 < C ∧ 0 < lam ∧
      ∀ t : ℝ, 0 ≤ t →
        1 - (fun _ : ℝ => (1 : ℝ)) t ≤ C * Real.exp (-(lam * t)) :=
  equiangular_local_rate 2 (by norm_num) β hβ (fun _ => 1) (equiangularSA_const_one 2 β)
    (by norm_num)

/-- **The hypothesis `1 ≤ n` cannot be dropped.**

At `n = 0` and `β = 0` the denominator `e^β + (n - 1) e^{βρ}` vanishes at
`ρ = 0`, so the right-hand side of the ODE reads `2 / 0 = 0` and the constant
`ρ ≡ 0` is a solution of `equiangularSA 0 0` — one that satisfies the basin
condition `(n - 1) ρ(0) + 1 = 1 > 0` and whose gap `1 - ρ ≡ 1` no decaying
exponential dominates.  The vanishing denominator is an artifact of `n = 0`:
`e^β + (n - 1) e^{βρ}` is the attention normalization of `n` tokens. -/
theorem not_equiangular_local_rate_zero :
    equiangularSA 0 0 (fun _ : ℝ => 0) ∧ 0 < (((0 : ℕ) : ℝ) - 1) * (fun _ : ℝ => (0 : ℝ)) 0 + 1 ∧
      ¬ ∃ (C lam : ℝ), 0 < C ∧ 0 < lam ∧
          ∀ t : ℝ, 0 ≤ t →
            1 - (fun _ : ℝ => (0 : ℝ)) t ≤ C * Real.exp (-(lam * t)) := by
  refine ⟨fun t => ?_, by norm_num, ?_⟩
  · -- `e^0 + (0 - 1) e^0 = 0`, so the right-hand side reads `2 / 0 = 0`
    simpa using hasDerivAt_const t (0 : ℝ)
  · rintro ⟨C, lam, hC, hlam, h⟩
    set t : ℝ := max 0 ((Real.log C + 1) / lam) with ht
    have ht0 : 0 ≤ t := le_max_left _ _
    have hdiv : Real.log C + 1 ≤ lam * t := by
      have hle := mul_le_mul_of_nonneg_left (le_max_right (0 : ℝ) ((Real.log C + 1) / lam)) hlam.le
      have hcancel : lam * ((Real.log C + 1) / lam) = Real.log C + 1 := by field_simp
      rw [hcancel] at hle
      exact hle
    have hstep : C * Real.exp (-(lam * t)) ≤ C * Real.exp (-(Real.log C + 1)) :=
      mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by linarith)) hC.le
    have hval : C * Real.exp (-(Real.log C + 1)) = Real.exp (-1) := by
      rw [show -(Real.log C + 1) = -Real.log C + -1 by ring, Real.exp_add, Real.exp_neg,
        Real.exp_log hC, ← mul_assoc, mul_inv_cancel₀ (ne_of_gt hC), one_mul]
    have hlt : Real.exp (-1 : ℝ) < 1 := Real.exp_lt_one_iff.mpr (by norm_num)
    have := h t ht0
    simp only [sub_zero] at this
    linarith

end MeanField
end Transformer
