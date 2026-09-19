/-
# The equiangular phase transition at `β_n = γ log n`

`Transformer.MeanField.EquiangularWeights` reduced the cosine between the two
output directions to a rational function of the two attention values

  `a = e^β / Z`,   `b = e^{βρ} / Z`,   `Z = e^β + (n-1) e^{βρ}`,

namely `(ρ + (1-ρ)(2ab + (n-2)b²)) / (ρ + (1-ρ)(a² + (n-1)b²))`.

The whole phase transition is then one observation: `a + (n-1) b = 1` forces
`b ≤ 1/(n-1)`, so *every* term carrying a factor `b` vanishes in the limit and
only `a` survives.  The cosine tends to `ρ / (ρ + (1-ρ) L²)` where `L` is the
limit of `a`, and the three phases are the three values of `L`.

Source: arXiv:2512.01868v4, §6, `thm: long-context-phase-transition`.
-/

import Transformer.MeanField.EquiangularPhases

open scoped BigOperators
open Filter Topology

namespace Transformer
namespace MeanField

/-- **The cosine is governed by `a` alone.**  Stated for abstract sequences
`a, b` subject only to positivity and the row-sum identity `a + (n-1) b = 1`
with `n = m + 2`. -/
theorem tendsto_cos_of_tendsto (ρ L : ℝ) (hρ₀ : 0 < ρ) (hρ₁ : ρ < 1)
    (a b : ℕ → ℝ) (hpa : ∀ m, 0 < a m) (hpb : ∀ m, 0 < b m)
    (hsum : ∀ m, a m + ((m : ℝ) + 1) * b m = 1)
    (hL : Tendsto a atTop (nhds L)) :
    Tendsto (fun m : ℕ =>
        (ρ + (1 - ρ) * (2 * a m * b m + (m : ℝ) * b m ^ 2))
          / (ρ + (1 - ρ) * (a m ^ 2 + ((m : ℝ) + 1) * b m ^ 2)))
      atTop (nhds (ρ / (ρ + (1 - ρ) * L ^ 2))) := by
  have hmpos : ∀ m : ℕ, (0 : ℝ) < (m : ℝ) + 1 := fun m => by positivity
  have hmb : ∀ m : ℕ, ((m : ℝ) + 1) * b m ≤ 1 := fun m => by linarith [hsum m, hpa m]
  have ha1 : ∀ m, a m ≤ 1 := fun m => by
    have := hsum m
    have h0 : (0 : ℝ) ≤ ((m : ℝ) + 1) * b m := mul_nonneg (hmpos m).le (hpb m).le
    linarith
  have hb1 : ∀ m, b m ≤ 1 / ((m : ℝ) + 1) := fun m => by
    rw [le_div_iff₀ (hmpos m)]
    linarith [hmb m, mul_comm ((m : ℝ) + 1) (b m)]
  have htb : Tendsto b atTop (nhds 0) :=
    squeeze_zero (fun m => (hpb m).le) hb1 tendsto_one_div_add_atTop_nhds_zero_nat
  -- the cross term
  have hcross : Tendsto (fun m : ℕ => 2 * a m * b m + (m : ℝ) * b m ^ 2)
      atTop (nhds 0) := by
    refine squeeze_zero (fun m => ?_) (fun m => ?_) (by simpa using htb.const_mul 3)
    · have := (hpa m).le
      have := (hpb m).le
      positivity
    · have h1 : 2 * a m * b m ≤ 2 * b m := by nlinarith [ha1 m, (hpb m).le]
      have h2 : (m : ℝ) * b m ^ 2 ≤ b m := by
        nlinarith [mul_le_mul_of_nonneg_right (hmb m) (hpb m).le, (hpb m).le]
      linarith
  -- the diagonal term
  have hself : Tendsto (fun m : ℕ => a m ^ 2 + ((m : ℝ) + 1) * b m ^ 2)
      atTop (nhds (L ^ 2)) := by
    have h0 : Tendsto (fun m : ℕ => ((m : ℝ) + 1) * b m ^ 2) atTop (nhds 0) := by
      refine squeeze_zero (fun m => ?_) (fun m => ?_) htb
      · have := (hpb m).le
        positivity
      · nlinarith [mul_le_mul_of_nonneg_right (hmb m) (hpb m).le, (hpb m).le]
    simpa using (hL.pow 2).add h0
  have hden : ρ + (1 - ρ) * L ^ 2 ≠ 0 := by nlinarith [sq_nonneg L]
  have hnum : Tendsto (fun m : ℕ => ρ + (1 - ρ) * (2 * a m * b m + (m : ℝ) * b m ^ 2))
      atTop (nhds ρ) := by
    simpa using tendsto_const_nhds.add (hcross.const_mul (1 - ρ))
  have hdenT : Tendsto (fun m : ℕ => ρ + (1 - ρ) * (a m ^ 2 + ((m : ℝ) + 1) * b m ^ 2))
      atTop (nhds (ρ + (1 - ρ) * L ^ 2)) :=
    tendsto_const_nhds.add (hself.const_mul (1 - ρ))
  exact hnum.div hdenT hden

/-! ### The theorem -/

/-- The cosine of the two output directions, in the shape
`tendsto_cos_of_tendsto` consumes. -/
theorem equiOutCos_seq_eq (γ ρ : ℝ) (hρ₀ : 0 < ρ) (hρ₁ : ρ < 1) (m : ℕ) :
    equiOutCos (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ 0 1
      = (ρ + (1 - ρ) * (2 * equiDiag (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ
              * equiOff (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ
            + (m : ℝ) * equiOff (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ ^ 2))
        / (ρ + (1 - ρ) * (equiDiag (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ ^ 2
            + ((m : ℝ) + 1) * equiOff (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ ^ 2)) := by
  have h01 : (0 : Idx (m + 2)) ≠ 1 := by simp
  rw [equiOutCos_of_ne _ ρ hρ₀ hρ₁ h01]
  push_cast
  ring_nf

/-- The row-sum identity for the sequence. -/
theorem equiDiag_add_equiOff_seq (γ ρ : ℝ) (m : ℕ) :
    equiDiag (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ
      + ((m : ℝ) + 1) * equiOff (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ = 1 := by
  have h := equiDiag_add_equiOff (n := m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ (by omega)
  push_cast at h
  rw [show ((m : ℝ) + 1) = (m : ℝ) + 2 - 1 from by ring]
  exact h

/-- **Theorem (thm: long-context-phase-transition).**  *Phase transition at
`β_n = γ log n`.*

In the equiangular initialization with `⟨x_i, x_j⟩ = ρ`, after a single
attention layer, the output directions satisfy

  `lim_{n → ∞} ⟨θ_i, θ_j⟩ =
      1,                          if γ < 1/(1 - ρ),
      4ρ / (1 + 3 ρ),             if γ = 1/(1 - ρ),
      ρ,                          if γ > 1/(1 - ρ).`

The configuration is entered through its Gram matrix (`equiGram`), and the
sequence is indexed so that `n = m + 2` always admits the two distinct tokens
`0` and `1` the statement compares.

Source: arXiv:2512.01868v4, §6, `thm: long-context-phase-transition`
(Chen et al. 2025). -/
theorem long_context_phase_transition (γ ρ : ℝ) (hρ₀ : 0 < ρ) (hρ₁ : ρ < 1) :
    (γ < 1 / (1 - ρ) →
      Tendsto (fun m : ℕ => equiOutCos (m + 2) (γ * Real.log (m + 2 : ℕ)) ρ 0 1)
        atTop (nhds 1))
    ∧ (γ = 1 / (1 - ρ) →
      Tendsto (fun m : ℕ => equiOutCos (m + 2) (γ * Real.log (m + 2 : ℕ)) ρ 0 1)
        atTop (nhds (4 * ρ / (1 + 3 * ρ))))
    ∧ (1 / (1 - ρ) < γ →
      Tendsto (fun m : ℕ => equiOutCos (m + 2) (γ * Real.log (m + 2 : ℕ)) ρ 0 1)
        atTop (nhds ρ)) := by
  have hcast : ∀ m : ℕ, ((m + 2 : ℕ) : ℝ) = (m : ℝ) + 2 := fun m => by push_cast; ring
  have h1ρ : (0 : ℝ) < 1 - ρ := by linarith
  have main : ∀ L : ℝ,
      Tendsto (fun m : ℕ => equiDiag (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ)
        atTop (nhds L) →
      Tendsto (fun m : ℕ => equiOutCos (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ 0 1)
        atTop (nhds (ρ / (ρ + (1 - ρ) * L ^ 2))) := by
    intro L hL
    refine Filter.Tendsto.congr (fun m => (equiOutCos_seq_eq γ ρ hρ₀ hρ₁ m).symm) ?_
    exact tendsto_cos_of_tendsto ρ L hρ₀ hρ₁ _ _
      (fun m => equiDiag_pos _ _ (by omega)) (fun m => equiOff_pos _ _ (by omega))
      (equiDiag_add_equiOff_seq γ ρ) hL
  simp only [hcast]
  refine ⟨fun hγ => ?_, fun hγ => ?_, fun hγ => ?_⟩
  · have hc : γ * (1 - ρ) < 1 := by rw [lt_div_iff₀ h1ρ] at hγ; linarith
    have hval : ρ / (ρ + (1 - ρ) * (0 : ℝ) ^ 2) = 1 := by
      rw [show ((0 : ℝ) : ℝ) ^ 2 = 0 by norm_num]
      field_simp
      ring
    have hlim := main 0 (tendsto_equiDiag_of_lt γ ρ hc)
    rwa [hval] at hlim
  · have hc : γ * (1 - ρ) = 1 := by rw [hγ]; field_simp
    have hden : (1 : ℝ) + 3 * ρ ≠ 0 := by positivity
    have hval : ρ / (ρ + (1 - ρ) * ((1 : ℝ) / 2) ^ 2) = 4 * ρ / (1 + 3 * ρ) := by
      rw [div_eq_div_iff (by nlinarith) hden]
      ring
    have hlim := main (1 / 2) (tendsto_equiDiag_of_eq γ ρ hc)
    rwa [hval] at hlim
  · have hc : 1 < γ * (1 - ρ) := by rw [div_lt_iff₀ h1ρ] at hγ; linarith
    have hval : ρ / (ρ + (1 - ρ) * (1 : ℝ) ^ 2) = ρ := by norm_num
    have hlim := main 1 (tendsto_equiDiag_of_gt γ ρ hc)
    rwa [hval] at hlim

/-- The hypotheses of `long_context_phase_transition` are satisfiable: `ρ = 1/2`
lies strictly between `0` and `1`. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) / 2 < 1 := by norm_num

/-- The hypotheses are satisfiable: `a ≡ 1`, `b ≡ 0` is excluded by strict
positivity, so take `a m = 1/2` and `b m = 1/(2(m+1))`. -/
example : ∀ m : ℕ, (1 : ℝ) / 2 + ((m : ℝ) + 1) * (1 / (2 * ((m : ℝ) + 1))) = 1 := by
  intro m
  have : (0 : ℝ) < (m : ℝ) + 1 := by positivity
  field_simp
  ring

end MeanField
end Transformer
