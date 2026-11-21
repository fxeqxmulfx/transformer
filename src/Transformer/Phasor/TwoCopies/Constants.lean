module

public import Transformer.Phasor.TwoCopies.Product

/-!
# Two copies: the constants of the forward pass

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

Every RMS of the construction acts on a state whose squared length does not depend on the tokens
until the output plane is written (`sum_stateC_sq`), so it multiplies the coefficients by a
constant. These are the constants, for the width `D`, `ε`, the bias `b` of the embedding, the depth
scale `s` and the gain `g` of the second branch. All of them are positive once `D > 0`, `ε ≥ 0`
and `s > 0` (`aAttn_pos`, `ampPlanes_pos`, `ampProd_pos`), which is all the construction needs:
the heads decode at any positive logit scale, and the phases do not depend on the amplitudes.
-/

@[expose] public section

open Finset Real

namespace Transformer.Phasor.TwoCopies

variable {D : ℕ} {eps b s : ℝ}

/-- The first RMS, on an embedding of squared length `b² + 2`. -/
noncomputable def kEmb (D : ℕ) (eps b : ℝ) : ℝ := rmsScale D eps (b ^ 2 + 2)

/-- The coefficient of a token's code in the heads' read of the first layer. -/
noncomputable def aRead (D : ℕ) (eps b : ℝ) : ℝ :=
  rmsScale D eps (kEmb D eps b ^ 2 * (b ^ 2 + 2)) * kEmb D eps b

/-- The coefficient of the codes `(j, t_j)`, `j < 3`, after the attention's residual. -/
noncomputable def aAttn (D : ℕ) (eps b s : ℝ) : ℝ :=
  s * rmsScale D eps (3 * aRead D eps b ^ 2) * aRead D eps b

/-- The RMS before the first branch. -/
noncomputable def kFFN0 (D : ℕ) (eps b s : ℝ) : ℝ :=
  rmsScale D eps (kEmb D eps b ^ 2 * (b ^ 2 + 2) + 3 * aAttn D eps b s ^ 2)

/-- The amplitudes that the first branch writes into the planes: the product of the two operands'
code coefficients over `√2`. -/
noncomputable def ampFFN0 (D : ℕ) (eps b s : ℝ) (c : Fin 2) : ℝ :=
  codeCoef (kFFN0 D eps b s * kEmb D eps b) (kFFN0 D eps b s * aAttn D eps b s) (xPos c) *
    codeCoef (kFFN0 D eps b s * kEmb D eps b) (kFFN0 D eps b s * aAttn D eps b s) (yPos c) /
      Real.sqrt 2

/-- The amplitudes of the planes after the first branch's residual. -/
noncomputable def ampPlanes (D : ℕ) (eps b s : ℝ) : Fin 2 → ℝ :=
  (s * rmsScale D eps (ampFFN0 D eps b s 0 ^ 2 + ampFFN0 D eps b s 1 ^ 2)) • ampFFN0 D eps b s

/-- The RMS before the second branch. -/
noncomputable def kFFN1 (D : ℕ) (eps b s : ℝ) : ℝ :=
  rmsScale D eps (kEmb D eps b ^ 2 * (b ^ 2 + 2) + 3 * aAttn D eps b s ^ 2 +
    (ampPlanes D eps b s 0 ^ 2 + ampPlanes D eps b s 1 ^ 2))

/-- The product that the second branch computes, before its gain. -/
noncomputable def ampProd (D : ℕ) (eps b s : ℝ) : ℝ :=
  (kFFN1 D eps b s • ampPlanes D eps b s) 0 * (kFFN1 D eps b s • ampPlanes D eps b s) 1 /
    Real.sqrt 2

/-- The amplitude of the answer in the output plane. -/
noncomputable def ampOut (D : ℕ) (eps b s g : ℝ) : ℝ :=
  s * rmsScale D eps ((g * ampProd D eps b s) ^ 2) * (g * ampProd D eps b s)

theorem kEmb_pos (hD : 0 < D) (heps : 0 ≤ eps) : 0 < kEmb D eps b :=
  rmsScale_pos hD heps (by positivity)

theorem aRead_pos (hD : 0 < D) (heps : 0 ≤ eps) : 0 < aRead D eps b :=
  mul_pos (rmsScale_pos hD heps (by have := kEmb_pos (b := b) hD heps; positivity))
    (kEmb_pos hD heps)

theorem aAttn_pos (hD : 0 < D) (heps : 0 ≤ eps) (hs : 0 < s) : 0 < aAttn D eps b s := by
  have := aRead_pos (b := b) hD heps
  exact mul_pos (mul_pos hs (rmsScale_pos hD heps (by positivity))) this

theorem kFFN0_pos (hD : 0 < D) (heps : 0 ≤ eps) : 0 < kFFN0 D eps b s :=
  rmsScale_pos hD heps (by have := kEmb_pos (b := b) hD heps; positivity)

theorem ampFFN0_pos (hD : 0 < D) (heps : 0 ≤ eps) (hs : 0 < s) (c : Fin 2) :
    0 < ampFFN0 D eps b s c := by
  have h1 := mul_pos (kFFN0_pos (b := b) (s := s) hD heps) (kEmb_pos (b := b) hD heps)
  have h2 := mul_pos (kFFN0_pos (b := b) (s := s) hD heps) (aAttn_pos (b := b) hD heps hs)
  have hc (j : Fin 4) : 0 < codeCoef (kFFN0 D eps b s * kEmb D eps b)
      (kFFN0 D eps b s * aAttn D eps b s) j := by
    unfold codeCoef; split_ifs <;> assumption
  unfold ampFFN0
  exact div_pos (mul_pos (hc _) (hc _)) (by positivity)

theorem ampPlanes_pos (hD : 0 < D) (heps : 0 ≤ eps) (hs : 0 < s) (c : Fin 2) :
    0 < ampPlanes D eps b s c := by
  have h := ampFFN0_pos (b := b) hD heps hs
  simp only [ampPlanes, Pi.smul_apply, smul_eq_mul]
  exact mul_pos (mul_pos hs (rmsScale_pos hD heps (by have := h 0; positivity))) (h c)

theorem kFFN1_pos (hD : 0 < D) (heps : 0 ≤ eps) : 0 < kFFN1 D eps b s :=
  rmsScale_pos hD heps (by have := kEmb_pos (b := b) hD heps; positivity)

theorem ampProd_pos (hD : 0 < D) (heps : 0 ≤ eps) (hs : 0 < s) : 0 < ampProd D eps b s := by
  have h := ampPlanes_pos (b := b) hD heps hs
  have hk := kFFN1_pos (b := b) (s := s) hD heps
  simp only [ampProd, Pi.smul_apply, smul_eq_mul]
  exact div_pos (mul_pos (mul_pos hk (h 0)) (mul_pos hk (h 1))) (by positivity)

/-- Main's width `D = 128`, `ε = 2⁻²³` and depth scale `1/√8` (`L = 4`), bias `1`. -/
example : 0 < kEmb 128 (1 / 8388608) 1 ∧ 0 < aRead 128 (1 / 8388608) 1 ∧
    0 < aAttn 128 (1 / 8388608) 1 (1 / Real.sqrt 8) ∧ 0 < kFFN0 128 (1 / 8388608) 1 1 ∧
    0 < ampFFN0 128 (1 / 8388608) 1 (1 / Real.sqrt 8) 0 ∧
    0 < ampPlanes 128 (1 / 8388608) 1 (1 / Real.sqrt 8) 1 ∧ 0 < kFFN1 128 (1 / 8388608) 1 1 ∧
    0 < ampProd 128 (1 / 8388608) 1 (1 / Real.sqrt 8) := by
  have hD : 0 < 128 := by norm_num
  have heps : (0 : ℝ) ≤ 1 / 8388608 := by norm_num
  have hs : 0 < 1 / Real.sqrt 8 := by positivity
  exact ⟨kEmb_pos hD heps, aRead_pos hD heps, aAttn_pos hD heps hs, kFFN0_pos hD heps,
    ampFFN0_pos hD heps hs 0, ampPlanes_pos hD heps hs 1, kFFN1_pos hD heps, ampProd_pos hD heps hs⟩

end Transformer.Phasor.TwoCopies
