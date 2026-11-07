module

public import Transformer.Rotate90.ComplexProduct.Defs

/-!
# Exact real/complex correspondence for the executed architecture

Source: `review/flat_spectrum.md`, *Complex-product Rotate90 bridge*, CP1–CP2.
The independently defined expanded matrices and scalar-product kernels agree
with complex arithmetic. The correspondence includes positive-epsilon RMSNorm,
the fixed J, the residual and the tied readout; it is not a floating-point claim.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open Landscape.Neural
open scoped BigOperators

variable {V K H : Type*}

/-- Realification commutes with addition (CP1). -/
theorem realify_add (z u : K → ℂ) :
    realify (fun k => z k + u k) = fun i => realify z i + realify u i := by
  funext ⟨k, j⟩
  fin_cases j <;> simp [realify]

/-- The implementation's adjacent-pair rotation is exactly multiplication by i (CP1). -/
theorem realify_input (E : V → K → ℂ) (a b : V) :
    realify (fun k => E a k + Complex.I * E b k) =
      tiedInput (fun q => realify (E q)) a b := by
  funext ⟨k, j⟩
  fin_cases j <;> simp [realify, tiedInput, rotate90]

variable [Fintype K] [Fintype H]

/-- Squared complex magnitudes reproduce the mean over the actual real width (CP1). -/
theorem meanSquare_realify (z : K → ℂ) : meanSquare (realify z) = energy z := by
  simp only [meanSquare, energy, Channels, Fintype.sum_prod_type, Fin.sum_univ_two,
    realify, Fin.isValue, ↓reduceIte, Fin.reduceEq, Fintype.card_prod, Fintype.card_fin,
    Nat.cast_mul, Nat.cast_ofNat, Complex.normSq_apply, pow_two]
  congr 1
  ring

/-- Positive epsilon makes every source normalization denominator positive (CP2). -/
theorem denominator_pos {eps : ℝ} (heps : 0 < eps) (z : K → ℂ) :
    0 < denominator eps z := by
  have h := meanSquare_nonneg (realify z)
  rw [meanSquare_realify] at h
  exact Real.sqrt_pos.mpr (add_pos_of_pos_of_nonneg heps h)

/-- Normalization retains epsilon and agrees coordinate by coordinate (CP1). -/
theorem realify_norm (eps : ℝ) (z : K → ℂ) :
    realify (norm eps z) = rmsNorm eps (realify z) := by
  funext ⟨k, j⟩
  fin_cases j <;> simp [realify, norm, rmsNorm, denominator, meanSquare_realify]

omit [Fintype H] in
/-- Tied real blocks A,-B,B,A compute the genuine complex-linear map (CP1). -/
theorem realify_dense (A : K → H → ℂ) (z : K → ℂ) :
    realify (denseC A z) = dense (realMatrix A) (realify z) := by
  funext ⟨j, c⟩
  fin_cases c <;>
    simp [realify, denseC, dense, realMatrix, Fintype.sum_prod_type,
      Fin.sum_univ_two, sub_eq_add_neg]

omit [Fintype H] in
/-- Both mixed products, their signs and sqrt(2) calibration agree exactly (CP1). -/
theorem realify_product (g u : H → ℂ) :
    realify (product g u) = realProduct (realify g) (realify u) := by
  funext ⟨j, c⟩
  fin_cases c <;> simp [realify, product, realProduct]

/-- The independent real branch realizes the specified complex product block (CP1). -/
theorem realify_branch (w : Weights V K H) (z : K → ℂ) :
    realify (branch w z) = realBranch w (realify z) := by
  unfold branch realBranch
  rw [realify_dense, realify_product, realify_dense, realify_dense]

/-- Exact source-state correspondence, including the residual and all three preceding
normalizations (CP1). No answer or recovery premise enters this equality. -/
theorem realify_state (eps : ℝ) (w : Weights V K H) (a b : V) :
    realify (state eps w a b) = realState eps w a b := by
  unfold state realState
  rw [realify_add]
  simp only [realify_norm, realify_branch, realify_input]

/-- The real tied readout is the real part of the Hermitian readout (CP1). -/
theorem realify_readout (E : V → K → ℂ) (z : K → ℂ) (q : V) :
    tiedReadout (fun q => realify (E q)) (realify z) q = readout E z q := by
  simp [tiedReadout, readout, realify, Fintype.sum_prod_type,
    Fin.sum_univ_two]

/-- **Actual expanded logits equal complex-model logits**, source CP1. -/
theorem realLogit_eq_logit (eps : ℝ) (w : Weights V K H) (a b q : V) :
    realLogit eps w a b q = logit eps w a b q := by
  unfold realLogit logit
  rw [← realify_state, ← realify_norm, realify_readout]

/-- Final RMSNorm divides all class scores by the same positive denominator (CP2). -/
theorem readout_norm (eps : ℝ) (E : V → K → ℂ) (z : K → ℂ) (q : V) :
    readout E (norm eps z) q = readout E z q / denominator eps z := by
  rw [← realify_readout, realify_norm, Rotate90.readout_rmsNorm,
    realify_readout, meanSquare_realify]
  rfl

example := realify_add (fun k : Fin 2 => (k.val : ℂ) + Complex.I) (fun _ => 1 - Complex.I)
example := realify_input (fun (q : Bool) (_ : Fin 1) => if q then Complex.I else 1) false true
example := meanSquare_realify (fun _ : Fin 1 => (1 + Complex.I))
example := denominator_pos (by norm_num : (0 : ℝ) < 1 / 8388608)
  (fun _ : Fin 1 => (1 + Complex.I))
example := realify_norm (1 / 8388608) (fun _ : Fin 1 => (1 + Complex.I))
example := realify_dense (fun (_ : Fin 1) (_ : Fin 1) => (2 + Complex.I))
  (fun _ => (1 - Complex.I))
example := realify_product (fun _ : Fin 1 => (1 + Complex.I)) (fun _ => (2 - Complex.I))
example := realify_readout (fun (q : Bool) (_ : Fin 1) => if q then Complex.I else 1)
  (fun _ => 1 + Complex.I) true
example := readout_norm 1 (fun (q : Bool) (_ : Fin 1) => if q then Complex.I else 1)
  (fun _ => 1 + Complex.I) true

end Transformer.Rotate90.ComplexProduct
