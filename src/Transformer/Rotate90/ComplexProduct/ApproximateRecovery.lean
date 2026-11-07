module

public import Transformer.Rotate90.ComplexProduct.Readout
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.ApproximateRecovery

/-!
# Approximate spectral recovery through the actual complex-product source

Source: `review/flat_spectrum.md`, *Approximate nonunique recovery*, QR4.
Recovery and alignment errors are measured on raw correct-versus-wrong score
contrasts. A contrast error `B` and a per-logit execution error `η` preserve a
target raw margin `δ` when `B + 2ηD < δ`, where `D` is the actual final RMS
denominator. The spectral corollary derives `B` from the fitting correction,
objective gap, certificate slack and readout conditioning. Training, spectral
alignment and backend accuracy remain separate obligations.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open Matrix Grokking.FlatSpectrum Grokking.FlatSpectrum.FeatureCodes

variable {V K H : Type*} [Fintype K] [Fintype H]

/-- **Contrast error plus execution error**, QR4. The final RMS denominator
rescales the surviving raw margin, including the error in source alignment. -/
theorem executed_correct_of_contrast_error {eps : ℝ} (heps : 0 < eps)
    (w : Weights V K H) (a b target : V) (ideal executed : V → ℝ) {δ B η : ℝ}
    (hgap : ∀ v, v ≠ target → δ ≤ ideal target - ideal v)
    (hcontrast : ∀ v, v ≠ target →
      |(readout w.embedding (state eps w a b) target -
        readout w.embedding (state eps w a b) v) - (ideal target - ideal v)| ≤ B)
    (herr : ∀ v, |executed v - realLogit eps w a b v| ≤ η)
    (hbudget : B + 2 * η * denominator eps (state eps w a b) < δ) :
    ∀ v, v ≠ target → executed v < executed target := by
  apply executed_correct_of_raw_margin heps w a b target executed (δ := δ - B)
    (η := η) ?_ herr (by linarith)
  intro v hv
  have h := (abs_le.mp (hcontrast v hv)).1
  linarith [hgap v hv]

variable {Q m n r : Type*} [Fintype m] [Fintype n] [Fintype r]
  [DecidableEq m] [DecidableEq n] [DecidableEq r]

/-- **Quantitative recovery reaches the executed source classifier**, QR4.
The matrix need only have a bounded objective gap, its observations may be noisy,
and each source contrast may differ from its recovered value. No unique recovery
or exact optimality is assumed. All constants are explicit and may be pessimistic. -/
theorem executed_correct_of_approximate_recovery {targetMatrix : Matrix m n ℂ}
    {U : Matrix m r ℂ} {V₀ : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : V₀ᴴ * V₀ = 1)
    (hKU : U * Uᴴ * targetMatrix = targetMatrix)
    (hK : ((U * V₀ᴴ)ᴴ * targetMatrix).trace = nuclearNorm targetMatrix)
    {Ω : Set (m × n)} {W : Matrix m n ℂ}
    (hsupport : ∀ i j, (i, j) ∉ Ω → (U * V₀ᴴ + W) i j = 0)
    (hWU : Uᴴ * W = 0) (hWV : W * V₀ = 0)
    {c : ℝ} (hc : 1 < c) (hW : IsContraction ((c : ℂ) • W))
    (score : Q → V → Matrix m n ℂ →ₗ[ℝ] ℝ) (answer : Q → V)
    (κ ℓ : Q → V → ℝ) (hκ : ∀ q v, 0 ≤ κ q v) (hℓ : ∀ q v, 0 ≤ ℓ q v)
    (hcondition : ∀ q v M, (∀ ij ∈ Ω, M ij.1 ij.2 = 0) →
      |(score q (answer q) - score q v) M| ≤
        κ q v * nuclearNorm ((1 - U * Uᴴ) * M * (1 - V₀ * V₀ᴴ)))
    (hbound : ∀ q v M, |(score q (answer q) - score q v) M| ≤ ℓ q v * nuclearNorm M)
    {X E : Matrix m n ℂ} {ε ρ : ℝ}
    (hfit : ∀ ij ∈ Ω, (X - E) ij.1 ij.2 = targetMatrix ij.1 ij.2)
    (hobjective : nuclearNorm X ≤ nuclearNorm targetMatrix + ε)
    (hnoise : nuclearNorm E ≤ ρ)
    {eps : ℝ} (heps : 0 < eps) (w : Weights V K H) (inputs : Q → V × V)
    (α : Q → V → ℝ)
    (halign : ∀ q v, |(readout w.embedding (state eps w (inputs q).1 (inputs q).2)
      (answer q) - readout w.embedding (state eps w (inputs q).1 (inputs q).2) v) -
        (score q (answer q) X - score q v X)| ≤ α q v)
    (executed : Q → V → ℝ) (δ η : Q → ℝ)
    (hgap : ∀ q v, v ≠ answer q →
      δ q ≤ score q (answer q) targetMatrix - score q v targetMatrix)
    (herr : ∀ q v, |executed q v - realLogit eps w (inputs q).1 (inputs q).2 v| ≤ η q)
    (hbudget : ∀ q v, v ≠ answer q →
      α q v + (κ q v * ((ε + ρ) / (1 - c⁻¹)) + ℓ q v * ρ) +
        2 * η q * denominator eps (state eps w (inputs q).1 (inputs q).2) < δ q) :
    ∀ q v, v ≠ answer q → executed q v < executed q (answer q) := by
  intro q v hv
  have hrecovery := linear_readout_error_le_of_approximate_minimum hU hV hKU hK
    hsupport hWU hWV hc hW (score q (answer q) - score q v) (hκ q v) (hℓ q v)
    (hcondition q v) (hbound q v) hfit hobjective hnoise
  simp only [LinearMap.sub_apply] at hrecovery
  have hcerror := contrast_error_le_of_alignment (halign q v) hrecovery
  have hlower := (abs_le.mp hcerror).1
  have htarget := hgap q v hv
  have hbudget' := hbudget q v hv
  have hd := denominator_pos heps (state eps w (inputs q).1 (inputs q).2)
  have he₁ := (abs_le.mp (herr q (answer q))).1
  have he₂ := (abs_le.mp (herr q v)).2
  rw [realLogit_eq_readout_div] at he₁ he₂
  have hraw : 2 * η q * denominator eps (state eps w (inputs q).1 (inputs q).2) <
      readout w.embedding (state eps w (inputs q).1 (inputs q).2) (answer q) -
      readout w.embedding (state eps w (inputs q).1 (inputs q).2) v := by linarith
  have hnormalized := (lt_div_iff₀ hd).mpr hraw
  rw [sub_div] at hnormalized
  linarith

end Transformer.Rotate90.ComplexProduct
