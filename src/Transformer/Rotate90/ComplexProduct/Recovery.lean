module

public import Transformer.Rotate90.ComplexProduct.Readout

/-!
# From a nonunique spectral optimum to the real source classifier

Source: `review/flat_spectrum.md`, *Complex-product Rotate90 bridge*, CP4.
The rank-r dual certificate recovers only correct-versus-incorrect linear score
contrasts. A separate, explicit alignment identifies those contrasts with the
computed complex-product residual state's tied readout differences. CP3 certifies an
accurate execution. No tangent injectivity, matrix uniqueness or convergence
of AdamW is assumed. Fitting, global optimality, alignment and numerical error
are hypotheses to establish for a particular learned checkpoint, not conclusions.
Concrete satisfiable instances are in `RecoveryWitness`.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open Matrix Grokking.FlatSpectrum

variable {V K H Q m n r : Type*} [Fintype K] [Fintype H]
  [Fintype m] [Fintype n] [Fintype r] [DecidableEq m] [DecidableEq n] [DecidableEq r]

/-- **Conditional spectral-to-execution bridge for the actual product model**, CP4.
The contrast alignment is checked before final RMSNorm, so normalization is derived,
not hidden in a supplied answer or score. Neither absolute scores nor the whole
feature matrix need agree. Unobserved tangent directions need only preserve the
indicated linear contrasts. No unique feature-recovery premise occurs. -/
theorem executed_correct_of_nonunique_minimum {targetMatrix : Matrix m n ℂ}
    {U : Matrix m r ℂ} {V₀ : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : V₀ᴴ * V₀ = 1)
    (hKU : U * Uᴴ * targetMatrix = targetMatrix)
    (hK : ((U * V₀ᴴ)ᴴ * targetMatrix).trace = nuclearNorm targetMatrix)
    {Ω : Set (m × n)} {W : Matrix m n ℂ}
    (hsupport : ∀ i j, (i, j) ∉ Ω → (U * V₀ᴴ + W) i j = 0)
    (hWU : Uᴴ * W = 0) (hWV : W * V₀ = 0)
    {c : ℝ} (hc : 1 < c) (hW : IsContraction ((c : ℂ) • W))
    (score : Q → V → Matrix m n ℂ →ₗ[ℝ] ℝ) (answer : Q → V)
    (hcontrast : ∀ (P : Matrix n r ℂ) (B : Matrix m r ℂ),
      (∀ ij ∈ Ω, (U * Pᴴ + B * V₀ᴴ) ij.1 ij.2 = 0) →
      ∀ q v, (score q (answer q) - score q v) (U * Pᴴ + B * V₀ᴴ) = 0)
    {X : Matrix m n ℂ} (hfit : ∀ ij ∈ Ω, X ij.1 ij.2 = targetMatrix ij.1 ij.2)
    (hmin : IsMinOn nuclearNorm {Y | ∀ ij ∈ Ω, Y ij.1 ij.2 = targetMatrix ij.1 ij.2} X)
    {eps : ℝ} (heps : 0 < eps) (w : Weights V K H) (inputs : Q → V × V)
    (halign : ∀ q v,
      readout w.embedding (state eps w (inputs q).1 (inputs q).2) (answer q) -
        readout w.embedding (state eps w (inputs q).1 (inputs q).2) v =
          score q (answer q) X - score q v X)
    (executed : Q → V → ℝ) (δ η : Q → ℝ)
    (hgap : ∀ q v, v ≠ answer q →
      δ q ≤ score q (answer q) targetMatrix - score q v targetMatrix)
    (herr : ∀ q v, |executed q v - realLogit eps w (inputs q).1 (inputs q).2 v| ≤ η q)
    (hbudget : ∀ q, 2 * η q * denominator eps
      (state eps w (inputs q).1 (inputs q).2) < δ q) :
    ∀ q v, v ≠ answer q → executed q v < executed q (answer q) := by
  intro q
  apply executed_correct_of_raw_margin heps w (inputs q).1 (inputs q).2 (answer q)
    (executed q) (δ := δ q) ?_ (herr q) (hbudget q)
  intro v hv
  rw [halign]
  have h := FeatureCodes.linear_readout_eq_of_certificate hU hV hKU hK hsupport
    hWU hWV hc hW (score q (answer q) - score q v)
    (fun P B hz => hcontrast P B hz q v) hfit hmin
  simp only [LinearMap.sub_apply] at h
  rw [h]
  exact hgap q v hv

end Transformer.Rotate90.ComplexProduct
