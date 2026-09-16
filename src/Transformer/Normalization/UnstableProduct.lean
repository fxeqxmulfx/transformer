/-
# Normalization — A positive-definite factor preserves instability (App. D)

`Lemma lem: matrix` of arXiv:2510.22026v2: the product of a symmetric
positive-definite matrix and a symmetric unstable one is unstable.  Step 4 of
Appendix D applies it to the energy Hessian, so "unstable" is read as "has a
positive real eigenvalue".

The proof is the one the paper sketches, with Sylvester's law of inertia
replaced by the single consequence of it that is used: conjugating by the
square root `P` of `D` turns `D A` into the symmetric `P A P`, which inherits
the sign of one value of the quadratic form of `A`, hence has an eigenvalue of
that sign.
-/

import Transformer.Basic
import Mathlib.Analysis.Matrix.Order
import Mathlib.Analysis.Matrix.PosDef
import Mathlib.Analysis.Matrix.Spectrum

open scoped MatrixOrder Matrix

namespace Transformer
namespace Normalization

/-- **Lemma (lem: matrix).** *A positive-definite factor preserves
instability.*

For a symmetric unstable `A` and a symmetric positive-definite `D`, the
product `D A` is unstable.  The proof conjugates by the square root `P` of `D`:
`P⁻¹ (D A) P = P A P` is symmetric and has the inertia of `A`, so it keeps a
positive eigenvalue, and similar matrices share their spectrum.

Instability is spelled out as the existence of a real eigenvalue `μ > 0` with
an eigenvector; the symmetry of `A` is what makes that the right reading, and
the lemma is false without it.

The inertia step is taken in the only form the lemma needs: at `w = P⁻¹ v` the
quadratic form of `P A P` equals `μ ‖v‖² > 0`, so `-(P A P)` is not positive
semidefinite, so `-(P A P)` has a negative eigenvalue, and its eigenvector is
carried by `P` to an eigenvector of `D A = P (P A P) P⁻¹`.

Source: arXiv:2510.22026v2, Appendix D, Step 4, `lem: matrix`. -/
theorem unstable_mul_of_posDef
    (N : ℕ) (A D : Matrix (Fin N) (Fin N) ℝ)
    (hA : A.IsHermitian) (hD : D.PosDef)
    (hAunstable : ∃ (μ : ℝ) (v : Fin N → ℝ),
      v ≠ 0 ∧ A.mulVec v = μ • v ∧ 0 < μ) :
    ∃ (μ : ℝ) (v : Fin N → ℝ), v ≠ 0 ∧ (D * A).mulVec v = μ • v ∧ 0 < μ := by
  classical
  obtain ⟨μ, v, hv, hAv, hμ⟩ := hAunstable
  have hD0 : (0 : Matrix (Fin N) (Fin N) ℝ) ≤ D := Matrix.nonneg_iff_posSemidef.mpr hD.posSemidef
  set P : Matrix (Fin N) (Fin N) ℝ := CFC.sqrt D with hPdef
  have hPsd : P.PosSemidef := Matrix.nonneg_iff_posSemidef.mp (CFC.sqrt_nonneg D)
  have hPP : P * P = D := by rw [hPdef, ← sq]; exact CFC.sq_sqrt D
  have hPT : Pᵀ = P := by
    have h : P.IsSymm := by simpa using hPsd.isHermitian
    exact h
  have hPdet : IsUnit P.det := by
    have hDdet : IsUnit D.det := (Matrix.isUnit_iff_isUnit_det D).mp hD.isUnit
    have hsq : P.det * P.det = D.det := by rw [← Matrix.det_mul, hPP]
    rw [isUnit_iff_ne_zero] at hDdet ⊢
    intro h
    rw [h, mul_zero] at hsq
    exact hDdet hsq.symm
  -- `M = P A P` is the symmetric matrix similar to `D A`.
  set M : Matrix (Fin N) (Fin N) ℝ := P * A * P with hMdef
  have hMH : M.IsHermitian := by
    show Mᴴ = M
    rw [hMdef, Matrix.conjTranspose_mul, Matrix.conjTranspose_mul, hPsd.isHermitian.eq, hA.eq,
      Matrix.mul_assoc]
  have hMneg : (-M).IsHermitian := hMH.neg
  -- At `w = P⁻¹ v` the quadratic form of `M` is the one of `A` at `v`.
  set w : Fin N → ℝ := P⁻¹ *ᵥ v with hwdef
  have hPw : P *ᵥ w = v := by
    rw [hwdef, Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv P hPdet, Matrix.one_mulVec]
  have hMw : M *ᵥ w = μ • (P *ᵥ v) := by
    rw [hMdef, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, hPw, hAv, Matrix.mulVec_smul]
  have hdot : w ⬝ᵥ (P *ᵥ v) = v ⬝ᵥ v := by
    rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, hPT, hPw]
  have hvv : 0 < v ⬝ᵥ v := by
    have h0 : 0 ≤ v ⬝ᵥ v := Finset.sum_nonneg fun i _ => mul_self_nonneg _
    rcases h0.lt_or_eq with h | h
    · exact h
    · exact absurd (dotProduct_self_eq_zero.mp h.symm) hv
  have hnot : ¬ (-M).PosSemidef := by
    intro hps
    have hge := hps.dotProduct_mulVec_nonneg w
    rw [star_trivial, Matrix.neg_mulVec, hMw] at hge
    simp only [dotProduct_neg, dotProduct_smul, hdot, smul_eq_mul] at hge
    nlinarith
  obtain ⟨i, hi⟩ : ∃ i, hMneg.eigenvalues i < 0 := by
    by_contra hcon
    push Not at hcon
    exact hnot (hMneg.posSemidef_iff_eigenvalues_nonneg.mpr fun i => hcon i)
  set u : Fin N → ℝ := ⇑(hMneg.eigenvectorBasis i) with hudef
  have hMu : (-M) *ᵥ u = hMneg.eigenvalues i • u := hMneg.mulVec_eigenvectorBasis i
  have hMu' : M *ᵥ u = (-(hMneg.eigenvalues i)) • u := by
    rw [Matrix.neg_mulVec] at hMu
    rw [neg_smul, ← hMu, neg_neg]
  have hu0 : u ≠ 0 := by
    intro h
    have hb : hMneg.eigenvectorBasis i ≠ 0 := (hMneg.eigenvectorBasis).toBasis.ne_zero i
    exact hb (by ext j; exact congrFun h j)
  refine ⟨-(hMneg.eigenvalues i), P *ᵥ u, ?_, ?_, by linarith⟩
  · intro h
    apply hu0
    have h2 := congrArg (fun x => P⁻¹ *ᵥ x) h
    simpa [Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul P hPdet] using h2
  · calc (D * A) *ᵥ (P *ᵥ u) = ((D * A) * P) *ᵥ u := by rw [Matrix.mulVec_mulVec]
      _ = (P * M) *ᵥ u := by rw [hMdef, ← hPP]; simp [Matrix.mul_assoc]
      _ = P *ᵥ (M *ᵥ u) := by rw [Matrix.mulVec_mulVec]
      _ = P *ᵥ ((-(hMneg.eigenvalues i)) • u) := by rw [hMu']
      _ = (-(hMneg.eigenvalues i)) • (P *ᵥ u) := Matrix.mulVec_smul _ _ _

/-- The hypotheses of `unstable_mul_of_posDef` are satisfiable: the identity
matrix is symmetric, positive-definite, and unstable — its only eigenvalue is
`1 > 0`. -/
example :
    (1 : Matrix (Fin 1) (Fin 1) ℝ).IsHermitian ∧
      (1 : Matrix (Fin 1) (Fin 1) ℝ).PosDef ∧
      ∃ (μ : ℝ) (v : Fin 1 → ℝ),
        v ≠ 0 ∧ (1 : Matrix (Fin 1) (Fin 1) ℝ).mulVec v = μ • v ∧ 0 < μ := by
  refine ⟨Matrix.isHermitian_one, Matrix.PosDef.one, 1, fun _ => 1, ?_, ?_, one_pos⟩
  · intro h
    simpa using congrFun h 0
  · simp

end Normalization
end Transformer
