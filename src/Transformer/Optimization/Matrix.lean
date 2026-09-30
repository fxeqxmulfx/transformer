/-
# Rectangular training parameters with Frobenius geometry

Source: arXiv:2502.16982, §2.1–2.2, and arXiv:2602.02016v2, §2–4.
Mathlib's ordinary function-space matrix norm is not the Frobenius norm.
Flattening into `EuclideanSpace` equips the actual parameter entries with
the inner product and norm used in both manuscripts and in the guard.
-/

import Transformer.DASH.Section2_Models
import Mathlib.Analysis.InnerProductSpace.PiL2

open scoped BigOperators

noncomputable section

namespace Transformer.Optimization

variable {m n : ℕ}

/-- Rectangular parameters with their Frobenius inner product.
Source: arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2.3. -/
abbrev MatrixSpace (m n : ℕ) := EuclideanSpace ℝ (Fin m × Fin n)

/-- Preserve every entry when passing parameters to a matrix optimizer.
Source: arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2.2. -/
def toMatrix (x : MatrixSpace m n) : Matrix (Fin m) (Fin n) ℝ :=
  fun i j => x (i, j)

/-- Preserve every entry when returning a matrix candidate to the
parameter space. Source: arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2.2–2.3. -/
def fromMatrix (X : Matrix (Fin m) (Fin n) ℝ) : MatrixSpace m n :=
  WithLp.toLp 2 (fun ij => X ij.1 ij.2)

/-- Matrix entry conversion is lossless,
arXiv:2502.16982, §2.1–2.2, and arXiv:2602.02016v2, §2.2–2.3. -/
theorem toMatrix_fromMatrix (X : Matrix (Fin m) (Fin n) ℝ) :
    toMatrix (fromMatrix X) = X := rfl

/-- Parameter entry conversion is lossless,
arXiv:2502.16982, §2.1–2.2, and arXiv:2602.02016v2, §2.2–2.3. -/
theorem fromMatrix_toMatrix (x : MatrixSpace m n) : fromMatrix (toMatrix x) = x := by
  ext ij
  rfl

/-- The training norm is exactly the source's Frobenius norm, rather than
the default matrix function-space norm. Source: arXiv:2502.16982, §2.1,
and arXiv:2602.02016v2, §2.3. -/
theorem fromMatrix_norm (X : Matrix (Fin m) (Fin n) ℝ) :
    ‖fromMatrix X‖ = DASH.frobeniusNorm X := by
  have h : ‖fromMatrix X‖ ^ 2 = Muon.squaredFrobenius X := by
    rw [EuclideanSpace.norm_sq_eq, Fintype.sum_prod_type]
    simp only [fromMatrix, PiLp.toLp_apply, Real.norm_eq_abs, sq_abs, Muon.squaredFrobenius]
  rw [DASH.frobeniusNorm, ← h, Real.sqrt_sq (norm_nonneg _)]

/-- Squared norm of the true gradient agrees with entrywise gradient
energy. Source: arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2.3. -/
theorem toMatrix_energy (x : MatrixSpace m n) :
    Muon.squaredFrobenius (toMatrix x) = ‖x‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq, Fintype.sum_prod_type]
  simp only [toMatrix, Muon.squaredFrobenius, Real.norm_eq_abs, sq_abs]

end Transformer.Optimization
