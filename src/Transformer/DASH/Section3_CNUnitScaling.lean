/-
# DASH — failure at the unit-scaled CN endpoint

arXiv:2602.02016v2, §3.4 chooses `(p+1)c^p=1`. Exact normalization
by the largest eigenvalue then puts a nonzero positive matrix at
the excluded CN endpoint. A one-dimensional Frobenius example has
the same failure, whereas NDB is exact there.
-/

import Transformer.DASH.Section3_MatrixConvergence
import Transformer.DASH.Section3_Scaling

open scoped Matrix

noncomputable section

namespace Transformer.DASH

/-- The CN scaling constant chosen in the source's comparison regime.
Source: arXiv:2602.02016v2, §3.4, `c=(1+p)^(-1/p)`. -/
def cnUnitScale (p : ℕ) : ℝ := ((p : ℝ) + 1) ^ (-(1 / (p : ℝ)))

/-- The chosen scalar scale is positive for every natural order,
arXiv:2602.02016v2, §3.4. -/
theorem cnUnitScale_pos (p : ℕ) : 0 < cnUnitScale p :=
  Real.rpow_pos_of_pos (by positivity) _

/-- The source's constant has precisely the claimed unit spectral bound
for every positive integer order. Source: arXiv:2602.02016v2, §3.4. -/
theorem cnUnitScale_power (p : ℕ) (hp : 0 < p) :
    ((p : ℝ) + 1) * cnUnitScale p ^ p = 1 := by
  have hpne : (p : ℝ) ≠ 0 := by exact_mod_cast hp.ne'
  have hbase : (0 : ℝ) < (p : ℝ) + 1 := by positivity
  have hpower : cnUnitScale p ^ p = ((p : ℝ) + 1) ^ (-1 : ℝ) := by
    rw [cnUnitScale, ← Real.rpow_mul_natCast hbase.le]
    congr 1
    field_simp
  rw [hpower, Real.rpow_neg_one, mul_inv_cancel₀ hbase.ne']

/-- The source's positive CN orders exist, arXiv:2602.02016v2, §3.4. -/
example : 0 < (2 : ℕ) := by norm_num

/-- At the upper spectral endpoint, both CN sequences become zero after
the first step, for every positive integer order and positive scale.
This generalizes the explicit order-two endpoint counterexample.
Source: arXiv:2602.02016v2, §3.2, the printed closed convergence interval. -/
theorem cn_general_upper_endpoint (p : ℕ) (c : ℝ) (hp : 0 < p) (hc : 0 < c) (k : ℕ) :
    cnScalarM p (((p : ℝ) + 1) * c ^ p) c (k + 1) = 0 ∧
      cnScalarX p (((p : ℝ) + 1) * c ^ p) c (k + 1) = 0 := by
  have hpne : (p : ℝ) ≠ 0 := by exact_mod_cast hp.ne'
  have hinitial : cnScalarM p (((p : ℝ) + 1) * c ^ p) c 0 = (p : ℝ) + 1 := by
    exact mul_div_cancel_right₀ _ (pow_ne_zero _ hc.ne')
  have hfactor : cnFactor p ((p : ℝ) + 1) = 0 := by
    unfold cnFactor
    field_simp
    ring
  induction k with
  | zero =>
    change cnMap p (cnScalarM p (((p : ℝ) + 1) * c ^ p) c 0) = 0 ∧
      c⁻¹ * cnFactor p (cnScalarM p (((p : ℝ) + 1) * c ^ p) c 0) = 0
    rw [hinitial]
    simp [cnMap, hfactor, hp.ne']
  | succ k ih =>
    change cnMap p (cnScalarM p (((p : ℝ) + 1) * c ^ p) c (k + 1)) = 0 ∧
      cnScalarX p (((p : ℝ) + 1) * c ^ p) c (k + 1) *
        cnFactor p (cnScalarM p (((p : ℝ) + 1) * c ^ p) c (k + 1)) = 0
    rw [ih.1, ih.2]
    simp [cnMap]

/-- The endpoint hypotheses are satisfiable for a Shampoo order,
arXiv:2602.02016v2, §3.2. -/
example : 0 < (4 : ℕ) ∧ (0 : ℝ) < 1 := by norm_num

/-- Normalizing an eigenvalue to one with the comparison scale makes CN
collapse to zero for every later step. Exact largest-eigenvalue scaling
therefore does not satisfy CN's corrected open domain in this regime.
Source: arXiv:2602.02016v2, §3.4, unit-domain choice and exact spectral scaling. -/
theorem cn_unit_scaled_endpoint (p : ℕ) (hp : 0 < p) (k : ℕ) :
    cnScalarM p 1 (cnUnitScale p) (k + 1) = 0 ∧
      cnScalarX p 1 (cnUnitScale p) (k + 1) = 0 := by
  have h := cn_general_upper_endpoint p (cnUnitScale p) hp (cnUnitScale_pos p) k
  rwa [cnUnitScale_power p hp] at h

/-- Positive unit-domain orders exist, arXiv:2602.02016v2, §3.4. -/
example : 0 < (4 : ℕ) := by norm_num

/-- The unit-scaled CN inverse-root error is exactly one at every positive
iteration count, rather than tending to zero.
Source: arXiv:2602.02016v2, §3.4, convergence after unit spectral normalization. -/
theorem cn_unit_scaled_inverse_error (p : ℕ) (hp : 0 < p) (k : ℕ) :
    |cnScalarX p 1 (cnUnitScale p) (k + 1) - (1 : ℝ) ^ (-(1 / (p : ℝ)))| = 1 := by
  rw [(cn_unit_scaled_endpoint p hp k).2]
  simp

/-- The inverse-error counterexample assumptions are satisfiable,
arXiv:2602.02016v2, §3.4. -/
example : 0 < (2 : ℕ) := by norm_num

/-- The one-dimensional identity has unit Frobenius norm, so Frobenius
normalization also leaves it at the failed endpoint in the source's unit
comparison regime. Source: arXiv:2602.02016v2, §3.4, Frobenius scaling. -/
theorem unit_scalar_frobeniusNorm :
    frobeniusNorm (1 : Matrix (Fin 1) (Fin 1) ℝ) = 1 := by
  norm_num [frobeniusNorm, Muon.squaredFrobenius]

/-- The actual matrix CN implementation collapses on the positive-definite
one-dimensional identity with the source's comparison scale.
Source: arXiv:2602.02016v2, §3.2–3.4, CN recurrence and unit scaling. -/
theorem cn_unit_matrix_endpoint (p : ℕ) (hp : 0 < p) (k : ℕ) :
    cnIterate p (1 : Matrix (Fin 1) (Fin 1) ℝ) (cnUnitScale p) (k + 1) = (0, 0) := by
  have hQ : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
    simp [Orthogonal, Muon.OrthonormalColumns]
  rw [← spectralMatrix_one _ hQ, cnIterate_spectrum p _ _ _ hQ]
  simp_rw [(cn_unit_scaled_endpoint p hp k).1, (cn_unit_scaled_endpoint p hp k).2]
  simp only [spectralMatrix_zero]

/-- The matrix endpoint counterexample has a positive order,
arXiv:2602.02016v2, §3.4. -/
example : 0 < (2 : ℕ) := by norm_num

/-- At the same unit scalar, NDB is exact at every iteration count.
The difference from unit-regime CN is its scaling, as the equal-scaling
equivalence theorem also shows. Source: arXiv:2602.02016v2, §3.3–3.4. -/
theorem ndb_unit_scalar_exact (k : ℕ) : ndbScalar 1 k = (1, 1) := by
  induction k with
  | zero => rfl
  | succ k ih => norm_num [ndbScalar, ih]

end Transformer.DASH
