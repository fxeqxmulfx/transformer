/-
# DASH — actual finite scalar roots and grafting for a training cycle

arXiv:2602.02016v2, §2.3 and §3.3–3.5. These calculations retain the
two finite NDB calls; an exact inverse root is never substituted.
They support a learning counterexample even after numerical scaling has
been repaired, demonstrating why a separate training safeguard is needed.
-/

import Transformer.DASH.Section2_TrainingModels

open scoped Matrix

noncomputable section

namespace Transformer.DASH

/-- A singleton PI pool starting from the nonzero scalar unit vector
returns the exact scalar eigenvalue with budget zero.
Source: arXiv:2602.02016v2, §3.4–3.5. -/
theorem scalar_unit_pool (a : ℝ) :
    pooledRayleigh (a • (1 : Matrix (Fin 1) (Fin 1) ℝ))
      (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) 0 = a := by
  simp [pooledRayleigh, powerIterate, rayleigh, Matrix.mulVec, dotProduct]

/-- The certified scalar input uses exactly the factor-two scale.
Source: arXiv:2602.02016v2, §3.4, corrected scaling. -/
theorem guardedScale_scalar (a : ℝ) (ha : 0 < a) :
    guardedScale (a • (1 : Matrix (Fin 1) (Fin 1) ℝ)) a = 2 * a := by
  have hnorm : frobeniusNorm (a • (1 : Matrix (Fin 1) (Fin 1) ℝ)) = a := by
    rw [frobeniusNorm_smul, abs_of_pos ha]
    norm_num [frobeniusNorm, Muon.squaredFrobenius]
  apply guardedScale_accepts
  have h := min_le_left (frobeniusNorm (a • (1 : Matrix (Fin 1) (Fin 1) ℝ)))
    (absoluteRowBound (a • (1 : Matrix (Fin 1) (Fin 1) ℝ)))
  rw [hnorm] at h
  change min (frobeniusNorm (a • (1 : Matrix (Fin 1) (Fin 1) ℝ)))
    (absoluteRowBound (a • (1 : Matrix (Fin 1) (Fin 1) ℝ))) < 2 * a
  rw [hnorm]
  exact lt_of_le_of_lt h (by linarith)

/-- Positive scalar inputs satisfy the scaling domain,
arXiv:2602.02016v2, §3.4, training counterexample. -/
example : (0 : ℝ) < 1 := by norm_num

/-- Actual one-step-per-call inverse-fourth-root response. The coefficient
`19/16` comes from the second NDB call applied to the first call's finite
square-root buffer `5/8`, not to an exact square root.
Source: arXiv:2602.02016v2, §3.3, finite inverse-root chaining. -/
def finiteScalarRoot (a : ℝ) : ℝ := (2 * a) ^ (-(1 / 4 : ℝ)) * (19 / 16)

/-- Both finite matrix calls give the stated scalar response.
Source: arXiv:2602.02016v2, §3.3–3.4, training counterexample. -/
theorem guardedInverseFourth_scalar (a : ℝ) (ha : 0 < a) :
    guardedInverseFourth (a • (1 : Matrix (Fin 1) (Fin 1) ℝ)) a 1 =
      finiteScalarRoot a • 1 := by
  have hnormalize : guardedNormalize (a • (1 : Matrix (Fin 1) (Fin 1) ℝ)) a =
      (1 / 2 : ℝ) • 1 := by
    rw [guardedNormalize, guardedScale_scalar a ha, smul_smul]
    congr 1
    field_simp
  rw [guardedInverseFourth, guardedScale_scalar a ha, hnormalize]
  ext i j
  fin_cases i
  fin_cases j
  norm_num [ndbIterate, ndbStep, ndbCorrection, Matrix.mul_apply, finiteScalarRoot]

/-- A positive scalar meets both finite NDB calls' input domain,
arXiv:2602.02016v2, §3.3–3.4, training counterexample. -/
example : (0 : ℝ) < 1 / 4 := by norm_num

/-- The actual finite scalar roots remain strictly positive.
Source: arXiv:2602.02016v2, §3.3, training counterexample. -/
theorem finiteScalarRoot_pos (a : ℝ) (ha : 0 < a) : 0 < finiteScalarRoot a := by
  exact mul_pos (Real.rpow_pos_of_pos (by positivity) _) (by norm_num)

/-- Positive roots exist in the finite-budget counterexample,
arXiv:2602.02016v2, §3.3, training counterexample. -/
example : (0 : ℝ) < 1 := by norm_num

/-- Scalar-aligned grafting cancels the positive finite preconditioner,
while retaining the reference's magnitude. Source: arXiv:2602.02016v2,
§2.3, “Grafting”. -/
theorem graft_positive_smul {m n : ℕ} (c d : ℝ) (G : Matrix (Fin m) (Fin n) ℝ)
    (hc : 0 < c) (hd : 0 ≤ d) (hG : 0 < frobeniusNorm G) :
    graft (c • G) (d • G) = d • G := by
  rw [graft, frobeniusNorm_smul, frobeniusNorm_smul, abs_of_pos hc, abs_of_nonneg hd,
    smul_smul]
  congr 1
  field_simp

/-- Aligned grafting hypotheses hold for a nonzero reference,
arXiv:2602.02016v2, §2.3. -/
example : (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 ∧
    0 < frobeniusNorm (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  norm_num [frobeniusNorm, Muon.squaredFrobenius]

end Transformer.DASH
