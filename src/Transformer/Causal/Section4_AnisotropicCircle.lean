/-
# An actual anisotropic two-token causal field

Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5`, and
`eqn:csa`. The value matrix is `diag(2,1)` and the query-key matrix
is `diag(0,4)`, with temperature one. The value matrix has distinct
positive eigenvalues, with leading unit eigenvector `(1,0)`.

The actual angular field below is derived from the original masked
softmax and tangent projection. Its denominator includes both the
leader and the follower's self-attention. The leader follows its
original single-token equation; the follower retains the anisotropic
self term, which the identity-value angular equation does not have.
No convergence or stability property is included in a definition.
-/

import Transformer.Causal.Basic
import Transformer.Perceptron.Section5_CircleGeodesics
import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.Calculus.ContDiff.RCLike
import Mathlib.Analysis.Calculus.Deriv.Prod

open Real
open scoped BigOperators

namespace Transformer.Causal

open Perceptron

/-- The actual value matrix for the §4 `thm1.5` test: eigenvalues two and one. -/
noncomputable def anisotropicValueMap : ParamMatrix 2 :=
  Matrix.toEuclideanCLM (𝕜 := ℝ) (Matrix.diagonal (![2, 1] : Fin 2 → ℝ))

/-- The actual key matrix; the query is the identity and temperature is one.
Source: arXiv:2411.04990v2, §4, arbitrary `Q,K` in `thm1.5`. -/
noncomputable def anisotropicKeyMap : ParamMatrix 2 :=
  Matrix.toEuclideanCLM (𝕜 := ℝ) (Matrix.diagonal (![0, 4] : Fin 2 → ℝ))

/-- The value map acts by its actual two diagonal entries.
Source: arXiv:2411.04990v2, §4, the value matrix in `eqn:csa`. -/
theorem anisotropicValueMap_apply (x : EucSpace 2) (i : Fin 2) :
    anisotropicValueMap x i = (![2, 1] : Fin 2 → ℝ) i * x i := by
  rw [anisotropicValueMap, Matrix.ofLp_toEuclideanCLM, Matrix.mulVec_diagonal]

/-- The key map acts by its actual two diagonal entries.
Source: arXiv:2411.04990v2, §4, the query-key matrix in `eqn:csa`. -/
theorem anisotropicKeyMap_apply (x : EucSpace 2) (i : Fin 2) :
    anisotropicKeyMap x i = (![0, 4] : Fin 2 → ℝ) i * x i := by
  rw [anisotropicKeyMap, Matrix.ofLp_toEuclideanCLM, Matrix.mulVec_diagonal]

/-- The source's distinct-positive-eigenvalue hypothesis holds for the
actual value matrix, on the genuine Euclidean basis in dimension two.
Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5`. -/
theorem anisotropicValueMap_eigenbasis :
    ∃ (e : Idx 2 → EucSpace 2) (μ : Idx 2 → ℝ), LinearIndependent ℝ e ∧
      (∀ i, anisotropicValueMap (e i) = μ i • e i) ∧ Function.Injective μ ∧
      (∀ i, 0 < μ i) ∧ ∀ i, μ i ≤ 2 := by
  refine ⟨EuclideanSpace.basisFun (Fin 2) ℝ, (![2, 1] : Fin 2 → ℝ), ?_, ?_, ?_, ?_, ?_⟩
  · exact (EuclideanSpace.basisFun (Fin 2) ℝ).toBasis.linearIndependent
  · intro i
    ext j
    rw [anisotropicValueMap_apply]
    fin_cases i <;> fin_cases j <;> simp [EuclideanSpace.basisFun_apply]
  · intro i j h
    fin_cases i <;> fin_cases j <;> simp_all
  · intro i; fin_cases i <;> norm_num
  · intro i; fin_cases i <;> norm_num

/-- The original leading unit vector really has eigenvalue two.
Source: arXiv:2411.04990v2, §4, the `ξ` in `thm1.5`. -/
theorem anisotropicValueMap_leading :
    anisotropicValueMap (basePoint 1 : EucSpace 2) = 2 • (basePoint 1 : EucSpace 2) := by
  ext i
  rw [anisotropicValueMap_apply]
  fin_cases i <;> simp [basePoint]

/-- The actual two-angle velocity, with the original self and predecessor
weights. Source: arXiv:2411.04990v2, §4, `eqn:csa` for the two matrices above. -/
noncomputable def anisotropicAngleField (φ : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![-Real.cos (φ 0) * Real.sin (φ 0),
    (-Real.exp (4 * Real.sin (φ 1) ^ 2) * Real.cos (φ 1) * Real.sin (φ 1) +
      Real.exp (4 * Real.sin (φ 0) * Real.sin (φ 1)) *
        (-2 * Real.cos (φ 0) * Real.sin (φ 1) + Real.sin (φ 0) * Real.cos (φ 1))) /
      (Real.exp (4 * Real.sin (φ 1) ^ 2) + Real.exp (4 * Real.sin (φ 0) * Real.sin (φ 1)))]

/-- The follower's actual normalization is positive at every pair of angles.
Source: arXiv:2411.04990v2, §4, the denominator of `eqn:csa`. -/
theorem anisotropicAngleDen_pos (φ : Fin 2 → ℝ) :
    0 < Real.exp (4 * Real.sin (φ 1) ^ 2) + Real.exp (4 * Real.sin (φ 0) * Real.sin (φ 1)) := by
  positivity

/-- The original angular field is continuously differentiable everywhere.
Source: arXiv:2411.04990v2, §4, smooth finite softmax dynamics `eqn:csa`.
Positivity of the actual denominator is proved, not assumed. -/
theorem contDiff_anisotropicAngleField : ContDiff ℝ 1 anisotropicAngleField := by
  apply contDiff_pi.mpr
  intro i
  fin_cases i
  · change ContDiff ℝ 1 (fun φ : Fin 2 → ℝ => -Real.cos (φ 0) * Real.sin (φ 0))
    fun_prop
  · change ContDiff ℝ 1 (fun φ : Fin 2 → ℝ => _ / _)
    apply ContDiff.div
    · fun_prop
    · fun_prop
    · exact fun φ => (anisotropicAngleDen_pos φ).ne'

/-- The actual circle projection is its scalar angular velocity times
the actual unit tangent. Source: arXiv:2411.04990v2, `eqn:csa`,
the tangent projection in its two-dimensional angular representation. -/
theorem proj_circlePoint_eq_speed (θ : ℝ) (v : EucSpace 2) :
    proj 2 (circlePoint θ : EucSpace 2) v =
      inner (𝕜 := ℝ) (circleVel θ) v • circleVel θ := by
  unfold proj
  exact sub_eq_iff_eq_add.mpr ((circle_frame_decomposition θ v).trans (add_comm _ _))

/-- The original masked softmax vector velocity equals the derived angular
field on the actual circle. Source: arXiv:2411.04990v2, §4, `eqn:csa`.
Both indices and both softmax weights are evaluated from the original matrices. -/
theorem anisotropicAngleField_eq_CSA_velocity (φ : Fin 2 → ℝ) (k : Fin 2) :
    proj 2 (circlePoint (φ k) : EucSpace 2)
      ((∑ j : Fin 2, if (j : ℕ) ≤ (k : ℕ) then
          Real.exp (inner (𝕜 := ℝ) (circlePoint (φ k) : EucSpace 2)
            (anisotropicKeyMap (circlePoint (φ j) : EucSpace 2))) else 0)⁻¹ •
        ∑ j : Fin 2, if (j : ℕ) ≤ (k : ℕ) then
          Real.exp (inner (𝕜 := ℝ) (circlePoint (φ k) : EucSpace 2)
            (anisotropicKeyMap (circlePoint (φ j) : EucSpace 2))) •
              anisotropicValueMap (circlePoint (φ j) : EucSpace 2) else 0) =
      anisotropicAngleField φ k • circleVel (φ k) := by
  rw [proj_circlePoint_eq_speed]
  congr 1
  have hin (x y : EucSpace 2) : inner (𝕜 := ℝ) x y = x 0 * y 0 + x 1 * y 1 := by
    simp [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Fin.sum_univ_two]
    ring
  have hkey (θ ψ : ℝ) : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2)
      (anisotropicKeyMap (circlePoint ψ : EucSpace 2)) = 4 * Real.sin ψ * Real.sin θ := by
    rw [hin, anisotropicKeyMap_apply, anisotropicKeyMap_apply]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one, zero_mul,
      mul_zero, zero_add, circlePoint_coord_one]
    ring
  have hval (θ ψ : ℝ) : inner (𝕜 := ℝ) (circleVel θ)
      (anisotropicValueMap (circlePoint ψ : EucSpace 2)) =
      -2 * Real.sin θ * Real.cos ψ + Real.cos θ * Real.sin ψ := by
    rw [hin, anisotropicValueMap_apply, anisotropicValueMap_apply]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one,
      circlePoint_coord_zero, circlePoint_coord_one, circleVel_coord_zero, circleVel_coord_one, one_mul]
    ring
  fin_cases k
  · norm_num only [Fin.sum_univ_two, Fin.val_zero, Fin.val_one]
    simp only [ite_true, ite_false, add_zero]
    change inner (𝕜 := ℝ) (circleVel (φ 0))
      ((Real.exp (inner (𝕜 := ℝ) (circlePoint (φ 0) : EucSpace 2)
        (anisotropicKeyMap (circlePoint (φ 0) : EucSpace 2))))⁻¹ •
          (Real.exp (inner (𝕜 := ℝ) (circlePoint (φ 0) : EucSpace 2)
            (anisotropicKeyMap (circlePoint (φ 0) : EucSpace 2))) •
              anisotropicValueMap (circlePoint (φ 0) : EucSpace 2))) = _
    rw [smul_smul, inv_mul_cancel₀ (Real.exp_ne_zero _), one_smul, hval]
    change _ = -Real.cos (φ 0) * Real.sin (φ 0)
    ring
  · norm_num only [Fin.sum_univ_two, Fin.val_zero, Fin.val_one]
    simp only [ite_true]
    change inner (𝕜 := ℝ) (circleVel (φ 1)) _ =
      (-Real.exp (4 * Real.sin (φ 1) ^ 2) * Real.cos (φ 1) * Real.sin (φ 1) +
        Real.exp (4 * Real.sin (φ 0) * Real.sin (φ 1)) *
          (-2 * Real.cos (φ 0) * Real.sin (φ 1) + Real.sin (φ 0) * Real.cos (φ 1))) /
        (Real.exp (4 * Real.sin (φ 1) ^ 2) + Real.exp (4 * Real.sin (φ 0) * Real.sin (φ 1)))
    simp only [real_inner_smul_right, inner_add_right, hkey, hval]
    change (Real.exp (4 * Real.sin (φ 0) * Real.sin (φ 1)) +
        Real.exp (4 * Real.sin (φ 1) * Real.sin (φ 1)))⁻¹ *
      (Real.exp (4 * Real.sin (φ 0) * Real.sin (φ 1)) *
        (-2 * Real.sin (φ 1) * Real.cos (φ 0) + Real.cos (φ 1) * Real.sin (φ 0)) +
       Real.exp (4 * Real.sin (φ 1) * Real.sin (φ 1)) *
        (-2 * Real.sin (φ 1) * Real.cos (φ 1) + Real.cos (φ 1) * Real.sin (φ 1))) = _
    rw [show 4 * Real.sin (φ 1) * Real.sin (φ 1) = 4 * Real.sin (φ 1) ^ 2 by ring,
      add_comm (Real.exp (4 * Real.sin (φ 0) * Real.sin (φ 1))) (Real.exp (4 * Real.sin (φ 1) ^ 2))]
    ring

/-- Every genuine global angular solution lifts to the original causal
attention equation. Source: arXiv:2411.04990v2, §4, `eqn:csa`.
The derivative is obtained from the actual circle chart and the proved velocity. -/
theorem CSA_of_anisotropicAngleFlow (φ : ℝ → Fin 2 → ℝ)
    (hφ : ∀ t, HasDerivAt φ (anisotropicAngleField (φ t)) t) :
    CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2)) anisotropicKeyMap anisotropicValueMap
      (fun t k => circlePoint (φ t k)) := by
  intro t k
  have hd := (hasDerivAt_circlePoint (φ t k)).scomp t ((hasDerivAt_pi.mp (hφ t)) k)
  simpa only [Function.comp_def, ContinuousLinearMap.id_apply, one_mul,
    anisotropicAngleField_eq_CSA_velocity] using hd

/-- The angular-solution hypothesis is genuinely satisfied at the leading
configuration, because the original field vanishes there.
Source: arXiv:2411.04990v2, §4, `eqn:csa`. -/
example : ∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : Fin 2 → ℝ))
    (anisotropicAngleField 0) t := by
  intro t
  have hz : anisotropicAngleField 0 = 0 := by
    ext i
    fin_cases i <;> simp [anisotropicAngleField]
  simpa only [hz] using hasDerivAt_const t (0 : Fin 2 → ℝ)

end Transformer.Causal
