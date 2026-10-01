/-
# Diagonal Gaussian affine states and their genuine derivatives

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The explicit diagonal continuous linear map identifies coordinate
derivatives in vector Gaussian integration by parts.
-/

import Transformer.BatchSize.Section4_VectorGaussianIntegration
import Mathlib.Analysis.Calculus.Deriv.Prod

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- The actual diagonal linear noise transformation, Section 4.3
(2)--(3), from independent real innovations to Euclidean states. -/
def gaussianDiagonalMap {d : ℕ} (A : Fin d → ℝ) : (Fin d → ℝ) →L[ℝ] EucSpace d :=
  (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi fun k => A k • (ContinuousLinearMap.proj k))

/-- Every coordinate is multiplied by its actual noise amplitude,
Section 4.3 (2)--(3). -/
theorem gaussianDiagonalMap_apply {d : ℕ} (A : Fin d → ℝ) (z : Fin d → ℝ) (k : Fin d) :
    gaussianDiagonalMap A z k = A k * z k := rfl

/-- A single-coordinate innovation is mapped to its scaled Euclidean
basis vector, Section 4.3 (2)--(3). -/
theorem gaussianDiagonalMap_single {d : ℕ} (A : Fin d → ℝ) (k : Fin d) (u : ℝ) :
    gaussianDiagonalMap A (Pi.single k u) = (A k * u) • EuclideanSpace.single k 1 := by
  ext j
  by_cases hj : j = k
  · subst j
    simp [gaussianDiagonalMap_apply, EuclideanSpace.single]
  · simp [gaussianDiagonalMap_apply, EuclideanSpace.single, hj]

/-- The actual affine Euclidean state with diagonal Gaussian noise,
Section 4.3 (2)--(3), with its base state and amplitudes explicit. -/
def gaussianAffineState {d : ℕ} (x : EucSpace d) (A : Fin d → ℝ) (z : Fin d → ℝ) : EucSpace d :=
  x + gaussianDiagonalMap A z

/-- The affine Gaussian state is continuous in the actual innovation,
Section 4.3 (2)--(3). -/
theorem gaussianAffineState_continuous {d : ℕ} (x : EucSpace d) (A : Fin d → ℝ) :
    Continuous (gaussianAffineState x A) := continuous_const.add (gaussianDiagonalMap A).continuous

/-- The genuine Frechet derivative of the affine Gaussian state is
its diagonal amplitude map, Section 4.3 (2)--(3). -/
theorem gaussianAffineState_hasFDerivAt {d : ℕ} (x : EucSpace d) (A : Fin d → ℝ) (z : Fin d → ℝ) :
    HasFDerivAt (gaussianAffineState x A) (gaussianDiagonalMap A) z :=
  (gaussianDiagonalMap A).hasFDerivAt.const_add x

/-- The genuine derivative when one innovation coordinate varies,
Section 4.3 (2)--(3), is precisely its scaled Euclidean basis vector. -/
theorem gaussianAffineState_slice_hasDerivAt {n : ℕ} (x : EucSpace (n + 1))
    (A : Fin (n + 1) → ℝ) (k : Fin (n + 1)) (z : Fin n → ℝ) (u : ℝ) :
    HasDerivAt (fun v => gaussianAffineState x A (k.insertNth v z))
      (A k • EuclideanSpace.single k 1) u := by
  have hz : HasDerivAt (fun v : ℝ => Fin.insertNth (α := fun _ : Fin (n + 1) => ℝ) k v z)
      (Pi.single k (1 : ℝ)) u := by
    apply hasDerivAt_pi.mpr
    apply (Fin.forall_iff_succAbove (P := fun i => HasDerivAt
      (fun w : ℝ => Fin.insertNth (α := fun _ : Fin (n + 1) => ℝ) k w z i)
        ((Pi.single k (1 : ℝ) : Fin (n + 1) → ℝ) i) u) k).mpr
    constructor
    · simp only [Fin.insertNth_apply_same, Pi.single_eq_same]
      exact hasDerivAt_id u
    · intro j
      simpa only [Fin.insertNth_apply_succAbove, Pi.single_eq_of_ne (Fin.succAbove_ne k j)] using
        hasDerivAt_const u (z j)
  have h := ((gaussianDiagonalMap A).hasFDerivAt.comp_hasDerivAt u hz).const_add x
  simpa only [gaussianAffineState, Function.comp_def, gaussianDiagonalMap_single, mul_one] using h

end Transformer.BatchSize
