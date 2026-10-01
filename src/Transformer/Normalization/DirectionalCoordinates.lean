/-
# Coordinates extending a nonzero real direction

The elementary pivot construction is adapted from Bochao Kong's
LocalComplexGeometry/Analytic/Regularization.lean, revision
029697242294aea7989bf5d391199696a43490df:
https://github.com/BochaoKong/nullstellensatz
Apache-2.0 license: third_party/LocalComplexGeometry/LICENSE.
The scalar field is real, and the target has the Euclidean norm.
-/

import Transformer.Basic
import Transformer.AnalyticPreparation.Basic
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Topology.Algebra.Module.FiniteDimension

noncomputable section

namespace Transformer.Normalization

open AnalyticPreparation

/-- The pivot coordinate `j` extends a nonzero direction to an invertible
linear coordinate system. Auxiliary construction for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
def directionExtensionLinearEquiv {n : ℕ}
    (v : Base (n + 1)) (j : Fin (n + 1)) (hvj : v j ≠ 0) :
    Ambient n ≃ₗ[ℝ] Base (n + 1) where
  toFun x := Fin.insertNth j (x.2 * v j)
    (fun i => x.1 i + x.2 * v (j.succAbove i))
  invFun y :=
    (fun i => y (j.succAbove i) - (y j / v j) * v (j.succAbove i), y j / v j)
  left_inv x := by
    ext i
    · simp [hvj]
    · simp [hvj]
  right_inv y := by
    funext k
    refine j.succAboveCases ?_ (fun i => ?_) k
    · simp [hvj]
    · simp [hvj]
  map_add' x y := by
    funext k
    refine j.succAboveCases ?_ (fun i => ?_) k
    · simp [add_mul]
    · simp [add_mul, add_assoc, add_left_comm, add_comm]
  map_smul' c x := by
    funext k
    refine j.succAboveCases ?_ (fun i => ?_) k
    · simp [mul_assoc]
    · simp [mul_add, mul_assoc]

/-- A continuous real coordinate equivalence whose distinguished axis
will be the prescribed Euclidean vector. Auxiliary construction for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def directionCoordinates {n : ℕ}
    (v : EucSpace (n + 1)) (j : Fin (n + 1)) (hvj : v j ≠ 0) :
    Ambient n ≃L[ℝ] EucSpace (n + 1) :=
  ((directionExtensionLinearEquiv (fun i => v i) j hvj).trans
    (WithLp.linearEquiv 2 ℝ (Base (n + 1))).symm).toContinuousLinearEquiv

/-- The distinguished coordinate axis maps to the specified line exactly,
with no rescaling of its parameter. Auxiliary for the general preparation
step in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem directionCoordinates_axis {n : ℕ}
    (v : EucSpace (n + 1)) (j : Fin (n + 1)) (hvj : v j ≠ 0) (t : ℝ) :
    directionCoordinates v j hvj (0, t) = t • v := by
  apply PiLp.ext
  intro k
  refine j.succAboveCases ?_ (fun i => ?_) k
  · simp [directionCoordinates, directionExtensionLinearEquiv]
  · simp [directionCoordinates, directionExtensionLinearEquiv]

/-- A nonzero two-dimensional vector supplies an actual invertible
coordinate chart and satisfies the axis lemma's hypotheses. Auxiliary
example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example (t : ℝ) : ∃ L : Ambient 1 ≃L[ℝ] EucSpace 2,
    L (0, t) = t • (WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) := by
  let v : EucSpace 2 := WithLp.toLp 2 (fun _ => 1)
  have hv : v 0 ≠ 0 := by simp [v]
  exact ⟨directionCoordinates v 0 hv, directionCoordinates_axis v 0 hv t⟩

end Transformer.Normalization
