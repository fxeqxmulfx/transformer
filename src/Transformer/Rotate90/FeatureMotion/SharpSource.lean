module

public import Transformer.Rotate90.FeatureMotion.SharpProbability
public import Transformer.Rotate90.FeatureMotion.Source

/-!
# Gram bounds for the source matrix maps

Source: `review/flat_spectrum.md`, *Pairwise encoder amplification*, PA2.
Apply the established FT2 bounds to the same real matrix maps as EM4.
The final theorem exposes the native Gram-energy upper-bound obligation.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.FeatureMotion

open Grokking.GradientEvidence
variable {I J : Type*} [Fintype I] [Fintype J]

/-- PA2: a smaller column Gram controls the source's actual linear map. -/
theorem matrixMap_column_gram (A : Matrix I J ℝ) :
    ‖matrixMap A‖ ≤ Real.sqrt (Real.sqrt (forecastMatrixEnergy
      (A.transpose.conjTranspose * A.transpose))) := by
  classical
  exact forecast_column_gram_op_le A.transpose

/-- PA2: the row Gram permits the opposite rectangular orientation. -/
theorem matrixMap_row_gram (A : Matrix I J ℝ) :
    ‖matrixMap A‖ ≤ Real.sqrt (Real.sqrt (forecastMatrixEnergy
      (A.transpose * A.transpose.conjTranspose))) := by
  classical
  exact forecast_row_gram_op_le A.transpose

/-- PA2: a verified numerical Gram-energy cap gives an operator cap.
It does not assume that a numerical SVD returns the exact leading value. -/
theorem matrixMap_gram_upper (A : Matrix I J ℝ) (U : ℝ)
    (hU : forecastMatrixEnergy (A.transpose.conjTranspose * A.transpose) ≤ U) :
    ‖matrixMap A‖ ≤ Real.sqrt (Real.sqrt U) := by
  classical
  exact forecast_gram_upper A.transpose U hU

example := matrixMap_gram_upper (fun _ _ : Fin 1 => (1 : ℝ)) 1
  (by norm_num [forecastMatrixEnergy, Matrix.mul_apply, Matrix.transpose])

end Transformer.Rotate90.FeatureMotion
