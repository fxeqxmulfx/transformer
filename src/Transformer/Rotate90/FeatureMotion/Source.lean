module

public import Transformer.Rotate90.FeatureMotion.Probability
public import Transformer.Grokking.GradientEvidence.ForecastGram

/-!
# Matrix and tied-source correspondence for encoder motion

Source: `review/flat_spectrum.md`, *Full encoder motion*, EM1 and EM4.
The real matrix maps used by the bound are ordinary dense layers. The
existing complex-linear source model embeds into this more general graph
with identical normalized features and logits. FT2 supplies its Frobenius
operator caps. Logical-coordinate identities do not certify CUDA execution.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.FeatureMotion

open Landscape.Neural Grokking.GradientEvidence
open scoped BigOperators
variable {I J V K H : Type*} [Fintype I] [Fintype J] [Fintype K] [Fintype H]

/-- EM1: input-major real arrays act on the Euclidean coordinate vectors. -/
def matrixMap (A : I → J → ℝ) : EuclideanSpace ℝ I →L[ℝ] EuclideanSpace ℝ J := by
  classical
  exact LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin (fun j i => A i j))

/-- EM1: the continuous map is the source's ordinary real dense layer. -/
theorem matrixMap_apply (A : I → J → ℝ) (x : I → ℝ) :
    matrixMap A (WithLp.toLp 2 x) = WithLp.toLp 2 (dense A x) := by
  classical
  ext j
  change (∑ i, A i j * x i) = ∑ i, x i * A i j
  exact Finset.sum_congr rfl (fun i _ => mul_comm _ _)

/-- EM1–FT2: the native Frobenius cap bounds the operator used by EM4. -/
theorem matrixMap_norm_le (A : I → J → ℝ) :
    ‖matrixMap A‖ ≤ Real.sqrt (∑ j, ∑ i, A i j ^ 2) := by
  classical
  exact forecast_matrix_op_le (fun j i => A i j)

/-- EM1: differences of arrays are differences of their represented maps. -/
theorem matrixMap_sub (A B : I → J → ℝ) :
    matrixMap (fun i j => A i j - B i j) = matrixMap A - matrixMap B := by
  ext x j
  change (∑ i, (A i j - B i j) * x i) = (∑ i, A i j * x i) - ∑ i, B i j * x i
  simp only [sub_mul, Finset.sum_sub_distrib]

/-- EM4: embed the existing tied complex-linear source into the real graph. -/
def ofComplexWeights (w : ComplexProduct.Weights V K H) : Weights V K H where
  embedding v := WithLp.toLp 2 (ComplexProduct.realify (w.embedding v))
  gate := matrixMap (ComplexProduct.realMatrix w.gate)
  value := matrixMap (ComplexProduct.realMatrix w.value)
  output := matrixMap (ComplexProduct.realMatrix w.output)

variable [Nonempty K]

/-- EM4: the final features agree with the independently expanded source,
including the four RMSNorms, hidden calibration, residual and parameter ties. -/
theorem feature_eq_source (eps : ℝ) (w : ComplexProduct.Weights V K H) (a b : V) :
    (trace eps (ofComplexWeights w) a b).feature =
      WithLp.toLp 2 (rmsNorm eps (ComplexProduct.realState eps w a b)) := by
  simp only [trace, ofComplexWeights, rms_eq_source, pairedProduct, matrixMap_apply]
  rfl

/-- EM4: the tied readout also agrees with the existing source logits. -/
theorem logit_eq_source (eps : ℝ) (w : ComplexProduct.Weights V K H) (a b c : V) :
    logit eps (ofComplexWeights w) a b c = ComplexProduct.realLogit eps w a b c := by
  rw [logit, feature_eq_source]
  unfold Grokking.FlatSpectrum.FeatureCodes.linearScore ComplexProduct.realLogit tiedReadout
  apply Finset.sum_congr rfl
  intro i hi
  exact mul_comm _ _

example := feature_eq_source 1
  (⟨fun (v : Bool) (_ : Fin 1) => if v then Complex.I else 1,
    fun _ (_ : Fin 1) => 1+Complex.I, fun _ _ => 1-Complex.I, fun _ _ => 1⟩ :
    ComplexProduct.Weights Bool (Fin 1) (Fin 1)) false true
example := logit_eq_source 1
  (⟨fun (v : Bool) (_ : Fin 1) => if v then Complex.I else 1,
    fun _ (_ : Fin 1) => 1+Complex.I, fun _ _ => 1-Complex.I, fun _ _ => 1⟩ :
    ComplexProduct.Weights Bool (Fin 1) (Fin 1)) false true false

end Transformer.Rotate90.FeatureMotion
