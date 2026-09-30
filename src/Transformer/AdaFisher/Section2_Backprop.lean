/-
# AdaFisher: affine layers and backpropagation identities

arXiv:2405.16397v3, §2, the forward and backward layer equations.
-/

import Transformer.AdaFisher.Section3_ConvexFalse
import Transformer.AdaFisher.SectionA_Descent
import Transformer.AdaFisher.Section2_Fisher
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Data.Fin.Tuple.Basic

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.AdaFisher

variable {a b : ℕ}

/-- The affine layer before its elementwise nonlinearity, §2. -/
def affineLayer (W : Matrix (Fin b) (Fin a) ℝ) (bias : Fin b → ℝ)
    (h : Fin a → ℝ) : Fin b → ℝ := W.mulVec h + bias

/-- Backpropagated input sensitivity, §2, `∇h L = θᵀs`. -/
def backpropInput (W : Matrix (Fin b) (Fin a) ℝ) (s : Fin b → ℝ) : Fin a → ℝ :=
  W.transpose.mulVec s

/-- Appending a constant one to the activation and the bias as a final
weight column gives the paper's augmented affine layer, §2. -/
theorem affineLayer_augmented (W : Matrix (Fin b) (Fin a) ℝ)
    (bias : Fin b → ℝ) (h : Fin a → ℝ) :
    affineLayer W bias h = fun i =>
      ∑ j : Fin (a + 1), Fin.snoc (W i) (bias i) j * Fin.snoc h 1 j := by
  funext i
  simp [affineLayer, Matrix.mulVec, dotProduct, Fin.sum_univ_castSucc]

/-- The transpose propagation is the adjoint of the layer's directional
response, §2, the last backpropagation equation. -/
theorem backpropInput_pairing (W : Matrix (Fin b) (Fin a) ℝ)
    (s : Fin b → ℝ) (u : Fin a → ℝ) :
    pairing s (W.mulVec u) = pairing (backpropInput W s) u := by
  simp only [pairing, backpropInput, Matrix.mulVec, dotProduct,
    Matrix.transpose_apply, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- The layer weight gradient is `s hᵀ`, §2: its pairing with an arbitrary
weight perturbation gives exactly the loss's first-order layer response. -/
theorem layer_weight_gradient_pairing (h : Fin a → ℝ) (s : Fin b → ℝ)
    (ΔW : Matrix (Fin b) (Fin a) ℝ) :
    pairing s (ΔW.mulVec h) = ∑ i, ∑ j, outer s h i j * ΔW i j := by
  simp only [pairing, Matrix.mulVec, dotProduct, Finset.mul_sum, outer]
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro j hj
  ring

/-- Actual one-coordinate chain rule behind the layer gradient formula,
§2. `h` is the input coordinate, `b` collects all other affine terms,
and `s` is the derivative with respect to the preactivation. -/
theorem affine_coordinate_chain_rule (loss : ℝ → ℝ) (θ h b s : ℝ)
    (hloss : HasDerivAt loss s (θ * h + b)) :
    HasDerivAt (fun u => loss (u * h + b)) (s * h) θ := by
  have ha : HasDerivAt (fun u : ℝ => u * h + b) h θ := by
    simpa using ((hasDerivAt_id θ).mul_const h).add_const b
  exact hloss.comp θ ha

example : HasDerivAt quadratic 1 (1 * 1 + 0 : ℝ) := by
  simpa using quadratic_hasDerivAt 1

/-- The elementwise nonlinearity backpropagation formula `s=r*φ'(a)`,
§2, is the chain rule for the actual loss and activation derivatives. -/
theorem nonlinearity_coordinate_chain_rule (loss φ : ℝ → ℝ) (a r d : ℝ)
    (hloss : HasDerivAt loss r (φ a)) (hφ : HasDerivAt φ d a) :
    HasDerivAt (loss ∘ φ) (r * d) a :=
  hloss.comp a hφ

example : HasDerivAt id (1 : ℝ) (id (2 : ℝ)) ∧ HasDerivAt id (1 : ℝ) 2 :=
  ⟨hasDerivAt_id 2, hasDerivAt_id 2⟩

/-- The full one-coordinate nonlinear layer weight derivative, §2.
The sensitivity is the loss derivative times the activation derivative,
then the weight derivative is sensitivity times the input activation. -/
theorem nonlinear_weight_chain_rule (loss φ : ℝ → ℝ) (θ h b r d : ℝ)
    (hloss : HasDerivAt loss r (φ (θ * h + b)))
    (hφ : HasDerivAt φ d (θ * h + b)) :
    HasDerivAt (fun u => loss (φ (u * h + b))) ((r * d) * h) θ := by
  exact affine_coordinate_chain_rule (loss ∘ φ) θ h b (r * d)
    (nonlinearity_coordinate_chain_rule loss φ (θ * h + b) r d hloss hφ)

example : HasDerivAt id (1 : ℝ) (id (1 * 1 + 0 : ℝ)) ∧
    HasDerivAt id (1 : ℝ) (1 * 1 + 0 : ℝ) := by
  constructor <;> simpa using hasDerivAt_id (1 : ℝ)

end Transformer.AdaFisher
