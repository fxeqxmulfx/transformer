/-
# Linear softmax classification under class imbalance

Kunstner, Yadav, Milligan, Schmidt, Bietti, arXiv:2402.19449v2,
Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent
on Language Models. Section 3.2, equations preceding Assumption 1.
-/

import Transformer.Perspective.Softmax
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Tactic

open scoped BigOperators

noncomputable section

namespace Transformer.Imbalance

variable {c d n : ℕ}

/-- Parameter matrix, with class rows and feature columns; Section 3.2. -/
abbrev Parameters (c d : ℕ) := Fin c → Fin d → ℝ

/-- Linear logits `Wx`; Section 3.2, definition of the softmax model. -/
def scores (W : Parameters c d) (x : Fin d → ℝ) (k : Fin c) : ℝ :=
  ∑ r, W k r * x r

/-- Predicted class probability; Section 3.2, equation (3). -/
noncomputable def probability (W : Parameters c d) (x : Fin d → ℝ) (k : Fin c) : ℝ :=
  Perspective.softmaxWeight (scores W x) k

/-- Negative log likelihood on logits; Section 3.2, equation (3). -/
noncomputable def crossEntropy (z : Fin c → ℝ) (y : Fin c) : ℝ :=
  -Real.log (Perspective.softmaxWeight z y)

/-- Per-sample loss `-log σ(Wx)_y`; Section 3.2, equation (3). -/
noncomputable def sampleLoss (W : Parameters c d) (x : Fin d → ℝ) (y : Fin c) : ℝ :=
  crossEntropy (scores W x) y

/-- Empirical loss, with all samples weighted by `1/n`; Section 3.2. -/
noncomputable def empiricalLoss (W : Parameters c d)
    (x : Fin n → Fin d → ℝ) (y : Fin n → Fin c) : ℝ :=
  (∑ i, sampleLoss W (x i) (y i)) / n

/-- Perturb one parameter, using class-first matrix indexing; Section 3.2. -/
def coordinateUpdate (W : Parameters c d) (k : Fin c) (r : Fin d) (t : ℝ) :
    Parameters c d := fun j s => W j s + if j = k ∧ s = r then t else 0

/-- Actual partial derivative of the sample loss; Section 3.2, equation (4). -/
noncomputable def sampleGradient (W : Parameters c d) (x : Fin d → ℝ)
    (y k : Fin c) (r : Fin d) : ℝ :=
  deriv (fun t => sampleLoss (coordinateUpdate W k r t) x y) 0

/-- Actual second partial derivative, including off-diagonal class blocks;
Section 3.2, equation (4), and Appendix H.1. -/
noncomputable def sampleHessian (W : Parameters c d) (x : Fin d → ℝ)
    (y k j : Fin c) (r s : Fin d) : ℝ :=
  deriv (fun t => sampleGradient (coordinateUpdate W j s t) x y k r) 0

/-- Samples of a particular class; Section 3.2, the data moment definitions. -/
def classSamples (y : Fin n → Fin c) (k : Fin c) : Finset (Fin n) :=
  Finset.univ.filter fun i => y i = k

/-- Class frequency `n_k/n`; Section 3.2. -/
def frequency (y : Fin n → Fin c) (k : Fin c) : ℝ :=
  (classSamples y k).card / (n : ℝ)

/-- Overall first moment; Section 3.2, before Assumption 1. -/
def meanInput (x : Fin n → Fin d → ℝ) (r : Fin d) : ℝ := (∑ i, x i r) / n

/-- Class-conditional first moment; Section 3.2, before Assumption 1. -/
def classMean (x : Fin n → Fin d → ℝ) (y : Fin n → Fin c)
    (k : Fin c) (r : Fin d) : ℝ :=
  (∑ i ∈ classSamples y k, x i r) / (classSamples y k).card

/-- Overall second moment; Section 3.2, before Assumption 1. -/
def secondMoment (x : Fin n → Fin d → ℝ) (r s : Fin d) : ℝ :=
  (∑ i, x i r * x i s) / n

/-- Class-conditional second moment; Section 3.2, before Assumption 1. -/
def classSecondMoment (x : Fin n → Fin d → ℝ) (y : Fin n → Fin c)
    (k : Fin c) (r s : Fin d) : ℝ :=
  (∑ i ∈ classSamples y k, x i r * x i s) / (classSamples y k).card

/-- Log-sum-exp form of the actual negative log likelihood;
Section 3.2, equation (3). -/
theorem crossEntropy_eq (z : Fin c → ℝ) (y : Fin c) :
    crossEntropy z y = Real.log (∑ j, Real.exp (z j)) - z y := by
  have hc : 0 < c := lt_of_le_of_lt (Nat.zero_le y.val) y.isLt
  rw [crossEntropy, Perspective.softmaxWeight,
    Real.log_div (Real.exp_ne_zero _) (Perspective.softmaxPartition_pos hc z).ne',
    Real.log_exp]
  ring

/-- Zero logits assign probability `1/c` to every class;
Proposition 2, equation (1). -/
theorem softmax_zero (k : Fin c) :
    Perspective.softmaxWeight (fun _ : Fin c => 0) k = 1 / (c : ℝ) := by
  simp [Perspective.softmaxWeight]

/-- Zero parameters give uniform predictions, as used in Proposition 2. -/
theorem probability_zero (x : Fin d → ℝ) (k : Fin c) :
    probability (fun _ _ => 0) x k = 1 / (c : ℝ) := by
  simp [probability, Perspective.softmaxWeight, scores]

/-- A coordinate perturbation changes just one logit; Section 3.2,
the chain rule connecting equation (3) to equation (4). -/
theorem scores_coordinateUpdate (W : Parameters c d) (x : Fin d → ℝ)
    (k : Fin c) (r : Fin d) (t : ℝ) (j : Fin c) :
    scores (coordinateUpdate W k r t) x j =
      scores W x j + t * (if j = k then x r else 0) := by
  simp only [scores, coordinateUpdate, add_mul, Finset.sum_add_distrib]
  congr 1
  by_cases hj : j = k
  · subst j
    simp [Finset.sum_ite_eq', mul_comm]
  · simp [hj]

end Transformer.Imbalance
