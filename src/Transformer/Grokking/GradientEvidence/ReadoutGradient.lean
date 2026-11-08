module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.SignedTransport
public import Transformer.Grokking.GradientEvidence.BinaryObjectives

/-!
# Actual cross-entropy gradient of a frozen-feature readout

Source: `review/flat_spectrum.md`, *Residual learning dynamics*, RD1.
The line-derivative calculation follows `Imbalance.hasDerivAt_crossEntropy_line`
(arXiv:2402.19449v2, §3.2) for the project's own finite-class CE definition.
Features are frozen here. This partial gradient omits the shared embedding's
encoder path and is not asserted to be the full tied-source gradient.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset Geometry FlatSpectrum.FeatureCodes Landscape.Neural

variable {I C K : Type*}

/-- Actual CE differentiated along arbitrary finite-logit directions, RD1. -/
theorem readout_ce_line_hasDerivAt [Fintype C] (z v : C → ℝ) (t : C) :
    HasDerivAt (fun s => crossEntropy (fun c => z c + s * v c) t)
      ((∑ c, probability z c * v c) - v t) 0 := by
  have hd : HasDerivAt (fun s => ∑ c, Real.exp (z c + s * v c))
      (∑ c, Real.exp (z c) * v c) 0 := by
    apply HasDerivAt.fun_sum
    intro c hc
    simpa using (((hasDerivAt_id (0 : ℝ)).mul_const (v c)).const_add (z c)).exp
  have hh := (hd.log (by simpa only [zero_mul, add_zero] using
    (ne_of_gt (sum_exp_pos z t)))).fun_sub
    (((hasDerivAt_id (0 : ℝ)).mul_const (v t)).const_add (z t))
  simp only [zero_mul, add_zero, one_mul] at hh
  convert hh using 1
  · rfl
  · congr 1
    rw [sum_div]
    apply sum_congr rfl
    intro c hc
    unfold probability
    ring

variable [Fintype I] [Fintype C] [Fintype K] [DecidableEq C] [DecidableEq K]

/-- Mean training CE of the final readout at fixed features, RD1. -/
def readoutLoss (T : I → K → ℝ) (y : I → C) (E : C → K → ℝ) : ℝ :=
  meanCrossEntropy univ (fun i c => linearScore (E c) (T i)) y

/-- Add a scalar to one actual readout parameter, RD1. -/
def readoutCoordinate (E : C → K → ℝ) (c : C) (k : K) (s : ℝ) (v : C) (l : K) : ℝ :=
  E v l + if v = c ∧ l = k then s else 0

/-- Actual coordinate derivative of the mean CE, not an assumed gradient, RD1. -/
def readoutGradient (T : I → K → ℝ) (y : I → C) (E : C → K → ℝ) (c : C) (k : K) : ℝ :=
  deriv (fun s => readoutLoss T y (readoutCoordinate E c k s)) 0

omit [Fintype I] [Fintype C] in
/-- Coordinate perturbation induces exactly the expected logit line, RD1. -/
theorem readoutCoordinate_score (E : C → K → ℝ) (x : K → ℝ)
    (c v : C) (k : K) (s : ℝ) :
    linearScore (readoutCoordinate E c k s v) x =
      linearScore (E v) x + s * (if v = c then x k else 0) := by
  by_cases h : v = c
  · simp [linearScore, readoutCoordinate, h, add_mul, sum_add_distrib, ite_mul]
  · simp [linearScore, readoutCoordinate, h]

/-- Differentiation includes every class and every training row, RD1. -/
theorem readoutLoss_coordinate_hasDerivAt (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (c : C) (k : K) :
    HasDerivAt (fun s => readoutLoss T y (readoutCoordinate E c k s))
      ((∑ i, (probability (fun v => linearScore (E v) (T i)) c -
        if c = y i then 1 else 0) * T i k) / Fintype.card I) 0 := by
  have hi (i : I) := readout_ce_line_hasDerivAt
    (fun v => linearScore (E v) (T i)) (fun v => if v = c then T i k else 0) (y i)
  have hs := (HasDerivAt.fun_sum (u := univ) (fun i _ => hi i)).div_const (Fintype.card I : ℝ)
  convert hs using 1
  · funext s
    unfold readoutLoss meanCrossEntropy
    simp only [card_univ, readoutCoordinate_score]
  · congr 1
    apply sum_congr rfl
    intro i hi
    simp only [mul_ite, mul_zero, sum_ite_eq', mem_univ, ite_true]
    by_cases h : c = y i <;> simp [h, Ne.symm, sub_mul]

/-- **The actual mean CE derivative is the probability-error feature mean**,
RD1. This is the readout part of backpropagation with states held fixed. -/
theorem readoutGradient_eq (T : I → K → ℝ) (y : I → C) (E : C → K → ℝ) (c : C) (k : K) :
    readoutGradient T y E c k =
      (∑ i, (probability (fun v => linearScore (E v) (T i)) c -
        if c = y i then 1 else 0) * T i k) / Fintype.card I := by
  exact (readoutLoss_coordinate_hasDerivAt T y E c k).deriv

example := readoutGradient_eq (fun (i : Bool) (_ : Fin 1) => if i then (2 : ℝ) else 1)
  id (fun (c : Bool) (_ : Fin 1) => if c then (1 : ℝ) else -1) true 0

end Transformer.Grokking.GradientEvidence
