module

public import Transformer.Grokking.GradientEvidence.UnusedClassDamping
public import Transformer.Grokking.GradientEvidence.ReadoutLeakage
public import Transformer.Grokking.Geometry.ProbabilityLoss

/-!
# Quantitative CE damping for unused pairs

Source: `review/flat_spectrum.md`, *Loss-specific momentum control*, CL1.
The gradient is the actual partial derivative. Probability bounds and the
feature energy give a finite-step coefficient, without assuming fixed
features across training. The tied encoder path is not this derivative.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset Geometry FlatSpectrum.FeatureCodes Landscape.Neural
variable {I C K : Type*} [Fintype I] [Fintype C] [Fintype K]

/-- CL1: the exponential tangent bounds a signed difference at either endpoint. -/
theorem exp_difference_le_tangent (a b : ℝ) :
    Real.exp a - Real.exp b ≤ (a-b)*Real.exp a := by
  have h := mul_le_mul_of_nonneg_right (Real.add_one_le_exp (b-a)) (Real.exp_pos a).le
  rw [Real.exp_sub, div_mul_cancel₀ _ (ne_of_gt (Real.exp_pos a))] at h
  nlinarith

omit [Fintype I] [Fintype K] in
/-- CL1: a signed softmax difference is bounded by its upper endpoint slope. -/
theorem probability_difference_le_tangent (z : C → ℝ) (c d : C) :
    probability z c - probability z d ≤ (z c-z d)*probability z c := by
  have h := div_le_div_of_nonneg_right (exp_difference_le_tangent (z c) (z d))
    (sum_exp_pos z c).le
  simpa only [probability, sub_div, mul_div_assoc] using h

omit [Fintype I] [Fintype K] in
/-- CL1: probability differences are cocoercive in logit differences. The
common probability ceiling q can be one or a tighter current-state bound. -/
theorem probability_pair_cocoercive (z : C → ℝ) (c d : C) (q : ℝ)
    (hc : probability z c ≤ q) (hd : probability z d ≤ q) :
    (probability z c-probability z d)^2 ≤
      q*(z c-z d)*(probability z c-probability z d) := by
  have hsign := logit_probability_difference_nonneg z c d
  rcases le_total (probability z d) (probability z c) with h | h
  · have ht := mul_le_mul_of_nonneg_right (probability_difference_le_tangent z c d)
      (sub_nonneg.mpr h)
    have hq := mul_le_mul_of_nonneg_right hc hsign
    nlinarith
  · have ht := mul_le_mul_of_nonneg_right (probability_difference_le_tangent z d c)
      (sub_nonneg.mpr h)
    have hq := mul_le_mul_of_nonneg_right hd hsign
    nlinarith

variable [DecidableEq C] [DecidableEq K]

/-- CL1: labels cancel from the pair gradient for two unused classes. -/
theorem unused_pair_gradient_formula (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (c d : C)
    (hc : ∀ i, c ≠ y i) (hd : ∀ i, d ≠ y i) (k : K) :
    readoutGradient T y E c k-readoutGradient T y E d k =
      (∑ i, (probability (fun v => linearScore (E v) (T i)) c-
        probability (fun v => linearScore (E v) (T i)) d)*T i k) / Fintype.card I := by
  simp_rw [readoutGradient_eq, ite_eq_right (hc _), ite_eq_right (hd _)]
  simp only [sub_zero, ← sub_div, ← sum_sub_distrib, ← sub_mul]

/-- CL1: actual unused-pair gradient energy is bounded by its dissipative
radial product. The coefficient is probability ceiling times mean feature energy. -/
theorem unused_pair_cocoercive (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (c d : C) (q : ℝ)
    (hc : ∀ i, c ≠ y i) (hd : ∀ i, d ≠ y i)
    (hqc : ∀ i, probability (fun v => linearScore (E v) (T i)) c ≤ q)
    (hqd : ∀ i, probability (fun v => linearScore (E v) (T i)) d ≤ q) :
    (∑ k, (readoutGradient T y E c k-readoutGradient T y E d k)^2) ≤
      (q*(matrixEnergy T/Fintype.card I)) *
        ∑ k, (E c k-E d k)*(readoutGradient T y E c k-readoutGradient T y E d k) := by
  let A := fun i => probability (fun v => linearScore (E v) (T i)) c-
    probability (fun v => linearScore (E v) (T i)) d
  let D := ∑ k, (E c k-E d k) *
    (readoutGradient T y E c k-readoutGradient T y E d k)
  have hp : (∑ i, A i^2)/Fintype.card I ≤ q*D := by
    have hsum := sum_le_sum (fun i (_ : i ∈ (univ : Finset I)) =>
      probability_pair_cocoercive (fun v => linearScore (E v) (T i)) c d q (hqc i) (hqd i))
    have hmean := div_le_div_of_nonneg_right hsum
      (Nat.cast_nonneg (Fintype.card I) : (0 : ℝ) ≤ Fintype.card I)
    dsimp only [D]
    rw [unused_pair_gradient_identity T y E c d hc hd]
    simpa only [A, ← mul_sum, mul_div_assoc, mul_assoc] using hmean
  have hm := mean_product_energy_le (fun i (_ : Fin 1) => A i) T
  have ht : 0 ≤ matrixEnergy T / Fintype.card I :=
    div_nonneg (matrixEnergy_nonneg T) (Nat.cast_nonneg _)
  calc
    _ = ∑ k, ((∑ i, A i*T i k)/Fintype.card I)^2 := by
      simp only [unused_pair_gradient_formula T y E c d hc hd, A]
    _ ≤ ((∑ i, A i^2)/Fintype.card I)*(matrixEnergy T/Fintype.card I) := by
      simpa only [matrixEnergy, Fin.sum_univ_one] using hm
    _ ≤ (q*D)*(matrixEnergy T/Fintype.card I) := mul_le_mul_of_nonneg_right hp ht
    _ = _ := by dsimp only [D]; ring

variable {U : Type*} [Fintype U] [Nonempty U]

omit [Fintype I] [Fintype C] [DecidableEq C] [DecidableEq K] in
/-- CL1: the matrix version of CG3's centered pair-product identity. -/
theorem centered_block_pairing (A B : U → K → ℝ) :
    (∑ u, ∑ k, (A u k-classMean (fun v => A v k)) *
      (B u k-classMean (fun v => B v k))) =
      (∑ u, ∑ v, ∑ k, (A u k-A v k)*(B u k-B v k))/(2*Fintype.card U) := by
  rw [sum_comm]
  simp_rw [centered_pair_product]
  rw [← sum_div]
  congr 1
  rw [sum_comm]
  apply sum_congr rfl
  intro u hu
  rw [sum_comm]

/-- CL1: quantitative control of the entire centered unused-class gradient.
Every class and minibatch row enters; changing features remain permitted. -/
theorem unused_group_cocoercive (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (classes : U → C) (q : ℝ)
    (hu : ∀ u i, classes u ≠ y i)
    (hq : ∀ u i, probability (fun v => linearScore (E v) (T i)) (classes u) ≤ q) :
    (∑ u, ∑ k, (readoutGradient T y E (classes u) k-
      classMean (fun v => readoutGradient T y E (classes v) k))^2) ≤
      (q*(matrixEnergy T/Fintype.card I)) *
        ∑ u, ∑ k, (E (classes u) k-classMean (fun v => E (classes v) k)) *
          (readoutGradient T y E (classes u) k-
            classMean (fun v => readoutGradient T y E (classes v) k)) := by
  have he := centered_block_pairing (fun u => readoutGradient T y E (classes u))
    (fun u => readoutGradient T y E (classes u))
  simp only [← pow_two] at he
  rw [he, centered_block_pairing, ← mul_div_assoc]
  apply div_le_div_of_nonneg_right _ (by positivity)
  rw [mul_sum]
  apply sum_le_sum
  intro u hmem
  rw [mul_sum]
  exact sum_le_sum fun v hv => unused_pair_cocoercive T y E (classes u) (classes v) q
    (hu u) (hu v) (hq u) (hq v)

example := unused_group_cocoercive (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val) (fun u : Fin 2 => u.succ) 1
  (by intro u i; exact Fin.succ_ne_zero u) (by intro u i; exact probability_le_one _ _)

example := probability_pair_cocoercive (fun c : Bool => if c then 2 else -1)
  true false 1 (probability_le_one _ _) (probability_le_one _ _)
example := unused_pair_gradient_formula (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val) 1 2
  (by intro i; decide) (by intro i; decide) 0
example := unused_pair_cocoercive (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val) 1 2 1
  (by intro i; decide) (by intro i; decide)
  (by intro i; exact probability_le_one _ _) (by intro i; exact probability_le_one _ _)

end Transformer.Grokking.GradientEvidence
