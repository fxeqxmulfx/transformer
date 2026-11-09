module

public import Transformer.Grokking.GradientEvidence.CEGram.Basic

/-!
# Actual CE damping with a probability-weighted feature operator

Source: `review/flat_spectrum.md`, *Query-weighted feature covariance*, QG2.
The gradient is still the true partial CE derivative. A different probability
ceiling may be supplied for every example, without changing labels, unused
classes, clipping or the fixed feature metric. No encoder term is discarded
from a claim about the full optimizer: these statements concern its partial
readout reference only.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence.CEGram

open Finset Geometry FlatSpectrum.FeatureCodes
variable {I C K U : Type*} [Fintype I] [Fintype C] [Fintype K]
  [DecidableEq I] [DecidableEq C] [DecidableEq K]

omit [Fintype I] [Fintype K] [DecidableEq I] [DecidableEq C] [DecidableEq K] in
/-- QG2: divide by the probability ceiling before averaging. Its positivity
follows from the actual finite-logit softmax probability, not a new premise. -/
theorem probability_pair_div (z : C → ℝ) (c d : C) (q : ℝ)
    (hc : probability z c ≤ q) (hd : probability z d ≤ q) :
    (probability z c-probability z d)^2 / q ≤
      (z c-z d)*(probability z c-probability z d) := by
  have hq : 0 < q := (probability_pos z c).trans_le hc
  apply (div_le_iff₀ hq).mpr
  have h := probability_pair_cocoercive z c d q hc hd
  nlinarith

/-- QG2: the actual unused-pair gradient uses the full probability-weighted
feature operator, rather than a maximum probability times a feature trace. -/
theorem pair_cocoercive (T : I → K → ℝ) (y : I → C) (E : C → K → ℝ)
    (c d : C) (q : I → ℝ) (r : K → ℝ)
    (hr : ∀ k, 0 ≤ r k) (hc : ∀ i, c ≠ y i) (hd : ∀ i, d ≠ y i)
    (hqc : ∀ i, probability (fun v => linearScore (E v) (T i)) c ≤ q i)
    (hqd : ∀ i, probability (fun v => linearScore (E v) (T i)) d ≤ q i) :
    (∑ k, r k * (readoutGradient T y E c k-readoutGradient T y E d k)^2) ≤
      coefficient q r T *
        ∑ k, (E c k-E d k)*(readoutGradient T y E c k-readoutGradient T y E d k) := by
  let a := fun i => probability (fun v => linearScore (E v) (T i)) c-
    probability (fun v => linearScore (E v) (T i)) d
  let D := ∑ k, (E c k-E d k)*
    (readoutGradient T y E c k-readoutGradient T y E d k)
  have hq (i : I) : 0 < q i :=
    (probability_pos (fun v => linearScore (E v) (T i)) c).trans_le (hqc i)
  have hp : (∑ i, (a i)^2 / q i) / Fintype.card I ≤ D := by
    have h := sum_le_sum (fun i (_ : i ∈ (univ : Finset I)) =>
      probability_pair_div (fun v => linearScore (E v) (T i)) c d (q i) (hqc i) (hqd i))
    have hmean := div_le_div_of_nonneg_right h
      (Nat.cast_nonneg (Fintype.card I) : (0 : ℝ) ≤ Fintype.card I)
    dsimp only [D]
    rw [unused_pair_gradient_identity T y E c d hc hd]
    exact hmean
  calc
    _ = ∑ k, r k * ((∑ i, a i * T i k) / Fintype.card I)^2 := by
      simp only [unused_pair_gradient_formula T y E c d hc hd, a]
    _ ≤ coefficient q r T * ((∑ i, (a i)^2 / q i) / Fintype.card I) :=
      weighted_mean_energy_le q r T a hq hr
    _ ≤ coefficient q r T * D := mul_le_mul_of_nonneg_left hp (coefficient_nonneg q r T)

variable [Fintype U] [Nonempty U]

/-- QG2: the same operator controls the whole centered unused-class block.
The probability ceiling depends on the example but covers every unused class. -/
theorem group_cocoercive (T : I → K → ℝ) (y : I → C) (E : C → K → ℝ)
    (classes : U → C) (q : I → ℝ) (r : K → ℝ)
    (hr : ∀ k, 0 ≤ r k) (hu : ∀ u i, classes u ≠ y i)
    (hq : ∀ u i, probability (fun v => linearScore (E v) (T i)) (classes u) ≤ q i) :
    (∑ j : U × K, r j.2 * (unusedCoordinates (readoutGradient T y E) classes j)^2) ≤
      coefficient q r T *
        ∑ j, unusedCoordinates E classes j * unusedCoordinates (readoutGradient T y E) classes j := by
  simp only [Fintype.sum_prod_type, unusedCoordinates]
  have he := metric_centered_block_pairing r (fun u => readoutGradient T y E (classes u))
    (fun u => readoutGradient T y E (classes u))
  simp only [← pow_two] at he
  rw [he, centered_block_pairing, ← mul_div_assoc]
  apply div_le_div_of_nonneg_right _ (by positivity)
  rw [mul_sum]
  apply sum_le_sum
  intro u hmem
  rw [mul_sum]
  exact sum_le_sum fun v hv => pair_cocoercive T y E (classes u) (classes v) q r
    hr (hu u) (hu v) (hq u) (hq v)

/-- QG2: a certified numerical or transported operator coefficient can be
used directly because the actual unused-gradient radial product is nonnegative. -/
theorem group_cocoercive_of_upper (T : I → K → ℝ) (y : I → C) (E : C → K → ℝ)
    (classes : U → C) (q : I → ℝ) (r : K → ℝ) (L : ℝ)
    (hr : ∀ k, 0 ≤ r k) (hu : ∀ u i, classes u ≠ y i)
    (hq : ∀ u i, probability (fun v => linearScore (E v) (T i)) (classes u) ≤ q i)
    (hL : coefficient q r T ≤ L) :
    (∑ j : U × K, r j.2 * (unusedCoordinates (readoutGradient T y E) classes j)^2) ≤
      L * ∑ j, unusedCoordinates E classes j *
        unusedCoordinates (readoutGradient T y E) classes j := by
  exact (group_cocoercive T y E classes q r hr hu hq).trans
    (mul_le_mul_of_nonneg_right hL (unused_gradient_radial_nonneg T y E classes hu))

example := probability_pair_div (fun c : Bool => if c then 2 else -1) true false 1
  (probability_le_one _ _) (probability_le_one _ _)
example := pair_cocoercive (I := Bool) (C := Fin 3) (K := Bool)
  (fun i k => if i == k then 2 else -1) (fun _ => 0) (fun c _ => c.val) 1 2
  (fun i => if i then 1 else 2) (fun k => if k then 3 else 1)
  (by intro k; cases k <;> norm_num) (by intro i; decide) (by intro i; decide)
  (by intro i; exact (probability_le_one _ _).trans (by cases i <;> norm_num))
  (by intro i; exact (probability_le_one _ _).trans (by cases i <;> norm_num))
example := group_cocoercive (I := Bool) (C := Fin 3) (K := Bool)
  (fun i k => if i == k then 2 else -1) (fun _ => 0) (fun c _ => c.val)
  (fun u : Fin 2 => u.succ) (fun i => if i then 1 else 2) (fun k => if k then 3 else 1)
  (by intro k; cases k <;> norm_num) (by intro u i; exact Fin.succ_ne_zero u)
  (by intro u i; exact (probability_le_one _ _).trans (by cases i <;> norm_num))
example := group_cocoercive_of_upper (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 3) (fun _ => 0) (fun c _ => c.val)
  (fun u : Fin 2 => u.succ) (fun _ => 1) (fun _ => 2) 18
  (by intro k; norm_num) (by intro u i; exact Fin.succ_ne_zero u)
  (by intro u i; exact probability_le_one _ _)
  (by
    have h := coefficient_le_trace (I := Fin 1) (K := Fin 1)
      (fun _ => 1) (fun _ => 2) (fun _ _ => 3) (by intro i; norm_num) (by intro k; norm_num)
    norm_num [traceCoefficient] at h
    exact h)

end Transformer.Grokking.GradientEvidence.CEGram
