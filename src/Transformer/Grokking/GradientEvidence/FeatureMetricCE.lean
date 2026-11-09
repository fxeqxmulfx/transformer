module

public import Transformer.Grokking.GradientEvidence.CEMomentumControl

/-!
# Actual CE damping in a shared feature metric

Source: `review/flat_spectrum.md`, *Checkpoint-fixed feature metrics*, FM1.
Weights may vary by feature but are shared across unused classes. This is
essential: it preserves the computed class projection and the CE radial sign.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset Geometry FlatSpectrum.FeatureCodes
variable {I C K U : Type*} [Fintype I] [Fintype C] [Fintype K] [Fintype U]

/-- FM1: the actual feature table's energy in the declared metric. -/
def metricFeatureEnergy (r : K → ℝ) (T : I → K → ℝ) : ℝ :=
  ∑ i, ∑ k, r k*(T i k)^2

/-- FM1: nonnegative feature weights give a nonnegative computed energy. -/
theorem metric_feature_energy_nonneg (r : K → ℝ) (T : I → K → ℝ)
    (hr : ∀ k, 0 ≤ r k) : 0 ≤ metricFeatureEnergy r T :=
  sum_nonneg fun i _ => sum_nonneg fun k _ => mul_nonneg (hr k) (sq_nonneg (T i k))

/-- FM1: weighted Cauchy–Schwarz for the minibatch gradient mean. -/
theorem metric_mean_product_energy_le (r : K → ℝ) (A : I → ℝ) (T : I → K → ℝ)
    (hr : ∀ k, 0 ≤ r k) :
    (∑ k, r k*((∑ i, A i*T i k)/Fintype.card I)^2) ≤
      ((∑ i, (A i)^2)/Fintype.card I)*(metricFeatureEnergy r T/Fintype.card I) := by
  have hk (k : K) := mean_product_energy_le (fun i (_ : Fin 1) => A i)
    (fun i (_ : Fin 1) => T i k)
  simp only [matrixEnergy, Fin.sum_univ_one] at hk
  calc
    _ ≤ ∑ k, r k*(((∑ i, (A i)^2)/Fintype.card I)*
        ((∑ i, (T i k)^2)/Fintype.card I)) :=
      sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hk k) (hr k)
    _ = _ := by
      unfold metricFeatureEnergy
      simp only [mul_sum, mul_div_assoc, sum_div]
      rw [sum_comm]
      apply sum_congr rfl
      intro i hi
      apply sum_congr rfl
      intro k hk
      ring

variable [DecidableEq C] [DecidableEq K]

/-- FM1: feature-dependent scaling changes the quantitative coefficient,
while the actual CE pair's radial product remains unweighted. -/
theorem metric_unused_pair_cocoercive (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (c d : C) (q : ℝ) (r : K → ℝ)
    (hr : ∀ k, 0 ≤ r k) (hc : ∀ i, c ≠ y i) (hd : ∀ i, d ≠ y i)
    (hqc : ∀ i, probability (fun v => linearScore (E v) (T i)) c ≤ q)
    (hqd : ∀ i, probability (fun v => linearScore (E v) (T i)) d ≤ q) :
    (∑ k, r k*(readoutGradient T y E c k-readoutGradient T y E d k)^2) ≤
      (q*(metricFeatureEnergy r T/Fintype.card I)) *
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
  have ht : 0 ≤ metricFeatureEnergy r T/Fintype.card I :=
    div_nonneg (metric_feature_energy_nonneg r T hr) (Nat.cast_nonneg _)
  calc
    _ = ∑ k, r k*((∑ i, A i*T i k)/Fintype.card I)^2 := by
      simp only [unused_pair_gradient_formula T y E c d hc hd, A]
    _ ≤ ((∑ i, A i^2)/Fintype.card I)*(metricFeatureEnergy r T/Fintype.card I) :=
      metric_mean_product_energy_le r A T hr
    _ ≤ (q*D)*(metricFeatureEnergy r T/Fintype.card I) := mul_le_mul_of_nonneg_right hp ht
    _ = _ := by dsimp only [D]; ring

variable [Nonempty U]

omit [Fintype I] [Fintype C] [DecidableEq C] [DecidableEq K] in
/-- FM1: class centering and a class-shared feature weight commute in the
pair-product identity. There is no analogous claim for arbitrary row weights. -/
theorem metric_centered_block_pairing (r : K → ℝ) (A B : U → K → ℝ) :
    (∑ u, ∑ k, r k*((A u k-classMean (fun v => A v k)) *
      (B u k-classMean (fun v => B v k)))) =
      (∑ u, ∑ v, ∑ k, r k*((A u k-A v k)*(B u k-B v k)))/(2*Fintype.card U) := by
  rw [sum_comm]
  simp_rw [← mul_sum, centered_pair_product, ← mul_div_assoc]
  rw [← sum_div]
  congr 1
  simp only [mul_sum]
  rw [sum_comm]
  apply sum_congr rfl
  intro u hu
  rw [sum_comm]

/-- FM1: the whole unused block has weighted gradient energy controlled
by the actual unweighted radial product. -/
theorem metric_unused_group_cocoercive (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (classes : U → C) (q : ℝ) (r : K → ℝ)
    (hr : ∀ k, 0 ≤ r k) (hu : ∀ u i, classes u ≠ y i)
    (hq : ∀ u i, probability (fun v => linearScore (E v) (T i)) (classes u) ≤ q) :
    (∑ j : U × K, r j.2*(unusedCoordinates (readoutGradient T y E) classes j)^2) ≤
      (q*(metricFeatureEnergy r T/Fintype.card I)) *
        ∑ j, unusedCoordinates E classes j*unusedCoordinates (readoutGradient T y E) classes j := by
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
  exact sum_le_sum fun v hv => metric_unused_pair_cocoercive T y E (classes u) (classes v) q r
    hr (hu u) (hu v) (hq u) (hq v)

omit [Fintype C] [Fintype U] [Nonempty U] [DecidableEq C] [DecidableEq K] in
/-- FM1: a declared feature-energy bound yields a conservative coefficient
before the next feature table or probability ceiling is observed. -/
theorem metric_feature_energy_le (r : K → ℝ) (T : I → K → ℝ) (R : ℝ)
    (hr : ∀ k, r k ≤ R) : metricFeatureEnergy r T ≤ R*matrixEnergy T := by
  unfold metricFeatureEnergy matrixEnergy
  simp only [mul_sum]
  exact sum_le_sum fun i _ => sum_le_sum fun k _ =>
    mul_le_mul_of_nonneg_right (hr k) (sq_nonneg _)

example := metric_mean_product_energy_le (K := Bool) (I := Bool)
  (fun k => if k then 2 else 1) (fun i => if i then 3 else -1)
  (fun i k => if i == k then 2 else -1) (by intro k; cases k <;> norm_num)
example := metric_feature_energy_nonneg (I := Fin 1) (K := Fin 1)
  (fun _ => 2) (fun _ _ => 3) (by intro k; norm_num)
example := metric_unused_pair_cocoercive (I := Fin 1) (C := Fin 3) (K := Bool)
  (fun _ k => if k then 2 else 1) (fun _ => 0) (fun c _ => c.val) 1 2 1
  (fun k => if k then 2 else 1) (by intro k; cases k <;> norm_num)
  (by intro i; decide) (by intro i; decide)
  (by intro i; exact probability_le_one _ _) (by intro i; exact probability_le_one _ _)
example := metric_unused_group_cocoercive (I := Fin 1) (C := Fin 3) (K := Bool)
  (fun _ k => if k then 2 else 1) (fun _ => 0) (fun c _ => c.val)
  (fun u : Fin 2 => u.succ) 1 (fun k => if k then 2 else 1)
  (by intro k; cases k <;> norm_num) (by intro u i; exact Fin.succ_ne_zero u)
  (by intro u i; exact probability_le_one _ _)
example := metric_feature_energy_le (I := Fin 1) (K := Bool)
  (fun k => if k then 2 else 1) (fun _ k => if k then 2 else 1) 2
  (by intro k; cases k <;> norm_num)

end Transformer.Grokking.GradientEvidence
