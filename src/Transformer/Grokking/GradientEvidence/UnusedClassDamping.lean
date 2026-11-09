module

public import Transformer.Grokking.GradientEvidence.ReadoutGradient
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.ClassDecomposition

/-!
# Supervised damping of unused class contrasts

Source: `review/flat_spectrum.md`, *Class-contrast discovery and persistence*,
CG1–CG2. The actual partial CE derivative damps each pair of unused readout
rows. Features are fixed while taking a partial derivative, but may change
between updates. Adaptive scaling, tied encoder motion and memory are not
identified with this isotropic contribution.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset Geometry FlatSpectrum.FeatureCodes Landscape.Neural

variable {I C K : Type*} [Fintype I] [Fintype C] [Fintype K]
  [DecidableEq C] [DecidableEq K]

omit [Fintype I] [Fintype K] [DecidableEq C] [DecidableEq K] in
/-- Softmax preserves the ordering of every pair of logits, CG1. -/
theorem logit_probability_difference_nonneg (z : C → ℝ) (c d : C) :
    0 ≤ (z c - z d) * (probability z c - probability z d) := by
  have hp : 0 ≤ (z c - z d) * (Real.exp (z c) - Real.exp (z d)) := by
    rcases le_total (z c) (z d) with h | h
    · exact mul_nonneg_of_nonpos_of_nonpos (sub_nonpos.mpr h)
        (sub_nonpos.mpr (Real.exp_le_exp.mpr h))
    · exact mul_nonneg (sub_nonneg.mpr h)
        (sub_nonneg.mpr (Real.exp_le_exp.mpr h))
  simpa only [probability, ← sub_div, ← mul_div_assoc] using
    div_nonneg hp (sum_exp_pos z c).le

/-- If neither class is a target, the exact partial-gradient pair product
is a mean of same-sign logit/probability differences, CG1. -/
theorem unused_pair_gradient_identity (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (c d : C)
    (hc : ∀ i, c ≠ y i) (hd : ∀ i, d ≠ y i) :
    (∑ k, (E c k - E d k) * (readoutGradient T y E c k - readoutGradient T y E d k)) =
      (∑ i, (linearScore (E c) (T i) - linearScore (E d) (T i)) *
        (probability (fun v => linearScore (E v) (T i)) c -
          probability (fun v => linearScore (E v) (T i)) d)) / Fintype.card I := by
  have hg (k : K) : readoutGradient T y E c k - readoutGradient T y E d k =
      (∑ i, (probability (fun v => linearScore (E v) (T i)) c -
        probability (fun v => linearScore (E v) (T i)) d) * T i k) / Fintype.card I := by
    simp_rw [readoutGradient_eq, ite_eq_right (hc _), ite_eq_right (hd _)]
    simp only [sub_zero, ← sub_div, ← sum_sub_distrib, ← sub_mul]
  simp_rw [hg, ← mul_div_assoc, mul_sum]
  rw [← sum_div, sum_comm]
  congr 1
  apply sum_congr rfl
  intro i hi
  simp only [linearScore, ← sum_sub_distrib, ← sub_mul, sum_mul]
  apply sum_congr rfl
  intro k hk
  ring

/-- Actual partial CE gives a dissipative unused-class direction, CG1.
This holds for every current feature table, hence does not require a feature
table to stay fixed over the complete training trajectory. -/
theorem unused_pair_gradient_damping (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (c d : C)
    (hc : ∀ i, c ≠ y i) (hd : ∀ i, d ≠ y i) :
    0 ≤ ∑ k, (E c k - E d k) *
      (readoutGradient T y E c k - readoutGradient T y E d k) := by
  rw [unused_pair_gradient_identity T y E c d hc hd]
  exact div_nonneg (sum_nonneg fun i hi =>
    logit_probability_difference_nonneg (fun v => linearScore (E v) (T i)) c d)
    (Nat.cast_nonneg _)

omit [Fintype I] [Fintype C] [DecidableEq C] [DecidableEq K] in
/-- The finite isotropic step keeps its quadratic term, CG2. Damping of a
derivative alone is insufficient at an arbitrarily large learning rate. -/
theorem isotropic_pair_energy_identity (w g : K → ℝ) (η : ℝ) :
    (∑ k, (w k - η * g k) ^ 2) = (∑ k, w k ^ 2) -
      2 * η * (∑ k, w k * g k) + η ^ 2 * ∑ k, g k ^ 2 := by
  simp only [mul_sum]
  rw [← sum_sub_distrib, ← sum_add_distrib]
  apply sum_congr rfl
  intro k hk
  ring

omit [Fintype I] [Fintype C] [DecidableEq C] [DecidableEq K] in
/-- A computable finite-step condition makes isotropic damping sufficient
for contraction, CG2. The condition uses current weights and gradients. -/
theorem isotropic_pair_energy_nonincrease (w g : K → ℝ) {η : ℝ}
    (hη : 0 ≤ η) (hstep : η * (∑ k, g k ^ 2) ≤ 2 * ∑ k, w k * g k) :
    (∑ k, (w k - η * g k) ^ 2) ≤ ∑ k, w k ^ 2 := by
  rw [isotropic_pair_energy_identity]
  have h := mul_le_mul_of_nonneg_left hstep hη
  nlinarith

example := unused_pair_gradient_identity (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val) 1 2
  (by intro i; decide) (by intro i; decide)

example := unused_pair_gradient_damping (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val) 1 2
  (by intro i; decide) (by intro i; decide)

example := isotropic_pair_energy_nonincrease (K := Bool)
  (fun k => if k then 2 else 1) (fun k => if k then 1 else 2) (η := 1 / 2)
  (by norm_num) (by norm_num)

variable {U : Type*} [Fintype U] [Nonempty U]

/-- Centered energy of the entire unused class group has a dissipative
partial CE gradient, CG1. Class means are computed, not assumed zero. -/
theorem unused_group_gradient_damping (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (classes : U → C) (h : ∀ u i, classes u ≠ y i) :
    0 ≤ ∑ u, ∑ k,
      (E (classes u) k - classMean (fun v => E (classes v) k)) *
      (readoutGradient T y E (classes u) k -
        classMean (fun v => readoutGradient T y E (classes v) k)) := by
  rw [sum_comm]
  simp_rw [centered_pair_product]
  rw [← sum_div]
  apply div_nonneg
  · rw [sum_comm]
    apply sum_nonneg
    intro u hu
    rw [sum_comm]
    exact sum_nonneg fun v hv => unused_pair_gradient_damping T y E
      (classes u) (classes v) (h u) (h v)
  · positivity

/-- The signed isotropic readout credit remains nonpositive after summing
any number of updates with changing features and minibatches, CG1/CG4. Other optimizer
components and the finite quadratic term are still separate. -/
theorem cumulative_unused_readout_credit_nonpos
    (T : ℕ → I → K → ℝ) (y : ℕ → I → C) (E : ℕ → C → K → ℝ)
    (classes : U → C) (h : ∀ u t i, classes u ≠ y t i)
    (η α : ℕ → ℝ) (hη : ∀ t, 0 ≤ η t) (hα : ∀ t, 0 ≤ α t) (n : ℕ) :
    (∑ t ∈ range n, -2 * η t * α t * (∑ u, ∑ k,
      (E t (classes u) k - classMean (fun v => E t (classes v) k)) *
      (readoutGradient (T t) (y t) (E t) (classes u) k -
        classMean (fun v => readoutGradient (T t) (y t) (E t) (classes v) k)))) ≤ 0 := by
  apply sum_nonpos
  intro t ht
  have hg := unused_group_gradient_damping (T t) (y t) (E t) classes (fun u => h u t)
  have hp := mul_nonneg (mul_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2)
    (hη t)) (hα t)) hg
  nlinarith

example := unused_group_gradient_damping (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val) (fun u : Fin 2 => u.succ)
  (by intro u i; exact Fin.succ_ne_zero u)

example (n : ℕ) := cumulative_unused_readout_credit_nonpos
  (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun t _ _ => t+1) (fun _ _ => 0) (fun t c _ => t+c.val)
  (fun u : Fin 2 => u.succ) (by intro u t i; exact Fin.succ_ne_zero u)
  (fun _ => 1/100) (fun _ => 2) (by intro t; norm_num) (by intro t; norm_num) n

end Transformer.Grokking.GradientEvidence
