module

public import Transformer.Grokking.GradientEvidence.FeatureMetricCE

/-!
# Momentum control with a fixed positive diagonal

Source: `review/flat_spectrum.md`, *Checkpoint-fixed feature metrics*, FM2–FM3.
The metric is fixed, not recomputed from a future gradient. In the actual-CE
application its entries are shared by unused classes at each feature.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset
variable {K J : Type*} [Fintype K] [Fintype J]

/-- FM2: an inverse-metric position energy and metric-weighted memory energy. -/
def metricMomentumPotential (r : K → ℝ) (s : ℝ) (w m : K → ℝ) : ℝ :=
  ∑ k, ((w k-9*s*r k*m k)^2/r k+90*s^2*r k*(m k)^2)

/-- FM2: positivity requires the declared metric entries to be nonnegative. -/
theorem metric_momentum_nonneg (r : K → ℝ) (s : ℝ) (w m : K → ℝ)
    (hr : ∀ k, 0 ≤ r k) : 0 ≤ metricMomentumPotential r s w m := by
  unfold metricMomentumPotential
  apply sum_nonneg
  intro k hk
  exact add_nonneg (div_nonneg (sq_nonneg _) (hr k))
    (mul_nonneg (mul_nonneg (by positivity) (hr k)) (sq_nonneg _))

/-- FM2: the metric potential controls inverse-weighted parameter energy. -/
theorem metric_momentum_coercive_identity (r : K → ℝ) (s : ℝ) (w m : K → ℝ)
    (hr : ∀ k, r k ≠ 0) :
    19*metricMomentumPotential r s w m-10*(∑ k, (w k)^2/r k) =
      9*∑ k, (w k-19*s*r k*m k)^2/r k := by
  unfold metricMomentumPotential
  simp only [mul_sum, ← sum_sub_distrib]
  apply sum_congr rfl
  intro k hk
  field_simp [hr k]
  ring

/-- FM2: retain the largest-weight penalty when returning to ordinary energy.
A small weighted potential alone need not be a tight Euclidean certificate. -/
theorem metric_parameter_energy_le (r : K → ℝ) (s R : ℝ) (w m : K → ℝ)
    (hr : ∀ k, 0 < r k) (hR : 0 ≤ R) (hmax : ∀ k, r k ≤ R) :
    (∑ k, (w k)^2) ≤ (19/10)*R*metricMomentumPotential r s w m := by
  have he := metric_momentum_coercive_identity r s w m (fun k => ne_of_gt (hr k))
  have hp : 0 ≤ ∑ k, (w k-19*s*r k*m k)^2/r k :=
    sum_nonneg (fun k _ => div_nonneg (sq_nonneg _) (hr k).le)
  have hb : (∑ k, (w k)^2/r k) ≤ (19/10)*metricMomentumPotential r s w m := by linarith
  have hsum : (∑ k, (w k)^2) ≤ R*(∑ k, (w k)^2/r k) := by
    rw [mul_sum]
    apply sum_le_sum
    intro k hk
    have h := mul_le_mul_of_nonneg_right (hmax k) (div_nonneg (sq_nonneg (w k)) (hr k).le)
    simpa only [mul_div_cancel₀ _ (ne_of_gt (hr k))] using h
  have h := mul_le_mul_of_nonneg_left hb hR
  nlinarith

/-- FM2: exact .9 EMA identity for the feature-scaled reference update. -/
theorem metric_momentum_step_identity (r : K → ℝ) (s : ℝ) (w m g : K → ℝ)
    (hr : ∀ k, r k ≠ 0) :
    metricMomentumPotential r s (fun k => w k-s*r k*mainFirst (m k) (g k))
      (fun k => mainFirst (m k) (g k)) - metricMomentumPotential r s w m =
      -2*s*(∑ k, w k*g k)+19*s^2*(∑ k, r k*(g k)^2)-
        (171/10)*s^2*∑ k, r k*(m k-g k)^2 := by
  unfold metricMomentumPotential
  simp only [mul_sum, ← sum_add_distrib, ← sum_sub_distrib]
  apply sum_congr rfl
  intro k hk
  unfold mainFirst
  field_simp [hr k]
  ring

/-- FM2: quantitative weighted damping absorbs arbitrary saved momentum. -/
theorem metric_momentum_step_nonincrease (r : K → ℝ) (s L : ℝ) (w m g : K → ℝ)
    (hr : ∀ k, 0 < r k) (hs : 0 ≤ s) (hd : 0 ≤ ∑ k, w k*g k)
    (hc : (∑ k, r k*(g k)^2) ≤ L*(∑ k, w k*g k)) (hstep : 19*s*L ≤ 2) :
    metricMomentumPotential r s (fun k => w k-s*r k*mainFirst (m k) (g k))
      (fun k => mainFirst (m k) (g k)) ≤ metricMomentumPotential r s w m := by
  have he := metric_momentum_step_identity r s w m g (fun k => ne_of_gt (hr k))
  have h1 := mul_le_mul_of_nonneg_left hc (by positivity : (0 : ℝ) ≤ 19*s^2)
  have h2 := mul_le_mul_of_nonneg_right hstep (mul_nonneg hs hd)
  have h3 : 0 ≤ (171/10 : ℝ)*s^2*∑ k, r k*(m k-g k)^2 := by
    exact mul_nonneg (by positivity) (sum_nonneg (fun k _ => mul_nonneg (hr k).le (sq_nonneg _)))
  nlinarith

/-- FM2: common clipping rescales the weighted coefficient, not the metric. -/
theorem metric_scaled_cocoercivity (r w g : K → ℝ) (L a : ℝ)
    (hc : (∑ k, r k*(g k)^2) ≤ L*(∑ k, w k*g k)) :
    (∑ k, r k*(a*g k)^2) ≤ (a*L)*(∑ k, w k*(a*g k)) := by
  have h := mul_le_mul_of_nonneg_left hc (sq_nonneg a)
  convert h using 1 <;> simp only [mul_sum] <;>
    apply sum_congr rfl <;> intro k hk <;> ring

/-- FM3: exact perturbation formula, including the square of the total error. -/
theorem metric_momentum_perturbation (r : K → ℝ) (s : ℝ) (w m dw dm : K → ℝ) :
    metricMomentumPotential r s (fun k => w k+dw k) (fun k => m k+dm k) -
      metricMomentumPotential r s w m =
      (∑ k, (2*(w k-9*s*r k*m k)*(dw k-9*s*r k*dm k)/r k+
        180*s^2*r k*m k*dm k)) + metricMomentumPotential r s dw dm := by
  unfold metricMomentumPotential
  rw [← sum_sub_distrib, ← sum_add_distrib]
  apply sum_congr rfl
  intro k hk
  ring

/-- FM3: source credits add linearly; quadratic cross terms are preserved. -/
theorem metric_momentum_sources (r : K → ℝ) (s : ℝ) (w m : K → ℝ)
    (dw dm : J → K → ℝ) :
    metricMomentumPotential r s (fun k => w k+∑ j, dw j k) (fun k => m k+∑ j, dm j k) -
      metricMomentumPotential r s w m =
      (∑ j, ∑ k, (2*(w k-9*s*r k*m k)*(dw j k-9*s*r k*dm j k)/r k+
        180*s^2*r k*m k*dm j k)) +
        metricMomentumPotential r s (fun k => ∑ j, dw j k) (fun k => ∑ j, dm j k) := by
  rw [metric_momentum_perturbation]
  congr 1
  rw [sum_comm]
  apply sum_congr rfl
  intro k hk
  simp only [sum_add_distrib, ← sum_div, ← mul_sum, sum_sub_distrib]

/-- FM3: the full signed change is the controlled reference plus its complete
perturbation. The last two terms still require prospective full-model control. -/
theorem metric_full_step_identity (r : K → ℝ) (s : ℝ) (w m g dw dm : K → ℝ)
    (hr : ∀ k, r k ≠ 0) :
    let mr := fun k => mainFirst (m k) (g k)
    let wr := fun k => w k-s*r k*mr k
    metricMomentumPotential r s (fun k => wr k+dw k) (fun k => mr k+dm k) -
      metricMomentumPotential r s w m =
      -2*s*(∑ k, w k*g k)+19*s^2*(∑ k, r k*(g k)^2)-
        (171/10)*s^2*(∑ k, r k*(m k-g k)^2) +
      (∑ k, (2*(wr k-9*s*r k*mr k)*(dw k-9*s*r k*dm k)/r k+
        180*s^2*r k*mr k*dm k)) + metricMomentumPotential r s dw dm := by
  dsimp only
  have hp := metric_momentum_perturbation r s
    (fun k => w k-s*r k*mainFirst (m k) (g k)) (fun k => mainFirst (m k) (g k)) dw dm
  have hb := metric_momentum_step_identity r s w m g hr
  linarith

example := metric_full_step_identity (K := Fin 1) (fun _ => 2) (1/100)
  (fun _ => 2) (fun _ => -1) (fun _ => 1) (fun _ => 1/100) (fun _ => -1/20)
  (by intro k; norm_num)

example := metric_momentum_nonneg (K := Bool) (fun k => if k then 2 else 1)
  (1/100) (fun k => if k then 2 else -1) (fun k => if k then -3 else 4)
  (by intro k; cases k <;> norm_num)
example := metric_momentum_coercive_identity (K := Fin 1) (fun _ => 2)
  (1/100) (fun _ => 2) (fun _ => -3) (by intro k; norm_num)
example := metric_parameter_energy_le (K := Fin 1) (fun _ => 2)
  (1/100) 3 (fun _ => 2) (fun _ => -3) (by intro k; norm_num)
  (by norm_num) (by intro k; norm_num)
example := metric_momentum_step_identity (K := Fin 1) (fun _ => 2)
  (1/100) (fun _ => 2) (fun _ => -3) (fun _ => 1) (by intro k; norm_num)
example := metric_momentum_step_nonincrease (K := Fin 1) (fun _ => 2) (1/100) 1
  (fun _ => 2) (fun _ => -3) (fun _ => 1) (by intro k; norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := metric_scaled_cocoercivity (K := Fin 1) (fun _ => 2)
  (fun _ => 2) (fun _ => 1) 1 (1/2) (by norm_num)

end Transformer.Grokking.GradientEvidence
