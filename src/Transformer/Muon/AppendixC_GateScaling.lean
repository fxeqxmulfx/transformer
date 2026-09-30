/-
# Muon — gate scaling

arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. The code
renormalizes its selected nonnegative gate weights and uses the reciprocal
Euclidean norm. The numerical value `2.446` is a Monte Carlo estimate, not
an exact constant implied by the formula.
-/

import Transformer.Muon.Section2_Models

open scoped BigOperators

noncomputable section

namespace Transformer.Muon

variable {k : ℕ}

/-- Renormalized selected gate weights, arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
def normalizedGateWeights (w : Fin k → ℝ) (i : Fin k) : ℝ := w i / ∑ j, w j

/-- The sample gate scaling factor in the source's Python code,
arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
def gateScale (p : Fin k → ℝ) : ℝ := (Real.sqrt (∑ i, p i ^ 2))⁻¹

/-- Renormalization yields a probability vector when selected weights are
nonnegative and have positive total mass.
Source: arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`, “renormalize”. -/
theorem normalizedGateWeights_probability (w : Fin k → ℝ) (hw : ∀ i, 0 ≤ w i)
    (hs : 0 < ∑ i, w i) :
    (∀ i, 0 ≤ normalizedGateWeights w i) ∧ (∑ i, normalizedGateWeights w i) = 1 := by
  refine ⟨fun i => div_nonneg (hw i) hs.le, ?_⟩
  simp only [normalizedGateWeights, ← Finset.sum_div]
  exact div_self hs.ne'

/-- Selected positive weights exist, arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
example : (∀ i : Fin 2, (0 : ℝ) ≤ (fun _ : Fin 2 => (1 : ℝ)) i) ∧
    0 < ∑ i : Fin 2, (fun _ : Fin 2 => (1 : ℝ)) i := by norm_num

/-- Squared probability mass is positive, at most one, and at least `1/k`,
arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`, the gate-factor denominator. -/
theorem gate_energy_bounds (p : Fin k → ℝ) (hp : ∀ i, 0 ≤ p i) (hs : ∑ i, p i = 1) :
    0 < ∑ i, p i ^ 2 ∧ (∑ i, p i ^ 2) ≤ 1 ∧ 1 ≤ (k : ℝ) * ∑ i, p i ^ 2 := by
  have hupper := Finset.sum_sq_le_sq_sum_of_nonneg (s := Finset.univ) (fun i _ => hp i)
  rw [hs] at hupper
  have hlower := Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul (s := Finset.univ)
    (r := p) (f := fun _ : Fin k => (1 : ℝ)) (g := fun i => p i ^ 2)
    (fun _ _ => zero_le_one) (fun i _ => sq_nonneg (p i)) (fun _ _ => by simp)
  simp only [hs, one_pow, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, mul_one] at hlower
  have hnonneg : 0 ≤ (k : ℝ) := Nat.cast_nonneg _
  have hpos : 0 < ∑ i, p i ^ 2 := by
    by_contra h
    have : (∑ i, p i ^ 2) ≤ 0 := le_of_not_gt h
    have hprod := mul_nonpos_of_nonneg_of_nonpos hnonneg this
    linarith
  exact ⟨hpos, by simpa using hupper, hlower⟩

/-- Probability gates exist, arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
example : (∀ i : Fin 2, (0 : ℝ) ≤ (fun _ : Fin 2 => (1 / 2 : ℝ)) i) ∧
    (∑ i : Fin 2, (fun _ : Fin 2 => (1 / 2 : ℝ)) i) = 1 := by norm_num

/-- Gate scaling cancels the weighted-output variance factor exactly,
arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. Interpreting this as variance
preservation assumes uncorrelated expert outputs with equal variance. -/
theorem gateScale_normalizes_energy (p : Fin k → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hs : ∑ i, p i = 1) : gateScale p ^ 2 * (∑ i, p i ^ 2) = 1 := by
  have hpos := (gate_energy_bounds p hp hs).1
  rw [gateScale, inv_pow, Real.sq_sqrt hpos.le]
  exact inv_mul_cancel₀ hpos.ne'

/-- Probability gates exist, arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
example : (∀ i : Fin 2, (0 : ℝ) ≤ (fun _ : Fin 2 => (1 / 2 : ℝ)) i) ∧
    (∑ i : Fin 2, (fun _ : Fin 2 => (1 / 2 : ℝ)) i) = 1 := by norm_num

/-- Every sample gate factor lies between one and `√k`,
arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
theorem gateScale_bounds (p : Fin k → ℝ) (hp : ∀ i, 0 ≤ p i) (hs : ∑ i, p i = 1) :
    1 ≤ gateScale p ∧ gateScale p ≤ Real.sqrt k := by
  have hb := gate_energy_bounds p hp hs
  have hroot : 0 < Real.sqrt (∑ i, p i ^ 2) := Real.sqrt_pos.mpr hb.1
  have hupper : Real.sqrt (∑ i, p i ^ 2) ≤ 1 := by
    nlinarith [Real.sq_sqrt hb.1.le]
  have hprod : 1 ≤ Real.sqrt k * Real.sqrt (∑ i, p i ^ 2) := by
    have hsquare := Real.sq_sqrt (by positivity : (0 : ℝ) ≤ k)
    have hqsquare := Real.sq_sqrt hb.1.le
    have hprod_nonneg := mul_nonneg (Real.sqrt_nonneg (k : ℝ)) hroot.le
    nlinarith [sq_nonneg (Real.sqrt k * Real.sqrt (∑ i, p i ^ 2) - 1)]
  unfold gateScale
  constructor
  · rw [inv_eq_one_div, one_le_div hroot]
    exact hupper
  · rw [inv_eq_one_div, div_le_iff₀ hroot]
    exact hprod

/-- Probability gates exist, arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
example : (∀ i : Fin 2, (0 : ℝ) ≤ (fun _ : Fin 2 => (1 / 2 : ℝ)) i) ∧
    (∑ i : Fin 2, (fun _ : Fin 2 => (1 / 2 : ℝ)) i) = 1 := by norm_num

/-- Uniform selected gates need the factor `√k`, arXiv:2502.16982,
Appendix C, `fig:gate_scaling_code`, the gate-factor formula. -/
theorem gateScale_uniform (hk : 0 < k) :
    gateScale (fun _ : Fin k => (k : ℝ)⁻¹) = Real.sqrt k := by
  have hk' : (k : ℝ) ≠ 0 := by positivity
  unfold gateScale
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have he : (k : ℝ) * ((k : ℝ)⁻¹) ^ 2 = (k : ℝ)⁻¹ := by field_simp
  rw [he, Real.sqrt_inv, inv_inv]

/-- A positive number of selected experts exists, arXiv:2502.16982, Appendix C. -/
example : 0 < (2 : ℕ) := by norm_num

/-- The finite sample mean returned by the source's Python function. Each
sample consists of selected, renormalized gates; the Gaussian draws are
inputs to the estimator rather than an assumed exact numerical result.
Source: arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`, `np.mean(factors)`. -/
def meanGateScale {trials : ℕ} (samples : Fin trials → Fin k → ℝ) : ℝ :=
  (∑ t, gateScale (samples t)) / trials

/-- The returned sample mean also lies between one and `√k`. This validates
the range of the Monte Carlo estimator without identifying its output
with the empirical rounded value 2.446.
Source: arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`, `np.mean(factors)`. -/
theorem meanGateScale_bounds {trials : ℕ} (samples : Fin trials → Fin k → ℝ)
    (ht : 0 < trials) (hp : ∀ t i, 0 ≤ samples t i) (hs : ∀ t, ∑ i, samples t i = 1) :
    1 ≤ meanGateScale samples ∧ meanGateScale samples ≤ Real.sqrt k := by
  have ht' : 0 < (trials : ℝ) := by exact_mod_cast ht
  have hlower := Finset.sum_le_sum (s := Finset.univ) fun t _ =>
    (gateScale_bounds (samples t) (hp t) (hs t)).1
  have hupper := Finset.sum_le_sum (s := Finset.univ) fun t _ =>
    (gateScale_bounds (samples t) (hp t) (hs t)).2
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one] at hlower hupper
  unfold meanGateScale
  constructor
  · exact (one_le_div ht').mpr hlower
  · apply (div_le_iff₀ ht').mpr
    simpa only [mul_comm] using hupper

/-- A positive sample count and normalized gates coexist,
arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
example : 0 < (1 : ℕ) ∧
    (∀ t : Fin 1, ∀ i : Fin 2, (0 : ℝ) ≤
      (fun _ : Fin 1 => fun _ : Fin 2 => (1 / 2 : ℝ)) t i) ∧
    (∀ t : Fin 1, (∑ i : Fin 2,
      (fun _ : Fin 1 => fun _ : Fin 2 => (1 / 2 : ℝ)) t i) = 1) := by
  norm_num

end Transformer.Muon
