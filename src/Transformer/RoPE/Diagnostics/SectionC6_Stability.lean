module

public import Transformer.RoPE.Diagnostics.SectionA4_Response

/-!
# Arbitrary-offset rotary-score stability

Wu, Du and Peng, arXiv:2609.39929v1, Appendix C.6, Eqs. (103)–(108),
with the unit displacement generalized to any real displacement. The
coefficients are arbitrary fixed Q/K activations, not an i.i.d. model.
These bounds do not describe recomputation of activations in preceding
layers after changing the input.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.RoPE.Diagnostics

/-- Appendix A.2: the score of a fully rotary head with arbitrary real
coefficient pairs, all realizable by actual query/key vectors. -/
noncomputable def headScore {n : ℕ} (a b omega : Fin n → ℝ) (distance : ℝ) : ℝ :=
  ∑ i, score (a i) (b i) (omega i) distance

/-- Frequency-weighted coefficient magnitude, the deterministic Lipschitz
budget obtained from Appendix C.6, Eq. (108). -/
noncomputable def spectralSlope {n : ℕ} (a b omega : Fin n → ℝ) : ℝ :=
  ∑ i, |omega i| * Real.sqrt (a i ^ 2 + b i ^ 2)

/-- The spectral stability budget is nonnegative for arbitrary coefficients. -/
theorem spectral_slope_nonneg {n : ℕ} (a b omega : Fin n → ℝ) :
    0 ≤ spectralSlope a b omega := by
  exact Finset.sum_nonneg fun i _ => mul_nonneg (abs_nonneg _) (Real.sqrt_nonneg _)

/-- A key's fixed reference phase can be absorbed into its coefficients.
Thus different baseline key positions admit the same query-shift coordinate. -/
theorem score_phase_shift (a b omega distance offset : ℝ) :
    score a b omega (distance + offset) =
      score (score a b omega offset) (score b (-a) omega offset) omega distance := by
  unfold score
  rw [mul_add, Real.cos_add, Real.sin_add]
  ring

/-- Appendix C.6's complex gain identity in real coordinates, with arbitrary
displacement in place of a one-token move. -/
theorem shift_difference (a b omega distance displacement : ℝ) :
    score a b omega (distance + displacement) - score a b omega distance =
      2 * Real.sin (omega * displacement / 2) *
        score b (-a) omega (distance + displacement / 2) := by
  have hc := Real.cos_sub_cos (x := omega * (distance + displacement))
    (y := omega * distance)
  have hs := Real.sin_sub_sin (x := omega * (distance + displacement))
    (y := omega * distance)
  have hsum : (omega * (distance + displacement) + omega * distance) / 2 =
      omega * (distance + displacement / 2) := by ring
  have hdiff : (omega * (distance + displacement) - omega * distance) / 2 =
      omega * displacement / 2 := by ring
  rw [hsum, hdiff] at hc hs
  unfold score
  linear_combination a * hc + b * hs

/-- Exact sine envelope for a single content-dependent score, Appendix C.6. -/
theorem shift_response_bound (a b omega distance displacement : ℝ) :
    |score a b omega (distance + displacement) - score a b omega distance| ≤
      2 * |Real.sin (omega * displacement / 2)| * Real.sqrt (a ^ 2 + b ^ 2) := by
  rw [shift_difference, abs_mul, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
  have h := score_amplitude_bound b (-a) omega (distance + displacement / 2)
  rw [show b ^ 2 + (-a) ^ 2 = a ^ 2 + b ^ 2 by ring] at h
  exact mul_le_mul_of_nonneg_left h (by positivity)

/-- Appendix C.6, Eq. (108): the envelope is at most displacement times the
frequency-weighted amplitude. No sign or distribution of Q/K is assumed. -/
theorem shift_frequency_bound (a b omega distance displacement : ℝ) :
    |score a b omega (distance + displacement) - score a b omega distance| ≤
      |displacement| * (|omega| * Real.sqrt (a ^ 2 + b ^ 2)) := by
  apply (shift_response_bound a b omega distance displacement).trans
  have h := Real.abs_sin_le_abs (x := omega * displacement / 2)
  rw [abs_div, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 2)] at h
  have hfactor : 2 * |Real.sin (omega * displacement / 2)| ≤ |omega| * |displacement| :=
    by linarith
  calc
    2 * |Real.sin (omega * displacement / 2)| * Real.sqrt (a ^ 2 + b ^ 2)
        ≤ (|omega| * |displacement|) * Real.sqrt (a ^ 2 + b ^ 2) :=
          mul_le_mul_of_nonneg_right hfactor (Real.sqrt_nonneg _)
    _ = _ := by ring

/-- The exact finite-head envelope for arbitrary position changes. -/
theorem head_shift_response_bound {n : ℕ} (a b omega : Fin n → ℝ)
    (distance displacement : ℝ) :
    |headScore a b omega (distance + displacement) - headScore a b omega distance| ≤
      ∑ i, 2 * |Real.sin (omega i * displacement / 2)| * Real.sqrt (a i ^ 2 + b i ^ 2) := by
  unfold headScore
  rw [← Finset.sum_sub_distrib]
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun i _ => shift_response_bound _ _ _ _ _)

/-- A whole rotary head is globally Lipschitz in its relative-position
coordinate with the explicit coefficient-dependent budget. -/
theorem head_score_lipschitz_bound {n : ℕ} (a b omega : Fin n → ℝ) (r s : ℝ) :
    |headScore a b omega r - headScore a b omega s| ≤
      |r - s| * spectralSlope a b omega := by
  have h : |headScore a b omega (s + (r - s)) - headScore a b omega s| ≤
      ∑ i, |r - s| * (|omega i| * Real.sqrt (a i ^ 2 + b i ^ 2)) := by
    unfold headScore
    rw [← Finset.sum_sub_distrib]
    exact (Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum fun i _ => shift_frequency_bound _ _ _ _ _)
  rw [show s + (r - s) = r by ring, ← Finset.mul_sum] at h
  exact h

/-- Uniformly scaling all frequencies scales the deterministic stability
budget by the magnitude of that factor, with the activations held fixed. -/
theorem spectral_slope_scale {n : ℕ} (a b omega : Fin n → ℝ) (factor : ℝ) :
    spectralSlope a b (fun i => factor * omega i) = |factor| * spectralSlope a b omega := by
  unfold spectralSlope
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [abs_mul]
  ring

/-- The maximum distance bounds the change everywhere in the window even
when the score itself is not monotone in distance. -/
theorem head_window_bound {n : ℕ} (a b omega : Fin n → ℝ) {width distance : ℝ}
    (hr : distance ∈ Set.Icc 0 width) :
    |headScore a b omega distance - headScore a b omega 0| ≤ width * spectralSlope a b omega := by
  apply (head_score_lipschitz_bound a b omega distance 0).trans
  rw [sub_zero, abs_of_nonneg hr.1]
  exact mul_le_mul_of_nonneg_right hr.2 (spectral_slope_nonneg a b omega)

example : |headScore (fun _ : Fin 1 => 4) (fun _ => 0) (fun _ => 1) (1 / 2) -
    headScore (fun _ : Fin 1 => 4) (fun _ => 0) (fun _ => 1) 0| ≤
      (1 / 2 : ℝ) * spectralSlope (fun _ : Fin 1 => 4) (fun _ => 0) (fun _ => 1) :=
  head_window_bound _ _ _ ⟨by norm_num, le_rfl⟩

end Transformer.RoPE.Diagnostics
