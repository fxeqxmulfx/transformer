/-
# Uniform backward test bounds on a physical time horizon

arXiv:2506.12543v1, Section 4.3, Theorem 1.
The comparison tests have a C2 bound independent of the step size
whenever their number of steps times the step size stays below T.
-/

import Transformer.BatchSize.Section4_DeterministicTestBounds

open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual Euler derivative growth is bounded by its physical
time exponential, Section 4.3, Theorem 1. -/
theorem deterministicEuler_growth_bound (η K : NNReal) (n : ℕ) (T : ℝ)
    (hn : (n : ℝ) * η ≤ T) :
    ((1 + η * K : NNReal) : ℝ) ^ n ≤ Real.exp (T * K) := by
  calc
    _ ≤ (Real.exp ((η : ℝ) * K)) ^ n :=
      pow_le_pow_left₀ (by positivity)
        (by simpa only [NNReal.coe_add, NNReal.coe_one, NNReal.coe_mul, add_comm] using
          Real.add_one_le_exp ((η : ℝ) * K)) n
    _ = Real.exp ((n : ℝ) * ((η : ℝ) * K)) := (Real.exp_nat_mul _ _).symm
    _ ≤ Real.exp (T * K) := Real.exp_le_exp.mpr (by
      simpa only [mul_assoc] using mul_le_mul_of_nonneg_right hn K.coe_nonneg)

/-- A mesh-independent bound for the first two derivatives of
actual deterministic backward tests, Section 4.3, Theorem 1. -/
def deterministicTestBound (K H : NNReal) (T : ℝ) : ℝ :=
  (1 + T * H) * Real.exp (2 * T * K)

/-- The physical-horizon comparison bound is at least one,
Section 4.3, Theorem 1. -/
theorem deterministicTestBound_one_le (K H : NNReal) (T : ℝ) (hT : 0 ≤ T) :
    1 ≤ deterministicTestBound K H T := by
  have he : 1 ≤ Real.exp (2 * T * K) := Real.one_le_exp_iff.mpr (by positivity)
  have ha : 1 ≤ 1 + T * H := le_add_of_nonneg_right (mul_nonneg hT H.coe_nonneg)
  simpa only [deterministicTestBound, one_mul] using
    mul_le_mul ha he (by norm_num : (0 : ℝ) ≤ 1) (by positivity)

/-- All derivatives through order two of the true n-step backward
test are bounded uniformly for n*eta <= T, Section 4.3, Theorem 1.
The bound is chosen before eta, n and the normalized observable. -/
theorem deterministicEulerTest_horizon_bounds {d : ℕ}
    (b : EucSpace d → EucSpace d) (K H : NNReal) (hb : ContDiff ℝ 2 b)
    (hK : LipschitzWith K b) (hH : LipschitzWith H (fderiv ℝ b))
    (T : ℝ) (hT : 0 ≤ T) (η : NNReal) (n : ℕ) (hn : (n : ℝ) * η ≤ T)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    ContDiff ℝ 2 (deterministicEulerTest b η φ n) ∧
      ∀ j ≤ 2, ∀ x, ‖iteratedFDeriv ℝ j (deterministicEulerTest b η φ n) x‖ ≤
        deterministicTestBound K H T := by
  obtain ⟨hs, hv, hd, hl⟩ := deterministicEulerTest_derivative_bounds b η K H hb hK hH φ hφ n
  have hpow : ((1 + η * K : NNReal) : ℝ) ^ (2 * n) ≤ Real.exp (2 * T * K) :=
    deterministicEuler_growth_bound η K (2 * n) (2 * T) (by
      push_cast
      nlinarith)
  have hq : 1 ≤ (1 + η * K : NNReal) := le_add_of_nonneg_right zero_le
  have hfirst : ((1 + η * K : NNReal) : ℝ) ^ n ≤ Real.exp (2 * T * K) := by
    refine (show ((1 + η * K : NNReal) : ℝ) ^ n ≤ ((1 + η * K : NNReal) : ℝ) ^ (2 * n) from
      pow_le_pow_right₀ (by exact_mod_cast hq) (by omega)).trans hpow
  have hcoef : 1 + (n : ℝ) * η * H ≤ 1 + T * H :=
    add_le_add le_rfl (mul_le_mul_of_nonneg_right hn H.coe_nonneg)
  have hsecond : (((1 + (n : NNReal) * η * H) * (1 + η * K) ^ (2 * n) : NNReal) : ℝ) ≤
      deterministicTestBound K H T := by
    simp only [NNReal.coe_mul, NNReal.coe_add, NNReal.coe_one, NNReal.coe_natCast,
      NNReal.coe_pow]
    exact mul_le_mul hcoef hpow (by positivity) (by positivity)
  refine ⟨hs, ?_⟩
  intro j hj x
  rcases (show j = 0 ∨ j = 1 ∨ j = 2 by omega) with rfl | rfl | rfl
  · rw [norm_iteratedFDeriv_zero, Real.norm_eq_abs]
    exact (hv x).trans (deterministicTestBound_one_le K H T hT)
  · rw [norm_iteratedFDeriv_one]
    refine (hd x).trans (hfirst.trans ?_)
    unfold deterministicTestBound
    nlinarith [mul_nonneg hT H.coe_nonneg, Real.exp_pos (2 * T * K)]
  · exact (iteratedFDeriv_two_le_of_fderiv_lipschitz _ _ hl x).trans hsecond

/-- Joint nonvacuity of the physical-horizon hypotheses,
Section 4.3: zero bounded drift, unit nonzero test and ten positive steps. -/
example : ContDiff ℝ 2 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    LipschitzWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    LipschitzWith 0 (fderiv ℝ (fun _ : EucSpace 1 => (0 : EucSpace 1))) ∧
    (0 : ℝ) ≤ 1 ∧ (10 : ℝ) * (1 / 1000 : NNReal) ≤ 1 ∧
    BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  have hd : fderiv ℝ (fun _ : EucSpace 1 => (0 : EucSpace 1)) = fun _ => 0 :=
    fderiv_const (𝕜 := ℝ) (E := EucSpace 1) _
  refine ⟨contDiff_const, LipschitzWith.const _, ?_, by norm_num, by norm_num,
    contDiff_const, ?_⟩
  · rw [hd]; exact LipschitzWith.const _
  · intro j hj x
    cases j with
    | zero => simp [norm_iteratedFDeriv_zero]
    | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
