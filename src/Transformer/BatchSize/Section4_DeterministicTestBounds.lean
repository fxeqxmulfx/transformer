/-
# Genuine C2 bounds for the deterministic backward Euler tests

arXiv:2506.12543v1, Section 4.3, Theorem 1's weak comparison.
The true propagated tests retain their value bound; their derivative
constants grow at most exponentially in physical time, not in mesh size.
-/

import Transformer.BatchSize.Section4_DeterministicEulerMap

open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual backward test obtained by n deterministic Euler
steps of the optimizer's mean drift, Section 4.3, Theorem 1. -/
def deterministicEulerTest {d : ℕ} (b : EucSpace d → EucSpace d) (η : NNReal)
    (φ : EucSpace d → ℝ) : ℕ → EucSpace d → ℝ
  | 0 => φ
  | n + 1 => deterministicEulerTest b η φ n ∘ deterministicEulerMap b η

/-- A normalized C2 test has a unit Lipschitz derivative in the
actual Frechet operator norm, Section 4.3, Theorem 1. -/
theorem boundedSmoothTest_fderiv_lipschitz {d : ℕ} (φ : EucSpace d → ℝ)
    (hφ : BoundedSmoothTest 2 φ) : LipschitzWith 1 (fderiv ℝ φ) := by
  apply lipschitzWith_of_nnnorm_fderiv_le
    ((hφ.1.fderiv_right (m := 1) (by norm_num)).differentiable (by norm_num))
  intro x
  change ‖fderiv ℝ (fderiv ℝ φ) x‖ ≤ 1
  rw [← norm_iteratedFDeriv_one (fderiv ℝ φ), norm_iteratedFDeriv_fderiv]
  exact hφ.2 2 le_rfl x

/-- The true deterministic backward tests have derivative bound
(1+eta*K)^n and derivative Lipschitz bound
(1+n*eta*H)*(1+eta*K)^(2*n), Section 4.3, Theorem 1.
The eta factors ensure uniform bounds on a finite physical horizon. -/
theorem deterministicEulerTest_derivative_bounds {d : ℕ}
    (b : EucSpace d → EucSpace d) (η K H : NNReal) (hb : ContDiff ℝ 2 b)
    (hK : LipschitzWith K b) (hH : LipschitzWith H (fderiv ℝ b))
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) (n : ℕ) :
    ContDiff ℝ 2 (deterministicEulerTest b η φ n) ∧
      (∀ x, |deterministicEulerTest b η φ n x| ≤ 1) ∧
      (∀ x, ‖fderiv ℝ (deterministicEulerTest b η φ n) x‖ ≤ (1 + η * K) ^ n) ∧
      LipschitzWith ((1 + (n : NNReal) * η * H) * (1 + η * K) ^ (2 * n))
        (fderiv ℝ (deterministicEulerTest b η φ n)) := by
  let q : NNReal := 1 + η * K
  have hq : 1 ≤ q := le_add_of_nonneg_right zero_le
  induction n with
  | zero =>
    refine ⟨hφ.1, ?_, ?_, ?_⟩
    · intro x
      simpa only [deterministicEulerTest, norm_iteratedFDeriv_zero, Real.norm_eq_abs] using
        hφ.2 0 (by norm_num) x
    · intro x
      simpa only [deterministicEulerTest, pow_zero, NNReal.coe_one, norm_iteratedFDeriv_one] using
        hφ.2 1 (by norm_num) x
    · simpa only [deterministicEulerTest, Nat.cast_zero, zero_mul, add_zero, Nat.mul_zero,
        pow_zero, mul_one, one_mul] using
        boundedSmoothTest_fderiv_lipschitz φ hφ
  | succ n ih =>
    have hmap := deterministicEulerMap_contDiff b η hb
    have hmapK := deterministicEulerMap_lipschitz b η K hK
    have hmapH := deterministicEulerMap_fderiv_lipschitz b η H hb hH
    refine ⟨ih.1.comp hmap, (fun x => ih.2.1 _), ?_, ?_⟩
    · intro x
      have h := composed_fderiv_norm_le (deterministicEulerMap b η)
        (deterministicEulerTest b η φ n) hmap ih.1 q (q ^ n) hmapK ih.2.2.1 x
      simpa only [deterministicEulerTest, pow_succ, q, NNReal.coe_pow, NNReal.coe_mul,
        NNReal.coe_add, NNReal.coe_one] using h
    · have h := composed_fderiv_lipschitz (deterministicEulerMap b η)
        (deterministicEulerTest b η φ n) hmap ih.1 q (η * H) (q ^ n)
        ((1 + (n : NNReal) * η * H) * q ^ (2 * n)) hmapK hmapH ih.2.2.1 ih.2.2.2
      have hpow : q ^ n ≤ q ^ (2 * n + 2) := pow_le_pow_right₀ hq (by omega)
      have hc : q ^ n * (η * H) + ((1 + (n : NNReal) * η * H) * q ^ (2 * n)) * q ^ 2 ≤
          (1 + (n.succ : NNReal) * η * H) * q ^ (2 * n.succ) := by
        calc
          _ ≤ q ^ (2 * n + 2) * (η * H) + ((1 + (n : NNReal) * η * H) * q ^ (2 * n)) * q ^ 2 :=
            add_le_add (mul_le_mul_of_nonneg_right hpow zero_le) le_rfl
          _ = _ := by
            rw [show 2 * n.succ = 2 * n + 2 by omega, pow_add]
            simp only [Nat.cast_succ]
            ring
      exact h.weaken hc

/-- Joint nonvacuity of the backward-test derivative hypotheses,
Section 4.3: bounded zero drift, a constant derivative map and a
normalized nonzero C2 observable, at positive step size. -/
example : ContDiff ℝ 2 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    LipschitzWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    LipschitzWith 0 (fderiv ℝ (fun _ : EucSpace 1 => (0 : EucSpace 1))) ∧
    BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  have hd : fderiv ℝ (fun _ : EucSpace 1 => (0 : EucSpace 1)) = fun _ => 0 := by
    funext x
    exact congrFun (fderiv_const (𝕜 := ℝ) (E := EucSpace 1) (c := (0 : EucSpace 1))) x
  refine ⟨contDiff_const, LipschitzWith.const _, ?_, contDiff_const, ?_⟩
  · rw [hd]
    exact LipschitzWith.const _
  · intro j hj y
    cases j with
    | zero => simp [norm_iteratedFDeriv_zero]
    | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
