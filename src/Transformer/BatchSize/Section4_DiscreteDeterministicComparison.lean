/-
# Finite-horizon weak comparison for the actual discrete optimizers

arXiv:2506.12543v1, Section 4.3, Theorem 1.
The genuine Gaussian sampler is compared with the deterministic Euler
orbit using uniformly bounded C2 backward tests and the proved Markov
contraction. Every local defect and derivative bound is established.
-/

import Transformer.BatchSize.Section4_C2DiscreteError
import Transformer.BatchSize.Section4_DeterministicHorizonBounds
import Transformer.BatchSize.Section4_DriftDerivativeBounds
import Transformer.BatchSize.Section4_IteratedExpectation

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Pullback by the actual deterministic mean Euler step,
Section 4.3, Theorem 1's comparison argument. -/
def deterministicExpectationOperator {d : ℕ} (b : EucSpace d → EucSpace d)
    (η : NNReal) (hb : Continuous b) :
    BoundedObservable (EucSpace d) → BoundedObservable (EucSpace d) :=
  fun φ => φ.comp (deterministicEulerMap b η)
    (continuous_id.add (hb.const_smul (η : ℝ))).measurable

/-- The comparison operator's true iterates are precisely the
deterministic Euler backward tests, Section 4.3, Theorem 1. -/
theorem deterministicExpectationOperator_iterate {d : ℕ} (b : EucSpace d → EucSpace d)
    (η : NNReal) (hb : Continuous b) (φ : BoundedObservable (EucSpace d)) (n : ℕ) :
    (((deterministicExpectationOperator b η hb)^[n]) φ).val =
      deterministicEulerTest b η φ.val n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [Function.iterate_succ_apply']
    change (fun x => (((deterministicExpectationOperator b η hb)^[n]) φ).val
      (deterministicEulerMap b η x)) = _
    rw [ih]
    rfl

/-- Nonvacuity of the deterministic expectation regularity,
Section 4.3: the identity drift is continuous. -/
example : Continuous (id : EucSpace 1 → EucSpace 1) := continuous_id

/-- On a fixed physical horizon, the actual SGD and SignSGD laws
differ from their deterministic mean Euler orbit by O(eta) on every
normalized C2 observable, Section 4.3, Theorem 1. The constant precedes
the mesh and test; no propagated smoothness premise is assumed. -/
theorem sgd_sign_discrete_deterministic_weak_error {d : ℕ} (method : UpdateKind)
    (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (T : ℝ) (hT : 0 ≤ T) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ η : NNReal, ∀ φ : EucSpace d → ℝ,
      BoundedSmoothTest 2 φ → ∀ n : ℕ, (n : ℝ) * η ≤ T → ∀ x,
      |(∫ y, φ y ∂discreteLaw method η B f σ x n) -
        deterministicEulerTest (diffusionDrift method B f σ) η φ n x| ≤ C * η := by
  let b := diffusionDrift method B f σ
  have hb : ContDiff ℝ 2 b :=
    (regularGaussianModel_smooth_coefficients method 0 B f σ hmodel).1.of_le (by norm_num)
  obtain ⟨K, hK⟩ := diffusionDrift_lipschitz method B f σ hmodel
  obtain ⟨H, hH⟩ := diffusionDrift_fderiv_lipschitz method B f σ hmodel
  obtain ⟨C₀, hC₀, hlocal⟩ := sgd_sign_C2_deterministic_local_error method B f σ hB hmodel
  let R := deterministicTestBound K H T
  have hR : 0 ≤ R := (deterministicTestBound_one_le K H T hT).trans' (by norm_num)
  refine ⟨C₀ * R * T, by positivity, ?_⟩
  intro η φ hφ n hn x
  have hf : ContDiff ℝ 1 f := hmodel.1.of_le (by norm_num)
  have hσ : Continuous σ := hmodel.2.1.continuous
  let A := discreteExpectationOperator method η B f σ hf hσ
  let S := deterministicExpectationOperator b η hb.continuous
  let q : BoundedObservable (EucSpace d) := ⟨φ, hφ.1.continuous.measurable,
    1, fun y => by simpa only [norm_iteratedFDeriv_zero, Real.norm_eq_abs] using hφ.2 0 (by norm_num) y⟩
  have hstep (j : ℕ) (hj : j < n) (y : EucSpace d) :
      |(A ((S^[j]) q)).val y - (S ((S^[j]) q)).val y| ≤ (C₀ * R) * (η : ℝ) ^ 2 := by
    have hjT : (j : ℝ) * η ≤ T := (mul_le_mul_of_nonneg_right
      (by exact_mod_cast (Nat.le_of_lt hj)) η.coe_nonneg).trans hn
    obtain ⟨hs, hd⟩ := deterministicEulerTest_horizon_bounds b K H hb hK hH T hT η j hjT φ hφ
    change |(∫ z, ((S^[j]) q).val (stochasticStep method η B f σ y z)
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) -
        ((S^[j]) q).val (deterministicEulerMap b η y)| ≤ _
    have heq : ((S^[j]) q).val = deterministicEulerTest b η φ j :=
      deterministicExpectationOperator_iterate b η hb.continuous q j
    simp_rw [heq]
    exact hlocal η R hR (deterministicEulerTest b η φ j) hs (hd 2 le_rfl) y
  have h := finite_horizon_weak_error A S q (C₀ * R) η T (mul_nonneg hC₀ hR) η.coe_nonneg
    (fun ψ χ r hr hclose y => discreteExpectationOperator_contraction method η B f σ hf hσ ψ χ r hclose y)
    n hn hstep x
  have hA : ((A^[n]) q).val x = ∫ y, φ y ∂discreteLaw method η B f σ x n :=
    iteratedExpectation_eq_discreteLaw method η B f σ hf hσ q n x
  have hS : ((S^[n]) q).val = deterministicEulerTest b η φ n :=
    deterministicExpectationOperator_iterate b η hb.continuous q n
  rw [hA, hS] at h
  exact h

/-- Joint nonvacuity of the actual finite-horizon comparison,
Section 4.3: positive batch and horizon, flat regular loss, unit noise. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧ (0 : ℝ) ≤ 1 :=
  ⟨by norm_num, regularGaussianModel_flat 1, by norm_num⟩

end Transformer.BatchSize
