/-
# Conditional global weak approximation of the actual optimizer

arXiv:2506.12543v1, Section 4.3, Theorem 1.
The discrete part of the global error bound is proved. The comparison
operator must give uniformly bounded propagated smooth tests and have
the stated one-step generator expansion on the finite horizon. Neither
condition is silently attributed to a continuous diffusion whose existence
has not been established.
-/

import Transformer.BatchSize.Section4_IteratedExpectation
import Transformer.BatchSize.Section4_ScaledWeakConsistency

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Conditional proved part of Section 4.3, Theorem 1: the actual n-step
Gaussian SGD/SignSGD law has O(eta) weak error against any comparison
operator with the stated smoothness and local generator expansion.
The model constant is independent of the state, step and observable;
the propagated derivative bound R enters explicitly and may exceed one.
Only propagation steps before n are constrained.
The continuous SDE operator, its smoothness and its expansion still need
to be constructed to deduce the paper's unconditional assertion. -/
theorem optimizer_conditional_weak_approximation {d : ℕ} (method : UpdateKind)
    (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ η : ℝ, 0 ≤ η →
      ∀ S : BoundedObservable (EucSpace d) → BoundedObservable (EucSpace d),
      ∀ φ : BoundedObservable (EucSpace d), ∀ R : ℝ, 0 < R → ∀ K : ℝ, 0 ≤ K →
        ∀ T : ℝ, ∀ n : ℕ, (n : ℝ) * η ≤ T →
        (∀ j : ℕ, j < n → ContDiff ℝ 6 (((S^[j]) φ).val) ∧
          ∀ i ≤ 6, ∀ x, ‖iteratedFDeriv ℝ i (((S^[j]) φ).val) x‖ ≤ R) →
        (∀ j : ℕ, j < n → ∀ x,
          |(S ((S^[j]) φ)).val x - ((S^[j]) φ).val x -
            η * diffusionGenerator method η B f σ (((S^[j]) φ).val) x| ≤ K * η ^ 2) →
        ∀ x : EucSpace d,
          |(∫ y, φ.val y ∂discreteLaw method η B f σ x n) -
            ((S^[n]) φ).val x| ≤ (C * R + K) * T * η := by
  obtain ⟨C, hC, hstep⟩ := sgd_sign_scaled_local_weak_consistency method B f σ hB hmodel
  refine ⟨C, hC, ?_⟩
  intro η hη S φ R hR K hK T n horizon hprop hS x
  have hf : ContDiff ℝ 1 f := hmodel.1.of_le (by norm_num)
  have hσ : Continuous σ := hmodel.2.1.continuous
  let A := discreteExpectationOperator method η B f σ hf hσ
  have hcontract : ∀ ψ χ : BoundedObservable (EucSpace d), ∀ r : ℝ, 0 ≤ r →
      (∀ y, |ψ.val y - χ.val y| ≤ r) → ∀ y, |(A ψ).val y - (A χ).val y| ≤ r := by
    intro ψ χ r hr hclose y
    exact discreteExpectationOperator_contraction method η B f σ hf hσ ψ χ r hclose y
  have hlocal (j : ℕ) (hj : j < n) (y : EucSpace d) :
      |(A ((S^[j]) φ)).val y - (S ((S^[j]) φ)).val y| ≤ (C * R + K) * η ^ 2 := by
    let q := ((S^[j]) φ).val
    let v := q y + η * diffusionGenerator method η B f σ q y
    have ha : |(A ((S^[j]) φ)).val y - v| ≤ C * R * η ^ 2 := by
      simpa only [A, discreteExpectationOperator, v, q, sub_add_eq_sub_sub] using
        hstep η hη R hR q (hprop j hj).1 (hprop j hj).2 y
    have hs : |v - (S ((S^[j]) φ)).val y| ≤ K * η ^ 2 := by
      rw [abs_sub_comm]
      simpa only [v, q, sub_add_eq_sub_sub] using hS j hj y
    calc
      |(A ((S^[j]) φ)).val y - (S ((S^[j]) φ)).val y| ≤
          |(A ((S^[j]) φ)).val y - v| + |v - (S ((S^[j]) φ)).val y| := abs_sub_le _ _ _
      _ ≤ C * R * η ^ 2 + K * η ^ 2 := add_le_add ha hs
      _ = (C * R + K) * η ^ 2 := by ring
  rw [← iteratedExpectation_eq_discreteLaw method η B f σ hf hσ φ n x]
  exact finite_horizon_weak_error A S φ (C * R + K) η T
    (add_nonneg (mul_nonneg hC hR.le) hK) hη
    hcontract n horizon hlocal x

/-- Joint nonvacuity of the model and comparison hypotheses,
Section 4.3: flat loss, positive unit noise, the identity comparison and
zero test, with a positive step on a nonzero finite horizon. -/
example : ∃ (f : EucSpace 1 → ℝ) (σ : EucSpace 1 → EucSpace 1)
    (S : BoundedObservable (EucSpace 1) → BoundedObservable (EucSpace 1))
    (φ : BoundedObservable (EucSpace 1)),
    0 < (1 : ℕ) ∧ RegularGaussianModel f σ ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧
    (10 : ℝ) * (1 / 1000) ≤ 1 ∧
    (∀ j : ℕ, j < 10 → ContDiff ℝ 6 (((S^[j]) φ).val) ∧
      ∀ i ≤ 6, ∀ x, ‖iteratedFDeriv ℝ i (((S^[j]) φ).val) x‖ ≤ 1) ∧
    (∀ j : ℕ, j < 10 → ∀ x,
      |(S ((S^[j]) φ)).val x - ((S^[j]) φ).val x -
        (1 / 1000) * diffusionGenerator .sign (1 / 1000) 1 f σ (((S^[j]) φ).val) x| ≤
        (0 : ℝ) * (1 / 1000) ^ 2) := by
  let φ : BoundedObservable (EucSpace 1) := ⟨fun _ => 0, measurable_const, 0, by simp⟩
  refine ⟨fun _ => 0, fun _ => WithLp.toLp 2 (fun _ : Fin 1 => 1), id, φ,
    by norm_num, regularGaussianModel_flat 1, by norm_num, by norm_num,
    le_rfl, by norm_num, ?_, ?_⟩
  · intro j hj
    simp only [Function.iterate_id, id_eq]
    refine ⟨contDiff_const, ?_⟩
    intro k hk x
    simp [φ]
  · intro j hj x
    simp [φ, diffusionGenerator]

end Transformer.BatchSize
