/-
# Mean-square convergence estimate for paired Euler refinements

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
This is an unconditional comparison of two actual Brownian Euler chains.
The error bound vanishes with the mesh and retains the small-noise factor
in the amplitudes, needed for the optimizer's O(eta) strong error.
-/

import Transformer.BatchSize.Section4_PairedEuler

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual lag of the interpolated coarse chain has the bounded
Euler displacement estimate, Section 4.3 (2)--(3). -/
theorem pairedEulerChain_lag_bound {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka M A ε : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (hmesh : ∀ j, (t (j + 1) : ℝ) - t j ≤ ε) (x₀ : EucSpace d) (n : ℕ) :
    (∫ ω, ‖pairedEulerChain b a t x₀ n ω -
      eulerChain b a (pairedGrid t) x₀ (n / 2) ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
        M ^ 2 * (ε : ℝ) ^ 2 + (d : ℝ) * A ^ 2 * ε := by
  have hleft : t (2 * (n / 2)) ≤ t n := hmono (by omega)
  have hδ : 0 ≤ (t n : ℝ) - t (2 * (n / 2)) :=
    sub_nonneg.mpr (NNReal.coe_le_coe.mpr hleft)
  have hδε : (t n : ℝ) - t (2 * (n / 2)) ≤ ε := by
    calc
      _ ≤ (t (2 * (n / 2) + 1) : ℝ) - t (2 * (n / 2)) :=
        sub_le_sub_right (NNReal.coe_le_coe.mpr (hmono (by omega))) _
      _ ≤ _ := hmesh _
  have h := brownianEulerStep_displacement_bound b a Kb Ka M A hb ha hbM haA _ _ hleft
    (eulerChain b a (pairedGrid t) x₀ (n / 2))
    (eulerChain_adapted b a hb.continuous (fun k => (ha k).continuous)
      (pairedGrid t) (pairedGrid_monotone t hmono) x₀ (n / 2))
    (eulerChain_memLp b a Kb Ka hb ha (pairedGrid t) (pairedGrid_monotone t hmono) x₀ (n / 2))
  exact h.trans (by gcongr)

/-- Refining every coarse interval into two fine intervals gives a
mean-square error of order M^2 epsilon^2 + d A^2 epsilon, uniformly in
the number of steps on a finite horizon, Section 4.3 (2)--(3).
No limit process or comparison assumption is used in this estimate. -/
theorem eulerChain_paired_refinement_bound {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka M A ε : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (t : ℕ → ℝ≥0) (hmono : Monotone t) (hε : ε ≤ 1)
    (hmesh : ∀ j, (t (j + 1) : ℝ) - t j ≤ ε) (x₀ : EucSpace d) (n : ℕ) :
    let C : ℝ := 4 * Kb ^ 2 + 2 * (d : ℝ) * Ka ^ 2
    (∫ ω, ‖eulerChain b a t x₀ n ω - pairedEulerChain b a t x₀ n ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
      C * (M ^ 2 * (ε : ℝ) ^ 2 + (d : ℝ) * A ^ 2 * ε) * ((t n : ℝ) - t 0) *
        Real.exp ((1 + C) * ((t n : ℝ) - t 0)) := by
  let C : ℝ := 4 * Kb ^ 2 + 2 * (d : ℝ) * Ka ^ 2
  let L : ℝ := M ^ 2 * (ε : ℝ) ^ 2 + (d : ℝ) * A ^ 2 * ε
  let E (j : ℕ) : ℝ := ∫ ω, ‖eulerChain b a t x₀ j ω -
    pairedEulerChain b a t x₀ j ω‖ ^ 2 ∂brownianNoiseLaw d
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hL : 0 ≤ L := by dsimp [L]; positivity
  have hrec (j : ℕ) : E (j + 1) ≤
      (1 + (1 + C) * ((t (j + 1) : ℝ) - t j)) * E j + C * ((t (j + 1) : ℝ) - t j) * L := by
    let X := eulerChain b a t x₀ j
    let Y := pairedEulerChain b a t x₀ j
    let U := eulerChain b a (pairedGrid t) x₀ (j / 2)
    have hδ : 0 ≤ (t (j + 1) : ℝ) - t j :=
      sub_nonneg.mpr (NNReal.coe_le_coe.mpr (hmono (Nat.le_succ j)))
    have hU : StronglyMeasurable[brownianFiltration d (t j)] U :=
      (eulerChain_adapted b a hb.continuous (fun k => (ha k).continuous)
        (pairedGrid t) (pairedGrid_monotone t hmono) x₀ (j / 2)).mono
          ((brownianFiltration d).mono (hmono (by omega)))
    have h := frozenEulerStep_stability b a Kb Ka hb ha (t j) (t (j + 1))
      (hmono (Nat.le_succ j)) ((hmesh j).trans (NNReal.coe_le_coe.mpr hε)) X Y U
      (eulerChain_adapted b a hb.continuous (fun k => (ha k).continuous) t hmono x₀ j)
      (pairedEulerChain_adapted b a hb.continuous (fun k => (ha k).continuous) t hmono x₀ j) hU
      (eulerChain_memLp b a Kb Ka hb ha t hmono x₀ j)
      (pairedEulerChain_memLp b a Kb Ka hb ha t hmono x₀ j)
      (eulerChain_memLp b a Kb Ka hb ha (pairedGrid t) (pairedGrid_monotone t hmono) x₀ (j / 2))
    have hfac : 1 + 4 * (Kb : ℝ) ^ 2 + 2 * (d : ℝ) * Ka ^ 2 = 1 + C := by
      dsimp [C]
      ring
    rw [hfac] at h
    have hlag := pairedEulerChain_lag_bound b a Kb Ka M A ε hb ha hbM haA t hmono hmesh x₀ j
    have heq : eulerChain b a t x₀ (j + 1) = brownianEulerStep b a (t j) (t (j + 1)) X := rfl
    dsimp only [E]
    rw [heq, pairedEulerChain_succ]
    exact h.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left hlag (mul_nonneg hC hδ)))
  change E n ≤ C * L * ((t n : ℝ) - t 0) * Real.exp ((1 + C) * ((t n : ℝ) - t 0))
  induction n with
  | zero => simp [E, pairedEulerChain, brownianEulerStep_same_time, eulerChain]
  | succ n ih =>
    let δ : ℝ := (t (n + 1) : ℝ) - t n
    let u : ℝ := (t n : ℝ) - t 0
    have hδ : 0 ≤ δ := sub_nonneg.mpr (NNReal.coe_le_coe.mpr (hmono (Nat.le_succ n)))
    have hu : 0 ≤ u := sub_nonneg.mpr (NNReal.coe_le_coe.mpr (hmono (Nat.zero_le n)))
    have hE : 0 ≤ E n := integral_nonneg fun _ => sq_nonneg _
    have hf : 1 + (1 + C) * δ ≤ Real.exp ((1 + C) * δ) := by
      simpa only [add_comm] using Real.add_one_le_exp ((1 + C) * δ)
    have hexp : 1 ≤ Real.exp ((1 + C) * (u + δ)) :=
      Real.one_le_exp_iff.mpr (mul_nonneg (by linarith) (add_nonneg hu hδ))
    calc
      _ ≤ (1 + (1 + C) * δ) * E n + C * δ * L := hrec n
      _ ≤ Real.exp ((1 + C) * δ) * E n + C * δ * L :=
        add_le_add (mul_le_mul_of_nonneg_right hf hE) le_rfl
      _ ≤ Real.exp ((1 + C) * δ) * (C * L * u * Real.exp ((1 + C) * u)) + C * δ * L :=
        add_le_add (mul_le_mul_of_nonneg_left ih (Real.exp_pos _).le) le_rfl
      _ ≤ Real.exp ((1 + C) * δ) * (C * L * u * Real.exp ((1 + C) * u)) +
          C * δ * L * Real.exp ((1 + C) * (u + δ)) :=
        add_le_add le_rfl (le_mul_of_one_le_right (by positivity) hexp)
      _ = C * L * ((t (n + 1) : ℝ) - t 0) * Real.exp ((1 + C) * ((t (n + 1) : ℝ) - t 0)) := by
        have he : (t (n + 1) : ℝ) - t 0 = u + δ := by dsimp [u, δ]; ring
        rw [he, mul_add (1 + C) u δ, Real.exp_add]
        ring

/-- Joint nonvacuity of refinement hypotheses, Section 4.3:
zero drift, unit amplitudes and a half-unit mesh. -/
example : LipschitzWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, LipschitzWith 0 (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : ℝ≥0)) ∧
    (∀ (x : EucSpace 1) (k : Fin 1), |(fun _ : EucSpace 1 => fun _ : Fin 1 => (1 : ℝ)) x k| ≤
      (1 : ℝ≥0)) ∧ Monotone (fun n : ℕ => (n : ℝ≥0) / 2) ∧
    (1 / 2 : ℝ≥0) ≤ 1 ∧
    (∀ j : ℕ, ((j + 1 : ℕ) : ℝ) / 2 - (j : ℝ) / 2 ≤ (1 / 2 : ℝ≥0)) := by
  refine ⟨LipschitzWith.const _, fun _ => LipschitzWith.const _, ?_, by simp, ?_, by norm_num, ?_⟩
  · intro x
    simp
  · intro i j hij
    exact div_le_div_of_nonneg_right (by exact_mod_cast hij) (by norm_num)
  · intro j
    simp only [Nat.cast_add, Nat.cast_one, NNReal.coe_div, NNReal.coe_one, NNReal.coe_ofNat]
    linarith

end Transformer.BatchSize
