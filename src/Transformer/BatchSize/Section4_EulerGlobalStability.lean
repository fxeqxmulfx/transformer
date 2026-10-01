/-
# Stability over finite horizons, uniformly in Euler resolution

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The constants depend on the model and dimension, not the number of grid
steps. For optimizer coefficients they can also be independent of eta in [0,1].
-/

import Transformer.BatchSize.Section4_EulerStability

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The accumulated mean-square Euler stability estimate,
Section 4.3 (2)--(3), on any increasing grid with steps at most one. -/
theorem eulerChain_global_stability {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (hstep : ∀ n, (t (n + 1) : ℝ) - t n ≤ 1) (x₀ y₀ : EucSpace d) (n : ℕ) :
    (∫ ω, ‖eulerChain b a t x₀ n ω - eulerChain b a t y₀ n ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
      Real.exp ((2 * Kb + Kb ^ 2 + (d : ℝ) * Ka ^ 2) * ((t n : ℝ) - t 0)) * ‖x₀ - y₀‖ ^ 2 := by
  let C : ℝ := 2 * Kb + Kb ^ 2 + (d : ℝ) * Ka ^ 2
  induction n with
  | zero => simp [eulerChain]
  | succ n ih =>
    let X := eulerChain b a t x₀ n
    let Y := eulerChain b a t y₀ n
    let δ : ℝ := (t (n + 1) : ℝ) - t n
    have hδ : 0 ≤ δ := sub_nonneg.mpr (NNReal.coe_le_coe.mpr (hmono (Nat.le_succ n)))
    have hδ1 : δ ≤ 1 := hstep n
    have he := brownianEulerStep_stability b a Kb Ka hb ha (t n) (t (n + 1))
      (hmono (Nat.le_succ n)) X Y
      (eulerChain_adapted _ _ hb.continuous (fun k => (ha k).continuous) t hmono x₀ n)
      (eulerChain_adapted _ _ hb.continuous (fun k => (ha k).continuous) t hmono y₀ n)
      (eulerChain_memLp _ _ Kb Ka hb ha t hmono x₀ n)
      (eulerChain_memLp _ _ Kb Ka hb ha t hmono y₀ n)
    have hfactor : (1 + δ * Kb) ^ 2 + (d : ℝ) * Ka ^ 2 * δ ≤ Real.exp (C * δ) := by
      calc
        _ ≤ 1 + C * δ := by
          dsimp [C]
          nlinarith [mul_nonneg (sq_nonneg (Kb : ℝ)) (mul_nonneg hδ (sub_nonneg.mpr hδ1))]
        _ ≤ _ := by simpa only [add_comm] using Real.add_one_le_exp (C * δ)
    have hnonneg : 0 ≤ ∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d :=
      integral_nonneg fun _ => sq_nonneg _
    calc
      _ ≤ ((1 + δ * Kb) ^ 2 + (d : ℝ) * Ka ^ 2 * δ) *
          (∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) := he
      _ ≤ Real.exp (C * δ) * (∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) :=
        mul_le_mul_of_nonneg_right hfactor hnonneg
      _ ≤ Real.exp (C * δ) *
          (Real.exp (C * ((t n : ℝ) - t 0)) * ‖x₀ - y₀‖ ^ 2) :=
        mul_le_mul_of_nonneg_left ih (Real.exp_pos _).le
      _ = _ := by
        rw [← mul_assoc, ← Real.exp_add]
        congr 2
        dsimp [C, δ]
        ring

/-- Joint nonvacuity of the global stability hypotheses, Section 4.3:
linear drift, positive constant amplitudes, and a grid with unit steps. -/
example : LipschitzWith 1 (id : EucSpace 2 → EucSpace 2) ∧
    (∀ k : Fin 2, LipschitzWith 0 (fun _ : EucSpace 2 => (k : ℝ) + 1)) ∧
    Monotone (fun n : ℕ => (n : ℝ≥0)) ∧
    (∀ n : ℕ, ((n + 1 : ℕ) : ℝ) - n ≤ 1) := by
  refine ⟨LipschitzWith.id, fun _ => LipschitzWith.const _,
    fun i j h => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast h, ?_⟩
  intro n
  simp

/-- A single horizon-stability constant works for the actual SGD and
SignSGD Euler chains for every eta in [0,1], Section 4.3 (2)--(3), Theorem 1. -/
theorem optimizerEulerChain_global_stability {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d) (hmodel : RegularGaussianModel f σ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ η : ℝ, 0 ≤ η → η ≤ 1 →
      ∀ t : ℕ → ℝ≥0, Monotone t → (∀ n, (t (n + 1) : ℝ) - t n ≤ 1) →
        ∀ x₀ y₀ : EucSpace d, ∀ n : ℕ,
          (∫ ω, ‖optimizerEulerChain method η B f σ t x₀ n ω -
            optimizerEulerChain method η B f σ t y₀ n ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
            Real.exp (C * ((t n : ℝ) - t 0)) * ‖x₀ - y₀‖ ^ 2 := by
  obtain ⟨Kb, hb⟩ := diffusionDrift_lipschitz method B f σ hmodel
  obtain ⟨Ka, ha⟩ := diffusionNoiseScale_lipschitz method B f σ hmodel
  refine ⟨2 * Kb + Kb ^ 2 + (d : ℝ) * Ka ^ 2, by positivity, ?_⟩
  intro η hη hη1 t ht hstep x₀ y₀ n
  have ha' k : LipschitzWith Ka (fun x => diffusionNoiseScale method η B f σ x k) := by
    apply LipschitzWith.of_dist_le_mul
    intro x y
    have hK : ((‖Real.sqrt η‖₊ * Ka : ℝ≥0) : ℝ) ≤ Ka := by
      change ‖Real.sqrt η‖ * (Ka : ℝ) ≤ Ka
      rw [Real.norm_eq_abs, abs_of_nonneg (Real.sqrt_nonneg _)]
      exact mul_le_of_le_one_left Ka.coe_nonneg (Real.sqrt_le_one.mpr hη1)
    exact ((ha η hη k).dist_le_mul x y).trans (mul_le_mul_of_nonneg_right hK dist_nonneg)
  exact eulerChain_global_stability _ _ Kb Ka hb ha' t ht hstep x₀ y₀ n

/-- Joint nonvacuity of model, rate and grid conditions in the optimizer
bound, Section 4.3 (2)--(3). -/
example : RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (1 / 1000 : ℝ) ≤ 1 ∧
    Monotone (fun n : ℕ => (n : ℝ≥0)) ∧
    (∀ n : ℕ, ((n + 1 : ℕ) : ℝ) - n ≤ 1) := by
  exact ⟨regularGaussianModel_flat 2, by norm_num, by norm_num,
    fun i j h => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast h, fun n => by simp⟩

end Transformer.BatchSize
