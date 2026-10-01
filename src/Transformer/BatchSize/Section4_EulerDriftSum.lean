/-
# Separating the bounded drift and martingale part of Euler chains

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The drift sum is bounded by the model bound times elapsed time.
The remaining term is the actual adapted vector Brownian sum.
-/

import Transformer.BatchSize.Section4_EulerPaths
import Transformer.BatchSize.Section4_VectorFourthMoment

open MeasureTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The accumulated state-dependent Euler drift, Section 4.3 (2)--(3). -/
def eulerDriftSum {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ)
    (ω : BrownianSample d) : EucSpace d :=
  ∑ j ∈ Finset.range n, ((t (j + 1) : ℝ) - t j) • b (eulerChain b a t x₀ j ω)

/-- Exact decomposition of the actual Euler displacement into its
drift and adapted Brownian sum, Section 4.3 (2)--(3). -/
theorem eulerChain_displacement_eq {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ)
    (ω : BrownianSample d) :
    eulerChain b a t x₀ n ω - x₀ = eulerDriftSum b a t x₀ n ω +
      vectorItoSum t (fun j ω => WithLp.toLp 2 (a (eulerChain b a t x₀ j ω))) n ω := by
  have hnoise : (∑ j ∈ Finset.range n, WithLp.toLp 2 (fun k =>
      a (eulerChain b a t x₀ j ω) k * brownianIncrement k (t j) (t (j + 1)) ω)) =
      vectorItoSum t (fun j ω => WithLp.toLp 2 (a (eulerChain b a t x₀ j ω))) n ω := by
    ext k
    exact map_sum (PiLp.proj (𝕜 := ℝ) (p := 2) (β := fun _ : Fin d => ℝ) k) _ _
  rw [eulerChain_eq_sum, Finset.sum_add_distrib, hnoise]
  dsimp only [eulerDriftSum]
  abel

/-- The bounded state-dependent drift accumulates at most linearly
in elapsed time, Section 4.3 (2)--(3), on any increasing grid. -/
theorem eulerDriftSum_norm_bound {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (M : ℝ≥0) (hbM : ∀ x, ‖b x‖ ≤ M)
    (t : ℕ → ℝ≥0) (hmono : Monotone t) (x₀ : EucSpace d) (n : ℕ)
    (ω : BrownianSample d) :
    ‖eulerDriftSum b a t x₀ n ω‖ ≤ M * ((t n : ℝ) - t 0) := by
  calc
    _ ≤ ∑ j ∈ Finset.range n, ‖((t (j + 1) : ℝ) - t j) • b (eulerChain b a t x₀ j ω)‖ :=
      norm_sum_le _ _
    _ ≤ ∑ j ∈ Finset.range n, ((t (j + 1) : ℝ) - t j) * M := by
      apply Finset.sum_le_sum
      intro j hj
      have hδ : 0 ≤ (t (j + 1) : ℝ) - t j :=
        sub_nonneg.mpr (NNReal.coe_le_coe.mpr (hmono (Nat.le_succ j)))
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hδ]
      exact mul_le_mul_of_nonneg_left (hbM _) hδ
    _ = _ := by rw [← Finset.sum_mul, Finset.sum_range_sub (fun j => (t j : ℝ)) n]; ring

/-- The accumulated drift is measurable in the joint terminal past,
Section 4.3 (2)--(3). -/
theorem eulerDriftSum_adapted {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (n : ℕ) :
    StronglyMeasurable[brownianFiltration d (t n)] (eulerDriftSum b a t x₀ n) := by
  apply Finset.stronglyMeasurable_fun_sum
  intro j hj
  have hjn := Finset.mem_range.mp hj
  exact ((hb.comp_stronglyMeasurable
    (eulerChain_adapted b a hb ha t hmono x₀ j)).mono
      ((brownianFiltration d).mono (hmono hjn.le))).const_smul ((t (j + 1) : ℝ) - t j)

/-- Joint nonvacuity of drift accumulation hypotheses, Section 4.3:
bounded constant drift and amplitudes on unit grid times. -/
example : Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, Continuous (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : ℝ≥0)) ∧
    Monotone (fun n : ℕ => (n : ℝ≥0)) :=
  ⟨continuous_const, fun _ => continuous_const, by simp,
    fun i j hij => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast hij⟩

end Transformer.BatchSize
