/-
# Fourth-moment displacement of actual Brownian Euler chains

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Bounded drift and amplitudes give a moment estimate independent of the
Euler mesh. The coefficients are evaluated at the actual random states.
-/

import Transformer.BatchSize.Section4_EulerDriftSum

open MeasureTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- A fourth-power norm inequality used to combine the drift and
Brownian parts in Section 4.3 (2)--(3). -/
theorem norm_add_four_le {E : Type*} [SeminormedAddCommGroup E] (x y : E) :
    ‖x + y‖ ^ 4 ≤ 8 * (‖x‖ ^ 4 + ‖y‖ ^ 4) := by
  have h2 : ‖x + y‖ ^ 2 ≤ 2 * (‖x‖ ^ 2 + ‖y‖ ^ 2) := by
    nlinarith [norm_add_le x y, norm_nonneg (x + y), norm_nonneg x, norm_nonneg y,
      sq_nonneg (‖x‖ - ‖y‖)]
  have h4 := pow_le_pow_left₀ (sq_nonneg ‖x + y‖) h2 2
  nlinarith [sq_nonneg (‖x‖ ^ 2 - ‖y‖ ^ 2)]

/-- Every actual Euler state has finite fourth moment for bounded
continuous coefficients, Section 4.3 (2)--(3). -/
theorem eulerChain_memLp_four {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (M A : ℝ≥0)
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (t : ℕ → ℝ≥0) (hmono : Monotone t) (x₀ : EucSpace d) (n : ℕ) :
    MemLp (eulerChain b a t x₀ n) 4 (brownianNoiseLaw d) := by
  induction n with
  | zero => exact memLp_const x₀
  | succ n ih =>
    have hX := eulerChain_adapted b a hb ha t hmono x₀ n
    have hD : MemLp (fun ω => b (eulerChain b a t x₀ n ω)) 4 (brownianNoiseLaw d) :=
      MemLp.of_bound ((hb.comp_stronglyMeasurable hX).mono
        ((brownianFiltration d).le (t n))).aestronglyMeasurable M
          (Eventually.of_forall fun ω => hbM (eulerChain b a t x₀ n ω))
    have hN : MemLp (fun ω => WithLp.toLp 2 (fun k => a (eulerChain b a t x₀ n ω) k *
        brownianIncrement k (t n) (t (n + 1)) ω)) 4 (brownianNoiseLaw d) := by
      apply MemLp.of_eval_piLp
      intro k
      exact brownianIncrement_bounded_memLp_four k (t n) (t (n + 1)) _
        ((ha k).comp_stronglyMeasurable hX) A (fun ω => haA (eulerChain b a t x₀ n ω) k)
    exact (ih.add (hD.const_smul ((t (n + 1) : ℝ) - t n))).add hN

/-- Bounded continuous Euler coefficients give fourth-moment
displacement at most 8 M^4 duration^4 + 24 d^2 A^4 duration^2,
Section 4.3 (2)--(3). The bound is independent of grid resolution. -/
theorem eulerChain_fourthMoment_bound {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (M A : ℝ≥0)
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (t : ℕ → ℝ≥0) (hmono : Monotone t) (x₀ : EucSpace d) (n : ℕ) :
    (∫ ω, ‖eulerChain b a t x₀ n ω - x₀‖ ^ 4 ∂brownianNoiseLaw d) ≤
      8 * M ^ 4 * ((t n : ℝ) - t 0) ^ 4 +
        24 * (d : ℝ) ^ 2 * A ^ 4 * ((t n : ℝ) - t 0) ^ 2 := by
  let D := eulerDriftSum b a t x₀ n
  let H (j : ℕ) (ω : BrownianSample d) : EucSpace d := WithLp.toLp 2 (a (eulerChain b a t x₀ j ω))
  let N := vectorItoSum t H n
  have hH (j : ℕ) : StronglyMeasurable[brownianFiltration d (t j)] (H j) := by
    have hX := eulerChain_adapted b a hb ha t hmono x₀ j
    let : MeasurableSpace (BrownianSample d) := brownianFiltration d (t j)
    apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp_stronglyMeasurable
    apply Measurable.stronglyMeasurable
    exact Measurable.of_eval fun k => ((ha k).comp_stronglyMeasurable hX).measurable
  have hHA (j : ℕ) (ω : BrownianSample d) (k : Fin d) : |H j ω k| ≤ A := haA _ k
  have hN4 : MemLp N 4 (brownianNoiseLaw d) := vectorItoSum_memLp_four t H hH A hHA n
  have hD4 : MemLp D 4 (brownianNoiseLaw d) := MemLp.of_bound
    ((eulerDriftSum_adapted b a hb ha t hmono x₀ n).mono
      ((brownianFiltration d).le (t n))).aestronglyMeasurable
    (M * ((t n : ℝ) - t 0)) (Eventually.of_forall (eulerDriftSum_norm_bound b a M hbM t hmono x₀ n))
  have hDb : (∫ ω, ‖D ω‖ ^ 4 ∂brownianNoiseLaw d) ≤ M ^ 4 * ((t n : ℝ) - t 0) ^ 4 := by
    have h := integral_mono (hD4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))
      (integrable_const ((M : ℝ) ^ 4 * ((t n : ℝ) - t 0) ^ 4)) fun ω => by
        simpa only [mul_pow] using pow_le_pow_left₀ (norm_nonneg _)
          (eulerDriftSum_norm_bound b a M hbM t hmono x₀ n ω) 4
    simpa using h
  have hNb := vectorItoSum_fourthMoment_bound t hmono H hH A hHA n
  have heq (ω : BrownianSample d) : eulerChain b a t x₀ n ω - x₀ = D ω + N ω :=
    eulerChain_displacement_eq b a t x₀ n ω
  simp_rw [heq]
  have h := integral_mono ((hD4.add hN4).integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))
    (((hD4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0)).add
      (hN4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))).const_mul 8)
    (fun ω => norm_add_four_le (D ω) (N ω))
  rw [integral_const_mul] at h
  simp only [Pi.add_apply] at h
  rw [integral_add (hD4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))
    (hN4.integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0))] at h
  calc
    _ ≤ 8 * ((∫ ω, ‖D ω‖ ^ 4 ∂brownianNoiseLaw d) +
        (∫ ω, ‖N ω‖ ^ 4 ∂brownianNoiseLaw d)) := h
    _ ≤ 8 * (M ^ 4 * ((t n : ℝ) - t 0) ^ 4 +
        3 * (d : ℝ) ^ 2 * A ^ 4 * ((t n : ℝ) - t 0) ^ 2) := by gcongr
    _ = _ := by ring

/-- Joint nonvacuity of fourth-moment Euler hypotheses, Section 4.3:
bounded constant drift and amplitudes on an increasing half-unit grid. -/
example : Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, Continuous (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : ℝ≥0)) ∧
    (∀ (x : EucSpace 1) (k : Fin 1), |(fun _ : EucSpace 1 => fun _ : Fin 1 => (1 : ℝ)) x k| ≤
      (1 : ℝ≥0)) ∧ Monotone (fun n : ℕ => (n : ℝ≥0) / 2) :=
  ⟨continuous_const, fun _ => continuous_const, by simp, by simp,
    fun i j hij => div_le_div_of_nonneg_right (by exact_mod_cast hij) (by norm_num)⟩

end Transformer.BatchSize
