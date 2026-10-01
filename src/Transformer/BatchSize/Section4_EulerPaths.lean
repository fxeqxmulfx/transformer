/-
# Continuous Brownian interpolation of finite Euler chains

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Each interval uses the left state's frozen coefficients and the actual
Brownian path. The interpolation is constant before zero and after the
finite chain's final time.
-/

import Transformer.BatchSize.Section4_EulerGlobalStability
import Transformer.BatchSize.Section4_BrownianPaths

open MeasureTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The value of the continuous Brownian Euler interpolation,
Section 4.3 (2)--(3). Clipped endpoints give a continuous formula on all
real times, without a discontinuous floor-time selector. -/
def eulerPathValue {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ)
    (ω : BrownianSample d) (u : ℝ) : EucSpace d :=
  x₀ + ∑ j ∈ Finset.range n,
    ((((min u.toNNReal (t (j + 1)) : ℝ≥0) : ℝ) - min u.toNNReal (t j)) •
      b (eulerChain b a t x₀ j ω) + WithLp.toLp 2 (fun k =>
        a (eulerChain b a t x₀ j ω) k *
          brownianIncrement k (min u.toNNReal (t j)) (min u.toNNReal (t (j + 1))) ω))

/-- Every interpolated Euler sample is an actual continuous path,
Section 4.3 (2)--(3). This uses continuity of the constructed Brownian
paths; no continuity assumption on an unconstructed SDE is introduced. -/
theorem eulerPathValue_continuous {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ)
    (ω : BrownianSample d) : Continuous (eulerPathValue b a t x₀ n ω) := by
  apply continuous_const.add
  apply continuous_finsetSum
  intro j hj
  have hc r : Continuous (fun u : ℝ => min u.toNNReal (t r)) :=
    continuous_real_toNNReal.min continuous_const
  have hd : Continuous (fun u : ℝ =>
      (((min u.toNNReal (t (j + 1)) : ℝ≥0) : ℝ) - min u.toNNReal (t j)) •
        b (eulerChain b a t x₀ j ω)) :=
    ((NNReal.continuous_coe.comp (hc (j + 1))).sub
      (NNReal.continuous_coe.comp (hc j))).smul continuous_const
  apply hd.add
  apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp
  apply continuous_pi
  intro k
  have hB : Continuous (fun r => coordinateBrownian k r ω) :=
    (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp (vectorBrownian_continuous d ω)
  exact continuous_const.mul ((hB.comp (hc (j + 1))).sub (hB.comp (hc j)))

/-- The finite Euler approximation as a continuous sample path,
Section 4.3 (2)--(3). -/
def eulerPath {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ)
    (ω : BrownianSample d) : DiffusionPath d :=
  ⟨eulerPathValue b a t x₀ n ω, eulerPathValue_continuous b a t x₀ n ω⟩

/-- The actual finite Euler path always starts at its prescribed state,
Section 4.3 (2)--(3). -/
theorem eulerPath_zero {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ)
    (ω : BrownianSample d) : eulerPath b a t x₀ n ω 0 = x₀ := by
  have hz : WithLp.toLp 2 (fun _ : Fin d => (0 : ℝ)) = (0 : EucSpace d) := by
    ext k
    rfl
  simp [eulerPath, eulerPathValue, brownianIncrement, hz]

/-- Unfolding an Euler chain into its actual finite stochastic sum,
Section 4.3 (2)--(3). -/
theorem eulerChain_eq_sum {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ)
    (ω : BrownianSample d) :
    eulerChain b a t x₀ n ω = x₀ + ∑ j ∈ Finset.range n,
      (((t (j + 1) : ℝ) - t j) • b (eulerChain b a t x₀ j ω) +
        WithLp.toLp 2 (fun k => a (eulerChain b a t x₀ j ω) k *
          brownianIncrement k (t j) (t (j + 1)) ω)) := by
  induction n with
  | zero => simp [eulerChain]
  | succ n ih =>
    change eulerChain b a t x₀ n ω +
      ((t (n + 1) : ℝ) - t n) • b (eulerChain b a t x₀ n ω) + _ = _
    rw [Finset.sum_range_succ, ← add_assoc, ← ih]
    abel

/-- The continuous interpolation agrees with the computed Euler
chain at its final grid time, Section 4.3 (2)--(3). -/
theorem eulerPath_endpoint {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (n : ℕ) (ω : BrownianSample d) :
    eulerPath b a t x₀ n ω (t n) = eulerChain b a t x₀ n ω := by
  rw [eulerChain_eq_sum]
  change eulerPathValue b a t x₀ n ω (t n) = _
  unfold eulerPathValue
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  have hjn := Finset.mem_range.mp hj
  simp only [Real.toNNReal_coe, min_eq_right (hmono hjn.le),
    min_eq_right (hmono (by omega : j + 1 ≤ n))]

/-- Measurability of the continuous Euler path law's sampler,
Section 4.3 (2)--(3), from its actual state and Brownian evaluations. -/
theorem eulerPath_measurable {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (n : ℕ) : Measurable (eulerPath b a t x₀ n) := by
  apply ContinuousMap.measurable_iff_eval.mpr
  intro u
  apply measurable_const.add
  apply Finset.measurable_fun_sum
  intro j hj
  have hX := (eulerChain_adapted b a hb ha t hmono x₀ j).mono ((brownianFiltration d).le (t j))
  have hd : Measurable (fun ω =>
      (((min u.toNNReal (t (j + 1)) : ℝ≥0) : ℝ) - min u.toNNReal (t j)) •
        b (eulerChain b a t x₀ j ω)) :=
    (hb.comp_stronglyMeasurable hX).measurable.const_smul
      (((min u.toNNReal (t (j + 1)) : ℝ≥0) : ℝ) - min u.toNNReal (t j))
  apply hd.add
  apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.measurable.comp
  apply Measurable.of_eval
  intro k
  exact ((ha k).comp_stronglyMeasurable hX).measurable.mul
    ((coordinateBrownian_measurable k _).sub (coordinateBrownian_measurable k _))

/-- Joint nonvacuity of interpolation hypotheses, Section 4.3:
linear drift, constant diagonal amplitudes and increasing unit times. -/
example : Continuous (id : EucSpace 2 → EucSpace 2) ∧
    (∀ k : Fin 2, Continuous (fun _ : EucSpace 2 => (k : ℝ) + 1)) ∧
    Monotone (fun n : ℕ => (n : ℝ≥0)) :=
  ⟨continuous_id, fun _ => continuous_const,
    fun i j h => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast h⟩

end Transformer.BatchSize
