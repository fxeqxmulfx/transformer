/-
# Coupling an Euler grid with its refinement into pairs

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The coarse chain is evaluated also at the intervening fine times, with
coefficients frozen at the preceding coarse state. Both chains use the
same actual vector Brownian sample.
-/

import Transformer.BatchSize.Section4_EulerStepBounds
import Transformer.BatchSize.Section4_EulerFrozenStability

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The coarse grid obtained by retaining every second fine time,
Section 4.3 (2)--(3). -/
def pairedGrid (t : ℕ → ℝ≥0) (n : ℕ) : ℝ≥0 := t (2 * n)

/-- The actual coarse Euler state, Brownian-interpolated to a fine
time using its preceding coarse state, Section 4.3 (2)--(3). -/
def pairedEulerChain {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d)
    (n : ℕ) : BrownianSample d → EucSpace d :=
  brownianEulerStep b a (t (2 * (n / 2))) (t n)
    (eulerChain b a (pairedGrid t) x₀ (n / 2))

/-- An Euler update over a zero interval leaves its state unchanged,
Section 4.3 (2)--(3). This holds pointwise, including on Brownian null sets. -/
theorem brownianEulerStep_same_time {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (s : ℝ≥0) (X : BrownianSample d → EucSpace d) :
    brownianEulerStep b a s s X = X := by
  ext ω k
  simp [brownianEulerStep, brownianKick, brownianIncrement]

/-- Splitting a Brownian update while retaining its original frozen
coefficients gives the same full update, Section 4.3 (2)--(3). -/
theorem brownianEulerStep_split {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (r s t : ℝ≥0)
    (X : BrownianSample d → EucSpace d) :
    frozenEulerStep b a s t (brownianEulerStep b a r s X) X =
      brownianEulerStep b a r t X := by
  ext ω k
  simp only [frozenEulerStep, brownianEulerStep, brownianKick, brownianIncrement,
    PiLp.add_apply, PiLp.smul_apply, smul_eq_mul]
  ring

/-- At each even fine time the interpolation is exactly its coarse
Euler state, Section 4.3 (2)--(3). -/
theorem pairedEulerChain_even {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ) :
    pairedEulerChain b a t x₀ (2 * n) = eulerChain b a (pairedGrid t) x₀ n := by
  have hn : 2 * n / 2 = n := by omega
  simp only [pairedEulerChain, hn, brownianEulerStep_same_time]

/-- At an odd fine time the coarse interpolation has made its first
partial Brownian update, Section 4.3 (2)--(3). -/
theorem pairedEulerChain_odd {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ) :
    pairedEulerChain b a t x₀ (2 * n + 1) =
      brownianEulerStep b a (t (2 * n)) (t (2 * n + 1))
        (eulerChain b a (pairedGrid t) x₀ n) := by
  have hn : (2 * n + 1) / 2 = n := by omega
  simp only [pairedEulerChain, hn]

/-- The interpolated coarse chain satisfies the genuine frozen-step
recurrence on the fine grid, Section 4.3 (2)--(3). -/
theorem pairedEulerChain_succ {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ) :
    pairedEulerChain b a t x₀ (n + 1) = frozenEulerStep b a (t n) (t (n + 1))
      (pairedEulerChain b a t x₀ n) (eulerChain b a (pairedGrid t) x₀ (n / 2)) := by
  obtain ⟨m, hm | hm⟩ := Nat.even_or_odd' n
  · subst n
    have hd : 2 * m / 2 = m := by omega
    rw [pairedEulerChain_odd, pairedEulerChain_even, hd]
    rfl
  · subst n
    have hd : (2 * m + 1) / 2 = m := by omega
    have he : 2 * m + 1 + 1 = 2 * (m + 1) := by omega
    rw [he, pairedEulerChain_even, pairedEulerChain_odd, hd]
    exact (brownianEulerStep_split b a (t (2 * m)) (t (2 * m + 1)) (t (2 * (m + 1)))
      (eulerChain b a (pairedGrid t) x₀ m)).symm

/-- Monotonicity of the actual retained coarse grid,
Section 4.3 (2)--(3). -/
theorem pairedGrid_monotone (t : ℕ → ℝ≥0) (hmono : Monotone t) :
    Monotone (pairedGrid t) :=
  fun _ _ hij => hmono (Nat.mul_le_mul_left 2 hij)

/-- Every interpolated coarse state is measurable in the whole
Brownian past at its fine time, Section 4.3 (2)--(3). -/
theorem pairedEulerChain_adapted {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (n : ℕ) :
    StronglyMeasurable[brownianFiltration d (t n)] (pairedEulerChain b a t x₀ n) :=
  brownianEulerStep_adapted b a hb ha _ _ (hmono (by omega)) _
    (eulerChain_adapted b a hb ha (pairedGrid t) (pairedGrid_monotone t hmono) x₀ (n / 2))

/-- Every interpolated coarse state is square integrable,
Section 4.3 (2)--(3). -/
theorem pairedEulerChain_memLp {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (t : ℕ → ℝ≥0) (hmono : Monotone t) (x₀ : EucSpace d) (n : ℕ) :
    MemLp (pairedEulerChain b a t x₀ n) 2 (brownianNoiseLaw d) :=
  brownianEulerStep_memLp b a Kb Ka hb ha _ _ (hmono (by omega)) _
    (eulerChain_adapted b a hb.continuous (fun k => (ha k).continuous)
      (pairedGrid t) (pairedGrid_monotone t hmono) x₀ (n / 2))
    (eulerChain_memLp b a Kb Ka hb ha (pairedGrid t) (pairedGrid_monotone t hmono) x₀ (n / 2))

/-- Joint nonvacuity of the retained-grid and interpolation hypotheses,
Section 4.3: linear drift, constant amplitudes and unit fine times. -/
example : LipschitzWith 1 (id : EucSpace 2 → EucSpace 2) ∧
    (∀ k : Fin 2, LipschitzWith 0 (fun _ : EucSpace 2 => (k : ℝ) + 1)) ∧
    Monotone (fun n : ℕ => (n : ℝ≥0)) :=
  ⟨LipschitzWith.id, fun _ => LipschitzWith.const _,
    fun i j hij => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast hij⟩

end Transformer.BatchSize
