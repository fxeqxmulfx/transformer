/-
# Actual finite stochastic sums are martingales

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
This verifies conditional mean zero for each new adapted stochastic
summand relative to the joint vector filtration.
-/

import Transformer.BatchSize.Section4_ItoSums
import Transformer.BatchSize.Section4_BrownianConditional

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual joint Brownian filtration sampled on an increasing grid,
Section 4.3 (2)--(3). -/
def brownianGridFiltration {d : ℕ} (t : ℕ → ℝ≥0) (hmono : Monotone t) :
    Filtration ℕ (inferInstance : MeasurableSpace (BrownianSample d)) where
  seq := fun n => brownianFiltration d (t n)
  mono' := fun _ _ h => (brownianFiltration d).mono (hmono h)
  le' n := (brownianFiltration d).le (t n)

/-- The actual finite stochastic sums form a martingale in the whole
vector driver's sampled filtration, Section 4.3 (2)--(3). All sums are
integrable by the proved adapted L2 isometry. -/
theorem brownianItoSum_martingale {d : ℕ} (k : Fin d) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (H : ℕ → BrownianSample d → ℝ)
    (hH : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (H n))
    (hHL2 : ∀ n, MemLp (H n) 2 (brownianNoiseLaw d)) :
    Martingale (brownianItoSum k t H) (brownianGridFiltration t hmono) (brownianNoiseLaw d) := by
  apply martingale_of_condExp_sub_eq_zero_nat
  · intro n
    exact brownianItoSum_adapted k t hmono H hH n
  · intro n
    exact (brownianItoSum_memLp k t hmono H hH hHL2 n).integrable (by norm_num)
  · intro n
    have heq : brownianItoSum k t H (n + 1) - brownianItoSum k t H n =
        (fun ω => H n ω * brownianIncrement k (t n) (t (n + 1)) ω) := by
      ext ω
      simp only [Pi.sub_apply, brownianItoSum, Finset.sum_range_succ, add_sub_cancel_left]
    rw [heq]
    exact brownianIncrement_adapted_conditional_mean k (H n) (t n) (t (n + 1))
      (hmono (Nat.le_succ n)) (hH n) (hHL2 n)

/-- Joint nonvacuity of finite-sum martingale hypotheses, Section 4.3:
unit times and coefficients taken from another Brownian coordinate. -/
example : Monotone (fun n : ℕ => (n : ℝ≥0)) ∧
    (∀ n : ℕ, StronglyMeasurable[brownianFiltration 2 n]
      (coordinateBrownian (1 : Fin 2) n)) ∧
    (∀ n : ℕ, MemLp (coordinateBrownian (1 : Fin 2) n) 2 (brownianNoiseLaw 2)) :=
  ⟨fun i j h => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast h,
    fun n => (coordinateBrownian_filtered (1 : Fin 2)).stronglyAdapted n,
    fun n => ((coordinateBrownian_isBrownian (1 : Fin 2)).isGaussianProcess.hasGaussianLaw_eval n).memLp_two⟩

end Transformer.BatchSize
