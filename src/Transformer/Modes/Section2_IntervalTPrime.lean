/-
# The number of modes of a Gaussian KDE — the interval `T'`

`eq:T'` of arXiv:2412.09080v3: `T' = [-√(2 log n - 3 log β), √(2 log n - 3 log β)]`
if `β ≤ n^{2/3}`, and `∅` otherwise.  Two facts about it are used in the proof of
`lem:main-int-phi` on `T'`:

* on `T'` the factor `β^{-3/2} n e^{-t²/2}` of the rate of `eq:int-phi-b` is at
  least `1` (`one_le_rateScale`), since there `e^{-t²/2} ≥ β^{3/2}/n`;
* for all large `β`, `T'` lies in `T` (for the window `ω(β) = √(log log β)`),
  so every uniform estimate proved on `T` holds on `T'`
  (`intervalT'_subset_intervalT`).

Source: arXiv:2412.09080v3, `eq:T'`, proof of `lem:main-int-phi`.
-/

import Mathlib.Analysis.Complex.ExponentialBounds
import Transformer.Modes.Section1_Sketch

open Real Filter
open scoped Topology

namespace Transformer
namespace Modes

/-- The factor `β^{-3/2} n e^{-t²/2}` of the rate `β^{-3/2} n t² e^{-t²/2}` of
`eq:int-phi-b`: the rate is `rateScale n β t * t²`.

Source: arXiv:2412.09080v3, `eq:int-phi-b`. -/
noncomputable def rateScale (n : ℕ) (β t : ℝ) : ℝ :=
  β ^ (-(3 : ℝ) / 2) * n * Real.exp (-(t ^ 2) / 2)

/-- `T'` is measurable. -/
theorem measurableSet_intervalT' (n : ℕ) (β : ℝ) : MeasurableSet (intervalT' n β) := by
  unfold intervalT'
  split_ifs
  · exact measurableSet_Icc
  · exact MeasurableSet.empty

/-- **On `T'`, `β^{-3/2} n e^{-t²/2} ≥ 1`**: there `t² ≤ 2 log n - 3 log β`, so
`e^{-t²/2} ≥ β^{3/2}/n`.

Source: arXiv:2412.09080v3, `eq:T'`, proof of `lem:main-int-phi`. -/
theorem one_le_rateScale {n : ℕ} {β t : ℝ} (hn : 1 ≤ n) (hβ : 1 ≤ β)
    (ht : t ∈ intervalT' n β) : 1 ≤ rateScale n β t := by
  unfold intervalT' at ht
  split_ifs at ht with h
  · have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
    have hβ0 : 0 < β := by linarith
    set a := 2 * Real.log n - 3 * Real.log β with ha
    -- `β ≤ n^{2/3}` is `a ≥ 0`
    have ha0 : 0 ≤ a := by
      have h1 : Real.log β ≤ Real.log ((n : ℝ) ^ ((2 : ℝ) / 3)) := Real.log_le_log hβ0 h
      rw [Real.log_rpow hn0] at h1
      linarith
    have hta : t ^ 2 ≤ a := by
      rw [← Real.sq_sqrt ha0, ← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) (abs_le.mpr ⟨ht.1, ht.2⟩) 2
    have hexp : β ^ ((3 : ℝ) / 2) / n ≤ Real.exp (-(t ^ 2) / 2) := by
      calc β ^ ((3 : ℝ) / 2) / n = Real.exp (-a / 2) := by
            rw [Real.rpow_def_of_pos hβ0, show -a / 2 = 3 / 2 * Real.log β - Real.log n by ring,
              Real.exp_sub, Real.exp_log hn0, mul_comm]
        _ ≤ Real.exp (-(t ^ 2) / 2) := Real.exp_le_exp.mpr (by linarith)
    calc (1 : ℝ) = β ^ (-(3 : ℝ) / 2) * n * (β ^ ((3 : ℝ) / 2) / n) := by
          rw [show -(3 : ℝ) / 2 = -((3 : ℝ) / 2) by ring, Real.rpow_neg hβ0.le]
          field_simp
      _ ≤ rateScale n β t := by unfold rateScale; gcongr
  · exact absurd ht (Set.notMem_empty t)

/-- The hypotheses of `one_le_rateScale` are satisfiable: `n = β = 1`, `t = 0`. -/
example : 1 ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ∈ intervalT' 1 1 := by
  refine ⟨le_rfl, le_rfl, ?_⟩
  unfold intervalT'
  simp only [show (1 : ℝ) ≤ ((1 : ℕ) : ℝ) ^ ((2 : ℝ) / 3) by simp, ite_true]
  exact ⟨by simp, Real.sqrt_nonneg _⟩

/-- **`T' ⊆ T` for all large `β`**, for the window `ω(β) = √(log log β)`:
`√(log log β) ≤ 2 log β` as soon as `β ≥ e`.

Source: arXiv:2412.09080v3, `eq:T`, `eq:T'`. -/
theorem intervalT'_subset_intervalT {n : ℕ} {β : ℝ} (hβ : Real.exp 1 ≤ β) :
    intervalT' n β ⊆ intervalT n β (Real.sqrt (Real.log (Real.log β))) := by
  intro t ht
  unfold intervalT' at ht
  split_ifs at ht with h
  · have hβ0 : 0 < β := lt_of_lt_of_le (Real.exp_pos 1) hβ
    have hL : 1 ≤ Real.log β := by rwa [Real.le_log_iff_exp_le hβ0]
    have hω : Real.sqrt (Real.log (Real.log β)) ≤ 2 * Real.log β := by
      rw [Real.sqrt_le_iff]
      refine ⟨by linarith, ?_⟩
      have := Real.log_le_sub_one_of_pos (by linarith : 0 < Real.log β)
      nlinarith
    have hle : Real.sqrt (2 * Real.log n - 3 * Real.log β) ≤
        Real.sqrt (2 * Real.log n - Real.log β - Real.sqrt (Real.log (Real.log β))) :=
      Real.sqrt_le_sqrt (by linarith)
    exact ⟨by linarith [ht.1], by linarith [ht.2]⟩
  · exact absurd ht (Set.notMem_empty t)

/-- The hypothesis of `intervalT'_subset_intervalT` is satisfiable with `T'`
nonempty: `n = 8`, `β = 3 ≥ e`, and `3 ≤ 8^{2/3} = 4`. -/
example : Real.exp 1 ≤ 3 ∧ (0 : ℝ) ∈ intervalT' 8 3 := by
  refine ⟨Real.exp_one_lt_three.le, ?_⟩
  unfold intervalT'
  have h : (3 : ℝ) ≤ ((8 : ℕ) : ℝ) ^ ((2 : ℝ) / 3) := by
    have h8 : ((8 : ℕ) : ℝ) ^ ((2 : ℝ) / 3) = 4 := by
      rw [Nat.cast_ofNat, show (8 : ℝ) = 2 ^ (3 : ℝ) by norm_num,
        ← Real.rpow_mul (by norm_num)]
      norm_num
    linarith
  simp only [h, ite_true]
  exact ⟨by simp, Real.sqrt_nonneg _⟩

/-- **`T' ⊆ T` eventually**, along a regime sequence. -/
theorem eventually_intervalT'_subset {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) :
    ∀ᶠ k in atTop, intervalT' (N k) (B k) ⊆
      intervalT (N k) (B k) (Real.sqrt (Real.log (Real.log (B k)))) := by
  filter_upwards [hreg.tendsto_B.eventually_ge_atTop (Real.exp 1)] with k hk
  exact intervalT'_subset_intervalT hk

/-- The hypotheses of `eventually_intervalT'_subset` are satisfiable. -/
example := eventually_intervalT'_subset isRegime_succ

end Modes
end Transformer
