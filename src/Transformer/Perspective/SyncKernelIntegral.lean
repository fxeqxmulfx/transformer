module

public import Transformer.Perspective.SyncKernelRegions
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-!
# `∫ |f_β'''|₊` through `f_β''`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §5, proof of `cor:conjecture`.

The criterion of `thm:main` involves `⟨1, |f'''|₊⟩ = ∫_{-π}^{π} |f'''|₊`. §5 reads
it off the regions of `lemma:positiveregion` and `lemma:positiveregion2`
(`Perspective.SyncKernelRegions`) by the fundamental theorem of calculus,
`f_β''` being a primitive of `f_β'''` and `f_β'''` being even:

* `β > 1/3`: `∫_{-π}^{π} |f_β'''|₊ = 2(f_β''(a) - f_β''(b))`;
* `0 < β ≤ 1/3`: `∫_{-π}^{π} |f_β'''|₊ = -2 f_β''(a)`.

For every `β > 0`, then, `∫_{-π}^{π} |f_β'''|₊ ≤ 2(U - L)` whenever
`L ≤ f_β'' ≤ U` on `[0, π]` (`integral_posPart_f_β'''_le`). Here `|y|₊` is
`max y 0`.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer
namespace Perspective

/-- `∫_{-π}^{π} |f_β'''|₊ = 2 ∫_0^π |f_β'''|₊`, as `f_β'''` is even. -/
theorem integral_posPart_f_β'''_eq_two_mul (β : ℝ) :
    ∫ x in (-π)..π, max (f_β''' β x) 0 = 2 * ∫ x in (0 : ℝ)..π, max (f_β''' β x) 0 := by
  have hc : Continuous fun x => max (f_β''' β x) 0 := (continuous_f_β''' β).max continuous_const
  have hneg := intervalIntegral.integral_comp_neg (a := 0) (b := π) fun x => max (f_β''' β x) 0
  simp only [f_β'''_neg, neg_zero] at hneg
  rw [← intervalIntegral.integral_add_adjacent_intervals (hc.intervalIntegrable (-π) 0)
    (hc.intervalIntegrable 0 π), ← hneg]
  ring

/-- **§5, case `β > 1/3`.**  With `b < a` of `lemma:positiveregion`,
`⟨1, |f_β'''|₊⟩ = ∫_{-π}^{π} |f_β'''|₊ = 2(f_β''(a) - f_β''(b))`.

Source: arXiv:2507.22857v1, §5, proof of `cor:conjecture`, case `β > 1/3`: "it
suffices to prove that `⟨τ, 1{(-a, -b) ∪ (b, a)} f'''⟩ < 4 ⇔ τ(f''(a) - f''(b)) < 2`". -/
theorem integral_posPart_f_β'''_of_one_third_lt {β : ℝ} (hβ : 1 / 3 < β) :
    ∃ b a, 0 < b ∧ b < a ∧ a < π ∧
      (∀ x ∈ Set.Ioc (-π) π, 0 < f_β''' β x ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a) ∧
      ∫ x in (-π)..π, max (f_β''' β x) 0 = 2 * (f_β'' β a - f_β'' β b) := by
  obtain ⟨b, a, hb, hba, ha, hfb, hfa, h⟩ := exists_f_β'''_pos_iff hβ
  refine ⟨b, a, hb, hba, ha, fun x hx => (h x ⟨hx.1.le, hx.2⟩).trans (abs_mem_Ioo_iff hb), ?_⟩
  have hc : Continuous fun x => max (f_β''' β x) 0 := (continuous_f_β''' β).max continuous_const
  -- on `[0, π]`, `|f_β'''|₊` vanishes off `(b, a)` and is `f_β'''` on `[b, a]`
  have hoff : ∀ x, 0 ≤ x → x ≤ π → (x ≤ b ∨ a ≤ x) → max (f_β''' β x) 0 = 0 := by
    intro x hx0 hxπ hx
    refine max_eq_right (le_of_not_gt fun hpos => ?_)
    have := (h x ⟨by linarith [Real.pi_pos], hxπ⟩).mp hpos
    rw [abs_of_nonneg hx0] at this
    rcases hx with hx | hx <;> linarith [this.1, this.2]
  have e₁ : Set.EqOn (fun x => max (f_β''' β x) 0) (fun _ => 0) (Set.uIcc 0 b) := by
    intro x hx
    rw [Set.uIcc_of_le hb.le] at hx
    exact hoff x hx.1 (by linarith [hx.2]) (Or.inl hx.2)
  have e₂ : Set.EqOn (fun x => max (f_β''' β x) 0) (f_β''' β) (Set.uIcc b a) := by
    intro x hx
    rw [Set.uIcc_of_le hba.le] at hx
    refine max_eq_left ?_
    rcases hx.1.eq_or_lt with h₁ | h₁
    · rw [← h₁, hfb]
    rcases hx.2.eq_or_lt with h₂ | h₂
    · rw [h₂, hfa]
    refine ((h x ⟨by linarith, by linarith⟩).mpr ?_).le
    rw [abs_of_pos (by linarith)]
    exact ⟨h₁, h₂⟩
  have e₃ : Set.EqOn (fun x => max (f_β''' β x) 0) (fun _ => 0) (Set.uIcc a π) := by
    intro x hx
    rw [Set.uIcc_of_le ha.le] at hx
    exact hoff x (by linarith [hx.1]) hx.2 (Or.inr hx.1)
  rw [integral_posPart_f_β'''_eq_two_mul,
    ← intervalIntegral.integral_add_adjacent_intervals (hc.intervalIntegrable 0 b)
      (hc.intervalIntegrable b π),
    ← intervalIntegral.integral_add_adjacent_intervals (hc.intervalIntegrable b a)
      (hc.intervalIntegrable a π),
    intervalIntegral.integral_congr e₁, intervalIntegral.integral_congr e₂,
    intervalIntegral.integral_congr e₃,
    intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hasDerivAt_f_β'' β x)
      ((continuous_f_β''' β).intervalIntegrable b a)]
  simp only [intervalIntegral.integral_zero]
  ring

/-- **§5, case `0 < β ≤ 1/3`.**  With `a` of `lemma:positiveregion2`,
`⟨1, |f_β'''|₊⟩ = ∫_{-π}^{π} |f_β'''|₊ = -2 f_β''(a)`.

Source: arXiv:2507.22857v1, §5, proof of `cor:conjecture`, case `0 < β ≤ 1/3`:
"it suffices to prove that `⟨τ, 1{(a, 2π - a)} f'''⟩ < 4 ⇔ τ f''(a) > -2`". -/
theorem integral_posPart_f_β'''_of_le_one_third {β : ℝ} (hβ : 0 < β) (hβ' : β ≤ 1 / 3) :
    ∃ a ∈ Set.Ioo 0 (π / 2),
      (∀ x ∈ Set.Ico 0 (2 * π), 0 ≤ f_β''' β x ↔ x ∈ Set.Icc a (2 * π - a)) ∧
      ∫ x in (-π)..π, max (f_β''' β x) 0 = -2 * f_β'' β a := by
  obtain ⟨a, ha0, ha, hfa, h⟩ := exists_f_β'''_nonneg_iff hβ hβ'
  refine ⟨a, ⟨ha0, ha⟩, nonneg_f_β'''_iff_of_abs ha h, ?_⟩
  have hπ := Real.pi_pos
  have hc : Continuous fun x => max (f_β''' β x) 0 := (continuous_f_β''' β).max continuous_const
  -- on `[0, π]`, `|f_β'''|₊` vanishes on `[0, a]` and is `f_β'''` on `[a, π]`
  have e₁ : Set.EqOn (fun x => max (f_β''' β x) 0) (fun _ => 0) (Set.uIcc 0 a) := by
    intro x hx
    rw [Set.uIcc_of_le ha0.le] at hx
    refine max_eq_right ?_
    rcases hx.2.eq_or_lt with h₁ | h₁
    · rw [h₁, hfa]
    refine le_of_not_gt fun hpos => ?_
    have := (h x ⟨by linarith [hx.1], by linarith⟩).mp hpos.le
    rw [abs_of_nonneg hx.1] at this
    linarith
  have e₂ : Set.EqOn (fun x => max (f_β''' β x) 0) (f_β''' β) (Set.uIcc a π) := by
    intro x hx
    rw [Set.uIcc_of_le (by linarith)] at hx
    refine max_eq_left ((h x ⟨by linarith [hx.1], hx.2⟩).mpr ?_)
    rw [abs_of_nonneg (by linarith [hx.1])]
    exact hx.1
  rw [integral_posPart_f_β'''_eq_two_mul,
    ← intervalIntegral.integral_add_adjacent_intervals (hc.intervalIntegrable 0 a)
      (hc.intervalIntegrable a π),
    intervalIntegral.integral_congr e₁, intervalIntegral.integral_congr e₂,
    intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hasDerivAt_f_β'' β x)
      ((continuous_f_β''' β).intervalIntegrable a π), f_β''_pi]
  simp only [intervalIntegral.integral_zero]
  ring

/-- **`∫ |f_β'''|₊` through bounds on `f_β''`.**  For `β > 0`, if `L ≤ f_β'' ≤ U`
on `[0, π]`, then `∫_{-π}^{π} |f_β'''|₊ ≤ 2(U - L)`.

For `β > 1/3` this is `f_β''(a) ≤ U` and `f_β''(b) ≥ L`; for `0 < β ≤ 1/3`,
`-2 f_β''(a) ≤ -2L ≤ 2(U - L)`, since `U ≥ f_β''(0) = 0`. Not a statement of the
source, which bounds `f''(a)` and `f''(b)` directly (`lemma:beta1`, …,
`lemma:beta_final`); any such bounds give `L` and `U`. -/
theorem integral_posPart_f_β'''_le {β L U : ℝ} (hβ : 0 < β)
    (hL : ∀ x ∈ Set.Icc 0 π, L ≤ f_β'' β x) (hU : ∀ x ∈ Set.Icc 0 π, f_β'' β x ≤ U) :
    ∫ x in (-π)..π, max (f_β''' β x) 0 ≤ 2 * (U - L) := by
  have hπ := Real.pi_pos
  rcases le_or_gt β (1 / 3) with h | h
  · obtain ⟨a, ha, -, hint⟩ := integral_posPart_f_β'''_of_le_one_third hβ h
    have h0 := hU 0 ⟨le_rfl, hπ.le⟩
    rw [f_β''_zero] at h0
    rw [hint]
    linarith [hL a ⟨ha.1.le, by linarith [ha.2]⟩]
  · obtain ⟨b, a, hb, hba, ha, -, hint⟩ := integral_posPart_f_β'''_of_one_third_lt h
    rw [hint]
    linarith [hU a ⟨by linarith, ha.le⟩, hL b ⟨hb.le, by linarith⟩]

/-- The hypotheses of `integral_posPart_f_β'''_le` are satisfiable: `f_β''` is
continuous, so bounded on `[0, π]`. -/
example : ∃ L U : ℝ, (∀ x ∈ Set.Icc 0 π, L ≤ f_β'' 1 x) ∧
    ∀ x ∈ Set.Icc 0 π, f_β'' 1 x ≤ U := by
  obtain ⟨C, hC⟩ := isCompact_Icc.exists_bound_of_continuousOn
    (continuous_iff_continuousAt.mpr fun x => (hasDerivAt_f_β'' 1 x).continuousAt).continuousOn
  refine ⟨-C, C, fun x hx => ?_, fun x hx => ?_⟩ <;>
    have h := hC x hx <;> rw [Real.norm_eq_abs] at h
  exacts [(abs_le.mp h).1, (abs_le.mp h).2]

/-- The other hypotheses above are satisfiable: `β = 1` for `1/3 < β`, and
`β = 1/4` for `0 < β ≤ 1/3`. -/
example : (1 : ℝ) / 3 < 1 ∧ (0 : ℝ) < 1 / 4 ∧ (1 : ℝ) / 4 ≤ 1 / 3 := by norm_num

end Perspective
end Transformer
