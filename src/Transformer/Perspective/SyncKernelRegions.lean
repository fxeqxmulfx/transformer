/-
# The positive regions of `f_β'''`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, Appendices A–B, `lemma:positiveregion` and
`lemma:positiveregion2`; §5.

`f_β'''(x) = p(cos x) e^{β(cos x - 1)}`, `cos` decreases on `[0, π]`, and
`f_β'''` is even, so the sign of `p` on `[-1, 1]` (`Perspective.SyncKernelSign`)
is the sign of `f_β'''` in `|x|`:

* `β > 1/3`: on `[-π, π]`, `f_β''' > 0` exactly where `b < |x| < a`, with
  `b = arccos r₂` and `a = arccos r₁` (`exists_f_β'''_pos_iff`).  On `(-π, π]`
  this is the positive region `(-a, -b) ⊔ (b, a)` of `lemma:positiveregion`.
* `0 < β ≤ 1/3`: on `[-π, π]`, `f_β''' ≥ 0` exactly where `a ≤ |x|`, with
  `a = arccos r` (`exists_f_β'''_nonneg_iff`).  By `2π`-periodicity this is the
  nonnegative region `[a, 2π - a]` on `[0, 2π)` of `lemma:positiveregion2`.

The endpoints `a` and `b` are zeros of `f_β'''`.
-/

import Transformer.Perspective.SyncKernelSign

open Real

namespace Transformer
namespace Perspective

/-- **`β > 1/3`: the sign of `f_β'''`.**  There are `0 < b < a < π`, zeros of
`f_β'''`, such that on `[-π, π]`, `f_β'''(x) > 0` iff `b < |x| < a`:
`b = arccos r₂` and `a = arccos r₁`. -/
theorem exists_f_β'''_pos_iff {β : ℝ} (hβ : 1 / 3 < β) :
    ∃ b a, 0 < b ∧ b < a ∧ a < π ∧ f_β''' β b = 0 ∧ f_β''' β a = 0 ∧
      ∀ x ∈ Set.Icc (-π) π, 0 < f_β''' β x ↔ b < |x| ∧ |x| < a := by
  obtain ⟨r₁, r₂, h1, h12, h2, hr₁, hr₂, hsign⟩ := exists_kernelQuartic_roots hβ
  have hc₁ : Real.cos (Real.arccos r₁) = r₁ := Real.cos_arccos h1.le (by linarith)
  have hc₂ : Real.cos (Real.arccos r₂) = r₂ := Real.cos_arccos (by linarith) h2.le
  have hm : ∀ r, Real.arccos r ∈ Set.Icc 0 π := fun r =>
    ⟨Real.arccos_nonneg r, Real.arccos_le_pi r⟩
  refine ⟨Real.arccos r₂, Real.arccos r₁, Real.arccos_pos.mpr h2,
    Real.arccos_lt_arccos h1.le h12 h2.le, Real.arccos_lt_pi.mpr h1,
    by rw [f_β'''_eq_kernelQuartic, hc₂, hr₂, zero_mul],
    by rw [f_β'''_eq_kernelQuartic, hc₁, hr₁, zero_mul], fun x hx => ?_⟩
  have hax : |x| ∈ Set.Icc 0 π := ⟨abs_nonneg x, abs_le.mpr hx⟩
  have e₁ := Real.strictAntiOn_cos.lt_iff_gt (hm r₁) hax
  have e₂ := Real.strictAntiOn_cos.lt_iff_gt hax (hm r₂)
  rw [hc₁] at e₁
  rw [hc₂] at e₂
  rw [f_β'''_eq_kernelQuartic, mul_pos_iff_of_pos_right (Real.exp_pos _), ← Real.cos_abs,
    hsign (Real.cos |x|) ⟨Real.neg_one_le_cos _, Real.cos_le_one _⟩, e₁, e₂, and_comm]

/-- **`0 < β ≤ 1/3`: the sign of `f_β'''`.**  There is `a ∈ (0, π/2)`, a zero of
`f_β'''`, such that on `[-π, π]`, `f_β'''(x) ≥ 0` iff `a ≤ |x|`: `a = arccos r`. -/
theorem exists_f_β'''_nonneg_iff {β : ℝ} (hβ : 0 < β) (hβ' : β ≤ 1 / 3) :
    ∃ a, 0 < a ∧ a < π / 2 ∧ f_β''' β a = 0 ∧
      ∀ x ∈ Set.Icc (-π) π, 0 ≤ f_β''' β x ↔ a ≤ |x| := by
  obtain ⟨r, hr0, hr1, hr, hsign⟩ := exists_kernelQuartic_root hβ hβ'
  have hc : Real.cos (Real.arccos r) = r := Real.cos_arccos (by linarith) hr1.le
  refine ⟨Real.arccos r, Real.arccos_pos.mpr hr1, Real.arccos_lt_pi_div_two.mpr hr0,
    by rw [f_β'''_eq_kernelQuartic, hc, hr, zero_mul], fun x hx => ?_⟩
  have hax : |x| ∈ Set.Icc 0 π := ⟨abs_nonneg x, abs_le.mpr hx⟩
  have e := Real.strictAntiOn_cos.le_iff_ge hax ⟨Real.arccos_nonneg r, Real.arccos_le_pi r⟩
  rw [hc] at e
  rw [f_β'''_eq_kernelQuartic, mul_nonneg_iff_of_pos_right (Real.exp_pos _), ← Real.cos_abs,
    hsign (Real.cos |x|) ⟨Real.neg_one_le_cos _, Real.cos_le_one _⟩, e]

/-- `b < |x| < a` says `x ∈ (-a, -b) ∪ (b, a)`, for `b > 0`. -/
theorem abs_mem_Ioo_iff {b a x : ℝ} (hb : 0 < b) :
    (b < |x| ∧ |x| < a) ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a := by
  simp only [Set.mem_union, Set.mem_Ioo]
  rcases le_or_gt 0 x with h | h
  · rw [abs_of_nonneg h]
    exact ⟨Or.inr, fun H => H.elim (fun H => False.elim (by linarith [H.2])) id⟩
  · rw [abs_of_neg h]
    exact ⟨fun H => Or.inl ⟨by linarith, by linarith⟩,
      fun H => H.elim (fun H => ⟨by linarith, by linarith⟩) fun H => False.elim (by linarith [H.1])⟩

/-- **Lemma (lemma:positiveregion).** *Suppose `β > 1/3`.  There exist
`0 < b < a < π` such that the positive region of `f_β'''` is `(-a, -b) ⊔ (b, a)`.*

The region is taken on `(-π, π]`, as §5 states it; §5 writes `0 < a < b < π`, the
lemma `0 < b < a < π`.  `b` and `a` are `arccos` of the two roots of the quartic
`p` in `(-1, 1)` (`exists_kernelQuartic_roots`, `exists_f_β'''_pos_iff`).

Source: arXiv:2507.22857v1, Appendix A, `lemma:positiveregion`; §5. -/
theorem positiveregion {β : ℝ} (hβ : 1 / 3 < β) :
    ∃ b a, 0 < b ∧ b < a ∧ a < π ∧
      ∀ x ∈ Set.Ioc (-π) π, 0 < f_β''' β x ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a := by
  obtain ⟨b, a, hb, hba, ha, -, -, h⟩ := exists_f_β'''_pos_iff hβ
  exact ⟨b, a, hb, hba, ha, fun x hx => (h x ⟨hx.1.le, hx.2⟩).trans (abs_mem_Ioo_iff hb)⟩

/-- The nonnegative region `a ≤ |x|` on `[-π, π]`, read on `[0, 2π)`: it is
`[a, 2π - a]`, by `2π`-periodicity. -/
theorem nonneg_f_β'''_iff_of_abs {β a : ℝ} (ha : a < π / 2)
    (h : ∀ x ∈ Set.Icc (-π) π, 0 ≤ f_β''' β x ↔ a ≤ |x|) :
    ∀ x ∈ Set.Ico 0 (2 * π), 0 ≤ f_β''' β x ↔ x ∈ Set.Icc a (2 * π - a) := by
  intro x hx
  have hπ := Real.pi_pos
  rcases le_or_gt x π with hxπ | hxπ
  · rw [h x ⟨by linarith [hx.1], hxπ⟩, abs_of_nonneg hx.1]
    exact ⟨fun h' => ⟨h', by linarith⟩, fun h' => h'.1⟩
  · rw [← (periodic_f_β''' β).sub_eq x, h (x - 2 * π) ⟨by linarith, by linarith [hx.2]⟩,
      abs_of_neg (by linarith [hx.2])]
    exact ⟨fun h' => ⟨by linarith, by linarith⟩, fun h' => by linarith [h'.2]⟩

/-- The hypotheses of `nonneg_f_β'''_iff_of_abs` are satisfiable: at `β = 1/4`,
`exists_f_β'''_nonneg_iff` provides them. -/
example : ∃ β a : ℝ, a < π / 2 ∧ ∀ x ∈ Set.Icc (-π) π, 0 ≤ f_β''' β x ↔ a ≤ |x| := by
  obtain ⟨a, -, ha, -, h⟩ := exists_f_β'''_nonneg_iff (β := 1 / 4) (by norm_num) (by norm_num)
  exact ⟨1 / 4, a, ha, h⟩

/-- **Lemma (lemma:positiveregion2).** *Suppose `β ∈ (0, 1/3]`.  Then there exists
`a ∈ (0, π/2)` such that the nonnegative region of `f_β'''` over `[0, 2π)` is
`[a, 2π - a]`.*

`a` is `arccos` of the root of the quartic `p` in `(0, 1)`
(`exists_kernelQuartic_root`, `exists_f_β'''_nonneg_iff`).  §5 speaks of the
positive region `(a, 2π - a)`; the source's remark after the lemma explains the
nonnegative one: at `β = 1/3`, `f_β'''(π) = 0`.

Source: arXiv:2507.22857v1, Appendix B, `lemma:positiveregion2`; §5. -/
theorem positiveregion2 {β : ℝ} (hβ : 0 < β) (hβ' : β ≤ 1 / 3) :
    ∃ a ∈ Set.Ioo 0 (π / 2),
      ∀ x ∈ Set.Ico 0 (2 * π), 0 ≤ f_β''' β x ↔ x ∈ Set.Icc a (2 * π - a) := by
  obtain ⟨a, ha0, ha, -, h⟩ := exists_f_β'''_nonneg_iff hβ hβ'
  exact ⟨a, ⟨ha0, ha⟩, nonneg_f_β'''_iff_of_abs ha h⟩

/-- The other hypotheses above are satisfiable: `β = 1` for `1/3 < β`, `β = 1/4`
for `0 < β ≤ 1/3`, and `b = 1` for `0 < b`. -/
example : (1 : ℝ) / 3 < 1 ∧ (0 : ℝ) < 1 / 4 ∧ (1 : ℝ) / 4 ≤ 1 / 3 ∧ (0 : ℝ) < 1 := by
  norm_num

end Perspective
end Transformer
