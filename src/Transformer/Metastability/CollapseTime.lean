/-
# Metastability — the collapse time `T₁` and the state of a cap after it

Steps 2–3 of the direct proof of `thm: metastability` (arXiv:2410.06833v1, §2).  A cap `I`
has `⟨x_i, x_j⟩ ≥ 1 - r` at time `0` (`r = 8ε`), and the goal is `⟨x_i, x_j⟩ ≥ 1 - δ`
(`δ = e^{-λβ}`) from `T₁` on.  With `F_β(s) = s(1 - s)e^{-βs}` and `σ₀ = max δ (min (1/β) r)`
the distance `s = 1 - ρ` to `1` is pushed down in three phases:

* `s` from `r` to `σ₀` at the constant speed `F_β(r)/n` (`phase_linear`; `F_β(s) ≥ F_β(r)`
  for `1/β ≤ s ≤ r`), which takes `t_a = n (r - σ₀)/F_β(r)`;
* `s` from `σ₀` to `δ` exponentially at the rate `c₂ = (β-1)/(β e n)` (`phase_exp`;
  `F_β(s) ≥ (β-1)/(β e) · s` for `s ≤ 1/β`), which takes `t_b = log(σ₀/δ)/c₂`;
* `s ≤ δ` for all later times (`phase_const`).

All three need `F_β` above the leakage level `2 n² e^{-(1-α)β}` on `[δ, r]`; by log-concavity
of `F_β` it suffices to have it at the two ends.
-/

import Transformer.Metastability.CollapseComparison

open scoped BigOperators InnerProductSpace
open Real Set

namespace Transformer
namespace Metastability

variable {d n : ℕ}

/-- The point `σ₀ = max δ (min (1/β) r)` where the collapse passes from the linear phase
(`1/β ≤ s ≤ r`) to the exponential phase (`δ ≤ s ≤ 1/β`). -/
noncomputable def collapseSwitch (β r δ : ℝ) : ℝ := max δ (min (1 / β) r)

/-- Duration of the linear phase: `s` falls from `r` to `σ₀` at the speed `F_β(r)/n`. -/
noncomputable def collapseTimeLinear (n : ℕ) (β r δ : ℝ) : ℝ :=
  (r - collapseSwitch β r δ) / (collapseRate β r / n)

/-- Duration of the exponential phase: `s` falls from `σ₀` to `δ` at the rate
`c₂ = (β-1)/(β e n)`. -/
noncomputable def collapseTimeExp (n : ℕ) (β r δ : ℝ) : ℝ :=
  Real.log (collapseSwitch β r δ / δ) / ((β - 1) / (β * Real.exp 1 * n))

/-- **The collapse time** `T₁ = t_a + t_b`. -/
noncomputable def collapseTime (n : ℕ) (β r δ : ℝ) : ℝ :=
  collapseTimeLinear n β r δ + collapseTimeExp n β r δ

theorem collapseSwitch_le {β r δ : ℝ} (hδr : δ ≤ r) : collapseSwitch β r δ ≤ r :=
  max_le hδr (min_le_right _ _)

theorem le_collapseSwitch (β r δ : ℝ) : δ ≤ collapseSwitch β r δ := le_max_left _ _

/-- Above the switching point, `β s ≥ 1`: the linear phase lives where `F_β` is decreasing. -/
theorem one_le_of_collapseSwitch_lt {β r δ s : ℝ} (hβ : 0 < β)
    (h : collapseSwitch β r δ < s) (hsr : s ≤ r) : 1 ≤ β * s := by
  have hm : min (1 / β) r < s := lt_of_le_of_lt (le_max_right _ _) h
  by_cases hb : 1 / β ≤ r
  · rw [min_eq_left hb] at hm
    rw [div_lt_iff₀ hβ] at hm
    linarith
  · rw [min_eq_right (not_le.1 hb).le] at hm
    linarith

/-- Below the switching point (and above `δ`), `s ≤ 1/β`: the exponential phase lives where
`F_β(s) ≥ (β-1)/(β e) · s`. -/
theorem le_inv_of_le_collapseSwitch {β r δ s : ℝ} (hδs : δ < s)
    (h : s ≤ collapseSwitch β r δ) : s ≤ 1 / β := by
  rcases le_max_iff.1 h with h1 | h1
  · linarith
  · exact h1.trans (min_le_left _ _)

/-- **Collapse of a cap.**  Let `X` be an attention flow and `I` a set of tokens that are
`α`-separated from all the others on `[0, T]`, with `⟨x_i, x_j⟩ ≥ 1 - r` on `I` at time `0`.
If `F_β` exceeds the leakage level `2 n² e^{-(1-α)β}` at `r` and at `δ < r`, then
`⟨x_i, x_j⟩ ≥ 1 - δ` for `i, j ∈ I` and all `t ∈ [T₁, T]`, `T₁ = collapseTime n β r δ ≤ T`.

Source: arXiv:2410.06833v1, §2, Steps 2–3 of the proof of `thm: metastability` (`lem: eminem`
for the collapse, `lem: collapsetime` for the propagation of smallness). -/
theorem collapse_dynamic {β α r δ : ℝ} (hβ : 1 < β) (hn : 1 ≤ n) (hδ : 0 < δ) (hδr : δ < r)
    (hr : r < 1) {X : ℝ → SphereTuple d n} (hX : IsAttnFlow d n β X) (I : Finset (Idx n))
    (T : ℝ)
    (hFr : 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < collapseRate β r)
    (hFδ : 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < collapseRate β δ)
    (hT : collapseTime n β r δ ≤ T)
    (hfar : ∀ t ∈ Icc 0 T, ∀ i ∈ I, ∀ k, k ∉ I →
      ⟪(X t i : EucSpace d), (X t k : EucSpace d)⟫_ℝ ≤ α)
    (hinit : ∀ i ∈ I, ∀ j ∈ I, 1 - r ≤ ⟪(X 0 i : EucSpace d), (X 0 j : EucSpace d)⟫_ℝ) :
    ∀ t ∈ Icc (collapseTime n β r δ) T, ∀ i ∈ I, ∀ j ∈ I,
      1 - δ ≤ ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ := by
  have hβ0 : 0 < β := by linarith
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hβ1 : β - 1 ≠ 0 := (sub_pos.2 hβ).ne'
  have hthr0 : (0 : ℝ) ≤ 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) := by positivity
  have hFr0 : 0 < collapseRate β r := lt_of_le_of_lt hthr0 hFr
  have hFall : ∀ s ∈ Icc δ r, 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < collapseRate β s :=
    fun s hs => lt_of_lt_of_le (lt_min hFδ hFr) (collapseRate_min_le hδ hr hs)
  have hσr : collapseSwitch β r δ ≤ r := collapseSwitch_le hδr.le
  have hδσ : δ ≤ collapseSwitch β r δ := le_collapseSwitch β r δ
  have hσpos : 0 < collapseSwitch β r δ := lt_of_lt_of_le hδ hδσ
  have hc₁ : 0 < collapseRate β r / n := div_pos hFr0 hn'
  have hc₂ : 0 < (β - 1) / (β * Real.exp 1 * n) := div_pos (by linarith) (by positivity)
  have hta : 0 ≤ collapseTimeLinear n β r δ := div_nonneg (by linarith) hc₁.le
  have htb : 0 ≤ collapseTimeExp n β r δ :=
    div_nonneg (Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)) hc₂.le
  have hT₁ : collapseTime n β r δ = collapseTimeLinear n β r δ + collapseTimeExp n β r δ := rfl
  have hbT : collapseTimeLinear n β r δ ≤ T := by linarith
  -- phase 1: `s` from `r` to `σ₀`, linearly
  have hcF1 : ∀ s ∈ Ioc (collapseSwitch β r δ) r,
      collapseRate β r / n ≤ collapseRate β s / n := by
    intro s hs
    refine div_le_div_of_nonneg_right ?_ hn'.le
    exact collapseRate_ge (one_le_of_collapseSwitch_lt hβ0 hs.1 hs.2)
      ((lt_of_lt_of_le hδ hδσ).trans hs.1) hs.2 hr.le
  have h1 := phase_linear (α := α) hβ0.le hn hX I 0 r (collapseSwitch β r δ)
    (collapseRate β r / n) hc₁ hσr hr.le hcF1
    (fun s hs => hFall s ⟨hδσ.trans hs.1.le, hs.2⟩)
    (fun t ht => hfar t ⟨ht.1, ht.2.trans (by rw [zero_add]; exact hbT)⟩) hinit
  have hend1 : ∀ i ∈ I, ∀ j ∈ I, 1 - collapseSwitch β r δ ≤
      ⟪(X (collapseTimeLinear n β r δ) i : EucSpace d),
        (X (collapseTimeLinear n β r δ) j : EucSpace d)⟫_ℝ := by
    intro i hi j hj
    have h := h1 (collapseTimeLinear n β r δ) ⟨hta, by rw [zero_add]; exact le_rfl⟩ i hi j hj
    have hc : collapseRate β r / n * (collapseTimeLinear n β r δ - 0)
        = r - collapseSwitch β r δ := by
      rw [sub_zero]
      unfold collapseTimeLinear
      field_simp
    rw [hc] at h
    linarith
  -- phase 2: `s` from `σ₀` to `δ`, exponentially
  have hcF2 : ∀ s ∈ Ioc δ (collapseSwitch β r δ),
      (β - 1) / (β * Real.exp 1 * n) * s ≤ collapseRate β s / n := by
    intro s hs
    have h := collapseRate_phase2 hβ (hδ.trans hs.1).le (le_inv_of_le_collapseSwitch hs.1 hs.2)
    calc (β - 1) / (β * Real.exp 1 * n) * s = (β - 1) / (β * Real.exp 1) * s / n := by
          field_simp
      _ ≤ collapseRate β s / n := div_le_div_of_nonneg_right h hn'.le
  have h2 := phase_exp (α := α) hβ0.le hn hX I (collapseTimeLinear n β r δ)
    (collapseSwitch β r δ) δ ((β - 1) / (β * Real.exp 1 * n)) hc₂ hδ hδσ (hσr.trans hr.le) hcF2
    (fun s hs => hFall s ⟨hs.1.le, hs.2.trans hσr⟩)
    (fun t ht => hfar t ⟨hta.trans ht.1, ht.2.trans hT⟩) hend1
  have hend2 : ∀ i ∈ I, ∀ j ∈ I, 1 - δ ≤
      ⟪(X (collapseTime n β r δ) i : EucSpace d), (X (collapseTime n β r δ) j : EucSpace d)⟫_ℝ := by
    intro i hi j hj
    have h := h2 (collapseTime n β r δ) ⟨by rw [hT₁]; linarith, le_rfl⟩ i hi j hj
    have hexp : collapseSwitch β r δ *
        Real.exp (-((β - 1) / (β * Real.exp 1 * n) *
          (collapseTime n β r δ - collapseTimeLinear n β r δ))) = δ := by
      have hlog : (β - 1) / (β * Real.exp 1 * n) *
          (collapseTime n β r δ - collapseTimeLinear n β r δ)
            = Real.log (collapseSwitch β r δ / δ) := by
        rw [hT₁, add_sub_cancel_left]
        unfold collapseTimeExp
        field_simp
      rw [hlog, Real.exp_neg, Real.exp_log (div_pos hσpos hδ), inv_div]
      field_simp
    rw [hexp] at h
    exact h
  -- phase 3: `s ≤ δ` from then on
  exact phase_const (α := α) hβ0.le hn hX I (collapseTime n β r δ) T δ hT (hδr.le.trans hr.le)
    hFδ (fun t ht => hfar t ⟨(hta.trans (by rw [hT₁]; linarith)).trans ht.1, ht.2⟩) hend2

end Metastability
end Transformer
