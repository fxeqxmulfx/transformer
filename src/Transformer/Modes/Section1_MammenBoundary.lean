import Transformer.Modes.Section1_IntervalCounts
import Transformer.Modes.Growth

/-
# The number of modes of a Gaussian KDE — the boundary case of Mammen's formula

The first bullet of `thm:mammen`, arXiv:2412.09080v3, §1.1, says that the
expected count in a fixed `[a,b]` is `1{0 ∈ [a,b]} + o(1)` when
`β ≪ n^(2/5)`. Its wording includes an interval with zero as an endpoint.
It therefore predicts limit one both for `[-1,1]` and for `[0,1]`.

For every positive bandwidth and nonempty Gaussian sample, reflection
preserves the law and zero is almost surely not a mode. Consequently,
twice the expected count in `[0,1]` is at most that in `[-1,1]`. Whenever
the latter count is eventually finite, these two limits would imply
`2 ≤ 1`. This contradiction applies to nondegenerate intervals and does
not need a Kac–Rice formula or a continuous joint density.

The explicit sample sizes `n = k + 1` and constant parameter `β = 1`
satisfy all hypotheses of the source's first bandwidth regime. We also
record the simpler singleton counterexample: the expectation in `[0,0]`
is identically zero. These are counterexamples to the formula as stated,
not a weakened positive theorem under its original name. A version with
zero in the interval's interior would require a separate proof.

Source: arXiv:2412.09080v3, §1.1, first bullet of `thm:mammen` and `eq:gkde`.
-/

open MeasureTheory Filter Asymptotics
open scoped Topology ENNReal

namespace Transformer.Modes

/-- Reflection rules out simultaneous limit one on `[-1,1]` and `[0,1]`.
Eventual finiteness is part of the conclusion being refuted, so the real
conversion cannot hide an infinite expected count.
Source: arXiv:2412.09080v3, §1.1, first bullet of `thm:mammen`. -/
theorem not_expectedModes_limits_one (N : ℕ → ℕ) (B : ℕ → ℝ)
    (hN : Tendsto (fun k => (N k : ℝ)) atTop atTop) (hB : ∀ k, 0 < B k) :
    ¬ ((∀ᶠ k in atTop, expectedModes (B k) (N k) (Set.Icc (-1) 1) ≠ ∞) ∧
      Tendsto (fun k => expectedModesReal (B k) (N k) (Set.Icc (-1) 1)) atTop (𝓝 1) ∧
      Tendsto (fun k => expectedModesReal (B k) (N k) (Set.Icc 0 1)) atTop (𝓝 1)) := by
  rintro ⟨hfinite, hwhole, hhalf⟩
  have hineq : ∀ᶠ k in atTop,
      2 * expectedModesReal (B k) (N k) (Set.Icc 0 1) ≤
        expectedModesReal (B k) (N k) (Set.Icc (-1) 1) := by
    filter_upwards [hfinite, hN.eventually_gt_atTop 0] with k hk hn
    have hn' : 0 < N k := by exact_mod_cast hn
    have hsum := twice_expectedModes_Icc_zero_le (hB k) hn' (b := 1) zero_le_one
    have hhalfFinite : expectedModes (B k) (N k) (Set.Icc 0 1) ≠ ∞ :=
      ne_top_of_le_ne_top hk (expectedModes_mono _ _
        (Set.Icc_subset_Icc (by norm_num : (-1 : ℝ) ≤ 0) le_rfl))
    have hreal := ENNReal.toReal_mono hk hsum
    rw [ENNReal.toReal_add hhalfFinite hhalfFinite] at hreal
    simpa only [expectedModesReal, two_mul] using hreal
  have hlimit : (2 : ℝ) ≤ 1 :=
    le_of_tendsto_of_tendsto (by simpa only [mul_one] using hhalf.const_mul 2)
      hwhole hineq
  norm_num at hlimit

/-- Increasing positive sample counts and constant positive bandwidth satisfy
the two hypotheses of the simultaneous-limit counterexample. -/
example : Tendsto (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop atTop ∧
    (∀ _ : ℕ, (0 : ℝ) < 1) := by
  exact ⟨by push_cast; exact tendsto_natSucc_atTop, fun _ => one_pos⟩

/-- The two explicit interval predictions cannot hold simultaneously along
any sequence of positive bandwidths and increasing sample sizes, including
bandwidths tending to infinity. No particular low-bandwidth rate is needed
for the reflection obstruction.
Source: arXiv:2412.09080v3, §1.1, first bullet of `thm:mammen`. -/
theorem not_mammen_lt_interval_pair (N : ℕ → ℕ) (B : ℕ → ℝ)
    (hN : Tendsto (fun k => (N k : ℝ)) atTop atTop) (hB : ∀ k, 0 < B k) :
    ¬ (((∀ᶠ k in atTop, expectedModes (B k) (N k) (Set.Icc (-1) 1) ≠ ∞) ∧
        (fun k => expectedModesReal (B k) (N k) (Set.Icc (-1) 1) - 1)
          =o[atTop] fun _ => (1 : ℝ)) ∧
      ((∀ᶠ k in atTop, expectedModes (B k) (N k) (Set.Icc 0 1) ≠ ∞) ∧
        (fun k => expectedModesReal (B k) (N k) (Set.Icc 0 1) - 1)
          =o[atTop] fun _ => (1 : ℝ))) := by
  rintro ⟨hw, hh⟩
  apply not_expectedModes_limits_one N B hN hB
  exact ⟨hw.1, tendsto_sub_nhds_zero_iff.mp ((isLittleO_one_iff ℝ).mp hw.2),
    tendsto_sub_nhds_zero_iff.mp ((isLittleO_one_iff ℝ).mp hh.2)⟩

/-- The paired-interval theorem's hypotheses have the same concrete witness. -/
example : Tendsto (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop atTop ∧
    (∀ _ : ℕ, (0 : ℝ) < 1) := by
  exact ⟨by push_cast; exact tendsto_natSucc_atTop, fun _ => one_pos⟩

/-- The first bullet of `thm:mammen` already fails when restricted to
nondegenerate intervals: its predictions for `[-1,1]` and `[0,1]` contradict
`not_expectedModes_limits_one`. The explicit sequences are `β = 1`, `n = k+1`.
Source: arXiv:2412.09080v3, §1.1, first bullet of `thm:mammen`. -/
theorem not_mammen_lt_nonzero_intervals :
    ¬ ∀ a b : ℝ, a < b →
      (∀ᶠ k : ℕ in atTop, expectedModes 1 (k + 1) (Set.Icc a b) ≠ ∞) ∧
      (fun k : ℕ => expectedModesReal 1 (k + 1) (Set.Icc a b) -
        (if (0 : ℝ) ∈ Set.Icc a b then 1 else 0)) =o[atTop] fun _ => (1 : ℝ) := by
  intro h
  have hw := h (-1) 1 (by norm_num)
  have hh := h 0 1 one_pos
  have hzeroWhole : (0 : ℝ) ∈ Set.Icc (-1) 1 := by norm_num
  have hzeroHalf : (0 : ℝ) ∈ Set.Icc 0 1 := by norm_num
  have hw' : (fun k : ℕ => expectedModesReal 1 (k + 1) (Set.Icc (-1) 1) - 1)
      =o[atTop] fun _ => (1 : ℝ) := by
    simpa only [hzeroWhole, ite_true] using hw.2
  have hh' : (fun k : ℕ => expectedModesReal 1 (k + 1) (Set.Icc 0 1) - 1)
      =o[atTop] fun _ => (1 : ℝ) := by
    simpa only [hzeroHalf, ite_true] using hh.2
  apply not_mammen_lt_interval_pair (fun k => k + 1) (fun _ => 1)
    (by push_cast; exact tendsto_natSucc_atTop) (fun _ => one_pos)
  exact ⟨⟨hw.1, hw'⟩, ⟨hh.1, hh'⟩⟩

/-- The counterexample sequences satisfy the source's original low-bandwidth
regime, rather than a boundary value of its asymptotic comparison.
Source: arXiv:2412.09080v3, §1.1, first bullet of `thm:mammen`. -/
theorem mammen_lt_counterexample_regime :
    Tendsto (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop atTop ∧
    (∀ _ : ℕ, (0 : ℝ) < 1) ∧
    (fun _ : ℕ => (1 : ℝ)) =o[atTop]
      fun k : ℕ => ((k + 1 : ℕ) : ℝ) ^ ((2 : ℝ) / 5) := by
  refine ⟨by push_cast; exact tendsto_natSucc_atTop, fun _ => one_pos, ?_⟩
  have hpow := isLittleO_rpow_rpow_nat (a := 0) (b := (2 : ℝ) / 5) (by norm_num)
  simpa using hpow

/-- A singleton supplies a second counterexample: its expected count is zero
at every positive sample size, so the claimed error is the constant `-1`.
Source: arXiv:2412.09080v3, §1.1, first bullet of `thm:mammen`, `[a,b]=[0,0]`. -/
theorem not_mammen_lt_singleton :
    ¬ (fun k : ℕ => expectedModesReal 1 (k + 1) (Set.Icc 0 0) - 1)
      =o[atTop] fun _ => (1 : ℝ) := by
  have hfun : (fun k : ℕ => expectedModesReal 1 (k + 1) (Set.Icc 0 0) - 1) =
      fun _ : ℕ => (-1 : ℝ) := by
    funext k
    rw [expectedModesReal, Set.Icc_self,
      expectedModes_singleton_eq_zero one_pos (by omega) 0, ENNReal.toReal_zero]
    ring
  intro h
  rw [hfun] at h
  have hzero : (-1 : ℝ) = 0 := isLittleO_const_const_iff.mp h
  norm_num at hzero

/-- The source's full range `a ≤ b` contains the nondegenerate contradiction.
This preserves the original quantification when replacing the false local
statement by a proved counterexample.
Source: arXiv:2412.09080v3, §1.1, first bullet of `thm:mammen`. -/
theorem not_mammen_lt_all_Icc :
    ¬ ∀ a b : ℝ, a ≤ b →
      (∀ᶠ k : ℕ in atTop, expectedModes 1 (k + 1) (Set.Icc a b) ≠ ∞) ∧
      (fun k : ℕ => expectedModesReal 1 (k + 1) (Set.Icc a b) -
        (if (0 : ℝ) ∈ Set.Icc a b then 1 else 0)) =o[atTop] fun _ => (1 : ℝ) := by
  intro h
  apply not_mammen_lt_nonzero_intervals
  intro a b hab
  exact h a b hab.le

end Transformer.Modes
