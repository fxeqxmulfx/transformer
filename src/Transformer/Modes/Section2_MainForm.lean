/-
# The number of modes of a Gaussian KDE — `eq:main-eq-form`

§2.1 of arXiv:2412.09080v3, `eq:main-eq-form`: the three bounds on the expected
number of upcrossings of `F_n` that `prop:main-int` and `prop:main-tail` "are
equivalent to".

**What the source says and what is carried here.**

* The equivalence runs through the identity mode `=` upcrossing, which holds in
  expectation (`Section2_Degenerate.lean`) and not pathwise.  That identity is
  unproved, so it is carried here as an explicit hypothesis `hid`, one per
  interval, and so are the three conclusions of `prop:main-int` and
  `prop:main-tail`.  What is proved is the passage from the one to the other.

* Lines 1 and 3 are the identity read off.  Line 2 is not: `prop:main-int` gives
  `E #modes(T') = O(√β)`, and the source's `≲ e^{-ω(β)/4}√(β log β)` is *weaker*
  than that, because `e^{ω(β)/4} ≪ √(log β)` — which is exactly what the right
  half of the window `ω(β) ≪ log log β` buys.  `isLittleO_sqrt_window` proves it,
  and line 2 comes out with `o` in place of the source's `≲`.

Source: arXiv:2412.09080v3, `eq:main-eq-form`, `sec: sketch`.
-/

import Transformer.Modes.Section2_Degenerate
import Transformer.Modes.Section1_Sketch

open Filter Asymptotics MeasureTheory
open scoped Topology ENNReal

namespace Transformer
namespace Modes

/-- `E U₀(F_n, T)`, the expected number of upcrossings of `F_n` at level `0` in
`T`, as a real number.

Source: arXiv:2412.09080v3, `eq:main-eq-form`. -/
noncomputable def expectedUpcrossingsReal (β : ℝ) (n : ℕ) (T : Set ℝ) : ℝ :=
  (expectedUpcrossings (gaussianSample n) (fun X => fieldF β X) 0 T).toReal

/-! ### The gap the window covers -/

/-- **`√β = o(e^{-ω(β)/4}√(β log β))`.**  This is why line 2 of
`eq:main-eq-form` is weaker than the `O(√β)` of `prop:main-int`: since
`ω(β) ≪ log log β`, one has `e^{ω(β)/4} ≪ √(log β)`.

Source: arXiv:2412.09080v3, `sec: sketch`, the closing paragraph. -/
theorem isLittleO_sqrt_window {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) {B : ℕ → ℝ}
    (hB : Tendsto B atTop atTop) :
    (fun k => Real.sqrt (B k)) =o[atTop]
      fun k => Real.exp (-(ω (B k)) / 4) * Real.sqrt (B k * Real.log (B k)) := by
  have hωB : Tendsto (fun k => ω (B k)) atTop atTop := hω.lower.comp hB
  have h1 : (fun k => Real.exp (ω (B k) / 4) * Real.sqrt (B k)) =O[atTop]
      fun k => Real.exp (ω (B k) / 2) * Real.sqrt (B k) := by
    refine IsBigO.of_bound 1 ?_
    filter_upwards [hωB.eventually_ge_atTop 0] with k hk
    rw [one_mul, Real.norm_of_nonneg (by positivity), Real.norm_of_nonneg (by positivity)]
    exact mul_le_mul_of_nonneg_right (Real.exp_le_exp.mpr (by linarith)) (Real.sqrt_nonneg _)
  have h3 := (isBigO_refl (fun k => Real.exp (-(ω (B k)) / 4)) atTop).mul_isLittleO
    (h1.trans_isLittleO (isLittleO_tail_sqrt hω hB))
  refine h3.congr' ?_ (by rfl)
  filter_upwards with k
  rw [← mul_assoc, ← Real.exp_add, show -(ω (B k)) / 4 + ω (B k) / 4 = 0 from by ring,
    Real.exp_zero, one_mul]

/-! ### The three lines of `eq:main-eq-form` -/

/-- The identity in expectation, read on the real shadows. -/
theorem expectedUpcrossingsReal_eq {β : ℝ} {n : ℕ} {T : Set ℝ}
    (hid : expectedUpcrossings (gaussianSample n) (fun X => fieldF β X) 0 T
      = expectedModes β n T) :
    expectedUpcrossingsReal β n T = expectedModesReal β n T := by
  rw [expectedUpcrossingsReal, hid, expectedModesReal]

/-- **Line 1 of `eq:main-eq-form`:** `E U₀(F_n, T) ≍ √(β log β)`, given
`prop:main-int` point 1 and the identity in expectation.

Source: arXiv:2412.09080v3, `eq:main-eq-form`. -/
theorem main_eq_form_T (N : ℕ → ℕ) (B : ℕ → ℝ) (ω : ℝ → ℝ)
    (hid : ∀ k, expectedUpcrossings (gaussianSample (N k)) (fun X => fieldF (B k) X) 0
      (intervalT (N k) (B k) (ω (B k))) = expectedModes (B k) (N k) (intervalT (N k) (B k) (ω (B k))))
    (hint : (fun k => expectedModesReal (B k) (N k) (intervalT (N k) (B k) (ω (B k))))
      =Θ[atTop] fun k => Real.sqrt (B k * Real.log (B k))) :
    (fun k => expectedUpcrossingsReal (B k) (N k) (intervalT (N k) (B k) (ω (B k))))
      =Θ[atTop] fun k => Real.sqrt (B k * Real.log (B k)) := by
  rw [funext fun k => expectedUpcrossingsReal_eq (hid k)]
  exact hint

/-- **Line 2 of `eq:main-eq-form`, in the sharper form the hypotheses give:**
`E U₀(F_n, T')` is finite and `o(e^{-ω(β)/4}√(β log β))`.  The source writes
`≲`; what `prop:main-int` point 2 and the window give is `o`.  Finiteness is
carried in and out, since `toReal` reads `∞` as `0`.

Source: arXiv:2412.09080v3, `eq:main-eq-form`. -/
theorem main_eq_form_T'_isLittleO (N : ℕ → ℕ) {B : ℕ → ℝ} (hB : Tendsto B atTop atTop)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω)
    (hid : ∀ k, expectedUpcrossings (gaussianSample (N k)) (fun X => fieldF (B k) X) 0
      (intervalT' (N k) (B k)) = expectedModes (B k) (N k) (intervalT' (N k) (B k)))
    (hint : (∀ᶠ k in atTop, expectedModes (B k) (N k) (intervalT' (N k) (B k)) ≠ ∞) ∧
      (fun k => expectedModesReal (B k) (N k) (intervalT' (N k) (B k))) =O[atTop]
        fun k => Real.sqrt (B k)) :
    (∀ᶠ k in atTop, expectedUpcrossings (gaussianSample (N k)) (fun X => fieldF (B k) X) 0
      (intervalT' (N k) (B k)) ≠ ∞) ∧
    (fun k => expectedUpcrossingsReal (B k) (N k) (intervalT' (N k) (B k))) =o[atTop]
      fun k => Real.exp (-(ω (B k)) / 4) * Real.sqrt (B k * Real.log (B k)) := by
  refine ⟨hint.1.mono fun k hk => (hid k).trans_ne hk, ?_⟩
  rw [funext fun k => expectedUpcrossingsReal_eq (hid k)]
  exact hint.2.trans_isLittleO (isLittleO_sqrt_window hω hB)

/-- **Line 2 of `eq:main-eq-form`, as the source writes it:**
`E U₀(F_n, T') ≲ e^{-ω(β)/4}√(β log β)`, with finiteness.

Source: arXiv:2412.09080v3, `eq:main-eq-form`. -/
theorem main_eq_form_T' (N : ℕ → ℕ) {B : ℕ → ℝ} (hB : Tendsto B atTop atTop)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω)
    (hid : ∀ k, expectedUpcrossings (gaussianSample (N k)) (fun X => fieldF (B k) X) 0
      (intervalT' (N k) (B k)) = expectedModes (B k) (N k) (intervalT' (N k) (B k)))
    (hint : (∀ᶠ k in atTop, expectedModes (B k) (N k) (intervalT' (N k) (B k)) ≠ ∞) ∧
      (fun k => expectedModesReal (B k) (N k) (intervalT' (N k) (B k))) =O[atTop]
        fun k => Real.sqrt (B k)) :
    (∀ᶠ k in atTop, expectedUpcrossings (gaussianSample (N k)) (fun X => fieldF (B k) X) 0
      (intervalT' (N k) (B k)) ≠ ∞) ∧
    (fun k => expectedUpcrossingsReal (B k) (N k) (intervalT' (N k) (B k))) =O[atTop]
      fun k => Real.exp (-(ω (B k)) / 4) * Real.sqrt (B k * Real.log (B k)) :=
  (main_eq_form_T'_isLittleO N hB hω hid hint).imp_right IsLittleO.isBigO

/-- **Line 3 of `eq:main-eq-form`:** `E U₀(F_n, ℝ ∖ T) ≲ e^{ω(β)/2}√β`, with
finiteness, given `prop:main-tail` and the identity in expectation.

Source: arXiv:2412.09080v3, `eq:main-eq-form`. -/
theorem main_eq_form_tail (N : ℕ → ℕ) (B : ℕ → ℝ) (ω : ℝ → ℝ)
    (hid : ∀ k, expectedUpcrossings (gaussianSample (N k)) (fun X => fieldF (B k) X) 0
      (intervalT (N k) (B k) (ω (B k)))ᶜ = expectedModes (B k) (N k) (intervalT (N k) (B k) (ω (B k)))ᶜ)
    (htail : (∀ᶠ k in atTop, expectedModes (B k) (N k) (intervalT (N k) (B k) (ω (B k)))ᶜ ≠ ∞) ∧
      (fun k => expectedModesReal (B k) (N k) (intervalT (N k) (B k) (ω (B k)))ᶜ)
        =O[atTop] fun k => Real.exp (ω (B k) / 2) * Real.sqrt (B k)) :
    (∀ᶠ k in atTop, expectedUpcrossings (gaussianSample (N k)) (fun X => fieldF (B k) X) 0
      (intervalT (N k) (B k) (ω (B k)))ᶜ ≠ ∞) ∧
    (fun k => expectedUpcrossingsReal (B k) (N k) (intervalT (N k) (B k) (ω (B k)))ᶜ)
      =O[atTop] fun k => Real.exp (ω (B k) / 2) * Real.sqrt (B k) := by
  refine ⟨htail.1.mono fun k hk => (hid k).trans_ne hk, ?_⟩
  rw [funext fun k => expectedUpcrossingsReal_eq (hid k)]
  exact htail.2

/-- The hypotheses above are satisfiable: the regime `β = n` and the window
`ω(β) = √(log log β)` of `Section1_Sketch.lean`, with the identity holding
trivially of the counts themselves. -/
example : Tendsto (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop atTop ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) ∧
    (fun _ : ℕ => (0 : ℝ)) =O[atTop] fun k : ℕ => Real.sqrt ((k + 1 : ℕ) : ℝ) := by
  refine ⟨by push_cast; exact tendsto_natSucc_atTop, ?_, isBigO_zero _ _⟩
  refine ⟨Real.tendsto_sqrt_atTop.comp (Real.tendsto_log_atTop.comp Real.tendsto_log_atTop), ?_⟩
  have hll : Tendsto (fun β : ℝ => Real.log (Real.log β)) atTop atTop :=
    Real.tendsto_log_atTop.comp Real.tendsto_log_atTop
  have h := (isLittleO_rpow_rpow_atTop (by norm_num : (1 : ℝ) / 2 < 1)).comp_tendsto hll
  simp only [Function.comp_def, Real.rpow_one] at h
  refine h.congr' ?_ (by rfl)
  filter_upwards with β
  rw [Real.sqrt_eq_rpow]

end Modes
end Transformer
