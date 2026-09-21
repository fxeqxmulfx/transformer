import Transformer.Modes.Section4_ScaleSpace
import Mathlib.Probability.Moments.Basic

/-
# The number of modes of a Gaussian KDE — the modes outside `T`

§4.2 of arXiv:2412.09080v3, `sec:tail`: the computation proving
`prop:main-tail` from `lem:scale-space`.

**What the source says and what is carried here.**

* The display is proved, `expectedModes_compl_le`, with `lem:scale-space`
  carried as a hypothesis: the tail bound `P(X ≥ a) ≤ e^{-a²/2}` on each side
  is the Chernoff bound, the expectation of the count is linearity, and the
  rest is arithmetic.  It needs `T` to be a genuine interval,
  `2 log n - log β - ω(β) > 0`, as `lem:scale-space` is only stated for
  `a > 0`.  The last step `2√(β e^{ω(β)}) ≪ √(β log β)` is
  `isLittleO_tail_sqrt`.

Source: arXiv:2412.09080v3, §4.2, proof of `prop:main-tail`.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal

namespace Transformer
namespace Modes

variable {n : ℕ}

/-! ### The Gaussian tail and the expected count -/

/-- **The Gaussian tail**, upper side: `P(X ≥ a) ≤ e^{-a²/2}` for `a ≥ 0`, by
Chernoff.  arXiv:2412.09080v3, proof of `prop:main-tail`. -/
theorem gaussianReal_Ici_le {a : ℝ} (ha : 0 ≤ a) :
    gaussianReal 0 1 (Set.Ici a) ≤ ENNReal.ofReal (Real.exp (-(a ^ 2) / 2)) := by
  have h := measure_ge_le_exp_mul_mgf (μ := gaussianReal 0 1) (X := id) a ha
    (integrable_exp_mul_gaussianReal a)
  rw [mgf_id_gaussianReal, ← Real.exp_add] at h
  rw [← ofReal_measureReal (measure_ne_top _ _)]
  refine ENNReal.ofReal_le_ofReal (h.trans_eq ?_)
  congr 1
  push_cast
  ring

/-- **The Gaussian tail**, lower side: `P(X ≤ -a) ≤ e^{-a²/2}` for `a ≥ 0`.
arXiv:2412.09080v3, proof of `prop:main-tail`. -/
theorem gaussianReal_Iic_le {a : ℝ} (ha : 0 ≤ a) :
    gaussianReal 0 1 (Set.Iic (-a)) ≤ ENNReal.ofReal (Real.exp (-(a ^ 2) / 2)) := by
  have hint : Integrable (fun x => Real.exp (a * (-id) x)) (gaussianReal 0 1) := by
    simpa [mul_neg, ← neg_mul] using integrable_exp_mul_gaussianReal (μ := 0) (v := 1) (-a)
  have h := measure_ge_le_exp_mul_mgf (μ := gaussianReal 0 1) (X := -id) a ha hint
  rw [mgf_neg, mgf_id_gaussianReal, ← Real.exp_add] at h
  have hset : {x : ℝ | a ≤ (-id) x} = Set.Iic (-a) := by
    ext x; simp [le_neg]
  rw [hset] at h
  rw [← ofReal_measureReal (measure_ne_top _ _)]
  refine ENNReal.ofReal_le_ofReal (h.trans_eq ?_)
  congr 1
  push_cast
  ring

/-- The count as a sum of indicators. -/
theorem countIn_eq_sum (X : Idx n → ℝ) (S : Set ℝ) :
    (countIn X S : ℝ≥0∞) = ∑ i : Idx n, (Function.eval i ⁻¹' S).indicator 1 X := by
  rw [countIn, Finset.card_filter]
  push_cast
  refine Finset.sum_congr rfl fun i _ => ?_
  by_cases h : X i ∈ S <;> simp [h]

/-- The count is a measurable function of the sample. -/
theorem measurable_countIn {S : Set ℝ} (hS : MeasurableSet S) :
    Measurable fun X : Idx n → ℝ => (countIn X S : ℝ≥0∞) := by
  simp_rw [countIn_eq_sum]
  exact Finset.measurable_sum _ fun i _ =>
    measurable_const.indicator (measurable_pi_apply i hS)

/-- **Linearity of expectation:** `𝔼|{i : Xᵢ ∈ S}| = n P(X ∈ S)`.
arXiv:2412.09080v3, proof of `prop:main-tail`. -/
theorem lintegral_countIn {S : Set ℝ} (hS : MeasurableSet S) :
    ∫⁻ X, (countIn X S : ℝ≥0∞) ∂gaussianSample n = n * gaussianReal 0 1 S := by
  simp_rw [countIn_eq_sum]
  rw [lintegral_finsetSum]
  · have h : ∀ i : Idx n, gaussianSample n (Function.eval i ⁻¹' S) = gaussianReal 0 1 S :=
      fun i => (measurePreserving_eval (μ := fun _ : Idx n => gaussianReal 0 1)
        i).measure_preimage hS.nullMeasurableSet
    have hm : ∀ i : Idx n, MeasurableSet (Function.eval i ⁻¹' S : Set (Idx n → ℝ)) :=
      fun i => hS.preimage (measurable_pi_apply i)
    rw [Finset.sum_congr rfl fun i _ => (lintegral_indicator_one (hm i)).trans (h i)]
    simp
  · exact fun i _ => measurable_const.indicator (measurable_pi_apply i hS)

/-! ### The proof of `prop:main-tail` -/

/-- `2n e^{-(2 log n - log β - w)/2} = 2√(β e^w)`. -/
theorem two_mul_exp_eq (hn : 0 < n) {β : ℝ} (hβ : 0 < β) (w : ℝ) :
    2 * n * Real.exp (-(2 * Real.log n - Real.log β - w) / 2)
      = 2 * Real.sqrt (β * Real.exp w) := by
  have hn' : (0 : ℝ) < n := Nat.cast_pos.2 hn
  rw [Real.sqrt_eq_rpow, Real.rpow_def_of_pos (by positivity),
    Real.log_mul hβ.ne' (Real.exp_pos w).ne', Real.log_exp]
  conv_lhs => rw [← Real.exp_log hn']
  rw [mul_assoc, ← Real.exp_add, Real.log_exp]
  congr 2
  ring

/-- **The proof of `prop:main-tail`.**  If `lem:scale-space` holds and `T` is a
genuine interval, `𝔼U₀(F_n, ℝ \ T) ≤ 𝔼|{i : Xᵢ ∉ T}| ≤ 2√(β e^{ω(β)})`.

Source: arXiv:2412.09080v3, proof of `prop:main-tail`. -/
theorem expectedModes_compl_le {β w : ℝ} (hβ : 0 < β) (hn : 0 < n)
    (hT : 0 < 2 * Real.log n - Real.log β - w)
    (hss : ∀ a : ℝ, 0 < a → ∀ X : Idx n → ℝ,
      modeCount (kde β X) (Set.Ioi a) ≤ countIn X (Set.Ici a)) :
    expectedModes β n (intervalT n β w)ᶜ ≤ ENNReal.ofReal (2 * Real.sqrt (β * Real.exp w)) := by
  set r := Real.sqrt (2 * Real.log n - Real.log β - w)
  have hr : 0 < r := Real.sqrt_pos.2 hT
  have hcompl : (intervalT n β w)ᶜ = Set.Iio (-r) ∪ Set.Ioi r := by
    ext t
    simp only [intervalT, r, Set.mem_compl_iff, Set.mem_Icc, not_and_or, not_le, Set.mem_union,
      Set.mem_Iio, Set.mem_Ioi]
  have hpt : ∀ X : Idx n → ℝ, modeCount (kde β X) (intervalT n β w)ᶜ
      ≤ countIn X (Set.Iic (-r)) + countIn X (Set.Ici r) := by
    intro X
    rw [hcompl]
    calc modeCount (kde β X) (Set.Iio (-r) ∪ Set.Ioi r)
        ≤ modeCount (kde β X) (Set.Iio (-r)) + modeCount (kde β X) (Set.Ioi r) := by
          have : modeSet (kde β X) (Set.Iio (-r) ∪ Set.Ioi r)
              = modeSet (kde β X) (Set.Iio (-r)) ∪ modeSet (kde β X) (Set.Ioi r) := by
            ext t; simp [modeSet, or_and_right]
          simp only [modeCount, this]
          exact_mod_cast Set.encard_union_le _ _
      _ ≤ _ := add_le_add (modeCount_kde_Iio_le (hss r hr) X) (hss r hr X)
  have hr2 : r ^ 2 = 2 * Real.log n - Real.log β - w := Real.sq_sqrt hT.le
  calc expectedModes β n (intervalT n β w)ᶜ
      ≤ ∫⁻ X, (countIn X (Set.Iic (-r)) + countIn X (Set.Ici r) : ℝ≥0∞) ∂gaussianSample n :=
        lintegral_mono fun X => by exact_mod_cast hpt X
    _ = n * gaussianReal 0 1 (Set.Iic (-r)) + n * gaussianReal 0 1 (Set.Ici r) := by
        rw [lintegral_add_left (measurable_countIn measurableSet_Iic),
          lintegral_countIn measurableSet_Iic, lintegral_countIn measurableSet_Ici]
    _ ≤ n * ENNReal.ofReal (Real.exp (-(r ^ 2) / 2))
          + n * ENNReal.ofReal (Real.exp (-(r ^ 2) / 2)) := by
        gcongr
        exacts [gaussianReal_Iic_le hr.le, gaussianReal_Ici_le hr.le]
    _ = ENNReal.ofReal (2 * n * Real.exp (-(r ^ 2) / 2)) := by
        rw [← two_mul, ← mul_assoc, ENNReal.ofReal_mul (by positivity),
          ENNReal.ofReal_mul (by positivity)]
        simp
    _ = ENNReal.ofReal (2 * Real.sqrt (β * Real.exp w)) := by
        rw [hr2, two_mul_exp_eq hn hβ]

/-- The hypotheses of `expectedModes_compl_le` are satisfiable: `n = 1`,
`β = 1`, `w = -1`. -/
example : (0 : ℝ) < 1 ∧ 0 < 1 ∧ 0 < 2 * Real.log (1 : ℕ) - Real.log 1 - (-1) ∧
    ∀ a : ℝ, 0 < a → ∀ X : Idx 1 → ℝ,
      modeCount (kde 1 X) (Set.Ioi a) ≤ countIn X (Set.Ici a) :=
  ⟨one_pos, one_pos, by simp, fun a _ => scale_space_one one_pos a⟩

end Modes
end Transformer
