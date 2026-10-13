import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryRatio

/-!
# Transfer from original retained measurements to physical parameters

Sources: Varma et al., arXiv:2309.02390v1, section 3 circuit strength
and appendix C physical products; native denominator and relative
measurement laws at lab commit 6ea7829.

The actual full denominator tends to epsilon=1 on the pinned legal-beta
path. Its generated tail ceiling 2 and the original native update bound
each newly retained negative first moment by 2000 times its own physical
parameter. This bound uses the actual next moment, not the current CE
input or a reset. Gen's 0.08-weighted complete measurement is therefore
at most 161 times the current true parameter sum on a generated tail.

Combine that coverage with the already derived actual relative
measurement selection. Every fixed positive multiple of Gen's true
parameter sum eventually exceeds Mem's true parameter sum permanently.
No future buffer, denominator, parameter or success condition is supplied.
Both retained buffers and original correction clocks remain in the path.

A parameter-sum advantage still needs within-pair information before it
implies a physical product margin. Positive partner mixing is the next
route; complete signed pair balance may be stronger than selection needs.
Fixed gained tables/uniform native decay differ from appendix C coupled
cost/GD. Learned stochastic/floating-point GPTMini transfer remains open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- A present ceiling on the original complete denominator bounds
the newly retained negative moment by its own new physical parameter.
Sources: native parameter/denominator law at 6ea7829 and appendix C
actual CE; the ceiling is a current condition, generated below. -/
theorem fixed_gain_memory_moment_parameter_ceiling (state : NativeSubweightState) (i : Fin 4)
    (hs : NonnegativeNativeState state)
    (hden : nextBufferDenominator (9 / 10) (49 / 50) 1 (state i).variance
      (appliedGainNativeGradient 0 3 2 1 state i) (state i).clock ≤ 2) :
    -(gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state i).moment ≤
      2000 * (gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state i).parameter := by
  have hg := gain_native_applied_gradient_nonpos 0 3 2 1 state i (by norm_num)
    (le_of_lt (gain_native_factor_gain_pos 3 2 (by norm_num) (by norm_num) i)) (hs _).1
  have hf := scalar_parameter_denominator_floor (9 / 10) (49 / 50) 1 100 (1 / 1000)
    (appliedGainNativeGradient 0 3 2 1 state i) 2 (state i)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (hs i).2.1 hg hden
  change (1 - (1 / 1000 : ℝ) * 100) * (state i).parameter + (1 / 1000) *
    (-(gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state i).moment) / 2 ≤
      (gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state i).parameter at hf
  nlinarith only [hf, (hs i).1]

example :
    let state := seededNativeSubweights ((0, 0), (0, 0))
    NonnegativeNativeState state ∧ nextBufferDenominator (9 / 10) (49 / 50) 1 (state 0).variance
      (appliedGainNativeGradient 0 3 2 1 state 0) (state 0).clock ≤ 2 := by
  dsimp only
  refine ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_⟩
  have hg : appliedGainNativeGradient 0 3 2 1 (seededNativeSubweights ((0, 0), (0, 0))) 0 = 0 := by
    rw [gain_native_applied_gradient_scale]
    norm_num [nativeFactorPartner, seededNativeSubweights, seededScalarState]
  rw [hg]
  norm_num [nextBufferDenominator, seededNativeSubweights, seededScalarState]

/-- Current own-coordinate moment ceilings give the numerical Gen
measurement ceiling 161. Sources: appendix C physical pair and the
0.08 native measurement at 6ea7829; no parameter sign or future
trajectory is needed for this present-state algebraic implication. -/
theorem fixed_gain_gen_memory_physical_ceiling (state : NativeSubweightState)
    (h0 : -(state 0).moment ≤ 2000 * (state 0).parameter)
    (h1 : -(state 1).moment ≤ 2000 * (state 1).parameter) :
    fixedGainGenMemoryMass state ≤ 161 * gainGenParameterMass state := by
  unfold fixedGainGenMemoryMass gainNativePairMemoryMass gainGenParameterMass
  change (state 0).parameter + (state 1).parameter + (2 / 25) * (-(state 0).moment - (state 1).moment) ≤ _
  linarith only [h0, h1]

example : -(seededNativeSubweights ((1, 2), (0, 0)) 0).moment ≤
    2000 * (seededNativeSubweights ((1, 2), (0, 0)) 0).parameter ∧
    -(seededNativeSubweights ((1, 2), (0, 0)) 1).moment ≤
      2000 * (seededNativeSubweights ((1, 2), (0, 0)) 1).parameter := by
  norm_num [seededNativeSubweights, seededScalarState]

/-- Original initialized denominator limits generate a complete
moment/parameter coverage tail and the Gen measurement ceiling 161.
Sources: appendix C physical factors and native limits at 6ea7829;
no future denominator or moment bound is independently supplied. -/
theorem fixed_gain_memory_physical_coverage_tail (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ∃ start : ℕ, ∀ n, start ≤ n →
      (∀ i, -(fixedGainMemoryPath initial (n + 1) i).moment ≤
        2000 * (fixedGainMemoryPath initial (n + 1) i).parameter) ∧
      fixedGainGenMemoryMass (fixedGainMemoryPath initial (n + 1)) ≤
        161 * gainGenParameterMass (fixedGainMemoryPath initial (n + 1)) := by
  have hd := gain_native_memory_denominator_and_clock_limits 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs (by norm_num)
  have hall : ∀ᶠ n in atTop, ∀ i : Fin 4, nextBufferDenominator (9 / 10) (49 / 50) 1
      (fixedGainMemoryPath initial n i).variance (appliedGainNativeGradient 0 3 2 1 (fixedGainMemoryPath initial n) i)
        (fixedGainMemoryPath initial n i).clock < 2 :=
    eventually_all.mpr (fun i => (hd i).1.eventually_lt_const (by norm_num))
  obtain ⟨start, htail⟩ := eventually_atTop.mp hall
  refine ⟨start, ?_⟩
  intro n hn
  have hm : ∀ i, -(fixedGainMemoryPath initial (n + 1) i).moment ≤
      2000 * (fixedGainMemoryPath initial (n + 1) i).parameter := by
    intro i
    exact fixed_gain_memory_moment_parameter_ceiling (fixedGainMemoryPath initial n) i
      (fixed_gain_memory_signs initial hs n) (le_of_lt (htail n hn i))
  exact ⟨hm, fixed_gain_gen_memory_physical_ceiling _ (hm 0) (hm 1)⟩

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Every positive fixed multiple of the actual Gen parameter sum
eventually exceeds the actual Mem parameter sum permanently. Sources:
section 3 physical strength and original native relative/coverage laws
at 6ea7829; no future positive Gen parameter limit or pair balance is
assumed. Product selection remains a separate obligation. -/
theorem fixed_gain_memory_physical_mass_dominance (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) (hgen : 0 < fixedGainGenMemoryMass initial)
    (multiple : ℝ) (hmultiple : 0 < multiple) :
    ∃ start : ℕ, ∀ n, start ≤ n →
      (fixedGainMemoryPath initial n 2).parameter + (fixedGainMemoryPath initial n 3).parameter <
        multiple * gainGenParameterMass (fixedGainMemoryPath initial n) := by
  have hfrac : 0 < multiple / 161 := div_pos hmultiple (by norm_num)
  obtain ⟨coverStart, hcover⟩ := fixed_gain_memory_physical_coverage_tail initial hs
  obtain ⟨relativeStart, hrelative⟩ := fixed_gain_memory_relative_dominance initial hs hgen (multiple / 161) hfrac
  refine ⟨max (coverStart + 1) relativeStart, ?_⟩
  intro n hn
  have hcn := le_trans (le_max_left _ _) hn
  have hrn := le_trans (le_max_right _ _) hn
  cases n with
  | zero => omega
  | succ k =>
    have hcg := (hcover k (by omega)).2
    have hr := hrelative (k + 1) hrn
    have hm := gain_native_pair_memory_mass_covers (19 / 200) (fixedGainMemoryPath initial (k + 1)) 2
      (by norm_num) (fixed_gain_memory_signs initial hs (k + 1))
    have hmBound := hm.2
    change (fixedGainMemoryPath initial (k + 1) 2).parameter + (fixedGainMemoryPath initial (k + 1) 3).parameter ≤
      fixedGainMemMemoryMass (fixedGainMemoryPath initial (k + 1)) at hmBound
    calc
      _ ≤ fixedGainMemMemoryMass (fixedGainMemoryPath initial (k + 1)) := hmBound
      _ < (multiple / 161) * fixedGainGenMemoryMass (fixedGainMemoryPath initial (k + 1)) := hr
      _ ≤ (multiple / 161) * (161 * gainGenParameterMass (fixedGainMemoryPath initial (k + 1))) :=
        mul_le_mul_of_nonneg_left hcg (le_of_lt hfrac)
      _ = multiple * gainGenParameterMass (fixedGainMemoryPath initial (k + 1)) := by ring

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    0 < fixedGainGenMemoryMass (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧ (0 : ℝ) < 1 / 100 := by
  exact ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [fixedGainGenMemoryMass, gainNativePairMemoryMass, seededNativeSubweights, seededScalarState, nativeFactorPartner], by norm_num⟩

end Transformer.Grokking.CircuitEfficiency
