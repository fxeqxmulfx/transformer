import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryIntervals

/-!
# Separated full retained pair rates on one actual legal-beta path

Sources: Varma et al., arXiv:2309.02390v1, section 3 competing
circuits and appendix C product factors; native generated pair
coefficient intervals at lab commit 9a67dbe.

Pin a binary task with Gen/Mem gains 3/2, native beta1=0.9,
beta2=0.98, epsilon/cap 1, decay 100 and rate 0.001. Both retained
buffers and completed clocks are kept. Initial numerical sign data
and the verified static strong-decay inequalities generate coordinate
interval tails, then full bounds on the original pair measurements.

The Gen measurement uses moment weight 0.08 and is nonincreasing but
falls no faster than factor 0.911. The Mem measurement uses weight
0.095 and falls at least as fast as factor 0.910. These are inequalities
on the actual original nonlinear clipped-CE path, not on a prescribed
constant matrix or a supplied future coefficient/parameter limit.

Comparison of initialized powers and ratios remains the next step.
Pair measurements include memory: their dominance alone is not a
physical product margin or a confidence result. Fixed gained tables
and uniform native decay differ from appendix C coupled-cost GD.
Learned stochastic/floating-point GPTMini transfer remains open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Original native binary gained-CE path at the explicit legal-beta
strong-decay configuration. Sources: appendix C factors and native
interval configuration at 9a67dbe; the full initialization is retained. -/
noncomputable def fixedGainMemoryPath (initial : NativeSubweightState) : ℕ → NativeSubweightState :=
  gainNativePath 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial

/-- Gen's present parameters and both negative retained moments.
Source: native pair measurement at 9a67dbe; no success is encoded. -/
noncomputable def fixedGainGenMemoryMass (state : NativeSubweightState) : ℝ :=
  gainNativePairMemoryMass (2 / 25) state 0

/-- Mem's present parameters and both negative retained moments.
Source: native pair measurement at 9a67dbe; no success is encoded. -/
noncomputable def fixedGainMemMemoryMass (state : NativeSubweightState) : ℝ :=
  gainNativePairMemoryMass (19 / 200) state 2

/-- The pinned original path keeps all four physical parameters and
both retained buffers in their numerical sign region. Sources: appendix
C actual partner inputs and native sign iteration at 9a67dbe; no future
parameter signs are supplied as an independent premise. -/
theorem fixed_gain_memory_signs (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ∀ n, NonnegativeNativeState (fixedGainMemoryPath initial n) := by
  intro n
  exact gain_native_nonnegative_path 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial n
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Both true partner pairs eventually obey their separate computed
coordinate intervals. Sources: section 3 gain ordering and original
native initialized limits at 9a67dbe; only current initial sign data
are assumed, not a future convergence or interval premise. -/
theorem fixed_gain_memory_coefficient_tail (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ∃ start : ℕ, ∀ n, start ≤ n →
      (∀ j : Fin 4, j = 0 ∨ j = 1 →
        ((911 / 1000 : ℝ) ≤ gainNativeWeightedParameterCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (2 / 25) (fixedGainMemoryPath initial n) j ∧
          gainNativeWeightedParameterCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (2 / 25) (fixedGainMemoryPath initial n) j ≤ 1) ∧
        ((911 / 1000 : ℝ) * (2 / 25) ≤ gainNativeWeightedMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) (2 / 25) (fixedGainMemoryPath initial n) j ∧
          gainNativeWeightedMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) (2 / 25) (fixedGainMemoryPath initial n) j ≤ 1 * (2 / 25))) ∧
      (∀ j : Fin 4, j = 2 ∨ j = 3 →
        (0 ≤ gainNativeWeightedParameterCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (19 / 200) (fixedGainMemoryPath initial n) j ∧
          gainNativeWeightedParameterCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (19 / 200) (fixedGainMemoryPath initial n) j ≤ 91 / 100) ∧
        (0 * (19 / 200 : ℝ) ≤ gainNativeWeightedMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) (19 / 200) (fixedGainMemoryPath initial n) j ∧
          gainNativeWeightedMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) (19 / 200) (fixedGainMemoryPath initial n) j ≤ (91 / 100) * (19 / 200))) := by
  have hgen (j : Fin 4) (hj : j = 0 ∨ j = 1) :
      ∃ start : ℕ, ∀ n, start ≤ n →
        ((911 / 1000 : ℝ) ≤ gainNativeWeightedParameterCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (2 / 25) (fixedGainMemoryPath initial n) j ∧
          gainNativeWeightedParameterCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (2 / 25) (fixedGainMemoryPath initial n) j ≤ 1) ∧
        ((911 / 1000 : ℝ) * (2 / 25) ≤ gainNativeWeightedMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) (2 / 25) (fixedGainMemoryPath initial n) j ∧
          gainNativeWeightedMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) (2 / 25) (fixedGainMemoryPath initial n) j ≤ 1 * (2 / 25)) := by
    rcases hj with rfl | rfl <;>
      apply gain_native_weighted_coefficient_interval_tail 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (2 / 25) (911 / 1000) 1 initial _ <;>
      first | exact hs | norm_num [coldGainWeightedParameterCoefficient, coldGainWeightedMomentCoefficient, coldGainCEGradientScale, nativeFactorGain]
  have hmem (j : Fin 4) (hj : j = 2 ∨ j = 3) :
      ∃ start : ℕ, ∀ n, start ≤ n →
        (0 ≤ gainNativeWeightedParameterCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (19 / 200) (fixedGainMemoryPath initial n) j ∧
          gainNativeWeightedParameterCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (19 / 200) (fixedGainMemoryPath initial n) j ≤ 91 / 100) ∧
        (0 * (19 / 200 : ℝ) ≤ gainNativeWeightedMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) (19 / 200) (fixedGainMemoryPath initial n) j ∧
          gainNativeWeightedMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) (19 / 200) (fixedGainMemoryPath initial n) j ≤ (91 / 100) * (19 / 200)) := by
    rcases hj with rfl | rfl <;>
      apply gain_native_weighted_coefficient_interval_tail 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (19 / 200) 0 (91 / 100) initial _ <;>
      first | exact hs | norm_num [coldGainWeightedParameterCoefficient, coldGainWeightedMomentCoefficient, coldGainCEGradientScale, nativeFactorGain]
  obtain ⟨s0, h0⟩ := hgen 0 (Or.inl rfl)
  obtain ⟨s1, h1⟩ := hgen 1 (Or.inr rfl)
  obtain ⟨s2, h2⟩ := hmem 2 (Or.inl rfl)
  obtain ⟨s3, h3⟩ := hmem 3 (Or.inr rfl)
  refine ⟨max (max s0 s1) (max s2 s3), ?_⟩
  intro n hn
  have h01 := le_trans (le_max_left _ _) hn
  have h23 := le_trans (le_max_right _ _) hn
  constructor
  · intro j hj
    rcases hj with rfl | rfl
    · exact h0 n (le_trans (le_max_left _ _) h01)
    · exact h1 n (le_trans (le_max_right _ _) h01)
  · intro j hj
    rcases hj with rfl | rfl
    · exact h2 n (le_trans (le_max_left _ _) h23)
    · exact h3 n (le_trans (le_max_right _ _) h23)

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Separated rates bound the complete original actual pair weights:
Gen is nonincreasing and falls no faster than 0.911, while Mem falls
at least as fast as 0.910. Sources: section 3 competition and native
pair law at 9a67dbe; a finite tail is generated from initial signs. -/
theorem fixed_gain_memory_pair_rate_tail (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ∃ start : ℕ, ∀ n, start ≤ n →
      (911 / 1000 : ℝ) * fixedGainGenMemoryMass (fixedGainMemoryPath initial n) ≤
        fixedGainGenMemoryMass (fixedGainMemoryPath initial (n + 1)) ∧
      fixedGainGenMemoryMass (fixedGainMemoryPath initial (n + 1)) ≤
        fixedGainGenMemoryMass (fixedGainMemoryPath initial n) ∧
      fixedGainMemMemoryMass (fixedGainMemoryPath initial (n + 1)) ≤
        (91 / 100 : ℝ) * fixedGainMemMemoryMass (fixedGainMemoryPath initial n) := by
  obtain ⟨start, htail⟩ := fixed_gain_memory_coefficient_tail initial hs
  refine ⟨start, ?_⟩
  intro n hn
  have hsnow := fixed_gain_memory_signs initial hs n
  have hc := htail n hn
  have hg := gain_native_pair_memory_comparison 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000)
    (2 / 25) (911 / 1000) 1 (fixedGainMemoryPath initial n) 0 hsnow
    (fun j hj => (hc.1 j hj).1) (fun j hj => (hc.1 j hj).2)
  have hm := gain_native_pair_memory_comparison 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000)
    (19 / 200) 0 (91 / 100) (fixedGainMemoryPath initial n) 2 hsnow
    (fun j hj => (hc.2 j hj).1) (fun j hj => (hc.2 j hj).2)
  change (911 / 1000 : ℝ) * fixedGainGenMemoryMass (fixedGainMemoryPath initial n) ≤
    fixedGainGenMemoryMass (fixedGainMemoryPath initial (n + 1)) ∧
    fixedGainGenMemoryMass (fixedGainMemoryPath initial (n + 1)) ≤
      fixedGainGenMemoryMass (fixedGainMemoryPath initial n) ∧
    fixedGainMemMemoryMass (fixedGainMemoryPath initial (n + 1)) ≤
      (91 / 100 : ℝ) * fixedGainMemMemoryMass (fixedGainMemoryPath initial n)
  exact ⟨hg.1, by simpa only [one_mul, fixedGainGenMemoryMass, fixedGainMemoryPath, gainNativePath] using hg.2, hm.2⟩

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.CircuitEfficiency
