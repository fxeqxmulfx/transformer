import Transformer.Grokking.CircuitEfficiency.SectionC_GainGrowthEnvelope

/-!
# Actual finite-clock Gen ceilings and coordinate decay floors

Sources: Varma et al., arXiv:2309.02390v1, section 3 Slow vs fast
learning and appendix C's product forward; native AdamW/clipping at
lab commit e0e5c6a. Iterate the checked retained growth envelope on
the actual closed CE path, deriving its numerical signs at each clock.
Separately iterate the parameter's pure-decay lower contribution.

The physical Gen product is controlled by the resulting parameter-mass
ceiling. These estimates retain all optimizer buffers and completed
clocks and require no parameter-limit hypothesis. The next comparison
must choose a small positive seed rather than assume future weak Gen.
No later generalization, quantitative GPTMini delay, floating-point
equivalence or convergence is proved. Fixed gains, plain CE and uniform
native decoupled decay differ from appendix C's coupled-cost GD.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- The actual retained growth weight is bounded by its seeded weight
times the derived factor to the actual number of steps. Sources:
section 3 Slow vs fast learning and native AdamW at e0e5c6a;
all signs come from the closed CE recurrence, not a future-input premise. -/
theorem gain_gen_growth_weight_path (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hdecay : 0 ≤ decay) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) :
    gainGenGrowthWeight b1 eps rate (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) ≤
      gainGenGrowthFactor b1 eps rate genGain ^ n * gainGenGrowthWeight b1 eps rate initial := by
  have hr := gain_gen_growth_factor_two_le b1 eps rate genGain h1 he heta (le_of_lt hgen)
  induction n with
  | zero => simp only [gainNativePath, pow_zero, one_mul, le_refl]
  | succ n ih =>
    have hn := gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
      hgen hmem hclip hb1 h1 hb2 h2 he heta hd hs
    calc
      gainGenGrowthWeight b1 eps rate (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial (n + 1)) ≤
          gainGenGrowthFactor b1 eps rate genGain *
            gainGenGrowthWeight b1 eps rate (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) :=
        gain_gen_growth_weight_step remaining genGain memGain bound b1 b2 eps decay rate _
          hgen hclip hb1 h1 he heta hdecay hn
      _ ≤ gainGenGrowthFactor b1 eps rate genGain *
          (gainGenGrowthFactor b1 eps rate genGain ^ n * gainGenGrowthWeight b1 eps rate initial) :=
        mul_le_mul_of_nonneg_left ih (by linarith only [hr])
      _ = gainGenGrowthFactor b1 eps rate genGain ^ (n + 1) * gainGenGrowthWeight b1 eps rate initial := by
        rw [pow_succ]
        ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 ≤ (1 / 10 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- Every actual coordinate stays above its seeded pure-decay floor.
Sources: appendix C's nonnegative partner inputs and native AdamW
at e0e5c6a; this lower bound admits decreases at positive decay. -/
theorem gain_native_parameter_path_decay_floor (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ) (i : Fin 4)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) :
    (1 - rate * decay) ^ n * (initial i).parameter ≤
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter := by
  induction n with
  | zero => simp only [pow_zero, one_mul, gainNativePath, le_refl]
  | succ n ih =>
    have hn := gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
      hgen hmem hclip hb1 h1 hb2 h2 he heta hd hs
    have hstep := gain_native_parameter_decay_floor remaining genGain memGain bound b1 b2 eps decay rate
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) i hclip
      (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)) (hn _).1
      hb1 h1 he heta (hn i).2.1
    have ht := mul_le_mul_of_nonneg_left ih hd
    rw [pow_succ]
    change (1 - rate * decay) ^ n * (1 - rate * decay) * (initial i).parameter ≤
      (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate
        (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) i).parameter
    nlinarith only [ht, hstep]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- Nonnegative physical Gen factors bound their product by squared
parameter mass. Source: appendix C's two-factor forward, with the
fixed physical gain at e0e5c6a; this intentionally uses a loose ceiling. -/
theorem gain_gen_physical_score_mass_ceiling (genGain : ℝ) (state : NativeSubweightState)
    (hg : 0 ≤ genGain) (hs : NonnegativeNativeState state) :
    physicalCircuitScore genGain (state 0).parameter (state 1).parameter ≤
      genGain * gainGenParameterMass state ^ 2 := by
  have hp := mul_nonneg (hs 0).1 (hs 1).1
  have hmass : (state 0).parameter * (state 1).parameter ≤ gainGenParameterMass state ^ 2 := by
    unfold gainGenParameterMass
    nlinarith only [hp, sq_nonneg (state 0).parameter, sq_nonneg (state 1).parameter]
  exact mul_le_mul_of_nonneg_left hmass hg

example : (0 : ℝ) ≤ 3 ∧ NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- The actual Gen logit is capped by the retained seeded weight and
the derived finite-clock factor. Sources: section 3's slow formation,
appendix C's physical product and native AdamW at e0e5c6a;
no weak future Gen score is supplied as a hypothesis. -/
theorem gain_gen_native_path_score_ceiling (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hdecay : 0 ≤ decay) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) :
    let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n
    physicalCircuitScore genGain (state 0).parameter (state 1).parameter ≤
      genGain * (gainGenGrowthFactor b1 eps rate genGain ^ n * gainGenGrowthWeight b1 eps rate initial) ^ 2 := by
  dsimp only
  have hn := gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
    hgen hmem hclip hb1 h1 hb2 h2 he heta hd hs
  have hm := gain_gen_growth_weight_covers_mass b1 eps rate _ h1 he heta hn
  have hw := gain_gen_growth_weight_path remaining genGain memGain bound b1 b2 eps decay rate initial n
    hgen hmem hclip hb1 h1 hb2 h2 he heta hdecay hd hs
  have hc := gain_gen_physical_score_mass_ceiling genGain _ (le_of_lt hgen) hn
  have hbound := le_trans hm.2 hw
  have hsq : gainGenParameterMass
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) ^ 2 ≤
      (gainGenGrowthFactor b1 eps rate genGain ^ n * gainGenGrowthWeight b1 eps rate initial) ^ 2 := by
    nlinarith only [hm.1, hbound]
  exact le_trans hc (mul_le_mul_of_nonneg_left hsq (le_of_lt hgen))

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 ≤ (1 / 10 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

end Transformer.Grokking.CircuitEfficiency
