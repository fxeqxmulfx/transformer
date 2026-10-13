import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryLimits

/-!
# Exact native retained-memory and partner coefficients

Sources: Varma et al., arXiv:2309.02390v1, appendix C product
partials and section 3 efficiency; native state/denominator/cold
feedback limits at lab commit d3bb8b8.

Rewrite the original actual parameter update as remaining decay times
its present parameter, plus a computed coefficient times its negative
retained moment, plus a computed coefficient times its current partner.
The coefficients use the complete actual denominator, generated shared
CE/clipping and physical gain. Second moments and clocks are retained.

Initialized strong native decay with legal betas makes those numerical
coefficients converge: rate*beta1/epsilon for memory, and
rate*(1-beta1)*coldCE*gain/epsilon for the partner. Greater Gen gain
gives a positive limiting difference and eventually greater actual
Gen partner coefficient than Mem on the same original path.

This is an exact current recurrence and its generated coefficient limits,
not a supplied linear gradient stream or a derivative at an invented
infinite-clock state. No parameter convergence or future coefficient
gap is assumed. A positive partner coefficient gap alone does not prove
coupled parameter/moment state dominance, pair balance, held-out success
or positive confidence. Those relative dynamics remain to be proved.
Fixed tables/uniform native decay differ from appendix C coupled-cost GD
and learned stochastic/floating-point GPTMini; no transfer is asserted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Actual coefficient of negative retained first moment. Sources:
native parameter/denominator at d3bb8b8; decay does not enter this
coefficient, and every supplied argument occurs in its expression. -/
noncomputable def gainNativeMomentCoefficient (remaining : ℕ) (genGain memGain bound b1 b2 eps rate : ℝ)
    (state : NativeSubweightState) (i : Fin 4) : ℝ :=
  rate * b1 / nextBufferDenominator b1 b2 eps (state i).variance
    (appliedGainNativeGradient remaining genGain memGain bound state i) (state i).clock

/-- Actual coefficient of the present physical partner. Sources:
appendix C true gained chain factor and native denominator at d3bb8b8;
it retains the original current shared clipping/CE computation. -/
noncomputable def gainNativePartnerCoefficient (remaining : ℕ) (genGain memGain bound b1 b2 eps rate : ℝ)
    (state : NativeSubweightState) (i : Fin 4) : ℝ :=
  rate * (1 - b1) * gainCEGradientScale remaining genGain memGain bound state * nativeFactorGain genGain memGain i /
    nextBufferDenominator b1 b2 eps (state i).variance
      (appliedGainNativeGradient remaining genGain memGain bound state i) (state i).clock

/-- The original actual parameter update has these exact retained
moment and partner coefficients. Sources: appendix C product partials
and native insertions/corrections at d3bb8b8; no buffer reset, prescribed
input or replacement of the full denominator is used. -/
theorem gain_native_parameter_memory_partner (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (i : Fin 4) :
    (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i).parameter =
      (1 - rate * decay) * (state i).parameter +
        gainNativeMomentCoefficient remaining genGain memGain bound b1 b2 eps rate state i * (-(state i).moment) +
        gainNativePartnerCoefficient remaining genGain memGain bound b1 b2 eps rate state i * (state (nativeFactorPartner i)).parameter := by
  dsimp only [gainNativeStep]
  rw [scalar_parameter_denominator]
  dsimp only [gainNativeMomentCoefficient, gainNativePartnerCoefficient, scalarNativeStep]
  rw [gain_native_applied_gradient_scale]
  ring

/-- Both exact current coefficients have their generated cold limits
under initialized legal-beta strong decay. Sources: appendix C gained
partners and native limits at d3bb8b8; no future parameter, buffer,
denominator or feedback convergence is supplied independently. -/
theorem gain_native_memory_coefficients_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∀ i, Tendsto (fun n => gainNativeMomentCoefficient remaining genGain memGain bound b1 b2 eps rate (path n) i)
      atTop (nhds (rate * b1 / eps)) ∧
      Tendsto (fun n => gainNativePartnerCoefficient remaining genGain memGain bound b1 b2 eps rate (path n) i)
        atTop (nhds (rate * (1 - b1) * coldGainCEGradientScale remaining bound * nativeFactorGain genGain memGain i / eps)) := by
  have hden := gain_native_memory_denominator_and_clock_limits remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  have hscale := gain_native_memory_ce_scale_tendsto_cold remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  have hconstant : Tendsto (fun _ : ℕ => rate * b1) atTop (nhds (rate * b1)) := tendsto_const_nhds
  dsimp only
  intro i
  constructor
  · simpa only [gainNativeMomentCoefficient, Pi.div_def] using hconstant.div (hden i).1 (ne_of_gt he)
  · have hn := (hscale.2.const_mul (rate * (1 - b1))).mul_const (nativeFactorGain genGain memGain i)
    simpa only [gainNativePartnerCoefficient, Pi.div_def] using hn.div (hden i).1 (ne_of_gt he)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

/-- Greater physical Gen gain gives a generated positive limiting
partner-coefficient gap and a strictly ordered actual coefficient tail.
Sources: section 3 efficiency and actual native coefficients at d3bb8b8;
this component advantage is not assumed to imply coupled state selection. -/
theorem gain_native_memory_eventual_partner_advantage (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hmem : 0 < memGain) (hbetter : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hstrong : genGain * (2 - b1) / ((1 - b1) * eps) < decay) :
    let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    let gap := rate * (1 - b1) * coldGainCEGradientScale remaining bound * (genGain - memGain) / eps
    0 < gap ∧ Tendsto (fun n =>
      gainNativePartnerCoefficient remaining genGain memGain bound b1 b2 eps rate (path n) 0 -
        gainNativePartnerCoefficient remaining genGain memGain bound b1 b2 eps rate (path n) 2) atTop (nhds gap) ∧
      ∃ start : ℕ, ∀ n, start ≤ n →
        gainNativePartnerCoefficient remaining genGain memGain bound b1 b2 eps rate (path n) 2 <
          gainNativePartnerCoefficient remaining genGain memGain bound b1 b2 eps rate (path n) 0 := by
  have hcoeff := gain_native_memory_coefficients_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
    (lt_trans hmem hbetter) hmem (le_of_lt hbetter) hclip hb1 h1 hb2 h2 he heta hd hs hstrong
  have hg := (hcoeff 0).2
  have hm := (hcoeff 2).2
  have hl := hg.sub hm
  let gap := rate * (1 - b1) * coldGainCEGradientScale remaining bound * (genGain - memGain) / eps
  have hgap : 0 < gap := div_pos (mul_pos (mul_pos
    (mul_pos heta (by linarith only [h1])) (cold_gain_ce_gradient_scale_pos remaining bound hclip))
    (by linarith only [hbetter])) he
  have heq : rate * (1 - b1) * coldGainCEGradientScale remaining bound * nativeFactorGain genGain memGain 0 / eps -
      rate * (1 - b1) * coldGainCEGradientScale remaining bound * nativeFactorGain genGain memGain 2 / eps = gap := by
    change rate * (1 - b1) * coldGainCEGradientScale remaining bound * genGain / eps -
      rate * (1 - b1) * coldGainCEGradientScale remaining bound * memGain / eps = gap
    dsimp only [gap]
    ring
  rw [heq] at hl
  obtain ⟨start, htail⟩ := eventually_atTop.mp (hl.eventually_const_lt hgap)
  refine ⟨hgap, hl, start, ?_⟩
  intro n hn
  linarith only [htail n hn]

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧
    (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    (3 : ℝ) * (2 - 9 / 10) / ((1 - 9 / 10) * 1) < 100 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

end Transformer.Grokking.CircuitEfficiency
