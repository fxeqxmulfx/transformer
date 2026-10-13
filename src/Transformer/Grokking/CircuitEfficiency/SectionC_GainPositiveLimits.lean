import Transformer.Grokking.CircuitEfficiency.SectionC_GainBalancedPath

/-!
# Efficient physical circuits win on actual positive finite-limit paths

Sources: Varma et al., arXiv:2309.02390v1, section 3 Efficiency
and appendix C's test logits; retained native AdamW/clipping at lab
commit 09694da. State selection directly on the actual closed CE
recurrence. Finite convergence of all four parameters to positive
limits is explicit. Actual input and moment convergence and native
limit balance are derived, without supplied future gradients or a
successful held-out margin premise.

With Gen's physical gain greater than Mem's, uniform native balance
forces positive decay and a strictly positive limiting test margin.
The actual gained Gen and Mem products converge to that margin, and
eventually the correct class strictly beats every held-out competitor.
Actual nonzero zero-buffer paths at beta1=0.9/beta2=0.98 jointly
witness every hypothesis, including evolved buffers and growing
clocks; their positive epsilon was chosen explicitly from actual CE.

This is conditional positive-interior selection, not global
convergence from source seeds or a proof of a long grokking delay.
Boundary allocations, learned stochastic GPTMini, numerical-kernel
transfer and the source's coupled norm-cost/GD deviation remain
separate. A limiting positive margin supplies eventual decisions,
whereas finite-point CE need not tend to zero.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Actual positive-limit native feedback has positive uniform decay
and a derived strictly positive test-margin limit. Sources: appendix C
test products and section 3 Efficiency, specialized to physical gains
and retained uniform native balance at 09694da. -/
theorem gain_native_closed_positive_margin_limit (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hstep : ∀ n, state (n + 1) = gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate (state n))
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hpos : ∀ i, 0 < (reference i).parameter)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    0 < decay ∧
    0 < physicalCircuitScore genGain (reference 0).parameter (reference 1).parameter -
      physicalCircuitScore memGain (reference 2).parameter (reference 3).parameter ∧
    Tendsto (fun n => physicalCircuitScore genGain (state n 0).parameter (state n 1).parameter -
      physicalCircuitScore memGain (state n 2).parameter (state n 3).parameter) atTop
      (nhds (physicalCircuitScore genGain (reference 0).parameter (reference 1).parameter -
        physicalCircuitScore memGain (reference 2).parameter (reference 3).parameter)) := by
  have hb : ∀ i, decay * (reference i).parameter + appliedGainNativeGradient remaining genGain memGain bound reference i /
      (|appliedGainNativeGradient remaining genGain memGain bound reference i| + eps) = 0 := by
    intro i
    exact gain_native_closed_limit_balance remaining genGain memGain bound b1 b2 eps decay rate state reference i
      hstep hb1 h1 hb2 h2 he heta hp
  have ha := gain_native_positive_balanced_allocation remaining _ _ _ _ _ reference hmem hgain hclip he hpos hb
  have hg := ((hp 0).mul (hp 1)).const_mul genGain
  have hm := ((hp 2).mul (hp 3)).const_mul memGain
  refine ⟨ha.1, sub_pos.mpr ha.2.2.2.2, ?_⟩
  simpa only [physicalCircuitScore] using hg.sub hm

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    let state := gainNativePath 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) point
    (∀ n, state (n + 1) = gainNativeStep 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) (state n)) ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    0 < eps ∧ (0 : ℝ) < 1 / 1000 ∧ (∀ i : Fin 4, 0 < (point i).parameter) ∧
    (∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (point i).parameter)) := by
  dsimp only
  have hw := gain_native_efficient_point_balance 111 1 (by norm_num)
  refine ⟨fun n => rfl, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, hw.1, by norm_num, ?_, ?_⟩
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]
  · intro i
    exact gain_native_balanced_path_parameter_tendsto 111 3 2 1 (9 / 10) (49 / 50) _ (1 / 2) (1 / 1000)
      ((1, 1), (1 / 2, 1 / 2)) i (by norm_num) (by norm_num) (by norm_num) (by norm_num) hw.2

/-- Positive finite-limit actual native paths have balanced limit
pairs and eventually larger Gen coordinates. Sources: section 3's
physical Efficiency and native balance at 09694da; pair geometry is
derived without symmetric seeds, buffers or finite-clock equality. -/
theorem gain_native_closed_positive_factor_order (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hstep : ∀ n, state (n + 1) = gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate (state n))
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hpos : ∀ i, 0 < (reference i).parameter)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    (reference 0).parameter = (reference 1).parameter ∧
    (reference 2).parameter = (reference 3).parameter ∧
    (∀ᶠ n in atTop, (state n 2).parameter < (state n 0).parameter) := by
  have hb : ∀ i, decay * (reference i).parameter + appliedGainNativeGradient remaining genGain memGain bound reference i /
      (|appliedGainNativeGradient remaining genGain memGain bound reference i| + eps) = 0 := by
    intro i
    exact gain_native_closed_limit_balance remaining genGain memGain bound b1 b2 eps decay rate state reference i
      hstep hb1 h1 hb2 h2 he heta hp
  have ha := gain_native_positive_balanced_allocation remaining _ _ _ _ _ reference hmem hgain hclip he hpos hb
  exact ⟨ha.2.1, ha.2.2.1, (hp 2).eventually_lt (hp 0) ha.2.2.2.1⟩

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    let state := gainNativePath 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) point
    (∀ n, state (n + 1) = gainNativeStep 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) (state n)) ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    0 < eps ∧ (0 : ℝ) < 1 / 1000 ∧ (∀ i : Fin 4, 0 < (point i).parameter) ∧
    (∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (point i).parameter)) := by
  dsimp only
  have hw := gain_native_efficient_point_balance 111 1 (by norm_num)
  refine ⟨fun n => rfl, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, hw.1, by norm_num, ?_, ?_⟩
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]
  · intro i
    exact gain_native_balanced_path_parameter_tendsto 111 3 2 1 (9 / 10) (49 / 50) _ (1 / 2) (1 / 1000)
      ((1, 1), (1 / 2, 1 / 2)) i (by norm_num) (by norm_num) (by norm_num) (by norm_num) hw.2

/-- Actual efficient positive finite-limit paths eventually answer
every fixed-table held-out example uniquely correctly. Sources:
appendix C test logits and section 3 Efficiency; all-class strict
correctness follows from derived output limits, not a margin premise. -/
theorem gain_native_closed_eventually_heldout_correct (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hstep : ∀ n, state (n + 1) = gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate (state n))
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hpos : ∀ i, 0 < (reference i).parameter)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    ∀ᶠ n in atTop, Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining
      (physicalCircuitScore genGain (state n 0).parameter (state n 1).parameter)
      (physicalCircuitScore memGain (state n 2).parameter (state n 3).parameter)) 0 := by
  have ha := gain_native_closed_positive_margin_limit remaining genGain memGain bound b1 b2 eps decay rate state reference
    hstep hmem hgain hclip hb1 h1 hb2 h2 he heta hpos hp
  have hg : Tendsto (fun n => physicalCircuitScore genGain (state n 0).parameter (state n 1).parameter)
      atTop (nhds (physicalCircuitScore genGain (reference 0).parameter (reference 1).parameter)) := by
    simpa only [physicalCircuitScore] using ((hp 0).mul (hp 1)).const_mul genGain
  have hm : Tendsto (fun n => physicalCircuitScore memGain (state n 2).parameter (state n 3).parameter)
      atTop (nhds (physicalCircuitScore memGain (reference 2).parameter (reference 3).parameter)) := by
    simpa only [physicalCircuitScore] using ((hp 2).mul (hp 3)).const_mul memGain
  have hscore := hm.eventually_lt hg (sub_pos.mp ha.2.1)
  have hzero : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0) := tendsto_const_nhds
  have hpositive := hzero.eventually_lt hg
    (mul_pos (lt_trans hmem hgain) (mul_pos (hpos 0) (hpos 1)))
  exact (hscore.and hpositive).mono fun n hh => heldout_table_strict_correct remaining _ _ hh.2 hh.1

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    let state := gainNativePath 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) point
    (∀ n, state (n + 1) = gainNativeStep 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) (state n)) ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    0 < eps ∧ (0 : ℝ) < 1 / 1000 ∧ (∀ i : Fin 4, 0 < (point i).parameter) ∧
    (∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (point i).parameter)) := by
  dsimp only
  have hw := gain_native_efficient_point_balance 111 1 (by norm_num)
  refine ⟨fun n => rfl, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, hw.1, by norm_num, ?_, ?_⟩
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]
  · intro i
    exact gain_native_balanced_path_parameter_tendsto 111 3 2 1 (9 / 10) (49 / 50) _ (1 / 2) (1 / 1000)
      ((1, 1), (1 / 2, 1 / 2)) i (by norm_num) (by norm_num) (by norm_num) (by norm_num) hw.2

end Transformer.Grokking.CircuitEfficiency
