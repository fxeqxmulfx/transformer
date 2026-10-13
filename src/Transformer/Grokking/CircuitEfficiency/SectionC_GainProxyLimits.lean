import Transformer.Grokking.CircuitEfficiency.SectionC_GainPairProxy

/-!
# Actual unequal native pair-mass laws approach their balanced proxies

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
compositional seeds and appendix C's product CE; actual retained
native feedback and relative-balance laws at lab commit 1b6785b.

Identify the exact unequal-pair correction on both actual native
coordinates. On initialized nonnegative zero-buffer paths, the box
and positive pair sums are derived. True current clipped CE is in
(0,1), and the squared relative-asymmetry bound makes both errors
per unit pair mass tend to zero without individual parameter limits.
The actual next mass laws therefore approach the same-current-scale
balanced proxies in relative residual, also on zero-first-factor seeds.

The proxy uses the original actual CE scale, rather than re-evaluating
CE/clipping on a balanced point or prescribing an alternate trajectory.
Efficient Gen/Mem crossing and positive limiting mass/margin are still
separate claims. Exact reals, fixed tables, zero betas and a sufficient
small rate differ from coupled-cost GD and preserved learned GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Filter

/-- Both actual native pair mass updates have the exact same-current-
CE balanced-proxy correction. Sources: appendix C true partner
partials and native zero-beta feedback at 1b6785b; buffers/clocks
remain arbitrary and no future balancing premise is supplied. -/
theorem gain_native_pair_mass_proxy (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hclip : 0 < bound) (he : 0 < eps) (hp : ∀ i, 0 ≤ (state i).parameter) :
    let next := gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state
    let scale := gainCEGradientScale remaining genGain memGain bound state
    (next 0).parameter + (next 1).parameter =
        2 * gainBalancedFactorStep genGain scale eps decay rate (((state 0).parameter + (state 1).parameter) / 2) -
          rate * gainPairBalancingError genGain scale eps (state 0).parameter (state 1).parameter ∧
      (next 2).parameter + (next 3).parameter =
        2 * gainBalancedFactorStep memGain scale eps decay rate (((state 2).parameter + (state 3).parameter) / 2) -
          rate * gainPairBalancingError memGain scale eps (state 2).parameter (state 3).parameter := by
  have hs := (gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip).1
  have hform := gain_native_zero_beta_pair_step remaining genGain memGain bound eps decay rate state hgen hmem hclip hp
  have hg := gain_pair_step_balanced_proxy genGain _ eps decay rate _ _ (le_of_lt hgen) (le_of_lt hs) he (hp 0) (hp 1)
  have hm := gain_pair_step_balanced_proxy memGain _ eps decay rate _ _ (le_of_lt hmem) (le_of_lt hs) he (hp 2) (hp 3)
  rw [← hform.1] at hg
  rw [← hform.2] at hm
  exact ⟨hg, hm⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    ∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- The actual current unequal-pair errors per positive mass vanish
on initialized native paths. Sources: section 3 unbalanced seeds
and closed CE estimates at 1b6785b; all box, sign, mass and relative-
balance premises are generated from initial data, not future inputs. -/
theorem gain_native_seeded_proxy_error_tendsto_zero (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    let scale := fun n => gainCEGradientScale remaining genGain memGain bound (path n)
    Tendsto (fun n => gainPairBalancingError genGain (scale n) eps (path n 0).parameter (path n 1).parameter /
      ((path n 0).parameter + (path n 1).parameter)) atTop (nhds 0) ∧
      Tendsto (fun n => gainPairBalancingError memGain (scale n) eps (path n 2).parameter (path n 3).parameter /
        ((path n 2).parameter + (path n 3).parameter)) atTop (nhds 0) := by
  obtain ⟨ceiling, _, _, _, hbox, hmass, _⟩ := gain_native_seeded_relative_path_bound remaining
    genGain memGain bound eps decay rate a b c d hgen hmem hgain hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall
  have hbalance := gain_native_seeded_relative_difference_tendsto_zero remaining
    genGain memGain bound eps decay rate a b c d hgen hmem hgain hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, b), (c, d)))
  have hp : ∀ n i, 0 ≤ (path n i).parameter ∧ (path n i).parameter ≤ ceiling := by
    intro n i
    exact ⟨((hbox n).1 i).1, ((hbox n).2 i).1⟩
  have hs := fun n => gain_ce_gradient_scale_unit_interval remaining genGain memGain bound (path n) hclip
  constructor
  · apply squeeze_zero
      (fun n => div_nonneg (gain_pair_balancing_error_nonneg genGain _ eps _ _
        (le_of_lt hgen) (le_of_lt (hs n).1) he (hp n 0).1 (hp n 1).1) (le_of_lt (hmass n).1))
      (fun n => gain_pair_balancing_error_relative_ceiling genGain _ eps ceiling _ _
        (le_of_lt hgen) (le_of_lt (hs n).1) (le_of_lt (hs n).2) he
        (hp n 0).1 (hp n 1).1 (hp n 0).2 (hp n 1).2 (hmass n).1)
    simpa using (hbalance.1.pow 2).const_mul (genGain ^ 2 * ceiling / eps ^ 2)
  · apply squeeze_zero
      (fun n => div_nonneg (gain_pair_balancing_error_nonneg memGain _ eps _ _
        (le_of_lt hmem) (le_of_lt (hs n).1) he (hp n 2).1 (hp n 3).1) (le_of_lt (hmass n).2))
      (fun n => gain_pair_balancing_error_relative_ceiling memGain _ eps ceiling _ _
        (le_of_lt hmem) (le_of_lt (hs n).1) (le_of_lt (hs n).2) he
        (hp n 2).1 (hp n 3).1 (hp n 2).2 (hp n 3).2 (hmass n).2)
    simpa using (hbalance.2.pow 2).const_mul (memGain ^ 2 * ceiling / eps ^ 2)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

/-- Actual next pair masses approach the same-current-CE balanced
proxy in relative residual. Sources: appendix C product dynamics
and closed native estimates at 1b6785b; parameter limits and a
successful learned circuit are not supplied or concluded. -/
theorem gain_native_seeded_mass_proxy_residual_tendsto_zero (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    let scale := fun n => gainCEGradientScale remaining genGain memGain bound (path n)
    Tendsto (fun n => (2 * gainBalancedFactorStep genGain (scale n) eps decay rate
      (((path n 0).parameter + (path n 1).parameter) / 2) -
        ((path (n + 1) 0).parameter + (path (n + 1) 1).parameter)) /
          ((path n 0).parameter + (path n 1).parameter)) atTop (nhds 0) ∧
      Tendsto (fun n => (2 * gainBalancedFactorStep memGain (scale n) eps decay rate
        (((path n 2).parameter + (path n 3).parameter) / 2) -
          ((path (n + 1) 2).parameter + (path (n + 1) 3).parameter)) /
            ((path n 2).parameter + (path n 3).parameter)) atTop (nhds 0) := by
  obtain ⟨_, _, _, _, hbox, _, _⟩ := gain_native_seeded_relative_path_bound remaining
    genGain memGain bound eps decay rate a b c d hgen hmem hgain hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall
  have herror := gain_native_seeded_proxy_error_tendsto_zero remaining genGain memGain bound eps decay rate a b c d
    hgen hmem hgain hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall
  have heq := fun n => gain_native_pair_mass_proxy remaining genGain memGain bound eps decay rate _
    hgen hmem hclip he (fun i => ((hbox n).1 i).1)
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, b), (c, d)))
  constructor
  · have hg := herror.1.const_mul rate
    simp only [mul_zero] at hg
    refine hg.congr (fun n => ?_)
    have hnext := (heq n).1
    change (path (n + 1) 0).parameter + (path (n + 1) 1).parameter = _ at hnext
    rw [hnext]
    ring
  · have hm := herror.2.const_mul rate
    simp only [mul_zero] at hm
    refine hm.congr (fun n => ?_)
    have hnext := (heq n).2
    change (path (n + 1) 2).parameter + (path (n + 1) 3).parameter = _ at hnext
    rw [hnext]
    ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
