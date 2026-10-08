import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryCoverage

/-!
# Positive partner mixing on the original nonzero-beta native path

Sources: Varma et al., arXiv:2309.02390v1, appendix C physical products
and section 3 circuit competition; native full-state relative selection
and physical coverage at lab commit eb67f82.

Generated original coefficient limits give every current partner a
positive coefficient floor 1/20000 on a finite tail. Current numerical
signs and the original complete denominator then bound each new physical
factor below by 1/20000 times its preceding full partner-pair sum.
Neither a moment reset nor a constant-coefficient update is substituted.

Actual Gen measurement monotonicity and generated physical coverage
bound the new Gen parameter sum by 161 times the preceding sum.
Both new Gen factors therefore stay at least 1/3220000 times their
own new parameter sum on a generated tail. This uniform positive cone
is much weaker than equal-factor or relative-balance convergence.
It may suffice for product selection against a vanishing Mem/Gen sum
ratio; that product/decision step is separate.

Every tail follows from full initial numerical signs. No future
coefficients, denominator, factor ratios or parameter limits are supplied.
Fixed gained tables and native uniform decay differ from appendix C
coupled-cost GD. Learned stochastic/floating-point GPTMini transfer and
positive confidence remain open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Computed actual cold partner limits generate a common positive
current partner floor for all four physical coordinates. Sources:
appendix C product chain factors and native limits at eb67f82;
no future coefficient bound is independently supplied. -/
theorem fixed_gain_memory_partner_floor_tail (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ∃ start : ℕ, ∀ n, start ≤ n → ∀ i : Fin 4,
      (1 / 20000 : ℝ) ≤ gainNativePartnerCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000)
        (fixedGainMemoryPath initial n) i := by
  have hc := gain_native_memory_coefficients_tendsto 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs (by norm_num)
  have hall : ∀ᶠ n in atTop, ∀ i : Fin 4,
      (1 / 20000 : ℝ) < gainNativePartnerCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000)
        (fixedGainMemoryPath initial n) i := by
    apply eventually_all.mpr
    intro i
    apply (hc i).2.eventually_const_lt
    fin_cases i <;> norm_num [coldGainCEGradientScale, nativeFactorGain]
  obtain ⟨start, htail⟩ := eventually_atTop.mp hall
  exact ⟨start, fun n hn i => le_of_lt (htail n hn i)⟩

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- A current partner floor and numerical signs mix the full present
pair into each own new physical factor. Sources: appendix C partner
partials and original native parameter law at eb67f82; the retained
negative moment contribution is kept and bounded below by zero. -/
theorem fixed_gain_memory_parameter_pair_floor (state : NativeSubweightState) (i : Fin 4)
    (hs : NonnegativeNativeState state)
    (hpartner : (1 / 20000 : ℝ) ≤ gainNativePartnerCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) state i) :
    (1 / 20000 : ℝ) * ((state i).parameter + (state (nativeFactorPartner i)).parameter) ≤
      (gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state i).parameter := by
  have hd := next_buffer_denominator_pos (9 / 10) (49 / 50) 1 (state i).variance
    (appliedGainNativeGradient 0 3 2 1 state i) (state i).clock (by norm_num) (by norm_num) (by norm_num)
  have hm : 0 ≤ gainNativeMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) state i := by
    unfold gainNativeMomentCoefficient
    exact div_nonneg (by norm_num) (le_of_lt hd)
  have hmemory := mul_nonneg hm (show 0 ≤ -(state i).moment by linarith only [(hs i).2.1])
  have hp := mul_le_mul_of_nonneg_right hpartner (hs (nativeFactorPartner i)).1
  have heq := gain_native_parameter_memory_partner 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state i
  nlinarith only [hmemory, hp, heq, (hs i).1]

example :
    let state := seededNativeSubweights ((0, 0), (0, 0))
    NonnegativeNativeState state ∧ (1 / 20000 : ℝ) ≤
      gainNativePartnerCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) state 0 := by
  dsimp only
  have hc := gain_ce_gradient_scale_zero_parameters 0 3 2 1 (seededNativeSubweights ((0, 0), (0, 0)))
    (by intro i; fin_cases i <;> rfl)
  have hg : appliedGainNativeGradient 0 3 2 1 (seededNativeSubweights ((0, 0), (0, 0))) 0 = 0 := by
    rw [gain_native_applied_gradient_scale]
    norm_num [nativeFactorPartner, seededNativeSubweights, seededScalarState]
  refine ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_⟩
  simp only [gainNativePartnerCoefficient, hc, hg]
  norm_num [coldGainCEGradientScale, nextBufferDenominator, nativeFactorGain, seededNativeSubweights, seededScalarState]

/-- Original generated coverage and Gen measurement monotonicity
bound each true Gen sum's next growth by 161. Sources: section 3
competition and native physical coverage at eb67f82; no future growth
ceiling or balance condition is independently supplied. -/
theorem fixed_gain_memory_gen_sum_step_tail (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ∃ start : ℕ, ∀ n, start ≤ n →
      gainGenParameterMass (fixedGainMemoryPath initial (n + 1)) ≤
        161 * gainGenParameterMass (fixedGainMemoryPath initial n) := by
  obtain ⟨rateStart, hrate⟩ := fixed_gain_memory_pair_rate_tail initial hs
  obtain ⟨coverStart, hcover⟩ := fixed_gain_memory_physical_coverage_tail initial hs
  refine ⟨max rateStart (coverStart + 1), ?_⟩
  intro n hn
  have hrn := le_trans (le_max_left _ _) hn
  have hcn := le_trans (le_max_right _ _) hn
  cases n with
  | zero => omega
  | succ k =>
    have hp := gain_native_pair_memory_mass_covers (2 / 25) (fixedGainMemoryPath initial (k + 1 + 1)) 0
      (by norm_num) (fixed_gain_memory_signs initial hs (k + 1 + 1))
    calc
      _ ≤ fixedGainGenMemoryMass (fixedGainMemoryPath initial (k + 1 + 1)) := hp.2
      _ ≤ fixedGainGenMemoryMass (fixedGainMemoryPath initial (k + 1)) := (hrate (k + 1) hrn).2.1
      _ ≤ 161 * gainGenParameterMass (fixedGainMemoryPath initial (k + 1)) := (hcover k (by omega)).2

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Both actual new Gen factors occupy a fixed positive fraction
of their own true new sum on a generated tail. Sources: appendix C
product factors and original native mixing/coverage at eb67f82;
no equal-factor, future pair ratio or positive parameter limit is assumed. -/
theorem fixed_gain_memory_gen_pair_cone_tail (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ∃ start : ℕ, ∀ n, start ≤ n →
      (1 / 3220000 : ℝ) * gainGenParameterMass (fixedGainMemoryPath initial (n + 1)) ≤
        (fixedGainMemoryPath initial (n + 1) 0).parameter ∧
      (1 / 3220000 : ℝ) * gainGenParameterMass (fixedGainMemoryPath initial (n + 1)) ≤
        (fixedGainMemoryPath initial (n + 1) 1).parameter := by
  obtain ⟨partnerStart, hpartner⟩ := fixed_gain_memory_partner_floor_tail initial hs
  obtain ⟨sumStart, hsum⟩ := fixed_gain_memory_gen_sum_step_tail initial hs
  refine ⟨max partnerStart sumStart, ?_⟩
  intro n hn
  have hpn := le_trans (le_max_left _ _) hn
  have hsn := le_trans (le_max_right _ _) hn
  have h0 := fixed_gain_memory_parameter_pair_floor (fixedGainMemoryPath initial n) 0
    (fixed_gain_memory_signs initial hs n) (hpartner n hpn 0)
  have h1 := fixed_gain_memory_parameter_pair_floor (fixedGainMemoryPath initial n) 1
    (fixed_gain_memory_signs initial hs n) (hpartner n hpn 1)
  change (1 / 20000 : ℝ) * ((fixedGainMemoryPath initial n 0).parameter + (fixedGainMemoryPath initial n 1).parameter) ≤
    (fixedGainMemoryPath initial (n + 1) 0).parameter at h0
  change (1 / 20000 : ℝ) * ((fixedGainMemoryPath initial n 1).parameter + (fixedGainMemoryPath initial n 0).parameter) ≤
    (fixedGainMemoryPath initial (n + 1) 1).parameter at h1
  have hu := hsum n hsn
  unfold gainGenParameterMass at hu ⊢
  constructor <;> linarith only [h0, h1, hu]

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.CircuitEfficiency
