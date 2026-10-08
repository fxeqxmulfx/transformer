import Transformer.Grokking.CircuitEfficiency.SectionC_GainCycleStep
import Mathlib.Order.Filter.AtTopBot.Tendsto
import Mathlib.Topology.Separation.Hausdorff

/-!
# An actual native CE path with period-two physical parameters

Sources: Varma et al., arXiv:2309.02390v1, appendix C's product
binary CE, and the native AdamW/clipping port at lab commit 842817b.
Use the previously checked positive constants, beta1=beta2=0,
uniform decoupled decay and the complete true gradient with cap ten.
The path below is the actual optimizer recurrence, not a separately
prescribed periodic input or parameter curve.

Gen factors alternate one/two and Mem stays zero. Prove both entire
parameter vectors by induction, then contradict a proposed finite
Gen parameter limit using the two actual infinite subsequences.
Physical parameters remain bounded, whereas optimizer clocks grow.
This refutes automatic parameter convergence from static cold
feedback and positive remaining decay alone. It does not refute
generalization: this already-formed Gen-only circuit is correct.

The large chosen rate and legal zero betas differ from the preserved
GPTMini runs. Fixed gained tables and plain CE with decoupled decay
also differ from appendix C's GD and coupled circuit norm cost.
Neither attraction nor a learned transformer cycle is asserted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- The actual retained native recurrence at the verified constants.
Sources: appendix C's binary product CE and native AdamW at 842817b;
the clock argument enters the ordinary recurrence without a phase rule. -/
noncomputable def gainCycleNativePath (n : ℕ) : NativeSubweightState :=
  gainNativePath 0 3 2 10 0 0 gainCycleEpsilon gainCycleDecay gainCycleRate
    (seededNativeSubweights ((1, 1), (0, 0))) n

/-- Every retained state at the low physical point moves to the high
point. Sources: actual binary CE and native AdamW at 842817b;
the hypothesis constrains present parameters, not future gradients. -/
theorem gain_cycle_one_step (state : NativeSubweightState)
    (hp : ∀ i, (state i).parameter =
      (seededNativeSubweights ((1, 1), (0, 0)) i).parameter) :
    ∀ i, (gainNativeStep 0 3 2 10 0 0 gainCycleEpsilon gainCycleDecay gainCycleRate state i).parameter =
      (seededNativeSubweights ((2, 2), (0, 0)) i).parameter := by
  have ht := gain_cycle_zero_beta_amplitude_step 1 gainCycleEpsilon gainCycleDecay gainCycleRate
    state (by norm_num) (by norm_num) hp
  have hs := gain_cycle_swapping_equations.1
  norm_num only [mul_one] at hs
  intro i
  rw [ht i]
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState, hs]

example : ∀ i : Fin 4, (seededNativeSubweights ((1, 1), (0, 0)) i).parameter =
    (seededNativeSubweights ((1, 1), (0, 0)) i).parameter := by
  intro i
  rfl

/-- Every retained state at the high physical point returns to the
low point. Sources: actual binary CE and native AdamW at 842817b;
completed clocks and both buffers are still updated by the actual step. -/
theorem gain_cycle_two_step (state : NativeSubweightState)
    (hp : ∀ i, (state i).parameter =
      (seededNativeSubweights ((2, 2), (0, 0)) i).parameter) :
    ∀ i, (gainNativeStep 0 3 2 10 0 0 gainCycleEpsilon gainCycleDecay gainCycleRate state i).parameter =
      (seededNativeSubweights ((1, 1), (0, 0)) i).parameter := by
  have ht := gain_cycle_zero_beta_amplitude_step 2 gainCycleEpsilon gainCycleDecay gainCycleRate
    state (by norm_num) (by norm_num) hp
  have hs := gain_cycle_swapping_equations.2
  unfold gainCycleHighDirection at hs
  have hswap : (1 - gainCycleRate * gainCycleDecay) * 2 +
      gainCycleRate * gainCycleInputMagnitude 2 / (gainCycleInputMagnitude 2 + gainCycleEpsilon) = 1 := by
    convert hs using 1
    ring
  intro i
  rw [ht i]
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState, hswap]

example : ∀ i : Fin 4, (seededNativeSubweights ((2, 2), (0, 0)) i).parameter =
    (seededNativeSubweights ((2, 2), (0, 0)) i).parameter := by
  intro i
  rfl

/-- The actual infinite path alternates both full physical vectors.
Sources: appendix C's binary CE and native AdamW at 842817b; this
derives periodic parameters from initial data and the ordinary step. -/
theorem gain_cycle_native_path_period_two (n : ℕ) :
    (∀ i, (gainCycleNativePath (2 * n) i).parameter =
      (seededNativeSubweights ((1, 1), (0, 0)) i).parameter) ∧
    (∀ i, (gainCycleNativePath (2 * n + 1) i).parameter =
      (seededNativeSubweights ((2, 2), (0, 0)) i).parameter) := by
  induction n with
  | zero =>
    constructor
    · intro i; rfl
    · exact gain_cycle_one_step _ (fun i => rfl)
  | succ n ih =>
    have heven : ∀ i, (gainCycleNativePath (2 * n + 2) i).parameter =
        (seededNativeSubweights ((1, 1), (0, 0)) i).parameter := by
      change ∀ i, (gainNativeStep 0 3 2 10 0 0 gainCycleEpsilon gainCycleDecay gainCycleRate
        (gainCycleNativePath (2 * n + 1)) i).parameter = _
      exact gain_cycle_two_step _ ih.2
    have hodd : ∀ i, (gainCycleNativePath (2 * n + 2 + 1) i).parameter =
        (seededNativeSubweights ((2, 2), (0, 0)) i).parameter := by
      change ∀ i, (gainNativeStep 0 3 2 10 0 0 gainCycleEpsilon gainCycleDecay gainCycleRate
        (gainCycleNativePath (2 * n + 2)) i).parameter = _
      exact gain_cycle_one_step _ heven
    rw [show 2 * (n + 1) = 2 * n + 2 by omega]
    exact ⟨heven, hodd⟩

/-- Physical parameter convergence fails for this actual legal
configuration. Sources: appendix C product CE and native AdamW at
842817b; counterexample to removing convergence solely from a static
cold feedback condition and a positive remaining decay factor. -/
theorem gain_cycle_native_parameter_not_convergent :
    ¬ ∃ value : ℝ, Tendsto (fun n => (gainCycleNativePath n 0).parameter) atTop (nhds value) := by
  rintro ⟨value, hlimit⟩
  have heven : Tendsto (fun n : ℕ => 2 * n) atTop atTop :=
    tendsto_atTop_mono (fun n => by change n ≤ 2 * n; omega) tendsto_id
  have hodd : Tendsto (fun n : ℕ => 2 * n + 1) atTop atTop :=
    tendsto_atTop_mono (fun n => by change n ≤ 2 * n + 1; omega) tendsto_id
  have hlow : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds value) := by
    apply (hlimit.comp heven).congr
    intro n
    exact (gain_cycle_native_path_period_two n).1 0
  have hhigh : Tendsto (fun _ : ℕ => (2 : ℝ)) atTop (nhds value) := by
    apply (hlimit.comp hodd).congr
    intro n
    exact (gain_cycle_native_path_period_two n).2 0
  have hone : value = 1 := tendsto_nhds_unique hlow tendsto_const_nhds
  have htwo : value = 2 := tendsto_nhds_unique hhigh tendsto_const_nhds
  linarith only [hone, htwo]

/-- All physical parameters are bounded on this nonconvergent path.
Sources: appendix C binary CE and native AdamW at 842817b; boundedness
is proved on the very same generated trajectory used above. -/
theorem gain_cycle_native_parameter_bounds (n : ℕ) (i : Fin 4) :
    0 ≤ (gainCycleNativePath n i).parameter ∧ (gainCycleNativePath n i).parameter ≤ 2 := by
  have hcases : n = 2 * (n / 2) ∨ n = 2 * (n / 2) + 1 := by omega
  rcases hcases with h | h
  · rw [h, (gain_cycle_native_path_period_two (n / 2)).1 i]
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]
  · rw [h, (gain_cycle_native_path_period_two (n / 2)).2 i]
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- The completed clocks grow on the same path; full optimizer states
are not periodic. Source: native retained AdamW at 842817b. -/
theorem gain_cycle_native_clocks (n : ℕ) (i : Fin 4) :
    (gainCycleNativePath n i).clock = n := by
  have h := gain_native_path_clocks 0 3 2 10 0 0 gainCycleEpsilon gainCycleDecay gainCycleRate
    (seededNativeSubweights ((1, 1), (0, 0))) n i
  fin_cases i <;> simpa [gainCycleNativePath, seededNativeSubweights, seededScalarState] using h

end Transformer.Grokking.CircuitEfficiency
