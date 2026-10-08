import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryDelayFamily
import Mathlib.Topology.Separation.Hausdorff

/-!
# Noncommuting original legal-beta time/initial-seed decision limits

Sources: Varma et al., arXiv:2309.02390v1, section 3 slow partners
and appendix C double-zero product logits; explicit numerical vanishing
family and actual formed-state comparison at lab commit 8e7f7d9,
initialized original retained selection at 4c76b7d.

Keep one binary task with Gen/Mem gains 3/2, beta1=0.9, beta2=0.98,
epsilon/cap 1, decay 100 and rate 0.001. Gen begins (0,seed_index)
and Mem (0,1); both products are initially zero. The explicit positive
bounded seeds tend to zero. All original buffers and completed clocks
then evolve by the actual clipped-CE callback without a reset.

Every member has strictly correct train and uniquely wrong held-out
answers at clocks one through index+1. Its complete clock-one retained
state supplies the original prefix theorem. Each fixed member later
permanently selects the true held-out class from generated selection.

The strict decision indicator is one when the actual true held-out
logit is uniquely largest, zero otherwise, including initial ties.
At each fixed clock, its small-seed family limit is zero. At each fixed
family member, its long-time limit is one. The actual iterated real
limits are zero and one, so they do not commute. No uniform finite
successful start exists across this fixed legal-memory seed family.

These are derived actual decision limits, not a prescribed staircase
or future success premise. They concern exact reals and time/initial
seed at fixed width/task, without temperature, Gibbs distribution or
thermodynamic size scaling. Strong decay can collapse margin/confidence;
this is not positive-confidence or numerical GPTMini convergence.
Fixed gained tables and uniform decoupled native decay differ from
appendix C coupled-cost GD and learned stochastic/numerical heads.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- Numerical double-zero initialization for an explicit bounded
vanishing Gen partner. Sources: section 3 initial factors and original
family at 8e7f7d9; the index changes current initial data alone. -/
noncomputable def fixedGainMemoryFamilyInitial (index : ℕ) : NativeSubweightState :=
  seededNativeSubweights ((0, fixedGainMemoryDelaySeed index), (0, 1))

/-- Strict true-target decision from the original native held-out
forward at a family index and clock. Sources: appendix C actual logits
and initialized original family at 8e7f7d9; initial ties receive zero. -/
noncomputable def fixedGainMemoryFamilyDecision (index clock : ℕ) : ℝ := by
  classical
  exact if StrictCorrect (fixedGainMemoryHeldoutLogits (fixedGainMemoryFamilyInitial index) clock) 0 then 1 else 0

/-- The actual retained native family is train-correct and uniquely
wrong-test through its own post-formation index. Sources: section 3
slow formation, appendix C tables and complete formed-state comparison
at 8e7f7d9; both buffers and clock one are retained, not initialized anew. -/
theorem fixed_gain_memory_family_actual_prefix (index clock : ℕ) (hclock : clock ≤ index) :
    StrictCorrect (fixedGainMemoryTrainLogits (fixedGainMemoryFamilyInitial index) (clock + 1)) 0 ∧
      StrictCorrect (fixedGainMemoryHeldoutLogits (fixedGainMemoryFamilyInitial index) (clock + 1)) 1 := by
  let initial := fixedGainMemoryFamilyInitial index
  have hf := fixed_gain_memory_delay_seed_formed_bounds index
  have hp := fixed_gain_memory_delay_seed_actual_formed_comparison index
  have hfloor := (fixed_gain_memory_seed_formation_scale 0 (by norm_num) (by norm_num)).1
  have hprefix := gain_native_train_correct_wrong_test_prefix 0 index 3 2 1
    (9 / 10) (49 / 50) 1 100 (1 / 1000) (gainCEFeedbackFloor 0 3 2 1 1 / 2000) (fixedGainMemoryPath initial 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hf.1 (div_pos hfloor (by norm_num)) hf.2.1.1 hf.2.1.2 hp
  dsimp only [fixedGainMemoryTrainLogits, fixedGainMemoryHeldoutLogits]
  rw [fixed_gain_memory_path_shift_one]
  exact hprefix clock hclock

example : (3 : ℕ) ≤ 5 := by norm_num

/-- The actual strict true-target trace is zero through index+1,
including the untrained all-zero-logit clock. Sources: appendix C
competing classes and original derived family prefixes at 8e7f7d9;
uniquely wrong successors exclude an argmax-tie explanation of the delay. -/
theorem fixed_gain_memory_family_trace_wrong_prefix (index clock : ℕ) (hclock : clock ≤ index + 1) :
    fixedGainMemoryFamilyDecision index clock = 0 := by
  classical
  have hnot : ¬StrictCorrect (fixedGainMemoryHeldoutLogits (fixedGainMemoryFamilyInitial index) clock) 0 := by
    intro ht
    cases clock with
    | zero =>
      have hc := ht 1 (by decide)
      norm_num [fixedGainMemoryHeldoutLogits, fixedGainMemoryPath, gainNativePath,
        fixedGainMemoryFamilyInitial, seededNativeSubweights, seededScalarState, physicalCircuitScore, heldoutTableLogits] at hc
    | succ n =>
      have hw := (fixed_gain_memory_family_actual_prefix index n (by omega)).2 0 (by decide)
      have hc := ht 1 (by decide)
      linarith only [hw, hc]
  simp only [fixedGainMemoryFamilyDecision, hnot, ite_false]

example : (3 : ℕ) ≤ 5 + 1 := by norm_num

/-- Each fixed explicit family member has permanently true actual
held-out decisions after a generated native clock. Sources: section 3
competition and original retained selection at 4c76b7d/8e7f7d9;
no future convergence, coefficient, margin or successful reference is supplied. -/
theorem fixed_gain_memory_family_trace_eventual_one (index : ℕ) :
    ∃ start : ℕ, ∀ clock, start ≤ clock → fixedGainMemoryFamilyDecision index clock = 1 := by
  have hseed := (fixed_gain_memory_delay_seed_bounds index).1
  have hs := native_seeded_nonnegative 0 (fixedGainMemoryDelaySeed index) 0 1
    le_rfl (le_of_lt hseed) le_rfl (by norm_num)
  have hg : 0 < fixedGainGenMemoryMass (fixedGainMemoryFamilyInitial index) := by
    simpa [fixedGainGenMemoryMass, gainNativePairMemoryMass, fixedGainMemoryFamilyInitial,
      seededNativeSubweights, seededScalarState, nativeFactorPartner] using hseed
  obtain ⟨start, htail⟩ := fixed_gain_memory_eventual_correct _ hs hg
  refine ⟨start, ?_⟩
  intro clock hc
  have ht : StrictCorrect (fixedGainMemoryHeldoutLogits (fixedGainMemoryFamilyInitial index) clock) 0 := (htail clock hc).2
  simp only [fixedGainMemoryFamilyDecision, ht, ite_true]

/-- At each fixed actual clock, shrinking explicit initialized Gen
partners give decision limit zero. Sources: section 3 slow partners
and original complete native prefixes at 8e7f7d9; no size scaling is supplied. -/
theorem fixed_gain_memory_family_seed_limit_zero (clock : ℕ) :
    Tendsto (fun index => fixedGainMemoryFamilyDecision index clock) atTop (nhds 0) := by
  have hc : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0) := tendsto_const_nhds
  apply hc.congr'
  exact (eventually_ge_atTop clock).mono (fun index hi =>
    (fixed_gain_memory_family_trace_wrong_prefix index clock (by omega)).symm)

/-- At each fixed initialized positive member, the actual long-time
strict decision limit is one. Sources: section 3 competition and native
generated permanent selection at 4c76b7d; parameter convergence is unneeded. -/
theorem fixed_gain_memory_family_clock_limit_one (index : ℕ) :
    Tendsto (fixedGainMemoryFamilyDecision index) atTop (nhds 1) := by
  obtain ⟨start, htail⟩ := fixed_gain_memory_family_trace_eventual_one index
  have hc : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) := tendsto_const_nhds
  apply hc.congr'
  exact (eventually_ge_atTop start).mono (fun clock hn => (htail clock hn).symm)

/-- Every explicit positive member succeeds, but its actual native
family admits no uniform finite true-decision start. Sources: section
3 slow partners and native full prefixes at 8e7f7d9; width/task/betas stay fixed. -/
theorem fixed_gain_memory_family_no_uniform_success_start :
    ¬∃ start : ℕ, ∀ index clock, start ≤ clock → fixedGainMemoryFamilyDecision index clock = 1 := by
  rintro ⟨start, hstart⟩
  have hz := fixed_gain_memory_family_trace_wrong_prefix start start (by omega)
  have ho := hstart start start le_rfl
  linarith only [hz, ho]

/-- The original nonzero-memory actual decision limits are zero for
seed-first and one for time-first iteration. Sources: section 3 slow
formation and complete native family/selection at 8e7f7d9/4c76b7d;
each inner limit is established on the actual clipped-feedback trace. -/
theorem fixed_gain_memory_family_iterated_limits :
    atTop.limUnder (fun clock : ℕ => atTop.limUnder (fun index : ℕ => fixedGainMemoryFamilyDecision index clock)) = 0 ∧
      atTop.limUnder (fun index : ℕ => atTop.limUnder (fixedGainMemoryFamilyDecision index)) = 1 := by
  have hseed : ∀ clock, atTop.limUnder (fun index => fixedGainMemoryFamilyDecision index clock) = 0 :=
    fun clock => (fixed_gain_memory_family_seed_limit_zero clock).limUnder_eq
  have hclock : ∀ index, atTop.limUnder (fixedGainMemoryFamilyDecision index) = 1 :=
    fun index => (fixed_gain_memory_family_clock_limit_one index).limUnder_eq
  constructor
  · simp only [hseed]
    exact (tendsto_const_nhds : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0)).limUnder_eq
  · simp only [hclock]
    exact (tendsto_const_nhds : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1)).limUnder_eq

/-- Small initialized seed and long-time actual native decision limits
do not commute, with legal nonzero retained beta memory. Sources:
section 3 slow formation and derived original limits at 8e7f7d9;
this is not a thermodynamic/system-size claim or learned GPTMini transfer. -/
theorem fixed_gain_memory_family_iterated_limits_ne :
    atTop.limUnder (fun clock : ℕ => atTop.limUnder (fun index : ℕ => fixedGainMemoryFamilyDecision index clock)) ≠
      atTop.limUnder (fun index : ℕ => atTop.limUnder (fixedGainMemoryFamilyDecision index)) := by
  have hlimits := fixed_gain_memory_family_iterated_limits
  rw [hlimits.1, hlimits.2]
  norm_num

end Transformer.Grokking.CircuitEfficiency
