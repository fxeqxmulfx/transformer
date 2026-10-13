import Transformer.Grokking.CircuitEfficiency.SectionC_GainSeedSequence
import Mathlib.Topology.Separation.Hausdorff

/-!
# Noncommuting small-initialization and long-time actual decision limits

Sources: Varma et al., arXiv:2309.02390v1, section 3's slow/efficient
Gen and appendix C's actual train/test tables; closed native delayed
selection and vanishing seed family at lab commit 4f52406.

Fix one task, two physical gains 3/2, cap 1, epsilon 1, decay 0.1,
rate 0.001, zero betas and unit Mem seed. The seed-family index varies
only initial Gen amplitude, with positive amplitudes tending to zero.
The decision trace is one exactly when the true held-out target is
strictly best, zero otherwise; no tie-handling implementation is assumed.

For each fixed native clock, increasingly small family members are
uniquely wrong and the trace tends to zero. For each fixed positive
family member, permanent actual success makes the clock limit one.
The two actual iterated real limits are therefore zero and one, and
there is no single finite successful tail start for the whole family.

All limits follow from actual generated feedback, not a prescribed
staircase or future success premise; parameter convergence is unneeded.
This is a nonuniform long-time/small-initialization singularity along
one explicit family, without a system-size limit, temperature or Gibbs
distribution. It does not establish a thermodynamic phase transition,
learned GPTMini transfer, loss convergence or numerical seed viability.
Fixed gained tables and decoupled native AdamW differ from coupled-cost GD.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss Filter

/-- Actual native path for one vanishing positive Gen family member.
Sources: appendix C product tables and native model at 4f52406;
task/native constants stay fixed and full buffers/clocks evolve. -/
noncomputable def gainSeedFamilyNativePath (remaining index : ℕ) : ℕ → NativeSubweightState :=
  gainNativePath remaining 3 2 1 0 0 1 (1 / 10) (1 / 1000)
    (seededNativeSubweights ((gainDelaySeedSequence 3 2 1 (1 / 10) (1 / 1000) 1 index,
      gainDelaySeedSequence 3 2 1 (1 / 10) (1 / 1000) 1 index), (1, 1)))

/-- True held-out logits on that actual native member and clock.
Sources: appendix C's competing target tables at 4f52406. -/
noncomputable def gainSeedFamilyHeldoutLogits (remaining index clock : ℕ) : Fin (remaining + 2) → ℝ :=
  heldoutTableLogits remaining
    (physicalCircuitScore 3 (gainSeedFamilyNativePath remaining index clock 0).parameter
      (gainSeedFamilyNativePath remaining index clock 1).parameter)
    (physicalCircuitScore 2 (gainSeedFamilyNativePath remaining index clock 2).parameter
      (gainSeedFamilyNativePath remaining index clock 3).parameter)

/-- Strict true-target success indicator from the actual forward.
Sources: section 3's held-out accuracy and native task tables at
4f52406; a tied true target gets zero by this stated convention. -/
noncomputable def gainSeedFamilyDecisionTrace (remaining index clock : ℕ) : ℝ := by
  classical
  exact if StrictCorrect (gainSeedFamilyHeldoutLogits remaining index clock) 0 then 1 else 0

/-- A family member is uniquely wrong through its own index.
Sources: actual seed-envelope comparison and appendix C tables at
4f52406; contradiction of two strict targets rules out tie artifacts. -/
theorem gain_seed_trace_wrong_prefix (remaining index clock : ℕ) (hclock : clock ≤ index) :
    gainSeedFamilyDecisionTrace remaining index clock = 0 := by
  classical
  have hw := (gain_native_delay_seed_sequence_prefix remaining index 3 2 1 1 (1 / 10) (1 / 1000) 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) clock hclock).2
  change StrictCorrect (gainSeedFamilyHeldoutLogits remaining index clock) (0 : Fin (remaining + 1)).succ at hw
  have hnot : ¬StrictCorrect (gainSeedFamilyHeldoutLogits remaining index clock) 0 := by
    intro hp
    have hwrong := hw 0 (fun heq => Fin.succ_ne_zero (0 : Fin (remaining + 1)) heq.symm)
    have hright := hp (0 : Fin (remaining + 1)).succ (Fin.succ_ne_zero _)
    linarith only [hwrong, hright]
  simp only [gainSeedFamilyDecisionTrace, hnot, ite_false]

example : (3 : ℕ) ≤ 5 := by norm_num

/-- Each fixed positive member permanently succeeds after a derived
finite start. Sources: actual balanced native selection and section 3
tables at 4f52406; no successful reference or parameter limit is assumed. -/
theorem gain_seed_trace_eventual_one (remaining index : ℕ) :
    ∃ start : ℕ, ∀ clock, start ≤ clock → gainSeedFamilyDecisionTrace remaining index clock = 1 := by
  have hs := gain_delay_seed_sequence_pos 3 2 1 (1 / 10) (1 / 1000) 1 index
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  obtain ⟨start, ht⟩ := gain_native_balanced_eventual_correct remaining 3 2 1 1 (1 / 10) (1 / 1000)
    (gainDelaySeedSequence 3 2 1 (1 / 10) (1 / 1000) 1 index) 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) hs (by norm_num)
  refine ⟨start, ?_⟩
  intro clock hclock
  have hc : StrictCorrect (gainSeedFamilyHeldoutLogits remaining index clock) 0 := (ht clock hclock).2
  simp only [gainSeedFamilyDecisionTrace, hc, ite_true]

/-- At each fixed actual clock, the vanishing-initialization family
decision limit is zero. Source: the actual uniquely wrong prefixes
at 4f52406; the family changes initial amplitude, not task or budget. -/
theorem gain_seed_trace_seed_limit_zero (remaining clock : ℕ) :
    Tendsto (fun index => gainSeedFamilyDecisionTrace remaining index clock) atTop (nhds 0) := by
  have hc : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0) := tendsto_const_nhds
  apply hc.congr'
  exact (eventually_ge_atTop clock).mono (fun index hi => (gain_seed_trace_wrong_prefix remaining index clock hi).symm)

/-- At each fixed positive initialization member, the actual long-time
decision limit is one. Source: permanent generated success at 4f52406;
the whole optimizer path is retained and no weight limit is presumed. -/
theorem gain_seed_trace_clock_limit_one (remaining index : ℕ) :
    Tendsto (gainSeedFamilyDecisionTrace remaining index) atTop (nhds 1) := by
  obtain ⟨start, ht⟩ := gain_seed_trace_eventual_one remaining index
  have hc : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) := tendsto_const_nhds
  apply hc.congr'
  exact (eventually_ge_atTop start).mono (fun clock hi => (ht clock hi).symm)

/-- The successful tail starts have no common finite bound over this
actual positive family. Source: section 3's slow initialization and
the checked native prefixes at 4f52406; every fixed member still succeeds. -/
theorem gain_seed_trace_no_uniform_success_start (remaining : ℕ) :
    ¬∃ start : ℕ, ∀ index clock, start ≤ clock → gainSeedFamilyDecisionTrace remaining index clock = 1 := by
  rintro ⟨start, hs⟩
  have hz := gain_seed_trace_wrong_prefix remaining start start le_rfl
  have ho := hs start start le_rfl
  linarith only [hz, ho]

/-- The two actual iterated real decision limits are zero and one.
Sources: section 3's delayed-selection mechanism and the closed native
family at 4f52406; every inner limit is first proved on the actual trace. -/
theorem gain_seed_trace_iterated_limits (remaining : ℕ) :
    atTop.limUnder (fun clock : ℕ => atTop.limUnder
      (fun index : ℕ => gainSeedFamilyDecisionTrace remaining index clock)) = 0 ∧
      atTop.limUnder (fun index : ℕ => atTop.limUnder
        (fun clock : ℕ => gainSeedFamilyDecisionTrace remaining index clock)) = 1 := by
  have hseed : ∀ clock, atTop.limUnder (fun index => gainSeedFamilyDecisionTrace remaining index clock) = 0 :=
    fun clock => (gain_seed_trace_seed_limit_zero remaining clock).limUnder_eq
  have hclock : ∀ index, atTop.limUnder (gainSeedFamilyDecisionTrace remaining index) = 1 :=
    fun index => (gain_seed_trace_clock_limit_one remaining index).limUnder_eq
  constructor
  · simp only [hseed]
    exact (tendsto_const_nhds : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0)).limUnder_eq
  · simp only [hclock]
    exact (tendsto_const_nhds : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1)).limUnder_eq

/-- Small-initialization and long-time decision limits do not commute
on this fixed actual native family. Source: the two derived limits
at 4f52406; this states no thermodynamic/system-size transition. -/
theorem gain_seed_trace_iterated_limits_ne (remaining : ℕ) :
    atTop.limUnder (fun clock : ℕ => atTop.limUnder
      (fun index : ℕ => gainSeedFamilyDecisionTrace remaining index clock)) ≠
      atTop.limUnder (fun index : ℕ => atTop.limUnder
        (fun clock : ℕ => gainSeedFamilyDecisionTrace remaining index clock)) := by
  have hlimits := gain_seed_trace_iterated_limits remaining
  rw [hlimits.1, hlimits.2]
  norm_num

end Transformer.Grokking.CircuitEfficiency
