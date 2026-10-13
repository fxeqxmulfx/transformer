import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryPowers

/-!
# Actual retained-memory relative selection despite absolute collapse

Sources: Varma et al., arXiv:2309.02390v1, section 3 competing
circuits and appendix C products; initialized original native
Gen/Mem powers at lab commit f8f8e92.

Positive initial numerical Gen parameter/moment mass remains positive
at every finite clock. The generated common tail bounds the actual
Mem/Gen measurement ratio by (910/911)^k times its current start ratio.
That ratio tends to zero on the complete original clipped native path
with beta1=0.9 and beta2=0.98, not on a prescribed gradient stream.

Every positive fixed multiple of the Gen measurement eventually exceeds
the Mem measurement permanently. Meanwhile all physical parameters and
both retained buffers tend to zero under the same static configuration.
Relative measurement selection and absolute collapse are compatible.
The generated clocks and actual full denominators are not reset.

The measurements include first moments with different fixed weights.
Their relative selection is not yet a physical product-margin result,
pair-balance theorem, confidence result or grokking detector. Fixed gained
tables and uniform native decay differ from appendix C coupled-cost GD.
Learned stochastic/floating-point GPTMini transfer remains open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Present Mem/Gen retained pair measurement ratio on the original
pinned path. Sources: appendix C competing factors and native weights
at f8f8e92; this numerical quotient does not encode successful logits. -/
noncomputable def fixedGainMemGenRatio (initial : NativeSubweightState) (n : ℕ) : ℝ :=
  fixedGainMemMemoryMass (fixedGainMemoryPath initial n) /
    fixedGainGenMemoryMass (fixedGainMemoryPath initial n)

/-- Initialized positive Gen mass generates an actual geometric
relative ceiling and a full-clock ratio limit of zero. Sources:
section 3 competition and native powers at f8f8e92; no future mass,
coefficient, buffer or success premise is independently supplied. -/
theorem fixed_gain_memory_ratio_tendsto_zero (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) (hgen : 0 < fixedGainGenMemoryMass initial) :
    ∃ start : ℕ, (∀ k, 0 ≤ fixedGainMemGenRatio initial (start + k) ∧
      fixedGainMemGenRatio initial (start + k) ≤ (910 / 911 : ℝ) ^ k * fixedGainMemGenRatio initial start) ∧
      Tendsto (fixedGainMemGenRatio initial) atTop (nhds 0) := by
  obtain ⟨start, hpowers⟩ := fixed_gain_memory_tail_power_bounds initial hs
  have hpositive := fixed_gain_gen_memory_positive_path initial hs hgen
  have hnonnegative := fixed_gain_memory_initial_power_floor initial hs
  have hbound : ∀ k, 0 ≤ fixedGainMemGenRatio initial (start + k) ∧
      fixedGainMemGenRatio initial (start + k) ≤ (910 / 911 : ℝ) ^ k * fixedGainMemGenRatio initial start := by
    intro k
    have hg := (hpowers k).1
    have hm := (hpowers k).2
    have hden : 0 < (911 / 1000 : ℝ) ^ k * fixedGainGenMemoryMass (fixedGainMemoryPath initial start) :=
      mul_pos (pow_pos (by norm_num) k) (hpositive start)
    have hupper := div_le_div_of_nonneg_right hm (le_of_lt (hpositive (start + k)))
    have hsmaller := div_le_div_of_nonneg_left
      (mul_nonneg (le_of_lt (pow_pos (by norm_num : (0 : ℝ) < 91 / 100) k)) (hnonnegative start).2) hden hg
    refine ⟨div_nonneg (hnonnegative (start + k)).2 (le_of_lt (hpositive (start + k))), ?_⟩
    change fixedGainMemMemoryMass (fixedGainMemoryPath initial (start + k)) /
      fixedGainGenMemoryMass (fixedGainMemoryPath initial (start + k)) ≤ _
    calc
      _ ≤ ((91 / 100 : ℝ) ^ k * fixedGainMemMemoryMass (fixedGainMemoryPath initial start)) /
          ((911 / 1000 : ℝ) ^ k * fixedGainGenMemoryMass (fixedGainMemoryPath initial start)) :=
        le_trans hupper hsmaller
      _ = (910 / 911 : ℝ) ^ k * fixedGainMemGenRatio initial start := by
        rw [mul_div_mul_comm, ← div_pow]
        norm_num [fixedGainMemGenRatio]
  have hpower := tendsto_pow_atTop_nhds_zero_of_lt_one
    (by norm_num : (0 : ℝ) ≤ 910 / 911) (by norm_num : (910 / 911 : ℝ) < 1)
  have hceiling : Tendsto (fun k : ℕ => (910 / 911 : ℝ) ^ k * fixedGainMemGenRatio initial start) atTop (nhds 0) := by
    simpa only [zero_mul] using hpower.mul_const (fixedGainMemGenRatio initial start)
  have hshift := squeeze_zero (fun k => (hbound k).1) (fun k => (hbound k).2) hceiling
  have hfull : Tendsto (fixedGainMemGenRatio initial) atTop (nhds 0) :=
    (tendsto_add_atTop_iff_nat start).mp (by simpa only [Nat.add_comm] using hshift)
  exact ⟨start, hbound, hfull⟩

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    0 < fixedGainGenMemoryMass (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [fixedGainGenMemoryMass, gainNativePairMemoryMass, seededNativeSubweights, seededScalarState, nativeFactorPartner]⟩

/-- Every fixed positive multiple of the actual Gen measurement
eventually exceeds Mem permanently. Sources: section 3 relative
competition and initialized native ratio at f8f8e92; the tail is
produced, and this measurement statement is not a product-margin claim. -/
theorem fixed_gain_memory_relative_dominance (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) (hgen : 0 < fixedGainGenMemoryMass initial)
    (multiple : ℝ) (hmultiple : 0 < multiple) :
    ∃ start : ℕ, ∀ n, start ≤ n →
      fixedGainMemMemoryMass (fixedGainMemoryPath initial n) <
        multiple * fixedGainGenMemoryMass (fixedGainMemoryPath initial n) := by
  have hl := (fixed_gain_memory_ratio_tendsto_zero initial hs hgen).choose_spec.2
  obtain ⟨start, htail⟩ := eventually_atTop.mp (hl.eventually_lt_const hmultiple)
  refine ⟨start, ?_⟩
  intro n hn
  have hpos := fixed_gain_gen_memory_positive_path initial hs hgen n
  exact (div_lt_iff₀ hpos).mp (htail n hn)

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    0 < fixedGainGenMemoryMass (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧ (0 : ℝ) < 1 / 100 := by
  exact ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [fixedGainGenMemoryMass, gainNativePairMemoryMass, seededNativeSubweights, seededScalarState, nativeFactorPartner], by norm_num⟩

/-- Relative retained measurement selection coexists with collapse
of every original physical parameter and both retained buffers on
that same legal-beta path. Sources: appendix C factors and native
static contraction at f8f8e92; no future convergence input is supplied. -/
theorem fixed_gain_memory_selection_with_collapse (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) (hgen : 0 < fixedGainGenMemoryMass initial) :
    Tendsto (fixedGainMemGenRatio initial) atTop (nhds 0) ∧
      ∀ i, Tendsto (fun n => (fixedGainMemoryPath initial n i).parameter) atTop (nhds 0) ∧
        Tendsto (fun n => (fixedGainMemoryPath initial n i).moment) atTop (nhds 0) ∧
        Tendsto (fun n => (fixedGainMemoryPath initial n i).variance) atTop (nhds 0) := by
  have hp := gain_native_memory_parameters_tendsto_zero 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs (by norm_num)
  have hb := gain_native_memory_buffers_tendsto_zero 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs (by norm_num)
  refine ⟨(fixed_gain_memory_ratio_tendsto_zero initial hs hgen).choose_spec.2, ?_⟩
  intro i
  exact ⟨hp i, (hb i).1, (hb i).2⟩

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    0 < fixedGainGenMemoryMass (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [fixedGainGenMemoryMass, gainNativePairMemoryMass, seededNativeSubweights, seededScalarState, nativeFactorPartner]⟩

/-- Both actual retained pair measurements tend to zero under the
pinned static decay, independently of positive Gen initialization.
Sources: appendix C coordinates and native full-state limits at f8f8e92;
component collapse alone does not determine their quotient limit. -/
theorem fixed_gain_memory_measurements_tendsto_zero (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    Tendsto (fun n => fixedGainGenMemoryMass (fixedGainMemoryPath initial n)) atTop (nhds 0) ∧
      Tendsto (fun n => fixedGainMemMemoryMass (fixedGainMemoryPath initial n)) atTop (nhds 0) := by
  have hp := gain_native_memory_parameters_tendsto_zero 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs (by norm_num)
  have hb := gain_native_memory_buffers_tendsto_zero 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs (by norm_num)
  have hpair (weight : ℝ) (i : Fin 4) :
      Tendsto (fun n => gainNativePairMemoryMass weight (fixedGainMemoryPath initial n) i) atTop (nhds 0) := by
    have ht := ((hp i).add (hp (nativeFactorPartner i))).add
      (((hb i).1.neg.sub (hb (nativeFactorPartner i)).1).const_mul weight)
    simpa only [gainNativePairMemoryMass, fixedGainMemoryPath, zero_add, neg_zero, sub_zero, mul_zero] using ht
  exact ⟨hpair (2 / 25) 0, hpair (19 / 200) 2⟩

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.CircuitEfficiency
