import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryFormedPrefix

/-!
# Explicit vanishing bounded Gen-seed family with original retained memory

Sources: Varma et al., arXiv:2309.02390v1, section 3 initial partner
slow learning and appendix C zero-first-factor initialization; uniform
actual Mem formation and retained Gen growth at lab commit fe86232,
original double-zero prefix comparison at c59d32c.

Construct a numerical family directly, rather than choosing a witness
from a future-success predicate. Let F be the complete current box CE/
clipping floor at double-zero initialization. The seed at budget n is
(F/16480)*(45/103)^n. Both F>0 and F<1 are generated from the original
current callback. Every seed is strictly positive, below one, decreases
strictly at successive budgets and tends to zero in exact reals.

The Gen envelope ratio is R=103/50, remaining native decay k=9/10,
and the uniform Mem amplitude formed at clock one is F/2000. Compute
R^n*(R*seed_n) = k^n*(F/2000)/4, then derive the strict gained initial
comparison with gains 3/2. This condition concerns actual initial data
and generated coefficients, not a future weak score or success premise.

The task, four factors and original beta1=0.9/beta2=0.98 native
configuration remain fixed as the budget changes. Complete CE, shared
clipping and full retained moment/clocks are used by the prefix theorem.
This family still needs to be connected to that actual retained prefix
and long-time selection before proving its iterated decision limits.
It is a time/initial-seed family, not thermodynamic size scaling.

The geometric seed scale can lie below floating-point range. This is
an exact-real construction, not a quantitative GPTMini delay prediction.
Fixed physical gained tables and uniform native decoupled decay differ
from appendix C coupled-cost GD and learned stochastic/numerical features.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Positive numerical initial partner indexed by the desired budget.
Sources: section 3 slow partners and original generated formation at
fe86232; the budget enters the explicit geometric initial value alone. -/
noncomputable def fixedGainMemoryDelaySeed (budget : ℕ) : ℝ :=
  gainCEFeedbackFloor 0 3 2 1 1 / 16480 * (45 / 103 : ℝ) ^ budget

/-- Each explicit retained-memory delay seed is positive and strictly
below one, so the original double-zero formation box applies. Sources:
section 3 initialization and full current CE/clipping floors at fe86232;
no future trajectory, success or limiting reference is supplied. -/
theorem fixed_gain_memory_delay_seed_bounds (budget : ℕ) :
    0 < fixedGainMemoryDelaySeed budget ∧ fixedGainMemoryDelaySeed budget < 1 := by
  have hf := fixed_gain_memory_seed_formation_scale 0 (by norm_num) (by norm_num)
  have hc : 0 < gainCEFeedbackFloor 0 3 2 1 1 / 16480 := div_pos hf.1 (by norm_num)
  have hp : 0 < (45 / 103 : ℝ) ^ budget := pow_pos (by norm_num) budget
  have hpower : (45 / 103 : ℝ) ^ budget ≤ 1 := pow_le_one₀ (by norm_num) (by norm_num)
  have hu := mul_le_mul_of_nonneg_left hpower (le_of_lt hc)
  have hcoef : gainCEFeedbackFloor 0 3 2 1 1 / 16480 < 1 := by linarith only [hf.2.2]
  refine ⟨mul_pos hc hp, ?_⟩
  exact lt_of_le_of_lt (by simpa only [fixedGainMemoryDelaySeed, mul_one] using hu) hcoef

/-- The actual clock-one Gen envelope adjustment cancels the geometric
budget seed into one quarter of the formed Mem pure-decay floor.
Sources: section 3 slow partners and original envelope R=103/50 at
fe86232; the identity is numerical and contains no future task predicate. -/
theorem fixed_gain_memory_delay_seed_envelope_identity (budget : ℕ) :
    gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ budget *
        ((103 / 50 : ℝ) * fixedGainMemoryDelaySeed budget) =
      (9 / 10 : ℝ) ^ budget * (gainCEFeedbackFloor 0 3 2 1 1 / 2000) / 4 := by
  have hr : gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 = (103 / 50 : ℝ) := by norm_num [gainGenGrowthFactor]
  rw [hr]
  unfold fixedGainMemoryDelaySeed
  calc
    (103 / 50 : ℝ) ^ budget * (103 / 50 * (gainCEFeedbackFloor 0 3 2 1 1 / 16480 * (45 / 103) ^ budget)) =
        ((103 / 50 : ℝ) * (45 / 103)) ^ budget * (gainCEFeedbackFloor 0 3 2 1 1 / 8000) := by
      rw [mul_pow]
      ring
    _ = (9 / 10 : ℝ) ^ budget * (gainCEFeedbackFloor 0 3 2 1 1 / 2000) / 4 := by
      norm_num only [show (103 / 50 : ℝ) * (45 / 103) = (9 / 10 : ℝ) by norm_num]
      ring

/-- Every explicit positive budget seed satisfies the strict original
formed-state gained comparison. Sources: section 3 delayed formation,
appendix C gained products 3/2 and native uniform formed amplitude at
fe86232; this bounds current initial data, not a prescribed future score. -/
theorem fixed_gain_memory_delay_seed_initial_comparison (budget : ℕ) :
    3 * (gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ budget *
        ((103 / 50 : ℝ) * fixedGainMemoryDelaySeed budget)) ^ 2 <
      2 * ((1 - (1 / 1000 : ℝ) * 100) ^ budget * (gainCEFeedbackFloor 0 3 2 1 1 / 2000)) ^ 2 := by
  have hf := fixed_gain_memory_seed_formation_scale 0 (by norm_num) (by norm_num)
  have hfloor : 0 < (9 / 10 : ℝ) ^ budget * (gainCEFeedbackFloor 0 3 2 1 1 / 2000) :=
    mul_pos (pow_pos (by norm_num) budget) (div_pos hf.1 (by norm_num))
  have hsquare := mul_pos hfloor hfloor
  rw [fixed_gain_memory_delay_seed_envelope_identity]
  norm_num only [show (1 - (1 / 1000 : ℝ) * 100) = (9 / 10 : ℝ) by norm_num]
  nlinarith only [hsquare]

/-- Explicit initialized partner seeds vanish as their delay budget
increases. Sources: section 3 small partners and original native budget
comparison at c59d32c; the geometric ratio is strictly below one and
no future success or independent seed convergence is supplied. -/
theorem fixed_gain_memory_delay_seed_tendsto_zero :
    Tendsto fixedGainMemoryDelaySeed atTop (nhds 0) := by
  have hp := tendsto_pow_atTop_nhds_zero_of_lt_one
    (show (0 : ℝ) ≤ 45 / 103 by norm_num) (show (45 / 103 : ℝ) < 1 by norm_num)
  have hs := hp.const_mul (gainCEFeedbackFloor 0 3 2 1 1 / 16480)
  change Tendsto (fun n => gainCEFeedbackFloor 0 3 2 1 1 / 16480 * (45 / 103 : ℝ) ^ n) atTop (nhds 0)
  simpa only [mul_zero] using hs

/-- Increasing the budget strictly decreases the positive explicit
initial Gen partner. Sources: section 3 slow partners and original
bounded seed/comparison at c59d32c; this is a parameter-family order
statement, not monotone training or monotone held-out accuracy. -/
theorem fixed_gain_memory_delay_seed_strict_successor (budget : ℕ) :
    0 < fixedGainMemoryDelaySeed (budget + 1) ∧
      fixedGainMemoryDelaySeed (budget + 1) < fixedGainMemoryDelaySeed budget := by
  have hn := fixed_gain_memory_delay_seed_bounds budget
  have hn1 := fixed_gain_memory_delay_seed_bounds (budget + 1)
  have heq : fixedGainMemoryDelaySeed (budget + 1) = (45 / 103 : ℝ) * fixedGainMemoryDelaySeed budget := by
    unfold fixedGainMemoryDelaySeed
    rw [pow_succ]
    ring
  refine ⟨hn1.1, ?_⟩
  rw [heq]
  linarith only [hn.1]

/-- Every explicit family member generates the uniform actual Mem
amplitude, retained Gen growth ceiling and full sign data at clock one.
Sources: section 3 double-zero formation and original generated first-
step bounds at fe86232; actual buffers and completed clocks are retained. -/
theorem fixed_gain_memory_delay_seed_formed_bounds (budget : ℕ) :
    let initial := seededNativeSubweights ((0, fixedGainMemoryDelaySeed budget), (0, 1))
    NonnegativeNativeState (fixedGainMemoryPath initial 1) ∧
      (gainCEFeedbackFloor 0 3 2 1 1 / 2000 ≤ (fixedGainMemoryPath initial 1 2).parameter ∧
        gainCEFeedbackFloor 0 3 2 1 1 / 2000 ≤ (fixedGainMemoryPath initial 1 3).parameter) ∧
      gainGenGrowthWeight (9 / 10) 1 (1 / 1000) (fixedGainMemoryPath initial 1) ≤
        (103 / 50 : ℝ) * fixedGainMemoryDelaySeed budget := by
  have hb := fixed_gain_memory_delay_seed_bounds budget
  have hs := native_seeded_nonnegative 0 (fixedGainMemoryDelaySeed budget) 0 1
    le_rfl (le_of_lt hb.1) le_rfl (by norm_num)
  exact ⟨fixed_gain_memory_signs _ hs 1,
    fixed_gain_memory_seed_mem_formation_floor _ (le_of_lt hb.1) (le_of_lt hb.2),
    fixed_gain_memory_seed_gen_formation_ceiling _ (le_of_lt hb.1)⟩

/-- The actual retained clock-one state of each explicit family member
satisfies the finite-budget gained comparison needed by the original
wrong-prefix theorem. Sources: section 3 slow learning, appendix C
products and complete native formation/envelope at fe86232/c59d32c;
no future weak scores, convergence or successful reference is supplied. -/
theorem fixed_gain_memory_delay_seed_actual_formed_comparison (budget : ℕ) :
    let initial := seededNativeSubweights ((0, fixedGainMemoryDelaySeed budget), (0, 1))
    3 * (gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ budget *
        gainGenGrowthWeight (9 / 10) 1 (1 / 1000) (fixedGainMemoryPath initial 1)) ^ 2 <
      2 * ((1 - (1 / 1000 : ℝ) * 100) ^ budget * (gainCEFeedbackFloor 0 3 2 1 1 / 2000)) ^ 2 := by
  let initial := seededNativeSubweights ((0, fixedGainMemoryDelaySeed budget), (0, 1))
  have hf := fixed_gain_memory_delay_seed_formed_bounds budget
  have hm := gain_gen_growth_weight_covers_mass (9 / 10) 1 (1 / 1000) (fixedGainMemoryPath initial 1)
    (by norm_num) (by norm_num) (by norm_num) hf.1
  have hw : 0 ≤ gainGenGrowthWeight (9 / 10) 1 (1 / 1000) (fixedGainMemoryPath initial 1) := le_trans hm.1 hm.2
  have hr : 0 ≤ gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 := by norm_num [gainGenGrowthFactor]
  have hupper := mul_le_mul_of_nonneg_left hf.2.2 (pow_nonneg hr budget)
  have hnn := mul_nonneg (pow_nonneg hr budget) hw
  have hsn : 0 ≤ gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ budget *
      ((103 / 50 : ℝ) * fixedGainMemoryDelaySeed budget) :=
    mul_nonneg (pow_nonneg hr budget) (mul_nonneg (by norm_num) (le_of_lt (fixed_gain_memory_delay_seed_bounds budget).1))
  have hsquare := mul_le_mul hupper hupper hnn hsn
  exact lt_of_le_of_lt (by simpa only [pow_two] using
    mul_le_mul_of_nonneg_left hsquare (show (0 : ℝ) ≤ 3 by norm_num))
    (fixed_gain_memory_delay_seed_initial_comparison budget)

end Transformer.Grokking.CircuitEfficiency
