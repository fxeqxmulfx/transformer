import Transformer.Grokking.AdamW.ScheduledHistory
import Transformer.Grokking.AdamW.SquareParameterBox

/-!
# Actual scheduled AdamW physical boxes and the original warmup

Sources: PyTorch 2.14.1 non-AMSGrad AdamW, adam.py lines 414--475
and 529--547, ported at 88aa892/0033b1b; domain.training.rate at
0033b1b, original rate=0.001 and ten completed-update warmup steps.
Changing rate does not reset either buffer or its bias clock.

Generate the physical box from zero native buffers and arbitrary
changing actual inputs. Positive fixed decay and nonnegative rates/
remaining decay suffice; no rate constancy, input bound, clipping
or future parameter/direction/convergence bound is supplied. The
same max(abs(initial),sqrt(K)/decay) ceiling works at every clock.

Verify the original warmup expression in exact reals: its first
rate is zero, its second rate is 0.0001, and it equals 0.001 after
ten completed updates. Every rate is nonnegative and keeps the
remaining decay nonnegative at decay=0.1. All original-beta
initialized coordinates of absolute size at most 17 stay within
17 under this actual changing-rate expression. Globally coupled
learned inputs are allowed by the coordinatewise application.

The exact update bridge stays explicit. This supplies neither the
Python float/rounding bridge nor learned transformer convergence,
Gen/Mem attraction or eventual held-out generalization. Frozen
training schedules, moments, checkpoints and measurements are unchanged.
-/

namespace Transformer.Grokking.AdamW

/-- Scheduled native paths inherit the initialized physical box.
Source: native decay/addcdiv at 88aa892 and ScheduledHistory;
rate legality is numerical at each clock, not a future state guard. -/
theorem scalar_scheduled_initialized_parameter_box (b1 b2 eps decay : ℝ) (rate : ℕ → ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay (rate n) (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0) (hc : (state 0).clock = 0)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2) (he : 0 < eps) (hdecay : 0 < decay)
    (hlegal : ∀ n, 0 ≤ rate n ∧ 0 ≤ 1 - rate n * decay) :
    ∀ n, |(state n).parameter| ≤
      max |(state 0).parameter| (Real.sqrt (nativeMomentSquareScale b1 b2) / decay) := by
  let ceiling := max |(state 0).parameter| (Real.sqrt (nativeMomentSquareScale b1 b2) / decay)
  have hcost : Real.sqrt (nativeMomentSquareScale b1 b2) ≤ ceiling * decay :=
    (div_le_iff₀ hdecay).mp (le_max_right _ _)
  have hdir := scalar_scheduled_initialized_direction_ceiling b1 b2 eps decay rate state gradient
    hstep hm hv hc hb1 h1 h2 hgap he
  intro n
  change |(state n).parameter| ≤ ceiling
  induction n with
  | zero => exact le_max_left _ _
  | succ n ih =>
    rw [hstep n]
    have ht := scalar_parameter_abs_step_ceiling b1 b2 eps decay (rate n) (gradient n)
      (Real.sqrt (nativeMomentSquareScale b1 b2)) (state n) (hlegal n).2 (hlegal n).1 (hdir n)
    have hold := mul_le_mul_of_nonneg_left ih (hlegal n).2
    have hcostrate := mul_le_mul_of_nonneg_left hcost (hlegal n).1
    calc
      _ ≤ (1 - rate n * decay) * |(state n).parameter| + rate n * Real.sqrt (nativeMomentSquareScale b1 b2) := ht
      _ ≤ (1 - rate n * decay) * ceiling + rate n * Real.sqrt (nativeMomentSquareScale b1 b2) := add_le_add hold le_rfl
      _ ≤ ceiling := by nlinarith only [hcostrate]

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) (if n < 10 then 0 else 1 / 1000) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧ (zeroScalarStateAt 0).clock = 0 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (49 / 50 : ℝ) < 1 ∧ (9 / 10 : ℝ) ^ 2 < 49 / 50 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 10 ∧
    (∀ n : ℕ, (0 : ℝ) ≤ (if n < 10 then 0 else 1 / 1000) ∧
      0 ≤ 1 - (if n < 10 then 0 else 1 / 1000) * (1 / 10)) := by
  refine ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm,
    rfl, rfl, rfl, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro n
  split_ifs <;> norm_num

/-- Original-beta scheduled paths have the rational physical box.
Source: native betas at 0033b1b and generated 17/10 direction
ceiling; the positive decay denominator never contains epsilon. -/
theorem scalar_scheduled_original_parameter_box (eps decay : ℝ) (rate : ℕ → ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep (9 / 10) (49 / 50) eps decay (rate n) (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0) (hc : (state 0).clock = 0) (he : 0 < eps)
    (hdecay : 0 < decay) (hlegal : ∀ n, 0 ≤ rate n ∧ 0 ≤ 1 - rate n * decay) :
    ∀ n, |(state n).parameter| ≤ max |(state 0).parameter| ((17 / 10) / decay) := by
  have ht := scalar_scheduled_initialized_parameter_box (9 / 10) (49 / 50) eps decay rate state gradient
    hstep hm hv hc (by norm_num) (by norm_num) (by norm_num) (by norm_num) he hdecay hlegal
  have hscale := div_le_div_of_nonneg_right (le_of_lt native_original_square_root_ceiling) (le_of_lt hdecay)
  intro n
  exact (ht n).trans (max_le_max_left _ hscale)

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) (if n < 10 then 0 else 1 / 1000) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧ (zeroScalarStateAt 0).clock = 0 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 10 ∧
    (∀ n : ℕ, (0 : ℝ) ≤ (if n < 10 then 0 else 1 / 1000) ∧
      0 ≤ 1 - (if n < 10 then 0 else 1 / 1000) * (1 / 10)) := by
  refine ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm,
    rfl, rfl, rfl, by norm_num, by norm_num, ?_⟩
  intro n
  split_ifs <;> norm_num

/-- Exact-real original completed-update warmup rate. Source:
domain.training.rate at 0033b1b, warmup=10, inclusive=False,
base rate=0.001 and no annealing; the completed clock enters it. -/
noncomputable def grokkingWarmupRate (n : ℕ) : ℝ := (1 / 1000) * min 1 ((n : ℝ) / 10)

/-- The exact original warmup rate stays within its cap and keeps
positive-decay updates legal. Source: domain.training.rate at
0033b1b; this does not bound future learned gradients or states. -/
theorem grokking_warmup_rate_legal (n : ℕ) :
    0 ≤ grokkingWarmupRate n ∧ grokkingWarmupRate n ≤ 1 / 1000 ∧
      0 ≤ 1 - grokkingWarmupRate n * (1 / 10) := by
  have hn : 0 ≤ min (1 : ℝ) ((n : ℝ) / 10) := le_min (by norm_num) (by positivity)
  have hc := min_le_left (1 : ℝ) ((n : ℝ) / 10)
  refine ⟨?_, ?_, ?_⟩
  · exact mul_nonneg (by norm_num) hn
  · unfold grokkingWarmupRate
    nlinarith only [hc]
  · unfold grokkingWarmupRate
    nlinarith only [hc]

/-- The original warmup starts at zero, then 0.0001, and reaches
the constant rate after ten completed updates. Source: actual
completed-update indexing in domain.training.rate at 0033b1b. -/
theorem grokking_warmup_rate_profile : grokkingWarmupRate 0 = 0 ∧
    grokkingWarmupRate 1 = 1 / 10000 ∧ ∀ n, 10 ≤ n → grokkingWarmupRate n = 1 / 1000 := by
  refine ⟨by norm_num [grokkingWarmupRate], by norm_num [grokkingWarmupRate], ?_⟩
  intro n hn
  have hc : (10 : ℝ) ≤ n := by exact_mod_cast hn
  have ht : (1 : ℝ) ≤ (n : ℝ) / 10 := (le_div_iff₀ (by norm_num)).mpr (by linarith only [hc])
  unfold grokkingWarmupRate
  rw [min_eq_left ht, mul_one]

example : 10 ≤ (17 : ℕ) := by norm_num

/-- The original exact-real warmup preserves the native physical
box 17 for every initialized coordinate. Source: original schedule
and betas/decay at 0033b1b, native step at 88aa892; only initial
data and the true scheduled recurrence, not future bounds, are inputs. -/
theorem scalar_original_warmup_parameter_box (eps : ℝ) (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep (9 / 10) (49 / 50) eps (1 / 10) (grokkingWarmupRate n)
      (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0) (hc : (state 0).clock = 0)
    (he : 0 < eps) (hi : |(state 0).parameter| ≤ 17) : ∀ n, |(state n).parameter| ≤ 17 := by
  have ht := scalar_scheduled_original_parameter_box eps (1 / 10) grokkingWarmupRate state gradient
    hstep hm hv hc he (by norm_num) (fun n => ⟨(grokking_warmup_rate_legal n).1, (grokking_warmup_rate_legal n).2.2⟩)
  intro n
  exact (ht n).trans (max_le hi (by norm_num))

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) (grokkingWarmupRate n) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧ (zeroScalarStateAt 0).clock = 0 ∧
    (0 : ℝ) < 1 / 100000000 ∧ |(zeroScalarStateAt 0).parameter| ≤ 17 := by
  refine ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm, rfl, rfl, rfl, by norm_num, ?_⟩
  norm_num [zeroScalarStateAt]

/-- Arbitrary globally coupled learned-coordinate inputs retain
the original warmup physical box. Source: native coordinate updates
and original rate at 0033b1b; the exact-real update bridge is explicit. -/
theorem native_original_warmup_coordinate_box (ι : Type*) (eps : ℝ)
    (state : ℕ → ι → ScalarState) (gradient : ℕ → ι → ℝ)
    (hstep : ∀ n i, state (n + 1) i = scalarNativeStep (9 / 10) (49 / 50) eps (1 / 10) (grokkingWarmupRate n)
      (state n i) (gradient n i))
    (hm : ∀ i, (state 0 i).moment = 0) (hv : ∀ i, (state 0 i).variance = 0)
    (hc : ∀ i, (state 0 i).clock = 0) (he : 0 < eps) (hi : ∀ i, |(state 0 i).parameter| ≤ 17) :
    ∀ n i, |(state n i).parameter| ≤ 17 := by
  intro n i
  exact scalar_original_warmup_parameter_box eps (fun k => state k i) (fun k => gradient k i)
    (fun k => hstep k i) (hm i) (hv i) (hc i) he (hi i) n

example :
    (∀ n : ℕ, ∀ i : Fin 4, (fun k (_ : Fin 4) => zeroScalarStateAt k) (n + 1) i =
      scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (grokkingWarmupRate n) (zeroScalarStateAt n) 0) ∧
    (∀ i : Fin 4, ((fun (_ : Fin 4) => zeroScalarStateAt 0) i).moment = 0) ∧
    (∀ i : Fin 4, ((fun (_ : Fin 4) => zeroScalarStateAt 0) i).variance = 0) ∧
    (∀ i : Fin 4, ((fun (_ : Fin 4) => zeroScalarStateAt 0) i).clock = 0) ∧
    (0 : ℝ) < 1 / 100000000 ∧ (∀ i : Fin 4, |((fun (_ : Fin 4) => zeroScalarStateAt 0) i).parameter| ≤ 17) := by
  refine ⟨fun n i => (scalar_native_zero_step _ _ _ _ _ n).symm,
    fun i => rfl, fun i => rfl, fun i => rfl, by norm_num, ?_⟩
  intro i
  norm_num [zeroScalarStateAt]

end Transformer.Grokking.AdamW
