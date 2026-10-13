import Transformer.Grokking.AdamW.SquareDirection

/-!
# Native parameter boxes independent of epsilon and input magnitudes

Source: PyTorch 2.14.1 non-AMSGrad AdamW, adam.py lines 414--422,
445--475 and 529--547, ported at 88aa892/0033b1b. Decoupled decay
multiplies the current physical parameter by 1-rate*decay, and
the actual newly corrected direction gives the adaptive insertion.

Generate a uniform absolute parameter box from the initialized
direction estimate, positive decay and nonnegative remaining decay.
The numerical ceiling is max(abs(initial),sqrt(K)/decay). Rate can
be zero; positive decay alone does not imply attraction or eventual
generalization. No future parameter, gradient, variance or direction
bound is a premise of the initialized path result. Inputs may vary
arbitrarily and may depend on the entire evolving learned model.

With the original retained beta1=0.9/beta2=0.98, rate=0.001 and
decay=0.1, initial absolute parameters at most 17 remain at most 17
for any positive epsilon. This replaces the much larger earlier
clip_bound/((1-beta1)*epsilon*decay) box. Both buffers and their
clock remain native; clipping need not be changed or even assumed.
Exact-real recurrence bounds do not certify floating-point kernels,
stochastic convergence or the transformer-to-recurrence bridge.
-/

namespace Transformer.Grokking.AdamW

/-- A current numerical direction ceiling caps the actual absolute
parameter after one native step. Source: decay/addcdiv at 88aa892;
the retained direction remains the actual expression, with every
buffer and completed clock present in the current hypothesis. -/
theorem scalar_parameter_abs_step_ceiling (b1 b2 eps decay rate gradient ceiling : ℝ)
    (state : ScalarState) (hkeep : 0 ≤ 1 - rate * decay) (hrate : 0 ≤ rate)
    (hdir : |nextBufferDirection b1 b2 eps state.moment state.variance gradient state.clock| ≤ ceiling) :
    |(scalarNativeStep b1 b2 eps decay rate state gradient).parameter| ≤
      (1 - rate * decay) * |state.parameter| + rate * ceiling := by
  change |(1 - rate * decay) * state.parameter - rate *
    nextBufferDirection b1 b2 eps state.moment state.variance gradient state.clock| ≤ _
  have ht := abs_sub ((1 - rate * decay) * state.parameter)
    (rate * nextBufferDirection b1 b2 eps state.moment state.variance gradient state.clock)
  rw [abs_mul, abs_mul, abs_of_nonneg hkeep, abs_of_nonneg hrate] at ht
  exact ht.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left hdir hrate))

example : (0 : ℝ) ≤ 1 - (1 / 1000) * (1 / 10) ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    |nextBufferDirection (9 / 10) (49 / 50) 1 (seededScalarState 1).moment
      (seededScalarState 1).variance (-1) (seededScalarState 1).clock| ≤ 2 := by
  norm_num [nextBufferDirection, seededScalarState]

/-- Initialized actual native paths remain in a computed physical
box independent of epsilon magnitude and all input magnitudes.
Source: decoupled native update at 88aa892 and SquareDirection;
zero initial buffers/clock generate every later direction bound. -/
theorem scalar_initialized_parameter_box (b1 b2 eps decay rate : ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay rate (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0) (hc : (state 0).clock = 0)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2) (he : 0 < eps)
    (hdecay : 0 < decay) (hrate : 0 ≤ rate) (hkeep : 0 ≤ 1 - rate * decay) :
    ∀ n, |(state n).parameter| ≤
      max |(state 0).parameter| (Real.sqrt (nativeMomentSquareScale b1 b2) / decay) := by
  let ceiling := max |(state 0).parameter| (Real.sqrt (nativeMomentSquareScale b1 b2) / decay)
  have hcost : Real.sqrt (nativeMomentSquareScale b1 b2) ≤ ceiling * decay :=
    (div_le_iff₀ hdecay).mp (le_max_right _ _)
  have hdir := scalar_initialized_direction_ceiling b1 b2 eps decay rate state gradient
    hstep hm hv hc hb1 h1 h2 hgap he
  intro n
  change |(state n).parameter| ≤ ceiling
  induction n with
  | zero => exact le_max_left _ _
  | succ n ih =>
    rw [hstep n]
    have ht := scalar_parameter_abs_step_ceiling b1 b2 eps decay rate (gradient n)
      (Real.sqrt (nativeMomentSquareScale b1 b2)) (state n) hkeep hrate (hdir n)
    have hold := mul_le_mul_of_nonneg_left ih hkeep
    have hcostrate := mul_le_mul_of_nonneg_left hcost hrate
    calc
      _ ≤ (1 - rate * decay) * |(state n).parameter| + rate * Real.sqrt (nativeMomentSquareScale b1 b2) := ht
      _ ≤ (1 - rate * decay) * ceiling + rate * Real.sqrt (nativeMomentSquareScale b1 b2) :=
        add_le_add hold le_rfl
      _ ≤ ceiling := by nlinarith only [hcostrate]

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) (1 / 1000) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧ (zeroScalarStateAt 0).clock = 0 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (49 / 50 : ℝ) < 1 ∧ (9 / 10 : ℝ) ^ 2 < 49 / 50 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) ≤ 1 - (1 / 1000) * (1 / 10) := by
  exact ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm,
    rfl, rfl, rfl, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- Original-beta paths inherit a rational physical parameter box.
Source: native optimizer at 0033b1b; positive decay balances the
generated 17/10 direction ceiling and never divides by epsilon. -/
theorem scalar_original_parameter_box (eps decay rate : ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep (9 / 10) (49 / 50) eps decay rate (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0) (hc : (state 0).clock = 0) (he : 0 < eps)
    (hdecay : 0 < decay) (hrate : 0 ≤ rate) (hkeep : 0 ≤ 1 - rate * decay) :
    ∀ n, |(state n).parameter| ≤ max |(state 0).parameter| ((17 / 10) / decay) := by
  have ht := scalar_initialized_parameter_box (9 / 10) (49 / 50) eps decay rate state gradient
    hstep hm hv hc (by norm_num) (by norm_num) (by norm_num) (by norm_num) he hdecay hrate hkeep
  have hscale := div_le_div_of_nonneg_right (le_of_lt native_original_square_root_ceiling) (le_of_lt hdecay)
  intro n
  exact (ht n).trans (max_le_max_left _ hscale)

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) (1 / 1000) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧ (zeroScalarStateAt 0).clock = 0 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) ≤ 1 - (1 / 1000) * (1 / 10) := by
  exact ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm,
    rfl, rfl, rfl, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- The original weak-decay constants keep every initialized
physical parameter of absolute size at most 17 in that same box.
Source: original beta1/beta2/rate/decay at 0033b1b; epsilon remains
arbitrary positive and all later learned-coordinate inputs may vary. -/
theorem scalar_original_weak_decay_box (eps : ℝ) (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000)
      (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0) (hc : (state 0).clock = 0)
    (he : 0 < eps) (hi : |(state 0).parameter| ≤ 17) : ∀ n, |(state n).parameter| ≤ 17 := by
  have ht := scalar_original_parameter_box eps (1 / 10) (1 / 1000) state gradient
    hstep hm hv hc he (by norm_num) (by norm_num) (by norm_num)
  intro n
  have hmax : max |(state 0).parameter| ((17 / 10 : ℝ) / (1 / 10)) ≤ 17 := by
    exact max_le hi (by norm_num)
  exact (ht n).trans hmax

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) (1 / 1000) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧ (zeroScalarStateAt 0).clock = 0 ∧
    (0 : ℝ) < 1 / 100000000 ∧ |(zeroScalarStateAt 0).parameter| ≤ 17 := by
  refine ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm, rfl, rfl, rfl, by norm_num, ?_⟩
  norm_num [zeroScalarStateAt]

/-- An arbitrary set of actual learned coordinates inherits the
original weak-decay physical box, even with globally coupled inputs.
Source: coordinatewise native updates at 0033b1b; the exact update
bridge is explicit and no future input bound is a hypothesis. -/
theorem native_original_weak_decay_coordinate_box (ι : Type*) (eps : ℝ)
    (state : ℕ → ι → ScalarState) (gradient : ℕ → ι → ℝ)
    (hstep : ∀ n i, state (n + 1) i = scalarNativeStep (9 / 10) (49 / 50) eps (1 / 10) (1 / 1000)
      (state n i) (gradient n i))
    (hm : ∀ i, (state 0 i).moment = 0) (hv : ∀ i, (state 0 i).variance = 0)
    (hc : ∀ i, (state 0 i).clock = 0) (he : 0 < eps) (hi : ∀ i, |(state 0 i).parameter| ≤ 17) :
    ∀ n i, |(state n i).parameter| ≤ 17 := by
  intro n i
  exact scalar_original_weak_decay_box eps (fun k => state k i) (fun k => gradient k i)
    (fun k => hstep k i) (hm i) (hv i) (hc i) he (hi i) n

example :
    (∀ n : ℕ, ∀ i : Fin 4, (fun k (_ : Fin 4) => zeroScalarStateAt k) (n + 1) i =
      scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000) (zeroScalarStateAt n) 0) ∧
    (∀ i : Fin 4, ((fun (_ : Fin 4) => zeroScalarStateAt 0) i).moment = 0) ∧
    (∀ i : Fin 4, ((fun (_ : Fin 4) => zeroScalarStateAt 0) i).variance = 0) ∧
    (∀ i : Fin 4, ((fun (_ : Fin 4) => zeroScalarStateAt 0) i).clock = 0) ∧
    (0 : ℝ) < 1 / 100000000 ∧ (∀ i : Fin 4, |((fun (_ : Fin 4) => zeroScalarStateAt 0) i).parameter| ≤ 17) := by
  refine ⟨fun n i => (scalar_native_zero_step _ _ _ _ _ n).symm,
    fun i => rfl, fun i => rfl, fun i => rfl, by norm_num, ?_⟩
  intro i
  norm_num [zeroScalarStateAt]

end Transformer.Grokking.AdamW
