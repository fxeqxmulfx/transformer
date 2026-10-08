import Transformer.Grokking.AdamW.SquareDirection

/-!
# Actual retained native histories with changing learning rates

Sources: PyTorch 2.14.1 non-AMSGrad AdamW, adam.py lines 414--475
and 529--547, ported at 88aa892/0033b1b; lab domain.training.rate
at 0033b1b, completed-update linear warmup. Reuse the actual scalar
update with the supplied rate at each clock, rather than introducing
a new optimizer. Both smoothing constants, epsilon and decay stay fixed.

Extend ScalarLimits.scalar_path_fields to changing rates. Neither
buffer insertion nor its completed clock depends on the learning
rate. Actual arbitrary changing inputs therefore generate the same
zero-buffer finite-history square budget and direction ceiling.
No future gradient, variance, direction or convergence bound is a
premise. No sign or clipping requirement enters those estimates.

The ordinary ten-update warmup starts with rate zero. A supplied
nonzero gradient still enters both buffers and advances the clock;
the next update uses those retained values and clock two. Check
both original-beta insertions with an explicit constant-input witness.
It is a numerical recurrence example, not an assigned GPTMini
gradient stream or a learned-task convergence theorem. Exact-real
identities do not certify the Python floating-point schedule or
kernels. Preserved rates, states and experiments are unchanged.

The history estimates alone allow arbitrary real rates because
the native buffers do not use them. A physical parameter-box
application must additionally supply nonnegative rates and
nonnegative remaining decay at each actual scheduled update.
-/

namespace Transformer.Grokking.AdamW

/-- Actual variable-rate native paths retain their causal moments
and completed clocks. Source: PyTorch insertions at 88aa892;
proof extends the fixed-rate scalar_path_fields at 9ff30f2, with
the current rate absent from the native buffer/clock projections. -/
theorem scalar_scheduled_path_fields (b1 b2 eps decay : ℝ) (rate : ℕ → ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ) (n : ℕ)
    (hstep : ∀ k, state (k + 1) = scalarNativeStep b1 b2 eps decay (rate k) (state k) (gradient k)) :
    (state n).moment = momentAt b1 gradient (state 0).moment n ∧
      (state n).variance = momentAt b2 (fun k => gradient k ^ 2) (state 0).variance n ∧
      (state n).clock = (state 0).clock + n := by
  induction n with
  | zero => exact ⟨rfl, rfl, rfl⟩
  | succ n ih =>
    rw [hstep n]
    change b1 * (state n).moment + (1 - b1) * gradient n = momentAt b1 _ _ (n + 1) ∧
      b2 * (state n).variance + (1 - b2) * gradient n ^ 2 = momentAt b2 _ _ (n + 1) ∧
      (state n).clock + 1 = (state 0).clock + (n + 1)
    rw [ih.1, ih.2.1, ih.2.2]
    refine ⟨rfl, rfl, ?_⟩
    omega

example : ∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50)
    (1 / 100000000) (1 / 10) ((1 / 1000) * min 1 ((n : ℝ) / 10)) (zeroScalarStateAt n) 0 := by
  intro n
  exact (scalar_native_zero_step _ _ _ _ _ n).symm

/-- Initialized scheduled native paths generate the exact finite
history square budget at every clock. Source: shared-input buffers
at 88aa892/0033b1b and SquareHistory; variable rates do not reset
moments or prescribe a future state inequality. -/
theorem scalar_scheduled_initialized_square (b1 b2 eps decay : ℝ) (rate : ℕ → ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay (rate n) (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0)
    (h1 : b1 < 1) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2) :
    ∀ n, (state n).moment ^ 2 ≤ nativeHistorySquareBudget b1 b2 n * (state n).variance := by
  intro n
  have hf := scalar_scheduled_path_fields b1 b2 eps decay rate state gradient n hstep
  rw [hf.1, hf.2.1, hm, hv]
  exact native_initialized_history_square b1 b2 gradient h1 h2 hgap n

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) ((1 / 1000) * min 1 ((n : ℝ) / 10)) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧
    (9 / 10 : ℝ) < 1 ∧ (49 / 50 : ℝ) < 1 ∧ (9 / 10 : ℝ) ^ 2 < 49 / 50 := by
  exact ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm,
    rfl, rfl, by norm_num, by norm_num, by norm_num⟩

/-- Zero-buffer/zero-clock scheduled native paths inherit the
epsilon-independent actual direction ceiling. Source: both native
corrections at 0033b1b and SquareDirection; arbitrary rates can
change the inputs but cannot change this generated history bound. -/
theorem scalar_scheduled_initialized_direction_ceiling (b1 b2 eps decay : ℝ) (rate : ℕ → ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay (rate n) (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0) (hc : (state 0).clock = 0)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2) (he : 0 < eps) :
    ∀ n, |nextBufferDirection b1 b2 eps (state n).moment (state n).variance (gradient n) (state n).clock| ≤
      Real.sqrt (nativeMomentSquareScale b1 b2) := by
  intro n
  have hf := scalar_scheduled_path_fields b1 b2 eps decay rate state gradient n hstep
  have ht := native_initialized_direction_ceiling b1 b2 eps gradient (n + 1) hb1 h1 h2 hgap he (by omega)
  simpa only [nextBufferDirection, hf.1, hf.2.1, hf.2.2, hm, hv, hc, zero_add,
    historyDirection, firstMomentAt, secondMomentAt, momentAt] using ht

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) ((1 / 1000) * min 1 ((n : ℝ) / 10)) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧ (zeroScalarStateAt 0).clock = 0 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (49 / 50 : ℝ) < 1 ∧
    (9 / 10 : ℝ) ^ 2 < 49 / 50 ∧ (0 : ℝ) < 1 / 100000000 := by
  exact ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm,
    rfl, rfl, rfl, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- Original-beta scheduled directions remain strictly below 1.7.
Source: original retained betas at 0033b1b; the estimate follows
from actual zero-buffer histories even during a zero-rate update. -/
theorem scalar_scheduled_original_direction_ceiling (eps decay : ℝ) (rate : ℕ → ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep (9 / 10) (49 / 50) eps decay (rate n) (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0) (hc : (state 0).clock = 0) (he : 0 < eps) :
    ∀ n, |nextBufferDirection (9 / 10) (49 / 50) eps (state n).moment (state n).variance
      (gradient n) (state n).clock| < (17 / 10 : ℝ) := by
  intro n
  exact lt_of_le_of_lt
    (scalar_scheduled_initialized_direction_ceiling (9 / 10) (49 / 50) eps decay rate state gradient
      hstep hm hv hc (by norm_num) (by norm_num) (by norm_num) (by norm_num) he n)
    native_original_square_root_ceiling

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) ((1 / 1000) * min 1 ((n : ℝ) / 10)) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧ (zeroScalarStateAt 0).clock = 0 ∧
    (0 : ℝ) < 1 / 100000000 := by
  exact ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm, rfl, rfl, rfl, by norm_num⟩

/-- The original zero-rate first warmup update keeps the physical
parameter but records both nonzero buffers and clock one. Source:
domain.training.rate and native insertions at 0033b1b/88aa892;
the supplied gradient -1 is an explicit numerical witness. -/
theorem native_original_zero_rate_first_buffers :
    let state := scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) 0 (seededScalarState 1) (-1)
    state.parameter = 1 ∧ state.moment = -(1 / 10) ∧ state.variance = 1 / 50 ∧ state.clock = 1 := by
  norm_num [scalarNativeStep, seededScalarState]

/-- The next original warmup insertion uses both retained buffers
and completed clock two. Source: rate=0.0001 after one completed
update at 0033b1b and native AdamW at 88aa892; both actual inputs
are -1, with no buffer or correction restart between them. -/
theorem native_original_second_warmup_buffers :
    let first := scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) 0 (seededScalarState 1) (-1)
    let second := scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 10000) first (-1)
    second.parameter = 99999 / 100000 + 10000 / 100000001 ∧
      second.moment = -(19 / 100) ∧ second.variance = 99 / 2500 ∧ second.clock = 2 := by
  norm_num [scalarNativeStep, seededScalarState, nextBufferDirection]

end Transformer.Grokking.AdamW
