import Transformer.Grokking.CircuitEfficiency.SectionC_GainRecurrence
import Mathlib.Topology.Order.OrderClosed

/-!
# Actual physical-gain CE feedback preserves numerical signs

Sources: Varma et al., arXiv:2309.02390v1, section 3's two-factor
formation and appendix C product CE; retained native AdamW/clipping
at lab commit 65ce284. Derive nonpositive coordinate inputs from
positive physical gains and present nonnegative partners. Close the
parameter/moment feedback by induction on the actual native path.

All coordinates retain one uniform decay, positive epsilon, both
buffers and growing clocks. A positive seeded coordinate persists
at every finite clock when the remaining decay factor is positive.
Its floor is the actual pure-decay contribution, not an assumed
constant positive bound or a future limiting margin.

Plain CE, fixed physical readouts and native decoupled decay differ
from appendix C's GD/coupled norm cost. These sign and finite-clock
noncollapse laws do not prove positive finite limits, attraction,
delayed selection or learned stochastic/numerical GPTMini transfer.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Every coordinate has its own positive physical readout factor.
Source: appendix C pair order and section 3's explicit gained forward. -/
theorem gain_native_factor_gain_pos (genGain memGain : ℝ) (hgen : 0 < genGain) (hmem : 0 < memGain) :
    ∀ i, 0 < nativeFactorGain genGain memGain i := by
  intro i
  fin_cases i
  · exact hgen
  · exact hgen
  · exact hmem
  · exact hmem

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 := by norm_num

/-- Actual clipped CE inputs are nonpositive on nonnegative partners.
Sources: appendix C true CE and native shared clipping at 65ce284;
the gain enters the actual chain factor, not a penalty or optimizer group. -/
theorem gain_native_applied_gradient_nonpos (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hclip : 0 < bound)
    (hg : 0 ≤ nativeFactorGain genGain memGain i)
    (hp : 0 ≤ (state (nativeFactorPartner i)).parameter) :
    appliedGainNativeGradient remaining genGain memGain bound state i ≤ 0 := by
  rw [gain_native_applied_gradient_scale]
  apply neg_nonpos.mpr
  exact mul_nonneg (mul_nonneg (le_of_lt (gain_ce_gradient_scale_pos remaining _ _ _ state hclip)) hg) hp

example : (0 : ℝ) < 1 ∧ 0 ≤ nativeFactorGain 3 2 0 ∧
    0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) (nativeFactorPartner 0)).parameter := by
  norm_num [nativeFactorGain, nativeFactorPartner, seededNativeSubweights, seededScalarState]

/-- One actual gained native step retains all numerical sign data.
Sources: appendix C product feedback and native AdamW at 65ce284;
no future gradient sign, moment reset or task-success premise is supplied. -/
theorem gain_native_nonnegative_step (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState state) :
    NonnegativeNativeState (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) := by
  intro i
  exact scalar_nonnegative_step b1 b2 eps decay rate _ (state i) hb1 h1 hb2 (le_of_lt h2) he heta hd
    (hs i) (gain_native_applied_gradient_nonpos remaining genGain memGain bound state i hclip
      (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)) (hs _).1)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- The complete retained gained-CE path stays in the numerical sign
region. Sources: appendix C feedback and native AdamW at 65ce284;
induction includes current partners, both buffers and completed clocks. -/
theorem gain_native_nonnegative_path (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ) (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial) :
    NonnegativeNativeState (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) := by
  induction n with
  | zero => exact hs
  | succ n ih =>
    exact gain_native_nonnegative_step remaining genGain memGain bound b1 b2 eps decay rate _
      hgen hmem hclip hb1 h1 hb2 h2 he heta hd ih

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- A coordinate's actual update stays above its pure-decay floor.
Sources: appendix C partner CE and native AdamW at 65ce284; this
permits positive-decay decreases and does not presume monotonicity. -/
theorem gain_native_parameter_decay_floor (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hclip : 0 < bound)
    (hg : 0 ≤ nativeFactorGain genGain memGain i) (hp : 0 ≤ (state (nativeFactorPartner i)).parameter)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate) (hm : (state i).moment ≤ 0) :
    (1 - rate * decay) * (state i).parameter ≤
      (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i).parameter := by
  exact scalar_parameter_decay_floor b1 b2 eps decay rate _ (state i) hb1 h1 he heta hm
    (gain_native_applied_gradient_nonpos remaining genGain memGain bound state i hclip hg hp)

example : (0 : ℝ) < 1 ∧ 0 ≤ nativeFactorGain 3 2 0 ∧
    0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) (nativeFactorPartner 0)).parameter ∧
    0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧ 0 ≤ (1 / 1000 : ℝ) ∧
    (seededNativeSubweights ((0, 1 / 200), (0, 1)) 0).moment ≤ 0 := by
  norm_num [nativeFactorGain, nativeFactorPartner, seededNativeSubweights, seededScalarState]

/-- A positive seeded coordinate stays strictly positive at all
finite actual gained native clocks. Sources: section 3's small seeds
and native AdamW at 65ce284; no positive limiting value is inferred. -/
theorem gain_native_positive_coordinate_path (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ) (i : Fin 4)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial i).parameter) :
    PositiveScalarState (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i) := by
  induction n with
  | zero => exact ⟨hi, (hs i).2⟩
  | succ n ih =>
    have hn := gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
      hgen hmem hclip hb1 h1 hb2 h2 he heta (le_of_lt hd) hs
    exact scalar_positive_step b1 b2 eps decay rate _ _ hb1 h1 hb2 (le_of_lt h2) he heta hd ih
      (gain_native_applied_gradient_nonpos remaining genGain memGain bound _ i hclip
        (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)) (hn _).1)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) 1).parameter := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState]⟩

/-- Finite physical parameter limits inherit the numerical signs
already established on the actual path. Sources: appendix C product
coordinates and native sign induction at 65ce284; convergence stays
explicit and no strictly positive limit follows from this weak bound. -/
theorem gain_native_nonnegative_parameter_limit (state : ℕ → NativeSubweightState)
    (reference : NativeSubweightState) (hs : ∀ n, NonnegativeNativeState (state n))
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    ∀ i, 0 ≤ (reference i).parameter := by
  intro i
  exact ge_of_tendsto (hp i) (Eventually.of_forall fun n => (hs n i).1)

example : (∀ _ : ℕ, NonnegativeNativeState (seededNativeSubweights ((1, 1), (1 / 2, 1 / 2)))) ∧
    (∀ i : Fin 4, Tendsto (fun _ : ℕ => (seededNativeSubweights ((1, 1), (1 / 2, 1 / 2)) i).parameter)
      atTop (nhds (seededNativeSubweights ((1, 1), (1 / 2, 1 / 2)) i).parameter)) := by
  exact ⟨fun _ => native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    fun i => tendsto_const_nhds⟩

end Transformer.Grokking.CircuitEfficiency
