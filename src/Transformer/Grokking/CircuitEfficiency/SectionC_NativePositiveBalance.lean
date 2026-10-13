import Transformer.Grokking.CircuitEfficiency.SectionC_NativeLimitBalance
import Transformer.Grokking.CircuitEfficiency.SectionC_NativeSymmetry

/-!
# Positive allocations allowed by uniform native finite-limit balance

Sources: Varma et al., arXiv:2309.02390v1, appendix C product logits
and true CE; native AdamW/clipping at lab commit cef67e4. Factor the
actual clipped CE derivative into its present partner and one shared
positive magnitude. Analyze the native normalized balance already
derived as a necessary condition of finite parameter convergence.

With four strictly positive parameters, positive clip bound and epsilon,
this balance forces positive uniform decay and equal parameters within
and across both circuits. Thus the positive interior balance cannot
favor Gen over Mem in this equal-coefficient fixed-table model. Its
held-out CE remains at least log two. A nonzero unit-parameter witness
with epsilon chosen from its actual clipped CE scale shows the point
balance hypotheses are jointly satisfiable.

This file analyzes necessary parameter-point equations, not convergence
or attraction from arbitrary seeds. Full-state symmetry is not assumed
at the limit. Native uniform decay differs from appendix C's asymmetric
coupled circuit-norm penalty. Physical forward efficiency, learned
stochastic features and finite-precision transfer remain separate work.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Common actual clipped CE magnitude multiplying every factor's
partner. Sources: appendix C's product chain rule and native shared
norm-two coefficient at cef67e4, including all q-1 competitors. -/
noncomputable def nativeCEGradientScale (remaining : ℕ) (bound : ℝ) (state : NativeSubweightState) : ℝ :=
  coordinateClipFactor bound (rawNativeSubweightGradient remaining state) *
    (((remaining : ℝ) + 1) / (Real.exp (nativeTotalScore state) + (remaining : ℝ) + 1))

/-- The actual shared magnitude is strictly positive at every finite
parameter point. Sources: appendix C's true CE and native clipping at
cef67e4; finite zero partners do not make the common slope vanish. -/
theorem native_ce_gradient_scale_pos (remaining : ℕ) (bound : ℝ) (state : NativeSubweightState)
    (hclip : 0 < bound) : 0 < nativeCEGradientScale remaining bound state := by
  unfold nativeCEGradientScale
  exact mul_pos (coordinate_clip_factor_pos bound _ hclip) (div_pos (by positivity) (by positivity))

example : (0 : ℝ) < 1 := by norm_num

/-- Every actual applied partial uses the same magnitude and its
current partner. Sources: appendix C product CE and clip_grad_norm_
at cef67e4; this is an identity, not an assigned gradient stream. -/
theorem native_applied_gradient_scale (remaining : ℕ) (bound : ℝ)
    (state : NativeSubweightState) (i : Fin 4) :
    appliedNativeSubweightGradient remaining bound state i =
      -(nativeCEGradientScale remaining bound state * (state (nativeFactorPartner i)).parameter) := by
  change coordinateClipFactor bound _ * rawNativeSubweightGradient remaining state i = _
  rw [native_raw_gradient_partner]
  unfold nativeCEGradientScale
  ring

/-- A positive product pair satisfying native normalized balance has
equal factors and positive decay. Source: native finite-limit balance
at cef67e4 applied to appendix C partners; it is not the paper's GD
stationarity equation or a coupled norm-cost allocation condition. -/
theorem native_positive_pair_balance (scale eps decay first second : ℝ)
    (hs : 0 < scale) (he : 0 < eps) (hf : 0 < first) (ht : 0 < second)
    (hfirst : decay * first = scale * second / (scale * second + eps))
    (hsecond : decay * second = scale * first / (scale * first + eps)) :
    0 < decay ∧ first = second ∧ decay * (scale * first + eps) = scale := by
  have hds : 0 < scale * second + eps := by positivity
  have hdf : 0 < scale * first + eps := by positivity
  have hp : 0 < decay * first := by rw [hfirst]; exact div_pos (mul_pos hs ht) hds
  have hd : 0 < decay := pos_of_mul_pos_left hp (le_of_lt hf)
  have hb := (eq_div_iff (ne_of_gt hds)).mp hfirst
  have ha := (eq_div_iff (ne_of_gt hdf)).mp hsecond
  have hz : (decay * eps + scale) * (first - second) = 0 := by nlinarith [hb, ha]
  have hc : decay * eps + scale ≠ 0 := by positivity
  have heq : first = second := by
    have hzero := (mul_eq_zero.mp hz).resolve_left hc
    linarith
  rw [← heq] at hb
  have hc : first * (decay * (scale * first + eps) - scale) = 0 := by nlinarith [hb]
  have hh := (mul_eq_zero.mp hc).resolve_left (ne_of_gt hf)
  exact ⟨hd, heq, by linarith⟩

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 / 2 : ℝ) * 1 = 1 * 1 / (1 * 1 + 1) ∧
    (1 / 2 : ℝ) * 1 = 1 * 1 / (1 * 1 + 1) := by norm_num

/-- Nonzero actual CE unit parameters satisfy uniform positive-decay
balance when epsilon equals their actual clipped CE scale. Sources:
appendix C's true CE and native normalization at cef67e4. This is a
parameter-point witness; no retained-path convergence is assumed. -/
theorem native_unit_point_balance (remaining : ℕ) (bound : ℝ) (hclip : 0 < bound) :
    let state := seededNativeSubweights ((1, 1), (1, 1))
    let eps := nativeCEGradientScale remaining bound state
    0 < eps ∧ ∀ i, (1 / 2 : ℝ) * (state i).parameter +
      appliedNativeSubweightGradient remaining bound state i /
        (|appliedNativeSubweightGradient remaining bound state i| + eps) = 0 := by
  dsimp only
  let state := seededNativeSubweights ((1, 1), (1, 1))
  let scale := nativeCEGradientScale remaining bound state
  have hp : ∀ i : Fin 4, (state i).parameter = 1 := by
    intro i
    fin_cases i <;> norm_num [state, seededNativeSubweights, seededScalarState]
  have he : 0 < scale := native_ce_gradient_scale_pos remaining bound state hclip
  refine ⟨he, ?_⟩
  intro i
  have hg : appliedNativeSubweightGradient remaining bound state i = -scale := by
    simpa only [hp, mul_one] using native_applied_gradient_scale remaining bound state i
  change (1 / 2 : ℝ) * (state i).parameter +
    appliedNativeSubweightGradient remaining bound state i / (|appliedNativeSubweightGradient remaining bound state i| + scale) = 0
  rw [hg, abs_of_neg (by linarith : -scale < 0), neg_neg, hp]
  field_simp
  ring

example : (0 : ℝ) < 1 := by norm_num

/-- A strictly positive native balanced point has all four factors
equal, so uniform native decay cannot prefer Gen in the positive
interior. Sources: appendix C's equal-coefficient tables and native
limit balance at cef67e4; no full-buffer symmetry hypothesis is used. -/
theorem native_positive_balanced_point_equal (remaining : ℕ) (bound eps decay : ℝ)
    (state : NativeSubweightState) (hclip : 0 < bound) (he : 0 < eps)
    (hp : ∀ i, 0 < (state i).parameter)
    (hb : ∀ i, decay * (state i).parameter + appliedNativeSubweightGradient remaining bound state i /
      (|appliedNativeSubweightGradient remaining bound state i| + eps) = 0) :
    0 < decay ∧ ∀ i, (state i).parameter = (state 0).parameter := by
  let scale := nativeCEGradientScale remaining bound state
  have hs : 0 < scale := native_ce_gradient_scale_pos remaining bound state hclip
  have hn : ∀ i, decay * (state i).parameter =
      scale * (state (nativeFactorPartner i)).parameter / (scale * (state (nativeFactorPartner i)).parameter + eps) := by
    intro i
    have hi := hb i
    rw [abs_of_neg (native_applied_gradient_negative remaining bound state i hclip (hp _)),
      native_applied_gradient_scale, neg_neg, neg_div] at hi
    linarith
  have hg := native_positive_pair_balance scale eps decay (state 0).parameter (state 1).parameter hs he
    (hp 0) (hp 1) (by simpa [nativeFactorPartner] using hn 0) (by simpa [nativeFactorPartner] using hn 1)
  have hm := native_positive_pair_balance scale eps decay (state 2).parameter (state 3).parameter hs he
    (hp 2) (hp 3) (by simpa [nativeFactorPartner] using hn 2) (by simpa [nativeFactorPartner] using hn 3)
  have hz : (decay * scale) * ((state 0).parameter - (state 2).parameter) = 0 := by nlinarith [hg.2.2, hm.2.2]
  have hc : decay * scale ≠ 0 := ne_of_gt (mul_pos hg.1 hs)
  have hdiff := (mul_eq_zero.mp hz).resolve_left hc
  have heq : (state 2).parameter = (state 0).parameter := by linarith only [hdiff]
  refine ⟨hg.1, ?_⟩
  intro i
  fin_cases i
  · rfl
  · exact hg.2.1.symm
  · exact heq
  · exact hm.2.1.symm.trans heq

example : (0 : ℝ) < 1 ∧
    0 < nativeCEGradientScale 111 1 (seededNativeSubweights ((1, 1), (1, 1))) ∧
    (∀ i : Fin 4, 0 < (seededNativeSubweights ((1, 1), (1, 1)) i).parameter) ∧
    (∀ i, (1 / 2 : ℝ) * (seededNativeSubweights ((1, 1), (1, 1)) i).parameter +
      appliedNativeSubweightGradient 111 1 (seededNativeSubweights ((1, 1), (1, 1))) i /
        (|appliedNativeSubweightGradient 111 1 (seededNativeSubweights ((1, 1), (1, 1))) i| +
          nativeCEGradientScale 111 1 (seededNativeSubweights ((1, 1), (1, 1)))) = 0) := by
  have hw := native_unit_point_balance 111 1 (by norm_num)
  refine ⟨by norm_num, hw.1, ?_, hw.2⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Every strictly positive native balanced point retains the true
held-out CE floor. Sources: appendix C's test CE and native uniform
decay balance at cef67e4; equal parameters are derived rather than
assumed, and no symmetric retained buffers are required. -/
theorem native_positive_balanced_point_heldout_ce_floor (remaining : ℕ) (bound eps decay : ℝ)
    (state : NativeSubweightState) (hclip : 0 < bound) (he : 0 < eps)
    (hp : ∀ i, 0 < (state i).parameter)
    (hb : ∀ i, decay * (state i).parameter + appliedNativeSubweightGradient remaining bound state i /
      (|appliedNativeSubweightGradient remaining bound state i| + eps) = 0) :
    Real.log 2 ≤ Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining
      ((state 0).parameter * (state 1).parameter) ((state 2).parameter * (state 3).parameter)) 0 := by
  have hparams := (native_positive_balanced_point_equal remaining bound eps decay state hclip he hp hb).2
  have hproducts : (state 0).parameter * (state 1).parameter =
      (state 2).parameter * (state 3).parameter := by rw [hparams 1, hparams 2, hparams 3]
  rw [hproducts]
  exact table_equal_weights_heldout_ce_floor remaining _

example :
    let state := seededNativeSubweights ((1, 1), (1, 1))
    (0 : ℝ) < 1 ∧ 0 < nativeCEGradientScale 111 1 state ∧
    (∀ i : Fin 4, 0 < (state i).parameter) ∧
    (∀ i, (1 / 2 : ℝ) * (state i).parameter + appliedNativeSubweightGradient 111 1 state i /
      (|appliedNativeSubweightGradient 111 1 state i| + nativeCEGradientScale 111 1 state) = 0) := by
  dsimp only
  have hw := native_unit_point_balance 111 1 (by norm_num)
  refine ⟨by norm_num, hw.1, ?_, hw.2⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency
