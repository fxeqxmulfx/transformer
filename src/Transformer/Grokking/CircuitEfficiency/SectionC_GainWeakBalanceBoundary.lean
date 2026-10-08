import Transformer.Grokking.CircuitEfficiency.SectionC_GainReferenceBalance
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# Original weak-decay balance excludes positive circuit coexistence

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C multiclass product CE; native necessary
balance at c5c2105 and physical-gain allocation at 4e10267.
Keep physical gains 3/2, uniform decay 0.1 and epsilon 1e-8.
For at most 113 classes, no nonnegative actual native balanced point
has all four physical factors positive. One complete pair is absent.

Positive Mem balance would require the actual shared CE scale above
5e-10. Simultaneous positive Gen balance would force each Gen factor
above 10/3, so the training logit exceeds 30. A checked exponential
lower bound and all finite competitors then put the complete CE
slope below 5e-10. Shared clipping cannot increase that slope.
The contradiction uses actual CE, not an assigned feedback value.

Both the experiment's 97 classes and the source's 113 classes lie
in this range. These are point and necessary-limit restrictions;
no trajectory convergence or eventual allocation is proved here.
The fully cold point witnesses the joint balanced hypotheses.
Positive boundary existence and selection require separate proofs.
Physical fixed gains, plain CE and native decoupled decay differ
from the source's coupled circuit-norm cost and GD. Learned GPTMini,
warmup/numerical and stochastic bridges remain explicit obligations.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Large Gen factors make the actual complete clipped CE scale
too small for positive original-decay Mem balance. Sources: appendix C
train CE with all competitors and native clipping at c268c1f;
even an arbitrary cap cannot raise its multiplier above one. -/
theorem original_gain_large_gen_feedback_ceiling (remaining : ℕ) (bound : ℝ)
    (state : NativeSubweightState) (hq : remaining ≤ 111)
    (hg0 : (10 / 3 : ℝ) ≤ (state 0).parameter) (hg1 : (10 / 3 : ℝ) ≤ (state 1).parameter)
    (hm0 : 0 ≤ (state 2).parameter) (hm1 : 0 ≤ (state 3).parameter) :
    gainCEGradientScale remaining 3 2 bound state < 1 / 2000000000 := by
  have hg := mul_le_mul hg0 hg1 (by norm_num) (le_trans (by norm_num) hg0)
  have hm := mul_nonneg hm0 hm1
  have hscore : (30 : ℝ) ≤ gainNativeTotalScore 3 2 state := by
    unfold gainNativeTotalScore physicalCircuitScore
    nlinarith only [hg, hm]
  have hone : (5 / 2 : ℝ) ≤ Real.exp 1 := by linarith only [Real.exp_one_gt_d9]
  have hpower := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 5 / 2) hone 30
  rw [Real.exp_one_pow] at hpower
  have hexp := hpower.trans (Real.exp_le_exp_of_le hscore)
  norm_num at hexp
  have hcount : (remaining : ℝ) ≤ 111 := by exact_mod_cast hq
  have hslope : ((remaining : ℝ) + 1) /
      (Real.exp (gainNativeTotalScore 3 2 state) + (remaining : ℝ) + 1) < 1 / 2000000000 := by
    apply (div_lt_iff₀ (by positivity)).mpr
    nlinarith only [hexp, hcount]
  have hclip := coordinate_clip_factor_le_one bound (rawGainNativeGradient remaining 3 2 state)
  have hscale : gainCEGradientScale remaining 3 2 bound state ≤
      ((remaining : ℝ) + 1) / (Real.exp (gainNativeTotalScore 3 2 state) + (remaining : ℝ) + 1) := by
    unfold gainCEGradientScale
    simpa only [one_mul] using mul_le_mul_of_nonneg_right hclip (by positivity :
      (0 : ℝ) ≤ ((remaining : ℝ) + 1) / (Real.exp (gainNativeTotalScore 3 2 state) + (remaining : ℝ) + 1))
  exact hscale.trans_lt hslope

example : (95 : ℕ) ≤ 111 ∧
    (10 / 3 : ℝ) ≤ (seededNativeSubweights ((10 / 3, 10 / 3), (0, 0)) 0).parameter ∧
    (10 / 3 : ℝ) ≤ (seededNativeSubweights ((10 / 3, 10 / 3), (0, 0)) 1).parameter ∧
    0 ≤ (seededNativeSubweights ((10 / 3, 10 / 3), (0, 0)) 2).parameter ∧
    0 ≤ (seededNativeSubweights ((10 / 3, 10 / 3), (0, 0)) 3).parameter := by
  norm_num [seededNativeSubweights, seededScalarState]

/-- Actual original-decay/epsilon balanced points cannot have all
four factors positive for the stated finite class range. Sources:
section 3 efficiency and appendix C full CE, using the necessary
native equations at c5c2105; positivity is denied, not assumed. -/
theorem original_gain_no_positive_interior (remaining : ℕ) (bound : ℝ)
    (state : NativeSubweightState) (hq : remaining ≤ 111) (hclip : 0 < bound)
    (hp : ∀ i, 0 ≤ (state i).parameter)
    (hb : ∀ i, (1 / 10 : ℝ) * (state i).parameter + appliedGainNativeGradient remaining 3 2 bound state i /
      (|appliedGainNativeGradient remaining 3 2 bound state i| + 1 / 100000000) = 0) :
    ¬∀ i, 0 < (state i).parameter := by
  intro hpositive
  let scale := gainCEGradientScale remaining 3 2 bound state
  have hs : 0 < scale := gain_ce_gradient_scale_pos remaining 3 2 bound state hclip
  have hn := gain_native_nonnegative_partner_balance remaining 3 2 bound (1 / 100000000) (1 / 10)
    state (by norm_num) (by norm_num) hclip hp hb
  have hg := native_positive_pair_balance (scale * 3) (1 / 100000000) (1 / 10)
    (state 0).parameter (state 1).parameter (by positivity) (by norm_num) (hpositive 0) (hpositive 1)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 0)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 1)
  have hm := native_positive_pair_balance (scale * 2) (1 / 100000000) (1 / 10)
    (state 2).parameter (state 3).parameter (by positivity) (by norm_num) (hpositive 2) (hpositive 3)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 2)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 3)
  have hmempos := mul_pos hs (hpositive 2)
  have hthreshold : (1 / 2000000000 : ℝ) < scale := by nlinarith only [hm.2.2, hmempos]
  have hgenfloor : (10 / 3 : ℝ) ≤ (state 0).parameter := by
    have hproduct : 0 < scale * ((state 0).parameter - 10 / 3) := by nlinarith only [hg.2.2, hthreshold]
    exact le_of_lt (sub_pos.mp ((mul_pos_iff_of_pos_left hs).mp hproduct))
  have hother : (10 / 3 : ℝ) ≤ (state 1).parameter := by rw [← hg.2.1]; exact hgenfloor
  have hce := original_gain_large_gen_feedback_ceiling remaining bound state hq hgenfloor hother (hp 2) (hp 3)
  linarith only [hthreshold, hce]

example :
    let state : NativeSubweightState := fun _ => zeroScalarStateAt 0
    (95 : ℕ) ≤ 111 ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 4, 0 ≤ (state i).parameter) ∧
    (∀ i : Fin 4, (1 / 10 : ℝ) * (zeroScalarStateAt 0).parameter +
      appliedGainNativeGradient 95 3 2 1 (fun _ => zeroScalarStateAt 0) i /
        (|appliedGainNativeGradient 95 3 2 1 (fun _ => zeroScalarStateAt 0) i| + 1 / 100000000) = 0) := by
  dsimp only
  refine ⟨by norm_num, by norm_num, fun _ => le_rfl, ?_⟩
  intro i
  rw [gain_native_cold_applied_gradient 95 3 2 1 0 i]
  norm_num [zeroScalarStateAt]

/-- One whole physical pair is absent at every nonnegative actual
original-decay/epsilon balance in this class range. Sources: appendix C
partners and native nonnegative pair classification at c5b8413;
this does not choose which pair survives or prove convergence. -/
theorem original_gain_balanced_pair_absent (remaining : ℕ) (bound : ℝ)
    (state : NativeSubweightState) (hq : remaining ≤ 111) (hclip : 0 < bound)
    (hp : ∀ i, 0 ≤ (state i).parameter)
    (hb : ∀ i, (1 / 10 : ℝ) * (state i).parameter + appliedGainNativeGradient remaining 3 2 bound state i /
      (|appliedGainNativeGradient remaining 3 2 bound state i| + 1 / 100000000) = 0) :
    ((state 0).parameter = 0 ∧ (state 1).parameter = 0) ∨
      ((state 2).parameter = 0 ∧ (state 3).parameter = 0) := by
  let scale := gainCEGradientScale remaining 3 2 bound state
  have hs : 0 < scale := gain_ce_gradient_scale_pos remaining 3 2 bound state hclip
  have hn := gain_native_nonnegative_partner_balance remaining 3 2 bound (1 / 100000000) (1 / 10)
    state (by norm_num) (by norm_num) hclip hp hb
  have hg := native_nonnegative_pair_balance (scale * 3) (1 / 100000000) (1 / 10)
    (state 0).parameter (state 1).parameter (by positivity) (by norm_num) (hp 0) (hp 1)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 0)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 1)
  have hm := native_nonnegative_pair_balance (scale * 2) (1 / 100000000) (1 / 10)
    (state 2).parameter (state 3).parameter (by positivity) (by norm_num) (hp 2) (hp 3)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 2)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 3)
  rcases hg with hz | ⟨_, hg0, hg1, _, _⟩
  · exact Or.inl hz
  rcases hm with hz | ⟨_, hm0, hm1, _, _⟩
  · exact Or.inr hz
  exfalso
  apply original_gain_no_positive_interior remaining bound state hq hclip hp hb
  intro i
  fin_cases i <;> assumption

example :
    let state : NativeSubweightState := fun _ => zeroScalarStateAt 0
    (111 : ℕ) ≤ 111 ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 4, 0 ≤ (state i).parameter) ∧
    (∀ i : Fin 4, (1 / 10 : ℝ) * (zeroScalarStateAt 0).parameter +
      appliedGainNativeGradient 111 3 2 1 (fun _ => zeroScalarStateAt 0) i /
        (|appliedGainNativeGradient 111 3 2 1 (fun _ => zeroScalarStateAt 0) i| + 1 / 100000000) = 0) := by
  dsimp only
  refine ⟨by norm_num, by norm_num, fun _ => le_rfl, ?_⟩
  intro i
  rw [gain_native_cold_applied_gradient 111 3 2 1 0 i]
  norm_num [zeroScalarStateAt]

end Transformer.Grokking.CircuitEfficiency
