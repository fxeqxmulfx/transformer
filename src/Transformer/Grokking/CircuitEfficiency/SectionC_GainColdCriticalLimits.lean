import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdCeiling
import Transformer.Grokking.CircuitEfficiency.SectionC_GainReferenceBalance

/-!
# Necessary finite native limits at and above the cold threshold

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product partials; actual normalized finite
balance at c5c2105, nonnegative pair classification at c5b8413 and
global native cold coefficient ceiling at 696630b.

A nonnegative pair satisfying normalized native parameter balance
must be zero when its positive shared coefficient is at most decay
times epsilon. A live balanced pair would require that same coefficient
to be strictly greater than decay times epsilon, a contradiction.

Apply this to both physical Gen/Mem pairs with their actual clipped
CE coefficients. At or above the cold greater-gain threshold every
nonnegative normalized balanced parameter point is fully zero.
Buffers and clocks do not enter this numerical point balance and
are not declared fixed or limiting whole-state coordinates.

On an initialized legal retained native trajectory, supplied finite
physical parameter convergence generates normalized balance and
nonnegative reference parameters. Hence any such finite reference
at or above the threshold is zero, including exact critical equality.
The critical all-zero native path jointly witnesses the hypotheses.

This is a necessary conditional classification at equality, not a
proof that positive-seed critical trajectories converge. It must not
replace the still-open critical attraction question by an assumed
limit. Fixed gained tables/plain CE/uniform native decay differ from
appendix C's assigned coupled norm-cost GD. Learned stochastic/numerical
GPTMini, stable weak-decay rule selection and thermodynamic size scaling
are separate; no optimizer, buffer or experiment is changed.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- A nonnegative normalized pair cannot have a live finite balance
at or above its coefficient threshold. Sources: appendix C partner
partials and native nonnegative pair classification at c5b8413;
the result permits exact equality and does not assert convergence. -/
theorem native_nonnegative_pair_balance_zero_at_ceiling (scale eps decay first second : ℝ)
    (hs : 0 < scale) (he : 0 < eps) (hf : 0 ≤ first) (ht : 0 ≤ second)
    (hfirst : decay * first = scale * second / (scale * second + eps))
    (hsecond : decay * second = scale * first / (scale * first + eps))
    (hceiling : scale ≤ decay * eps) :
    first = 0 ∧ second = 0 := by
  rcases native_nonnegative_pair_balance scale eps decay first second hs he hf ht hfirst hsecond with
    hz | ⟨hd, hpositive, _, _, hbalance⟩
  · exact hz
  · have hstrict := mul_pos (mul_pos hd hs) hpositive
    exfalso
    nlinarith only [hbalance, hstrict, hceiling]

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧
    (1 : ℝ) * 0 = 1 * 0 / (1 * 0 + 1) ∧ (1 : ℝ) * 0 = 1 * 0 / (1 * 0 + 1) ∧
    (1 : ℝ) ≤ 1 * 1 := by norm_num

/-- At or above the actual cold greater-gain threshold all nonnegative
normalized balanced physical parameters vanish. Sources: section 3
competition, appendix C actual partner partials and cold ceiling at
696630b; this is parameter-point balance, without a frozen native state. -/
theorem gain_native_cold_balanced_point_zero (remaining : ℕ) (genGain memGain bound eps decay : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps)
    (hp : ∀ i, 0 ≤ (state i).parameter)
    (hb : ∀ i, decay * (state i).parameter + appliedGainNativeGradient remaining genGain memGain bound state i /
      (|appliedGainNativeGradient remaining genGain memGain bound state i| + eps) = 0)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    ∀ i, (state i).parameter = 0 := by
  have hn := gain_native_nonnegative_partner_balance remaining genGain memGain bound eps decay state hgen hmem hclip hp hb
  have hs := gain_ce_gradient_scale_pos remaining genGain memGain bound state hclip
  have hscale := gain_ce_gradient_scale_cold_ceiling remaining genGain memGain bound state
    (le_of_lt hgen) (le_of_lt hmem) hclip hp
  have hgenBound := le_trans (mul_le_mul_of_nonneg_right hscale (le_of_lt hgen)) hthreshold
  have hmemBound := le_trans (mul_le_mul_of_nonneg_left hgain (le_of_lt hs)) hgenBound
  have hg := native_nonnegative_pair_balance_zero_at_ceiling
    (gainCEGradientScale remaining genGain memGain bound state * genGain) eps decay (state 0).parameter (state 1).parameter
    (mul_pos hs hgen) he (hp 0) (hp 1)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 0)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 1) hgenBound
  have hm := native_nonnegative_pair_balance_zero_at_ceiling
    (gainCEGradientScale remaining genGain memGain bound state * memGain) eps decay (state 2).parameter (state 3).parameter
    (mul_pos hs hmem) he (hp 2) (hp 3)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 2)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 3) hmemBound
  intro i
  fin_cases i
  · exact hg.1
  · exact hg.2
  · exact hm.1
  · exact hm.2

example :
    let state : NativeSubweightState := fun _ => { parameter := 0, moment := -1, variance := 1, clock := 37 }
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (∀ i, 0 ≤ (state i).parameter) ∧
      (∀ i, (3 / 2 : ℝ) * (state i).parameter + appliedGainNativeGradient 0 3 2 1 state i /
        (|appliedGainNativeGradient 0 3 2 1 state i| + 1) = 0) ∧
      coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, fun _ => le_rfl,
    ?_, by norm_num [coldGainCEGradientScale]⟩
  intro i
  rw [gain_native_applied_gradient_scale]
  norm_num

/-- Any supplied finite physical parameter limit on a legal initialized
native path at or above the cold threshold is zero. Sources: actual
finite-limit balance at c5c2105, appendix C feedback and cold ceiling
at 696630b; at equality existence of this convergence stays explicit. -/
theorem gain_native_cold_finite_parameter_limit_zero (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial reference : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps)
    (hp : ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds (reference i).parameter)) :
    ∀ i, (reference i).parameter = 0 := by
  let path := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  have hsign := fun n => gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
    hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) hkeep hs
  have href := gain_native_nonnegative_parameter_limit path reference hsign hp
  have hbalance := fun i => gain_native_closed_limit_balance remaining genGain memGain bound b1 b2 eps decay rate
    path reference i (fun _ => rfl) hb1 h1 hb2 h2 he heta hp
  exact gain_native_cold_balanced_point_zero remaining genGain memGain bound eps decay reference
    hgen hmem hgain hclip he href hbalance hthreshold

example :
    let point := seededNativeSubweights ((0, 0), (0, 0))
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
      (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
      NonnegativeNativeState point ∧ coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 ∧
      (∀ i, Tendsto (fun n => (gainNativePath 0 3 2 1 (9 / 10) (49 / 50) 1 (3 / 2) (1 / 1000) point n i).parameter)
        atTop (nhds (point i).parameter)) := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ le_rfl le_rfl le_rfl le_rfl, by norm_num [coldGainCEGradientScale], ?_⟩
  intro i
  apply gain_native_balanced_path_parameter_tendsto 0 3 2 1 (9 / 10) (49 / 50) 1 (3 / 2) (1 / 1000)
    ((0, 0), (0, 0)) i (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  intro j
  rw [gain_native_applied_gradient_scale]
  fin_cases j <;> norm_num [seededNativeSubweights, seededScalarState, nativeFactorPartner]

end Transformer.Grokking.CircuitEfficiency

