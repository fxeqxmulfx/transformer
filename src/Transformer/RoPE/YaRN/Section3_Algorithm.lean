module

public import Transformer.RoPE.YaRN.Section3_Temperature
public import Transformer.RoPE.Diagnostics.SectionC6_Stability

/-!
# Complete fixed-context YaRN scores

Peng et al., arXiv:2309.00071v3, Definition 2 and §3.4: combine the
rotation-count ramp, context-derived scale, and fitted attention temperature.
The stability corollary holds for fixed Q/K activations. It is not a model
quality guarantee and does not identify a task-optimal frequency or temperature.
-/

@[expose] public section

open scoped BigOperators
open Transformer.RoPE.Diagnostics

namespace Transformer.RoPE.YaRN

/-- Definition 2 with §3.4's fixed configured context. Coefficients encode
actual Q/K pairs as in `pair_score_expansion`; the ordinary `1/sqrt(d)`
attention normalization is already included in them. -/
noncomputable def fullScore {n : ℕ} (a b omega : Fin n → ℝ)
    (original target distance : ℝ) : ℝ :=
  logitGain (contextScale original target) *
    headScore a b (fun i => yarnFrequency (omega i) (contextScale original target) original)
      distance

/-- Definition 2 really is a dot product of both length-scaled rotary Q/K
vectors, not a different operation that scales each logit only once. -/
theorem full_score_query_key {n : ℕ} (query key : Fin n → ℝ × ℝ)
    (omega : Fin n → ℝ) (original target distance : ℝ) :
    fullScore (fun i => (query i).1 * (key i).1 + (query i).2 * (key i).2)
      (fun i => (query i).1 * (key i).2 - (query i).2 * (key i).1)
      omega original target distance =
    ∑ i, pairScore
      (lengthGain (contextScale original target) * (query i).1,
        lengthGain (contextScale original target) * (query i).2)
      (lengthGain (contextScale original target) * (key i).1,
        lengthGain (contextScale original target) * (key i).2)
      (yarnFrequency (omega i) (contextScale original target) original) distance := by
  unfold fullScore headScore
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [scaled_pair_score, pair_score_expansion]
  rfl

/-- A target context at or below training recovers every original score. -/
theorem full_score_unextended {n : ℕ} (a b omega : Fin n → ℝ)
    {original target : ℝ} (ho : 0 < original) (ht : target ≤ original) (distance : ℝ) :
    fullScore a b omega original target distance = headScore a b omega distance := by
  unfold fullScore
  rw [context_scale_unextended ho ht, gains_one.2, one_mul]
  congr 1
  funext i
  exact yarn_frequency_one _ _

example : fullScore (fun _ : Fin 1 => 3) (fun _ => 2) (fun _ => 1) 4096 2048 10 =
    headScore (fun _ : Fin 1 => 3) (fun _ => 2) (fun _ => 1) 10 :=
  full_score_unextended _ _ _ (by norm_num) (by norm_num) _

/-- The actual ramp compresses the fixed-activation spectral slope by a
factor between `1/s` and one, before the separate temperature multiplication. -/
theorem yarn_spectral_slope_bounds {n : ℕ} (a b omega : Fin n → ℝ)
    (ho : ∀ i, 0 < omega i) {scale : ℝ} (hs : 1 ≤ scale) (original : ℝ) :
    spectralSlope a b omega / scale ≤
        spectralSlope a b (fun i => yarnFrequency (omega i) scale original) ∧
      spectralSlope a b (fun i => yarnFrequency (omega i) scale original) ≤
        spectralSlope a b omega := by
  have hp : ∀ i, 0 < yarnFrequency (omega i) scale original :=
    fun i => yarn_frequency_positive (ho i) hs original
  unfold spectralSlope
  rw [Finset.sum_div]
  constructor
  · apply Finset.sum_le_sum
    intro i _
    rw [abs_of_pos (ho i), abs_of_pos (hp i)]
    calc
      omega i * Real.sqrt (a i ^ 2 + b i ^ 2) / scale =
          (omega i / scale) * Real.sqrt (a i ^ 2 + b i ^ 2) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_right
        (yarn_frequency_bounds (ho i) hs original).1 (Real.sqrt_nonneg _)
  · apply Finset.sum_le_sum
    intro i _
    rw [abs_of_pos (ho i), abs_of_pos (hp i)]
    exact mul_le_mul_of_nonneg_right
      (yarn_frequency_bounds (ho i) hs original).2 (Real.sqrt_nonneg _)

example : spectralSlope (fun _ : Fin 1 => 3) (fun _ => 4) (fun _ => 1) / 4 ≤
      spectralSlope (fun _ : Fin 1 => 3) (fun _ => 4) (fun _ => yarnFrequency 1 4 4096) ∧
    spectralSlope (fun _ : Fin 1 => 3) (fun _ => 4) (fun _ => yarnFrequency 1 4 4096) ≤
      spectralSlope (fun _ : Fin 1 => 3) (fun _ => 4) (fun _ => 1) :=
  yarn_spectral_slope_bounds _ _ _ (by intro i; norm_num) (by norm_num) _

/-- Combining the complete scheme with the deterministic shift theorem gives
an explicit bound including the squared temperature gain. -/
theorem full_score_lipschitz {n : ℕ} (a b omega : Fin n → ℝ)
    (original target r t : ℝ) :
    |fullScore a b omega original target r - fullScore a b omega original target t| ≤
      |r - t| * (logitGain (contextScale original target) *
        spectralSlope a b
          (fun i => yarnFrequency (omega i) (contextScale original target) original)) := by
  have hg := logit_gain_ge_one (context_scale_ge_one original target)
  unfold fullScore
  rw [← mul_sub, abs_mul, abs_of_nonneg (by linarith :
    0 ≤ logitGain (contextScale original target))]
  calc
    _ ≤ logitGain (contextScale original target) * (|r - t| *
        spectralSlope a b
          (fun i => yarnFrequency (omega i) (contextScale original target) original)) :=
      mul_le_mul_of_nonneg_left (head_score_lipschitz_bound _ _ _ r t) (by linarith)
    _ = _ := by ring

/-- Uniform temperature multiplies a reference margin and its Lipschitz
budget by the same positive number. It cannot improve this certificate. -/
theorem temperature_margin_certificate {scale : ℝ} (hs : 1 ≤ scale)
    (margin budget width : ℝ) :
    width * (logitGain scale * budget) < logitGain scale * margin ↔
      width * budget < margin := by
  rw [show width * (logitGain scale * budget) = logitGain scale * (width * budget) by ring]
  exact temperature_preserves_order hs (width * budget) margin

example : 2 * (logitGain 4 * 1) < logitGain 4 * 3 ↔ (2 : ℝ) * 1 < 3 :=
  temperature_margin_certificate (by norm_num) _ _ _

end Transformer.RoPE.YaRN
