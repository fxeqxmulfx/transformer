module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Fixed
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Order

/-!
# Feasible minibatch CE improvement can harm the combined training loss

Source: `review/flat_spectrum.md`, CM99. Distinct original prefixes have opposite
labels. A concrete admissible constant-class recurrence improves one CE loss
but worsens their mean. The full rotary family also contains a strict separator.
This is an implication counterexample, not a convergence claim about a schedule.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Minibatch

open Landscape.Neural (crossEntropy)

/-- CM99: actual fixed-frequency model with raw logits(log3,0). -/
noncomputable def tilted (frequency : ℝ × ℝ) : List Bool → Bool → ℝ :=
  Real.log 3 • score (Fixed.bias frequency false 1)

/-- CM99: the complete tilted model lies inside the fixed-frequency capacity. -/
theorem tilted_mem (frequency : ℝ × ℝ) :
    tilted frequency ∈ Fixed.domain frequency (Real.log 3) :=
  Fixed.generator_mem frequency _ (Fixed.bias frequency false 1).content
    (Fixed.bias_unit frequency false 1 (by norm_num))

/-- CM99: every original-class score is obtained from the actual recurrence. -/
theorem tilted_logits (frequency : ℝ × ℝ) (xs : List Bool) (c : Bool) :
    tilted frequency xs c = if c = false then Real.log 3 else 0 := by
  simp [tilted, Fixed.bias_score]

/-- CM99: the original uniform model has the usual two-class CE. -/
theorem zero_loss (target : Bool) : crossEntropy (0 : Bool → ℝ) target = Real.log 2 := by
  norm_num [crossEntropy, Fintype.sum_bool]

/-- CM99: exact full-class CE of the actual biased model on either target. -/
theorem tilted_loss (frequency : ℝ × ℝ) (xs : List Bool) (c : Bool) :
    crossEntropy (tilted frequency xs) c =
      Real.log 4 - if c = false then Real.log 3 else 0 := by
  cases c <;> norm_num [crossEntropy, Fintype.sum_bool, tilted_logits,
    Real.exp_log (by norm_num : (0 : ℝ) < 3)]

/-- CM99: mean CE on two distinct, consistently labeled original prefixes. -/
noncomputable def pairedLoss (r : List Bool → Bool → ℝ) : ℝ :=
  (crossEntropy (r [false, true]) false + crossEntropy (r [true, false]) true) / 2

/-- CM99: the first minibatch's actual CE strictly improves. -/
theorem batch_improves (frequency : ℝ × ℝ) :
    crossEntropy (tilted frequency [false, true]) false <
      crossEntropy (0 : Bool → ℝ) false := by
  rw [tilted_loss, zero_loss]
  have h23 := Real.log_lt_log (by norm_num : (0 : ℝ) < 2) (by norm_num : (2 : ℝ) < 3)
  have h4 : Real.log 4 = 2 * Real.log 2 := by
    have h := Real.log_mul (by norm_num : (2 : ℝ) ≠ 0) (by norm_num : (2 : ℝ) ≠ 0)
    norm_num at h
    linarith
  simp only [ite_true]
  linarith

/-- CM99: combined actual training CE strictly worsens despite that improvement. -/
theorem pair_worsens (frequency : ℝ × ℝ) : pairedLoss 0 < pairedLoss (tilted frequency) := by
  unfold pairedLoss
  rw [tilted_loss, tilted_loss]
  change (crossEntropy (0 : Bool → ℝ) false + crossEntropy (0 : Bool → ℝ) true) / 2 < _
  rw [zero_loss, zero_loss]
  have h34 := Real.log_lt_log (by norm_num : (0 : ℝ) < 3) (by norm_num : (3 : ℝ) < 4)
  have h4 : Real.log 4 = 2 * Real.log 2 := by
    have h := Real.log_mul (by norm_num : (2 : ℝ) ≠ 0) (by norm_num : (2 : ℝ) ≠ 0)
    norm_num at h
    linarith
  norm_num only [Bool.true_eq_false, ite_true, ite_false, sub_zero]
  linarith

/-- CM99: both endpoints are actual feasible models at the same budget. -/
theorem interference (frequency : ℝ × ℝ) :
    (0 : List Bool → Bool → ℝ) ∈ Fixed.domain frequency (Real.log 3) ∧
    tilted frequency ∈ Fixed.domain frequency (Real.log 3) ∧
    crossEntropy (tilted frequency [false, true]) false < crossEntropy (0 : Bool → ℝ) false ∧
    pairedLoss 0 < pairedLoss (tilted frequency) :=
  ⟨Fixed.zero_mem _ _, tilted_mem frequency, batch_improves frequency, pair_worsens frequency⟩

/-- CM99: a strict separating model uses the existing complete matrix witness. -/
noncomputable def separating : List Bool → Bool → ℝ :=
  Real.log 3 • score ⟨(0, 0), Pair.orderWitness⟩

/-- CM99: the same full capacity family contains the separating model. -/
theorem separating_mem : separating ∈ domain Bool (Real.log 3) :=
  generator_mem _ ⟨(0, 0), Pair.orderWitness⟩ Pair.order_unit

/-- CM99: both consistently labeled prefixes have a strict actual class margin. -/
theorem separating_margin (b : Bool) :
    0 < separating [b, !b] b - separating [b, !b] (!b) := by
  simp only [separating, Pi.smul_apply, smul_eq_mul, score_zero]
  rw [← mul_sub, Pair.order_margin]
  exact mul_pos (Real.log_pos (by norm_num)) (by norm_num)

/-- CM99: a zero-based update ceiling, not a convergence assertion. -/
noncomputable def ceiling (t : ℕ) : ℝ := 2 / ((t : ℝ) + 2)

/-- CM99: the proposed update ceiling never vanishes at a finite step. -/
theorem ceiling_pos (t : ℕ) : 0 < ceiling t :=
  div_pos (by norm_num) (by positivity)

/-- CM99: every ceiling lies inside the original convex update interval. -/
theorem ceiling_le_one (t : ℕ) : ceiling t ≤ 1 := by
  apply (div_le_one (by positivity : 0 < (t : ℝ) + 2)).mpr
  linarith [Nat.cast_nonneg (α := ℝ) t]

/-- CM99: any accepted nonnegative capped step preserves the full learned family.
Feasibility alone does not imply loss descent on other minibatches. -/
theorem capped_step {O : Type*} (t : ℕ) (B eta : ℝ) (r : List O → O → ℝ) (f : Parameters O)
    (hr : r ∈ domain O B) (hf : Unit f) (he : 0 ≤ eta) (hc : eta ≤ ceiling t) :
    (1 - eta) • r + eta • (B • score f) ∈ domain O B :=
  step_feasible B eta r f hr hf he (hc.trans (ceiling_le_one t))

/-- CM99: the same accepted step preserves the fixed-frequency control as well. -/
theorem capped_fixed_step {O : Type*} (frequency : ℝ × ℝ) (t : ℕ) (B eta : ℝ)
    (r : List O → O → ℝ) (f : LearnedRecurrent.Parameters O Pair.State)
    (hr : r ∈ Fixed.domain frequency B) (hf : LearnedRecurrent.Unit f)
    (he : 0 ≤ eta) (hc : eta ≤ ceiling t) :
    (1 - eta) • r + eta • (B • score ⟨frequency, f⟩) ∈ Fixed.domain frequency B :=
  Fixed.step_feasible frequency B eta r f hr hf he (hc.trans (ceiling_le_one t))

example : (1 - (1 / 2 : ℝ)) • (0 : List Bool → Bool → ℝ) +
    (1 / 2 : ℝ) • ((1 : ℝ) • score positionWitness) ∈ domain Bool 1 :=
  capped_step 2 1 (1 / 2) 0 positionWitness (zero_mem _) position_unit
    (by norm_num) (by norm_num [ceiling])

example : (1 - (1 / 2 : ℝ)) • (0 : List Bool → Bool → ℝ) +
    (1 / 2 : ℝ) • ((1 : ℝ) • score ⟨(1, 1 / 100), Pair.orderWitness⟩) ∈
      Fixed.domain (1, 1 / 100) 1 :=
  capped_fixed_step _ 2 1 (1 / 2) 0 Pair.orderWitness (Fixed.zero_mem _ _) Pair.order_unit
    (by norm_num) (by norm_num [ceiling])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Minibatch
