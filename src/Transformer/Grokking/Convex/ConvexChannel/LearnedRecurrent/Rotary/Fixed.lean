module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Queries

/-!
# Actual fixed-frequency rotary domains and feasible loss queries

Source: `review/convex/moments.md`, CM99. Fix only the positional frequencies;
all content coordinates remain learned. Identity transport makes the actual
signed CE query feasible within this restricted family at every input length.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Fixed

variable {O : Type*}

/-- CM99: actual identity transitions survive arbitrary fixed rotations. -/
theorem transport_identity (frequency : ℝ × ℝ) (position : ℝ) :
    transport frequency position (ContinuousLinearMap.id ℂ Pair.State) =
      ContinuousLinearMap.id ℂ Pair.State := by
  apply ContinuousLinearMap.ext
  intro z
  change turn frequency position (turn frequency (-position) z) = z
  simpa only [neg_neg] using turn_inverse frequency (-position) z

/-- CM99: frequencies are fixed, while all bounded content parameters vary. -/
def generators (frequency : ℝ × ℝ) (B : ℝ) : Set (List O → O → ℝ) :=
  {r | ∃ f : LearnedRecurrent.Parameters O Pair.State,
    LearnedRecurrent.Unit f ∧ r = B • score ⟨frequency, f⟩}

/-- CM99: the declared fixed-frequency function family. -/
def domain (frequency : ℝ × ℝ) (B : ℝ) : Set (List O → O → ℝ) :=
  convexHull ℝ (generators frequency B)

/-- CM99: fixing frequencies preserves joint score-domain convexity. -/
theorem domain_convex (frequency : ℝ × ℝ) (B : ℝ) :
    Convex ℝ (domain (O := O) frequency B) := convex_convexHull ℝ _

/-- CM99: this control is an explicit subset of the learned-frequency family. -/
theorem subset_full (frequency : ℝ × ℝ) (B : ℝ) :
    domain (O := O) frequency B ⊆ Rotary.domain O B := by
  apply convexHull_mono
  rintro r ⟨f, hf, hr⟩
  exact ⟨⟨frequency, f⟩, hf, hr⟩

/-- CM99: every complete admissible content model is feasible at the fixed frequency. -/
theorem generator_mem (frequency : ℝ × ℝ) (B : ℝ)
    (f : LearnedRecurrent.Parameters O Pair.State) (hf : LearnedRecurrent.Unit f) :
    B • score ⟨frequency, f⟩ ∈ domain frequency B :=
  subset_convexHull ℝ _ ⟨f, hf, rfl⟩

/-- CM99: the actual zero recurrence remains available at every fixed frequency. -/
theorem zero_score (frequency : ℝ × ℝ) (xs : List O) (c : O) :
    score ⟨frequency, LearnedRecurrent.zeroParameters⟩ xs c = 0 := by
  simp [score, complex, LearnedRecurrent.zeroParameters]

/-- CM99: unused mixture mass has an actual zero component in this family. -/
theorem zero_mem (frequency : ℝ × ℝ) (B : ℝ) :
    (0 : List O → O → ℝ) ∈ domain frequency B := by
  have h := generator_mem frequency B (LearnedRecurrent.zeroParameters (O := O))
    LearnedRecurrent.zero_unit
  have hz : score (⟨frequency, LearnedRecurrent.zeroParameters⟩ : Parameters O) = 0 := by
    funext xs c
    exact zero_score frequency xs c
  simpa only [hz, smul_zero] using h

/-- CM99: the same actual uniform output bound applies to the fixed control. -/
theorem domain_bound (frequency : ℝ × ℝ) (B : ℝ) (hB : 0 ≤ B)
    (r : List O → O → ℝ) (hr : r ∈ domain frequency B) (xs : List O) (c : O) :
    |r xs c| ≤ B := Rotary.domain_bound B hB r (subset_full frequency B hr) xs c

/-- CM99: ordinary full-class CE remains convex in the restricted scores. -/
theorem loss_convex [Fintype O] (frequency : ℝ × ℝ) (B : ℝ) (xs : List O) (c : O) :
    ConvexOn ℝ (domain frequency B) (fun r => Landscape.Neural.crossEntropy (r xs) c) :=
  (Rotary.loss_convex B xs c).subset (subset_full frequency B) (domain_convex frequency B)

/-- CM99: an actual bounded component step stays inside the fixed family. -/
theorem step_feasible (frequency : ℝ × ℝ) (B eta : ℝ) (r : List O → O → ℝ)
    (f : LearnedRecurrent.Parameters O Pair.State) (hr : r ∈ domain frequency B)
    (hf : LearnedRecurrent.Unit f) (he : 0 ≤ eta) (he1 : eta ≤ 1) :
    (1 - eta) • r + eta • (B • score ⟨frequency, f⟩) ∈ domain frequency B :=
  domain_convex frequency B hr (generator_mem frequency B f hf)
    (by linarith) he (by ring)

variable [DecidableEq O]

/-- CM99: signed original-class probe with the very same fixed frequencies. -/
noncomputable def bias (frequency : ℝ × ℝ) (target : O) (s : ℝ) : Parameters O :=
  ⟨frequency, (Rotary.bias target s).content⟩

/-- CM99: every actual fixed-frequency probe parameter satisfies capacity. -/
theorem bias_unit (frequency : ℝ × ℝ) (target : O) (s : ℝ) (hs : |s| ≤ 1) :
    Unit (bias frequency target s) := Rotary.bias_unit target s hs

/-- CM99: identity content keeps every actual state fixed despite rotary execution. -/
theorem bias_run (frequency : ℝ × ℝ) (target : O) (s : ℝ) (position : ℕ)
    (v : Pair.State) (xs : List O) : runFrom (bias frequency target s) position v xs = v := by
  induction xs generalizing position v with
  | nil => rfl
  | cons o xs ih =>
    change runFrom (bias frequency target s) (position + 1)
      (step (bias frequency target s) position o v) xs = v
    rw [ih]
    change transport frequency position (ContinuousLinearMap.id ℂ Pair.State) v = v
    rw [transport_identity]
    rfl

/-- CM99: complete raw class scores of the fixed-frequency bias recurrence. -/
theorem bias_score (frequency : ℝ × ℝ) (target : O) (s : ℝ) (xs : List O) (c : O) :
    score (bias frequency target s) xs c = if c = target then s else 0 := by
  unfold score complex
  rw [bias_run]
  by_cases hc : c = target <;>
    simp [bias, Rotary.bias, LearnedRecurrent.biasParameters, Pair.row_apply, hc]

/-- CM99: the exact CM98 loss queries are feasible even in the fixed family,
using the actual same-frequency bias model proved above. -/
theorem query_feasible (frequency : ℝ × ℝ) (B eta : ℝ) (r : List O → O → ℝ)
    (target : O) (s : ℝ) (hr : r ∈ domain frequency B) (hs : |s| ≤ 1)
    (he : 0 ≤ eta) (he1 : eta ≤ 1) :
    Rotary.query B eta r target s ∈ domain frequency B := by
  have h := step_feasible frequency B eta r (bias frequency target s).content hr
    (bias_unit frequency target s hs) he he1
  have hb : score (⟨frequency, (bias frequency target s).content⟩ : Parameters O) =
      score (Rotary.bias target s) := by
    funext xs c
    change score (bias frequency target s) xs c = _
    rw [bias_score, Rotary.bias_score]
  rw [hb] at h
  exact h

example : Unit (bias (1, 1 / 100) true (-1)) := bias_unit _ _ _ (by norm_num)

example : (1 : ℝ) • score ⟨(1, 1 / 100), Pair.orderWitness⟩ ∈
    domain (1, 1 / 100) 1 := generator_mem _ _ _ Pair.order_unit

example : |(0 : List Bool → Bool → ℝ) [true, false] true| ≤ 2 :=
  domain_bound (1, 1 / 100) 2 (by norm_num) 0 (zero_mem _ _) _ _

example : (1 - (1 / 2 : ℝ)) • (0 : List Bool → Bool → ℝ) +
    (1 / 2 : ℝ) • score ⟨(1, 1 / 100), Pair.orderWitness⟩ ∈ domain (1, 1 / 100) 1 := by
  simpa using step_feasible (1, 1 / 100) 1 (1 / 2) 0 Pair.orderWitness
    (zero_mem _ _) Pair.order_unit (by norm_num) (by norm_num)

example : Rotary.query 2 (1 / 2) (0 : List Bool → Bool → ℝ) true (-1) ∈
    domain (1, 1 / 100) 2 :=
  query_feasible _ _ _ _ _ _ (zero_mem _ _) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Fixed
