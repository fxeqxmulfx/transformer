module

public import Transformer.Grokking.Geometry.ProbabilityLoss

/-!
# Transfer target probability through competitor odds bounds

Source: `review/flat_spectrum.md`, *Probability transfer*, PT1–PT2.
Training probabilities supply competitor odds. Valid bounds on changes in
relative logits multiply those odds by explicit exponential penalties.
Their sum bounds query loss and target probability. Different competitors
may use different training anchors; no held-out target error is assumed.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.Geometry

open Finset Landscape.Neural

variable {C : Type*} [Fintype C] [DecidableEq C]

/-- Sum of wrong-class odds from upper bounds on relative logits, PT1. -/
def oddsEnvelope (U : C → ℝ) (t : C) : ℝ := ∑ c ∈ univ.erase t, Real.exp (U c)

/-- The summed wrong-class odds are nonnegative, including a one-class task, PT1. -/
theorem oddsEnvelope_nonneg (U : C → ℝ) (t : C) : 0 ≤ oddsEnvelope U t :=
  sum_nonneg fun c _ => (Real.exp_pos (U c)).le

omit [DecidableEq C] in
/-- Relative logit exponentials are exactly probability ratios, PT1. -/
theorem probability_ratio (z : C → ℝ) (c t : C) :
    probability z c / probability z t = Real.exp (z c - z t) := by
  have hd := ne_of_gt (sum_exp_pos z t)
  have ht := ne_of_gt (Real.exp_pos (z t))
  rw [probability, probability, Real.exp_sub]
  field_simp

/-- The target probability is the reciprocal of one plus the total wrong odds,
PT1. All classes are included in the same softmax denominator. -/
theorem probability_eq_one_div_odds (z : C → ℝ) (t : C) :
    probability z t = 1 / (1 + oddsEnvelope (fun c => z c - z t) t) := by
  have hs : (∑ c, Real.exp (z c - z t)) = 1 + oddsEnvelope (fun c => z c - z t) t := by
    rw [← sum_erase_add univ (fun c => Real.exp (z c - z t)) (mem_univ t)]
    simp [oddsEnvelope, add_comm]
  rw [← hs]
  simp_rw [Real.exp_sub]
  rw [← sum_div]
  simp [probability, div_eq_mul_inv]

/-- The same wrong-class odds reconstruct the actual CE, PT1. -/
theorem crossEntropy_eq_log_odds (z : C → ℝ) (t : C) :
    crossEntropy z t = Real.log (1 + oddsEnvelope (fun c => z c - z t) t) := by
  rw [crossEntropy_eq_neg_log_probability, probability_eq_one_div_odds]
  simp

/-- Every valid relative-logit envelope bounds the total wrong odds, PT2. -/
theorem odds_le_envelope (z U : C → ℝ) (t : C)
    (h : ∀ c, c ≠ t → z c - z t ≤ U c) :
    oddsEnvelope (fun c => z c - z t) t ≤ oddsEnvelope U t := by
  apply sum_le_sum
  intro c hc
  exact Real.exp_le_exp.mpr (h c (mem_erase.mp hc).1)

/-- **Training-derived odds give a query probability floor**, PT2. -/
theorem probability_ge_of_contrast_bounds (z U : C → ℝ) (t : C)
    (h : ∀ c, c ≠ t → z c - z t ≤ U c) :
    1 / (1 + oddsEnvelope U t) ≤ probability z t := by
  rw [probability_eq_one_div_odds]
  apply one_div_le_one_div_of_le (by have := oddsEnvelope_nonneg (fun c => z c - z t) t; linarith)
  linarith [odds_le_envelope z U t h]

/-- The same construction bounds query CE without measuring query CE, PT2. -/
theorem crossEntropy_le_of_contrast_bounds (z U : C → ℝ) (t : C)
    (h : ∀ c, c ≠ t → z c - z t ≤ U c) :
    crossEntropy z t ≤ Real.log (1 + oddsEnvelope U t) := by
  rw [crossEntropy_eq_log_odds]
  apply Real.log_le_log (by have := oddsEnvelope_nonneg (fun c => z c - z t) t; linarith)
  linarith [odds_le_envelope z U t h]

/-- Average the independently valid query ceilings over the declared query
set, PT2. Failed classification conditions are still included in this bound. -/
theorem meanCrossEntropy_le_of_contrast_bounds {I : Type*} (rows : Finset I)
    (z U : I → C → ℝ) (target : I → C)
    (h : ∀ i ∈ rows, ∀ c, c ≠ target i → z i c - z i (target i) ≤ U i c) :
    meanCrossEntropy rows z target ≤
      (∑ i ∈ rows, Real.log (1 + oddsEnvelope (U i) (target i))) / rows.card := by
  unfold meanCrossEntropy
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg rows.card)
  exact sum_le_sum fun i hi => crossEntropy_le_of_contrast_bounds (z i) (U i) (target i) (h i hi)

omit [Fintype C] [DecidableEq C] in
/-- Negative individual relative-logit bounds suffice for argmax correctness,
PT2. This weaker condition does not assert target probability above one half. -/
theorem strict_winner_of_negative_contrast_bounds (z U : C → ℝ) (t : C)
    (h : ∀ c, c ≠ t → z c - z t ≤ U c) (hU : ∀ c, c ≠ t → U c < 0) :
    ∀ c, c ≠ t → z c < z t := by
  intro c hc
  linarith [h c hc, hU c hc]

/-- Total wrong odds below one ensure a strict winner, PT2. This is stronger
than checking each individual wrong-class odds ratio below one. -/
theorem strict_winner_of_odds_envelope (z U : C → ℝ) (t : C)
    (h : ∀ c, c ≠ t → z c - z t ≤ U c) (hU : oddsEnvelope U t < 1) :
    ∀ c, c ≠ t → z c < z t := by
  intro c hc
  have he : Real.exp (U c) ≤ oddsEnvelope U t :=
    single_le_sum (fun v _ => (Real.exp_pos (U v)).le) (mem_erase.mpr ⟨hc, mem_univ c⟩)
  have hu : U c < 0 := Real.exp_lt_one_iff.mp (lt_of_le_of_lt he hU)
  linarith [h c hc]

/-- Nonzero permitted logit deterioration still yields a positive probability
and decision bound, PT5. The training target odds are multiplied by two. -/
theorem nonzero_probability_transfer_witness :
    let z : Bool → ℝ := fun c => if c then Real.log 4 - Real.log 2 else 0
    let U : Bool → ℝ := fun c => (if c then 0 else -Real.log 4) + Real.log 2
    1 / (1 + oddsEnvelope U true) ≤ probability z true ∧
      oddsEnvelope U true = 1 / 2 ∧ z false < z true := by
  dsimp
  have h : ∀ c : Bool, c ≠ true →
      (if c then Real.log 4 - Real.log 2 else 0) - (Real.log 4 - Real.log 2) ≤
        (if c then 0 else -Real.log 4) + Real.log 2 := by
    intro c hc
    cases c <;> simp_all
  have he : oddsEnvelope (fun c : Bool => (if c then 0 else -Real.log 4) + Real.log 2) true = 1 / 2 := by
    norm_num [oddsEnvelope, Real.exp_add, Real.exp_neg, Real.exp_log]
  refine ⟨probability_ge_of_contrast_bounds _ _ true h, he, ?_⟩
  exact strict_winner_of_odds_envelope _ _ true h (by rw [he]; norm_num) false (by decide)

example := nonzero_probability_transfer_witness
example := odds_le_envelope (fun _ : Bool => (0 : ℝ)) (fun _ => 1) true
  (by intro c hc; norm_num)
example := crossEntropy_le_of_contrast_bounds (fun _ : Bool => (0 : ℝ)) (fun _ => 1) true
  (by intro c hc; norm_num)
example := meanCrossEntropy_le_of_contrast_bounds (univ : Finset Bool)
  (fun _ c : Bool => if c then (1 : ℝ) else 0)
  (fun _ c : Bool => if c then (0 : ℝ) else -1) (fun _ => true)
  (by intro i hi c hc; cases c <;> simp_all)
example := strict_winner_of_negative_contrast_bounds
  (fun c : Bool => if c then (1 : ℝ) else 0)
  (fun c : Bool => if c then (0 : ℝ) else -1) true
  (by intro c hc; cases c <;> simp_all) (by intro c hc; cases c <;> simp_all)

end Transformer.Grokking.Geometry
