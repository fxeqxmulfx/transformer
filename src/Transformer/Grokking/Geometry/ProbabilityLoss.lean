module

public import Transformer.Grokking.Geometry.Basic
public import Transformer.Landscape.Neural.Loss

/-!
# The within-answer geometry controlled by actual cross-entropy

Source: `review/flat_spectrum.md`, *Loss-controlled geometry*, LG1–LG3.
Small mean CE bounds squared probability error and within-answer probability
scatter. These are objective consequences for arbitrary finite logits,
including the actual normalized tied models, not assumed feature contraction.
The bound applies on the rows where the loss is measured. It supplies neither
unseen-row control nor monotone scatter along every loss-decreasing update.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.Geometry

open Finset Landscape.Neural

variable {I C : Type*} [Fintype C]

/-- The actual finite softmax probability, LG1. -/
def probability (z : C → ℝ) (c : C) : ℝ := Real.exp (z c) / ∑ j, Real.exp (z j)

/-- Every finite-logit probability is positive, LG1. -/
theorem probability_pos (z : C → ℝ) (c : C) : 0 < probability z c :=
  div_pos (Real.exp_pos _) (sum_exp_pos z c)

/-- A target index witnesses a nonempty class set; probabilities sum to one, LG1. -/
theorem sum_probability (z : C → ℝ) (t : C) : ∑ c, probability z c = 1 := by
  simp only [probability, ← sum_div]
  exact div_self (ne_of_gt (sum_exp_pos z t))

/-- No softmax coordinate exceeds one, LG1. -/
theorem probability_le_one (z : C → ℝ) (c : C) : probability z c ≤ 1 := by
  apply (div_le_one (sum_exp_pos z c)).mpr
  exact single_le_sum (fun j _ => (Real.exp_pos (z j)).le) (mem_univ c)

/-- The project's log-sum-exp objective is negative log target probability, LG1. -/
theorem crossEntropy_eq_neg_log_probability (z : C → ℝ) (t : C) :
    crossEntropy z t = -Real.log (probability z t) := by
  rw [probability, Real.log_div (ne_of_gt (Real.exp_pos _))
    (ne_of_gt (sum_exp_pos z t)), Real.log_exp]
  unfold crossEntropy
  ring

/-- CE controls missing target probability without a margin hypothesis, LG1. -/
theorem missing_probability_le_loss (z : C → ℝ) (t : C) :
    1 - probability z t ≤ crossEntropy z t := by
  rw [crossEntropy_eq_neg_log_probability]
  have h := Real.log_le_sub_one_of_pos (probability_pos z t)
  linarith

/-- One-hot squared error is bounded by twice the actual row CE, LG2. -/
theorem probability_error_le_twice_loss [DecidableEq C] (z : C → ℝ) (t : C) :
    ∑ c, (probability z c - if c = t then 1 else 0) ^ 2 ≤ 2 * crossEntropy z t := by
  classical
  have hs : ∑ c, probability z c ^ 2 ≤ 1 := by
    calc
      _ ≤ ∑ c, probability z c := sum_le_sum fun c _ => by
        have hp := probability_pos z c
        have h1 := probability_le_one z c
        nlinarith
      _ = 1 := sum_probability z t
  have he : (∑ c, (probability z c - if c = t then 1 else 0) ^ 2) =
      (∑ c, probability z c ^ 2) + 1 - 2 * probability z t := by
    have hi (c : C) : (probability z c - if c = t then 1 else 0) ^ 2 =
        probability z c ^ 2 + (if c = t then 1 else 0) -
          2 * (if c = t then probability z c else 0) := by
      split_ifs <;> ring
    simp_rw [hi]
    simp [sum_sub_distrib, sum_add_distrib]
  rw [he]
  have hl := missing_probability_le_loss z t
  linarith

/-- Center probabilities at their computed group mean, LG3. The least-squares
property of that mean and LG2 bound within-answer scatter by the observed loss. -/
theorem probability_scatter_le_twice_loss (rows : Finset I) (z : I → C → ℝ) (t : C) :
    (∑ c, residualOver rows (fun i => probability (z i) c)) ≤
      2 * ∑ i ∈ rows, crossEntropy (z i) t := by
  classical
  calc
    _ ≤ ∑ c, energyOver rows (fun i => probability (z i) c -
        if c = t then 1 else 0) := by
      exact sum_le_sum fun c _ => mean_minimizes_error rows _ _
    _ = ∑ i ∈ rows, ∑ c, (probability (z i) c - if c = t then 1 else 0) ^ 2 := by
      unfold energyOver
      exact sum_comm
    _ ≤ ∑ i ∈ rows, 2 * crossEntropy (z i) t :=
      sum_le_sum fun i _ => probability_error_le_twice_loss (z i) t
    _ = _ := by rw [mul_sum]

/-- Mean training CE gives a decreasing ceiling on mean probability scatter,
LG3. This does not assert that the scatter itself decreases every step. -/
theorem mean_probability_scatter_le (rows : Finset I) (z : I → C → ℝ) (t : C)
    {ε : ℝ} (hloss : meanCrossEntropy rows z (fun _ => t) ≤ ε) :
    (∑ c, residualOver rows (fun i => probability (z i) c)) / rows.card ≤ 2 * ε := by
  have h := div_le_div_of_nonneg_right (probability_scatter_le_twice_loss rows z t)
    (Nat.cast_nonneg rows.card : (0 : ℝ) ≤ rows.card)
  unfold meanCrossEntropy at hloss
  calc
    _ ≤ (2 * ∑ i ∈ rows, crossEntropy (z i) t) / rows.card := h
    _ = 2 * ((∑ i ∈ rows, crossEntropy (z i) t) / rows.card) := by ring
    _ ≤ 2 * ε := by linarith

example := probability_error_le_twice_loss (fun c : Bool => if c then 2 else -1) true
example := probability_scatter_le_twice_loss (univ : Finset Bool)
  (fun i c : Bool => if c then (if i then 2 else 1) else 0) true
example : (∑ c : Bool, residualOver (univ : Finset Bool)
      (fun _ => probability (fun _ : Bool => (0 : ℝ)) c)) / 2 ≤ 2 * Real.log 2 := by
  simpa using mean_probability_scatter_le (univ : Finset Bool)
    (fun _ _ : Bool => (0 : ℝ)) true (ε := Real.log 2)
    (by norm_num [meanCrossEntropy, crossEntropy])

end Transformer.Grokking.Geometry
