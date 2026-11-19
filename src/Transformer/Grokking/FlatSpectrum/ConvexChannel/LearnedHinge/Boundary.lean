module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Witness

/-!
# Omitting an input from the cone does not prove its answer-loss convexity

Source: `review/convex/moments.md`, CM75. Two positive-coordinate points
obey both observed input cones. At a different input their learned ReLU
changes branch, and ordinary answer CE violates the midpoint inequality.
The exact factor-recovery identity still holds there. This distinguishes
global prediction inclusion from convexity outside the declared region.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Boundary

open Witness (query weights)

noncomputable section

/-- Two feasible joint representation endpoints for the scope check. -/
def left : ℂ × ℂ := (1, 0)

/-- The second endpoint uses the other learned input coordinate. -/
def right : ℂ × ℂ := (0, 1)

/-- A different linear input read, absent from the two constrained ones. -/
def unseen : (ℂ × ℂ) →ₗ[ℝ] ℂ := LinearMap.fst ℝ ℂ ℂ - LinearMap.snd ℝ ℂ ℂ

/-- The first point obeys every observed real/imaginary sign, CM75. -/
theorem left_mem : weights left ∈ complexDomain query Set.univ (fun _ _ => true) := by
  intro j c s x hx
  rcases x with ⟨x, k⟩
  cases x <;> cases k <;> cases c <;> cases s <;>
    norm_num [weights, components, query, left, orient]

/-- The second point obeys the same complete observed cone, CM75. -/
theorem right_mem : weights right ∈ complexDomain query Set.univ (fun _ _ => true) := by
  intro j c s x hx
  rcases x with ⟨x, k⟩
  cases x <;> cases k <;> cases c <;> cases s <;>
    norm_num [weights, components, query, right, orient]

/-- The missing input is genuinely distinct from both constrained reads,
CM75. Equality of linear maps would contradict their value at `(1,1)`. -/
theorem unseen_not_query (x : Bool) : unseen ≠ query x := by
  intro h
  have he := LinearMap.congr_fun h ((1, 1) : ℂ × ℂ)
  cases x <;> norm_num [unseen, query] at he

/-- The actual midpoint has the same answer CE as the inactive endpoint,
CM75. Every real and imaginary product is evaluated before this equality. -/
theorem midpoint_loss :
    complexLoss unseen true ((1 / 2 : ℝ) • weights left + (1 / 2 : ℝ) • weights right) =
      complexLoss unseen true (weights right) := by
  norm_num [complexLoss, Landscape.Neural.crossEntropy, Fintype.sum_bool,
    complexLogits, complexForward, complexNeuron_eq, weights, left, right, unseen, splitHinge, hinge]

/-- The active endpoint predicts the same target strictly better, CM75. -/
theorem left_loss_lt_right : complexLoss unseen true (weights left) < complexLoss unseen true (weights right) := by
  unfold complexLoss
  rw [ConvexRecall.crossEntropy_relative, ConvexRecall.crossEntropy_relative]
  norm_num [Fintype.sum_bool, complexLogits, complexForward, complexNeuron_eq,
    weights, left, right, unseen, splitHinge, hinge]
  apply Real.log_lt_log (by positivity)
  have h := Real.exp_lt_exp.mpr (by norm_num : (-1 : ℝ) < 0)
  rw [Real.exp_zero] at h
  linarith

/-- Convexity on all observed activation constraints does not extend to
this missing input's ordinary CE, CM75. Neither the exact lift nor balanced
recovery removes the input-membership premise. -/
theorem outside_loss_not_convex :
    ¬ ConvexOn ℝ (complexDomain query Set.univ (fun _ _ => true) :
      Set (Parameters Unit Bool (ℂ × ℂ))) (complexLoss unseen true) := by
  intro h
  have hj := h.2 left_mem right_mem (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  rw [midpoint_loss] at hj
  simp only [smul_eq_mul] at hj
  linarith [left_loss_lt_right]

end

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Boundary
