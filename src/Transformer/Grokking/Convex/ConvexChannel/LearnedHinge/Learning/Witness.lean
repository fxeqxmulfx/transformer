module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning.Batch

/-!
# Nonzero feasible-query and CE-slope witnesses

Source: `review/convex/moments.md`, CM82. A learned complex bias has nonzero
real and imaginary parts. A two-sided feasible perturbation changes one
original object's relative logit, with ordinary CE derivative minus one half.
This is an inhabited proof control, not a trained or task-specific predictor.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning.Witness

open scoped BigOperators

noncomputable section

/-- A literal learned bias, retaining every position/object coordinate, CM82. -/
def bias (z : ℂ) : ObjectWeights Bool 1
  | none => z
  | some _ => 0

/-- One complete complex neuron for the true object and positive sign, CM82. -/
def point (z : ℂ) : Parameters Unit Bool (ObjectWeights Bool 1) :=
  fun _ c s => if c && s then bias z else 0

/-- Every admitted prefix reads this actual learned bias, CM82. -/
theorem bias_read (tokens : Prefix Bool 1) (z : ℂ) : rawInput tokens (bias z) = z := by
  rw [rawInput_apply]
  change z + _ = z
  apply add_eq_left.mpr
  apply Finset.sum_eq_zero
  intro i hi
  cases tokens.val[i.val]? <;> rfl

/-- The actual input retains the selected class and readout sign, CM82. -/
theorem point_read (tokens : Prefix Bool 1) (z : ℂ) (j : Unit) (c s : Bool) :
    rawInput tokens (point z j c s) = if c && s then z else 0 := by
  cases c <;> cases s <;> simp [point, bias_read]

/-- A nonzero real and imaginary point admits these two-sided perturbations
at every original prefix, CM82, rather than at sampled training inputs alone. -/
theorem point_queries {amount : ℝ} (ha : |amount| ≤ 1) :
    point (3 + 3 * Complex.I) + amount • point 1 ∈
      complexDomain rawInput Set.univ (fun _ _ => true) := by
  have hb := abs_le.mp ha
  intro j c s x hx
  rcases x with ⟨tokens, k⟩
  change 0 ≤ orient true (components rawInput (tokens, k)
    (point (3 + 3 * Complex.I) j c s + amount • point 1 j c s))
  cases k <;> cases c <;> cases s <;>
    norm_num [orient, components, map_add, map_smul, point_read]
  linarith

/-- The unperturbed nonzero complex point satisfies the entire cone, CM82. -/
theorem point_mem : point (3 + 3 * Complex.I) ∈
    complexDomain rawInput Set.univ (fun _ _ => true) := by
  simpa using point_queries (amount := 0) (by norm_num)

/-- Real and imaginary contributions cancel at this point, giving equal
actual logits over both original output objects, CM82. -/
theorem point_logits (tokens : Prefix Bool 1) (c : Bool) :
    sequenceLogits (point (3 + 3 * Complex.I)) tokens c = 0 := by
  cases c <;> norm_num [sequenceLogits, complexLogits, complexForward, complexNeuron_eq,
    point_read, splitHinge, hinge]

/-- The feasible direction changes only the true object's relative logit,
CM82; it is not merely a common shift of all logits. -/
theorem point_direction (tokens : Prefix Bool 1) (c : Bool) :
    logitDirection (rawInput tokens) (fun _ _ => true) (point 1) c = if c then 1 else 0 := by
  cases c <;> norm_num [logitDirection, gatedForward, gated, point_read]

/-- The full actual ordinary CE slope is nonzero and depends on the original
target object, CM82. Both output classes enter its softmax normalization. -/
theorem point_slope (tokens : Prefix Bool 1) (target : Bool) :
    rawWork (rawGradient tokens (fun _ _ => true) target (point (3 + 3 * Complex.I))) (point 1) =
      if target then -(1 / 2) else 1 / 2 := by
  rw [rawGradient_work]
  change ceSlope (sequenceLogits (point (3 + 3 * Complex.I)) tokens)
    (logitDirection (rawInput tokens) (fun _ _ => true) (point 1)) target = _
  cases target <;> norm_num [ceSlope, Geometry.probability, point_logits, point_direction,
    Fintype.sum_bool]

example := point_queries (amount := 1) (by norm_num)

/-- One visible immutable object, CM82. -/
def sample (object : Bool) : Prefix Bool 1 := ⟨[object], by simp⟩

/-- Three indexed records, with the first and third equal, CM82. -/
def records (i : Fin 3) : Prefix Bool 1 := sample (i.val != 1)

/-- Original next-object targets for the three indexed records, CM82. -/
def targets (i : Fin 3) : Bool := i.val != 1

/-- The repeated true-target record contributes twice to the actual mean
batch slope: deduplication would incorrectly give zero instead, CM82. -/
theorem repeated_batch_slope :
    rawWork (meanBatchGradient (fun _ _ => true) records targets
      (point (3 + 3 * Complex.I))) (point 1) = -(1 / 6) := by
  rw [meanBatchGradient_work]
  simp only [point_slope]
  norm_num [targets, Fin.sum_univ_succ]

end

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning.Witness
