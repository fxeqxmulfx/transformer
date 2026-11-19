module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Replacement.Raw
public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning.Witness

/-!
# Nonzero replacements with both component and readout signs

Source: `review/convex/moments.md`, CM83. The full raw-input cone admits a
unit bias replacement in either component and either readout sign. The
actual complex output changes only for the chosen original object. These
inhabited controls establish no trained accuracy or floating error bound.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Replacement.Witness

open scoped BigOperators
open PatternReduction Projection

noncomputable section

/-- A full real vector with unit bias and all other addresses retained, CM83. -/
def value : Coordinate Bool 1 → ℝ
  | none => 1
  | some _ => 0

/-- Every original prefix reads the same unit bias, CM83. -/
theorem value_read (tokens : Prefix Bool 1) : realInput tokens value = 1 := by
  rw [realInput_apply]
  change 1 + _ = 1
  apply add_eq_left.mpr
  apply Finset.sum_eq_zero
  intro i hi
  cases tokens.val[i.val]? <;> rfl

/-- The replacement satisfies every original all-positive row, CM83. -/
theorem value_feasible : Feasible (fun x : Prefix Bool 1 => signedRow true x) value := by
  intro tokens
  rw [signedRow_dot, value_read]
  norm_num [orient]

/-- Recombination has the declared real or imaginary bias, CM83. -/
theorem component_zero (part : Bool) :
    component part value (0 : ObjectWeights Bool 1) =
      Learning.Witness.bias (if part then 1 else Complex.I) := by
  funext d
  cases d <;> cases part <;>
    simp [component, join, project, value, Learning.Witness.bias, Complex.ext_iff]

/-- Each component and sign replacement belongs to the entire raw cone,
CM83, including prefixes absent from any training batch. -/
theorem bias_mem (sign part : Bool) :
    replace (0 : Parameters Unit Bool (ObjectWeights Bool 1)) () true sign part value ∈
      complexDomain rawInput Set.univ (fun _ _ => true) :=
  replace_mem 0 (fun _ _ => true) () true sign part value (zero_mem _ _ _) value_feasible

/-- Both complex output components change with the unit replacement,
CM83. The imaginary case changes the real logit with the opposite sign;
the unselected original object's output remains zero. -/
theorem bias_forward (sign part : Bool) (tokens : Prefix Bool 1) (c : Bool) :
    complexForward
      (replace (0 : Parameters Unit Bool (ObjectWeights Bool 1)) () true sign part value)
      (rawInput tokens) c = if c then
        (if sign then (1 : ℂ) else -1) *
          (if part then 1 + Complex.I else -1 + Complex.I) else 0 := by
  rw [replace_forward_change]
  cases c <;> cases sign <;> cases part <;>
    norm_num [complexForward, component_zero, complexNeuron_eq,
      Learning.Witness.bias_read, splitHinge, hinge, Complex.ext_iff]

example := replace_mem (0 : Parameters Unit Bool (ObjectWeights Bool 1))
  (fun _ _ => true) () true false false value (zero_mem _ _ _) value_feasible

end

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Replacement.Witness
