module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ParameterSharing.Basic

/-!
# Nonzero shared coefficients with exact expansion costs

Source: `review/convex/moments.md`, CM55. Both parameters occur in both head
coordinates and both raw classes. The nonzero offset and a nonzero parameter
point attain the charged budget. Repeated records and a one-record tail are
concrete instances of the normalized CE result.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ParameterSharing

open scoped BigOperators

private noncomputable def costs (p : Fin 2) : ℝ := p.val + 1
private noncomputable def expansion (p k : Fin 2) (c : Bool) : ℝ :=
  (if c then 1 else -1) * (p.val + 1) * (k.val + 1) / 3
private noncomputable def offset (k : Fin 2) (c : Bool) : ℝ :=
  (if c then 1 else -1) * (k.val + 1) / 6
private noncomputable def point : Fin 2 → ℝ := ![1, 1 / 2]
private noncomputable def features (k x : Fin 2) : ℝ := (k.val + 1) * (x.val + 1) / 4
private def inputs (s : Fin 3) : Fin 2 := if s = 2 then 1 else 0
private def labels (s : Fin 3) : Bool := s == 2

private theorem costs_nonneg (p : Fin 2) : 0 ≤ costs p := by
  unfold costs
  positivity

private theorem expansion_bound (p : Fin 2) (c : Bool) :
    ∑ k, |expansion p k c| ≤ costs p := by
  fin_cases p <;> cases c <;> norm_num [expansion, costs, Fin.sum_univ_two]

private theorem offset_budget : offset ∈ headBudget (1 / 2) := by
  intro c
  cases c <;> norm_num [offset, Fin.sum_univ_two]

private theorem point_mem : point ∈ weightedDomain costs 2 := by
  norm_num [weightedDomain, point, costs, Fin.sum_univ_two]

example := weightedDomain_convex costs costs_nonneg 2

example : parameterMap expansion offset point ∈ headBudget (1 / 2 + 2) :=
  parameterMap_budget expansion offset costs expansion_bound offset_budget point_mem

example (c : Bool) : ∑ k, |parameterMap expansion offset point k c| = 5 / 2 := by
  cases c <;> norm_num [parameterMap, coefficientMap, expansion, offset, point,
    Fin.sum_univ_two]

example (x : Fin 2) (c : Bool) :
    |headLogits features (parameterMap expansion offset point) x c| ≤ (1 / 2 + 2) * 1 := by
  apply logits_bound expansion offset costs features expansion_bound offset_budget point_mem
    (by norm_num : (0 : ℝ) ≤ 1) x
  intro k
  fin_cases k <;> fin_cases x <;> norm_num [features]

example := loss_convex expansion offset costs costs_nonneg features inputs labels
  2 (1 / 2) expansion_bound offset_budget

example := batchLoss_convex expansion offset costs costs_nonneg features inputs labels
  (Finset.univ : Finset (Fin 3)) 2 (1 / 2) expansion_bound offset_budget

example := batchLoss_convex expansion offset costs costs_nonneg features inputs labels
  ({2} : Finset (Fin 3)) 2 (1 / 2) expansion_bound offset_budget

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ParameterSharing
