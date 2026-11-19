module

public import Transformer.Grokking.Convex.ConvexDepth.MarginGuard.Stability

/-!
# A nonzero Fourier witness for margin constraints

Source: `review/convex/moments.md`, CM47. Use CM46's actual complete Fourier
head, two fixed gates and two context maps. Halving local preferences changes
unseen logits from (15,9) to (13.5,10.5), retaining half the raw margin. A wrong
reference prediction remains wrong. Zero retention can instead create a tie
and change the first maximizing class.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.MarginGuard

open scoped BigOperators
open Landscape.Neural (crossEntropy)

/-- Actual complete complex-head scores with CM46's nonzero source, CM47. -/
noncomputable def scores (a : Bool → Bool → ℝ) : Fin 4 → Fin 4 → ℝ :=
  headLogits (GatedContext.features GatedContext.witnessGate ContextMixture.witnessBank
    GatedContext.witnessBase (1 / 2) a) ContextMixture.witnessHead

/-- Halfway between the original local rows and the uniform anchor, CM47. -/
noncomputable def softened (g j : Bool) : ℝ :=
  (ContextMixture.witnessMix g j + GatedContext.witnessBase j) / 2

theorem softened_rows : softened ∈ GatedContext.rows := by
  intro g
  constructor
  · intro j; cases g <;> cases j <;>
      norm_num [softened, GatedContext.witnessBase, ContextMixture.witnessMix]
  · cases g <;>
      norm_num [softened, GatedContext.witnessBase, ContextMixture.witnessMix, Fintype.sum_bool]

/-- Compute every reference score through the actual CM46 Fourier products,
CM47, including both original observed rows. -/
theorem reference_scores (x c : Fin 4) : scores ContextMixture.witnessMix x c =
    if x = 0 then (if c = 0 then 24 else 0)
    else if x = 1 then (if c = 2 then 24 else 0)
    else if c = ContextMixture.witnessLabels x then 15
    else if c = 0 ∨ c = 2 then 9 else 0 := by
  unfold scores
  rw [GatedContext.logits]
  simp only [ContextMixture.logits, Fintype.sum_bool, Pi.add_apply, Pi.smul_apply,
    smul_eq_mul, ContextMixture.witness_bank_logits]
  fin_cases x <;> fin_cases c <;>
    norm_num [GatedContext.witnessBase, GatedContext.witnessGate,
      ContextInterpolation.pointWeights, GatedContext.witnessRoute,
      ContextMixture.witnessMix, ContextMixture.witnessState, ContextMixture.witnessLabels]

/-- The candidate changes confidence but retains the observed logits,
CM47. All unused raw classes still receive their actual zero logits. -/
theorem softened_scores (x c : Fin 4) : scores softened x c =
    if x = 0 then (if c = 0 then 24 else 0)
    else if x = 1 then (if c = 2 then 24 else 0)
    else if c = ContextMixture.witnessLabels x then 27 / 2
    else if c = 0 ∨ c = 2 then 21 / 2 else 0 := by
  unfold scores
  rw [GatedContext.logits]
  simp only [ContextMixture.logits, Fintype.sum_bool, Pi.add_apply, Pi.smul_apply,
    smul_eq_mul, ContextMixture.witness_bank_logits]
  fin_cases x <;> fin_cases c <;>
    norm_num [softened, GatedContext.witnessBase, GatedContext.witnessGate,
      ContextInterpolation.pointWeights, GatedContext.witnessRoute,
      ContextMixture.witnessMix, ContextMixture.witnessState, ContextMixture.witnessLabels]

/-- The original chosen raw labels are first maximizers on all four inputs. -/
theorem reference_first (x : Fin 4) :
    firstMax (scores ContextMixture.witnessMix x) (ContextMixture.witnessLabels x) := by
  fin_cases x <;> constructor <;> intro c <;> fin_cases c <;>
    norm_num [reference_scores, ContextMixture.witnessLabels]

/-- Every actual raw margin retains at least one half of the reference,
CM47. This includes competitors outside the two active raw labels. -/
theorem softened_mem : scores softened ∈
    region (scores ContextMixture.witnessMix) ContextMixture.witnessLabels (1 / 2) := by
  intro x c
  fin_cases x <;> fin_cases c <;>
    norm_num [gap, softened_scores, reference_scores, ContextMixture.witnessLabels]

/-- The half-margin constraint permits an actual nonzero logit change,
CM47; it does not freeze the whole score vector. -/
theorem confidence_changes :
    scores softened 2 0 = 27 / 2 ∧ scores ContextMixture.witnessMix 2 0 = 15 := by
  norm_num [softened_scores, reference_scores, ContextMixture.witnessLabels]

/-- A held-out target of raw class2 remains wrong at query2 despite the
positive margin constraint, CM47. The constraint preserves reference choices. -/
theorem wrong_label_retained : firstMax (scores softened 2) 0 ∧ (0 : Fin 4) ≠ 2 := by
  constructor
  · have h := firstMax_preserved softened_mem (by norm_num : (0 : ℝ) < 1 / 2)
      2 (reference_first 2)
    simpa [ContextMixture.witnessLabels] using h
  · norm_num

/-- The same wrong target has CE exactly three above the predicted label,
CM47. This is full four-class CE, with an actual reduced raw margin. -/
theorem wrong_target_ce : crossEntropy (scores softened 2) 2 =
    crossEntropy (scores softened 2) 0 + 3 := by
  simp only [crossEntropy]
  norm_num [softened_scores, ContextMixture.witnessLabels]
  ring

/-- Uniform local rows produce a genuine top-score tie at unseen queries,
CM47. The complete head remains the same nonzero Fourier tensor. -/
theorem tied_scores (x c : Fin 4) : scores (fun _ => GatedContext.witnessBase) x c =
    if x = 0 then (if c = 0 then 24 else 0)
    else if x = 1 then (if c = 2 then 24 else 0)
    else if c = 0 ∨ c = 2 then 12 else 0 := by
  unfold scores
  rw [GatedContext.logits]
  simp only [ContextMixture.logits, Fintype.sum_bool, Pi.add_apply, Pi.smul_apply,
    smul_eq_mul, ContextMixture.witness_bank_logits]
  fin_cases x <;> fin_cases c <;>
    norm_num [GatedContext.witnessBase, GatedContext.witnessGate,
      ContextInterpolation.pointWeights, GatedContext.witnessRoute,
      ContextMixture.witnessMix, ContextMixture.witnessState, ContextMixture.witnessLabels]

/-- At retention zero all weak margin constraints hold, CM47. -/
theorem zero_retention_mem : scores (fun _ => GatedContext.witnessBase) ∈
    region (scores ContextMixture.witnessMix) ContextMixture.witnessLabels 0 := by
  intro x c
  fin_cases x <;> fin_cases c <;>
    norm_num [gap, tied_scores, reference_scores, ContextMixture.witnessLabels]

/-- Zero retention can change the deterministic raw decision from2to0,
CM47. This explains the strictly positive fraction in firstMax_preserved. -/
theorem zero_retention_flips : firstMax (scores ContextMixture.witnessMix 3) 2 ∧
    firstMax (scores (fun _ => GatedContext.witnessBase) 3) 0 ∧ (0 : Fin 4) ≠ 2 := by
  refine ⟨?_, ?_, by norm_num⟩
  · simpa [ContextMixture.witnessLabels] using reference_first 3
  · constructor <;> intro c <;> fin_cases c <;> norm_num [tied_scores]

end Transformer.Grokking.Convex.ConvexDepth.MarginGuard
