module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.SharedLearning.Witness
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Capacity

/-!
# Both labels and value coverage still permit ambiguous shared answers

Source: `review/flat_spectrum.md`, CM73. Two constant dictionaries contain both
labels and cover both values at every slot. Every selected slot with negation
is an attained ordinary-CE optimizer in the same label-independent domain.
Two such optimizers give opposite strict answers on an unobserved dictionary.
Only the collision condition is missing; flatness or more optimization of the
same loss cannot replace the missing identifying observations.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.SharedLearning

open scoped BigOperators

/-- Two ordinary observations, with distinct record indices and both labels. -/
def ambiguousData : Bool → Fin 3 → Bool :=
  fun b => ![b, b, b]

/-- A dictionary on which two empirical optimizers disagree. -/
def disagreementData : Fin 3 → Bool := ![true, false, false]

/-- Every selected slot realizes these same observed answer labels, CM73. -/
theorem ambiguous_realizable (k : Fin 3) :
    ∀ s, Bool.not s = Bool.not (ambiguousData s k) := by
  intro s
  fin_cases k <;> rfl

/-- Every slot has complete value coverage, CM73, so lack of coverage alone
does not explain the ambiguity. -/
theorem ambiguous_coverage (k : Fin 3) : Covers ambiguousData k := by
  intro v
  refine ⟨v, ?_⟩
  fin_cases k <;> rfl

/-- The observed labels do not separate the first slot from the second, CM73. -/
theorem ambiguous_not_separated :
    ¬ SeparatesSlots ambiguousData Bool.not (0 : Fin 3) := by
  unfold SeparatesSlots
  decide

/-- Each selected decoder is an attained optimum of precisely the same CE
objective over the complete shared domain, CM73. -/
theorem ambiguous_optimum (k : Fin 3) :
    IsMinOn (answerLoss (fun _ : Bool => ()) ambiguousData Bool.not)
      {A | Feasible A} (selected k Bool.not) := by
  exact selected_isMinOn ambiguousData Bool.not k Bool.not (ambiguous_realizable k)

/-- The conflicting dictionary is absent from both observations, CM73. -/
theorem disagreement_unobserved : ∀ s, ambiguousData s ≠ disagreementData := by
  decide

/-- Feasibility, both actual global optima and opposite strict unseen answers
hold together, CM73. The negative control uses the same joint model and loss
as the identifying example, with no failed optimizer hidden in the claim. -/
theorem ambiguous_optima_disagree :
    Feasible (selected (0 : Fin 3) Bool.not) ∧
    Feasible (selected (1 : Fin 3) Bool.not) ∧
    IsMinOn (answerLoss (fun _ : Bool => ()) ambiguousData Bool.not)
      {A | Feasible A} (selected (0 : Fin 3) Bool.not) ∧
    IsMinOn (answerLoss (fun _ : Bool => ()) ambiguousData Bool.not)
      {A | Feasible A} (selected (1 : Fin 3) Bool.not) ∧
    logits (selected (0 : Fin 3) Bool.not) () disagreementData true <
      logits (selected (0 : Fin 3) Bool.not) () disagreementData false ∧
    logits (selected (1 : Fin 3) Bool.not) () disagreementData false <
      logits (selected (1 : Fin 3) Bool.not) () disagreementData true := by
  refine ⟨feasible_selected 0 Bool.not, feasible_selected 1 Bool.not,
    ambiguous_optimum 0, ambiguous_optimum 1, ?_, ?_⟩ <;>
    norm_num [logits_selected, disagreementData]

/-- Enumerate every three-slot categorical dictionary for the operator audit,
CM73. This is an input enumeration, not an arithmetic target feature. -/
def allData (x : Bool × Bool × Bool) : Fin 3 → Bool := ![x.1, x.2.1, x.2.2]

/-- The first optimal completion's actual query-by-input operator is flat,
CM73. Its two rows select different slots through the declared alignment. -/
theorem first_ambiguous_output_flat :
    IsFlat (outputKernel boolSign
      (tiedChannel witnessAlign (selected (0 : Fin 3) Bool.not)) allData) 8 := by
  right
  ext q r
  cases q <;> cases r <;>
    norm_num [Matrix.mul_apply, Matrix.conjTranspose_apply, outputKernel,
      logits_tiedChannel, logits_selected, allData, alignedData, witnessAlign,
      boolSign, Fintype.sum_prod_type, Fintype.sum_bool, Matrix.one_apply]

/-- The competing optimum is flat with the same squared singular value,
CM73, while its unobserved strict answer is opposite. Flat capacity does not
select between these two already feasible, globally optimal shared tensors. -/
theorem second_ambiguous_output_flat :
    IsFlat (outputKernel boolSign
      (tiedChannel witnessAlign (selected (1 : Fin 3) Bool.not)) allData) 8 := by
  right
  ext q r
  cases q <;> cases r <;>
    norm_num [Matrix.mul_apply, Matrix.conjTranspose_apply, outputKernel,
      logits_tiedChannel, logits_selected, allData, alignedData, witnessAlign,
      boolSign, Fintype.sum_prod_type, Fintype.sum_bool, Matrix.one_apply]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.SharedLearning
