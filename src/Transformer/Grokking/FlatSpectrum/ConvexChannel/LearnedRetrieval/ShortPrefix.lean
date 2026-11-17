module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval.Witness

/-!
# Retrieval scores on a two-object prefix

Source: `review/flat_spectrum.md`, CM112. A single memory position carries
the entire mass of both learned softmaxes. This constrains the complete
score vector on the original two-operand arithmetic interface.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval

variable {O E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]

/-- CM112. With one past object, the sole path has mass one for every learned
key, relative potential and query. Its value supplies every class score. -/
theorem logit_one_past (p : Parameters O E) (x : Fin 1 → O) (q c : O) :
    logit p x q c = kernel p c (x 0) := by
  have mass := sum_pathMass (n := 1) (by decide) p x q
  simp only [Fintype.sum_prod_type, Fin.sum_univ_one] at mass
  simp only [logit, Fintype.sum_prod_type, Fin.sum_univ_one, mass, one_mul]

/-- CM112. On the actual serialized prefix of two objects, each logit
depends on the first object's value and the class readout. -/
theorem score_two_objects (p : Parameters O E) (a q c : O) :
    score p [a, q] c = kernel p c a := by
  change logit p (fun j : Fin 1 => [a][j]) q c = _
  rw [logit_one_past]
  rfl

/-- CM112. Any deterministic object selection from the complete scores
must miss at least one of two different answers required after the same
first object. The selection may include any fixed scaling and tie rule. -/
theorem incompatible_two_object_answers (p : Parameters O E) (choose : (O → ℝ) → O)
    (a q r y z : O) (hne : y ≠ z) :
    ¬ (choose (score p [a, q]) = y ∧ choose (score p [a, r]) = z) := by
  have equal : score p [a, q] = score p [a, r] := by
    funext c
    rw [score_two_objects, score_two_objects]
  intro ⟨hq, hr⟩
  exact hne (hq.symm.trans ((congrArg choose equal).trans hr))

example :
    ¬ ((fun z : Bool → ℝ => decide (z false < z true)) (score witness [false, false]) = false ∧
      (fun z : Bool → ℝ => decide (z false < z true)) (score witness [false, true]) = true) :=
  incompatible_two_object_answers witness (fun z : Bool → ℝ => decide (z false < z true))
    false false true false true (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval
