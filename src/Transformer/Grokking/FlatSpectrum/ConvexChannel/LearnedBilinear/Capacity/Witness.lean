module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Capacity.Sequence
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates.Unobserved

/-!
# A bounded complete model identifies a previously ambiguous missing pair

Source: `review/flat_spectrum.md`, CM91. Observe 00→false, 01→false and
10→true; withhold 11. These are immutable Boolean object names, not numbers
passed to arithmetic. All complete real/imaginary coordinates remain learned.
The budget 13/48 is feasible, and a finite residual certificate forces margin
at least 1/8 on the missing pair. Without this budget CM90 proves ambiguity.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Capacity.Witness

open scoped BigOperators
open Identification
open Updates (pairInput)

/-- Three distinct observed object pairs, CM91. -/
def tokens : Fin 3 → LearnedHinge.Prefix Bool (1 + 1) :=
  ![pairInput false false, pairInput false true, pairInput true false]

/-- Both original objects occur as observed labels, CM91. -/
def labels : Fin 3 → Bool := ![false, false, true]

/-- The fourth object pair is genuinely unseen, CM91. -/
def query : LearnedHinge.Prefix Bool (1 + 1) := pairInput true true

/-- A finite nonnegative witness, obtained without a query label, CM91.
Only the candidate comparison true-versus-false is being certified. -/
noncomputable def weights (i : Fin 3) (c : Bool) : ℝ :=
  if c = labels i then 0 else (![20 / 3, 8 / 3, 8] : Fin 3 → ℝ) i

/-- Feasible complete real parameters built from the three observed margin
forms, CM91. Both classes and both complex components remain present. -/
noncomputable def vector (z : RealCoordinate Bool 1) : ℝ :=
  5 / 48 * coefficient (tokens 0) false true z +
  1 / 24 * coefficient (tokens 1) false true z +
  1 / 8 * coefficient (tokens 2) true false z

/-- Actual complex parameters, to be executed through the learned encoders. -/
noncomputable def point : SequenceParameters Bool 1 := decode vector

/-- Every observed margin is feasible in the original next-object model,
CM91. This is independent of certificate soundness. -/
theorem point_fits : FitsObserved tokens labels point := by
  intro i c hc
  fin_cases i <;> cases c
  all_goals try exact False.elim (hc rfl)
  all_goals
    rw [show point = decode vector from rfl, margin_decode]
    norm_num [vector, coefficient, tokens, labels, pairInput, queryRow, contextRow,
      past, Fintype.sum_prod_type, Fintype.sum_option, Fintype.sum_sum_type,
      Fin.sum_univ_succ]

/-- The full squared budget is attained, CM91. It counts all 36 real
coordinates, including both components of both classes. -/
theorem point_energy : energy (encode point) = 13 / 48 := by
  rw [show point = decode vector from rfl, encode_decode]
  norm_num [energy, vector, coefficient, tokens, pairInput, queryRow, contextRow,
    past, Fintype.sum_prod_type, Fintype.sum_option, Fintype.sum_sum_type,
    Fin.sum_univ_succ]

/-- The witness itself has positive actual unseen margin, CM91. The next
theorem certifies every fitting model within capacity, not only this point. -/
theorem point_query : margin point query true false = 1 / 4 := by
  rw [show point = decode vector from rfl, margin_decode]
  norm_num [vector, coefficient, tokens, query, pairInput, queryRow, contextRow,
    past, Fintype.sum_prod_type, Fintype.sum_option, Fintype.sum_sum_type,
    Fin.sum_univ_succ]

/-- The full nonzero residual and squared slack satisfy a rational check,
CM91. The squared slack exceeds the capacity charge by 1/64. -/
theorem checked_certificate :
    BoundedCertificate tokens labels query true false (13 / 48) (1 / 8) weights := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i c
    fin_cases i <;> cases c <;> norm_num [weights, labels]
  · intro i
    simp [weights]
  · norm_num [weights, labels, Fin.sum_univ_succ]
  · norm_num [energy, residual, observedForms, coefficient, tokens, query, weights,
      labels, pairInput, queryRow, contextRow, past, Fintype.sum_prod_type,
      Fintype.sum_option, Fintype.sum_sum_type, Fin.sum_univ_succ]

/-- The capacity certificate has a genuinely nonzero residual, CM91. -/
theorem residual_energy : energy (residual (observedForms tokens labels)
    (coefficient query true false) (fun j => weights j.1 j.2)) = 3280 / 3 := by
  norm_num [energy, residual, observedForms, coefficient, tokens, query, weights,
    labels, pairInput, queryRow, contextRow, past, Fintype.sum_prod_type,
    Fintype.sum_option, Fintype.sum_sum_type, Fin.sum_univ_succ]

/-- Nonempty observations with nonconstant original labels, CM91. -/
theorem nonconstant : labels 0 ≠ labels 2 := by decide

/-- No observed prefix equals the query, CM91. -/
theorem held_out (i : Fin 3) : tokens i ≠ query := by fin_cases i <;> decide

/-- The bounded family is inhabited by a fitting actual model, CM91. -/
theorem feasible : ∃ p : SequenceParameters Bool 1,
    FitsObserved tokens labels p ∧ energy (encode p) ≤ 13 / 48 :=
  ⟨point, point_fits, point_energy.le⟩

/-- Capacity plus the observations forces a strict unseen margin, CM91. -/
theorem forced_margin (p : SequenceParameters Bool 1) (hp : FitsObserved tokens labels p)
    (hcap : energy (encode p) ≤ 13 / 48) : 1 / 8 ≤ margin p query true false :=
  bounded_certificate_sound tokens labels query true false (13 / 48) (1 / 8)
    weights checked_certificate p hp hcap

/-- The result is one original object, with any deterministic tie choice,
CM91. The positive margin rules out a tie here. -/
theorem forced_answer (p : SequenceParameters Bool 1) (hp : FitsObserved tokens labels p)
    (hcap : energy (encode p) ≤ 13 / 48) (witness : Bool) :
    nextObject p query witness = true := by
  apply nextObject_of_unique
  intro c hc
  cases c
  · have h := forced_margin p hp hcap
    unfold margin at h
    linarith
  · exact False.elim (hc rfl)

/-- The same missing pair has no unrestricted certificate, CM91/CM90.
This proves that the bounded certificate adds identification power. -/
theorem unrestricted_refusal (beta : ℝ) :
    ¬ ∃ w, Certificate tokens labels query true false beta w := by
  let inputs : Fin 3 → Bool × Bool := ![(false, false), (false, true), (true, false)]
  have ht : tokens = fun i => pairInput (inputs i).1 (inputs i).2 := by
    funext i
    fin_cases i <;> rfl
  have hm : ∀ i, inputs i ≠ (true, true) := by intro i; fin_cases i <;> decide
  have hf : ∃ p, FitsObserved (fun i => pairInput (inputs i).1 (inputs i).2) labels p := by
    rw [← ht]
    exact ⟨point, point_fits⟩
  rw [ht]
  exact Updates.missing_pair_no_certificate inputs labels true true true false
    (by decide) hm hf beta

example : 1 / 8 ≤ margin point query true false :=
  forced_margin point point_fits point_energy.le

example : nextObject point query false = true :=
  forced_answer point point_fits point_energy.le false

example (B : ℝ) : BoundedCertificate Identification.Witness.tokens Identification.Witness.labels
    Identification.Witness.query true false B 3 Identification.Witness.weights :=
  of_exact_certificate _ _ _ _ _ _ _ _ Identification.Witness.checked_certificate

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Capacity.Witness
