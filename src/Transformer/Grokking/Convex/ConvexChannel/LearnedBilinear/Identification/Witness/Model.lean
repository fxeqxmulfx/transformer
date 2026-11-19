module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Witness.Geometry
public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Quantitative

/-!
# Feasibility and a forced unseen original-object answer

Source: `review/convex/moments.md`, CM87. Explicit parameters prove that the
observations are feasible, with two distinct observed labels. Identification
then holds for every fitting parameter, not just this chosen example. The
free middle-position coefficient is retained for the ambiguity control.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Witness

open scoped BigOperators Matrix
open Restriction (cube)

/-- A family of feasible full joint parameters, CM87. The free real
coefficient changes an unobserved middle-object direction. -/
def point (r : ℝ) : SequenceParameters Bool 3 := fun out q d =>
  if out then
    match q, d with
    | none, none => -1
    | none, some (.inl (i, o)) =>
      if o then (if i = 1 then (r : ℂ) else 2) else 0
    | _, _ => 0
  else 0

/-- Actual learned logits of this feasible family, CM87. No coordinate
or class is removed from the underlying model. -/
theorem point_logits (r : ℝ) (a b c : Bool) :
    sequenceLogits (point r) (cube a b c) false = 0 ∧
    sequenceLogits (point r) (cube a b c) true =
      -1 + (if a then 2 else 0) + (if b then r else 0) + (if c then 2 else 0) := by
  simp only [sequenceLogits_eq, logits_eq]
  cases a <;> cases b <;> cases c <;>
    norm_num [point, queryRow, contextRow, past, cube, Matrix.mulVec, dotProduct,
      Fintype.sum_option, Fintype.sum_sum_type, Fintype.sum_prod_type, Fin.sum_univ_three] <;>
    ring

/-- Every value of the free coefficient fits all recorded margins,
CM87. This witnesses feasibility without using the duality theorem. -/
theorem point_fits (r : ℝ) : FitsObserved tokens labels (point r) := by
  intro i c hc
  fin_cases i <;> cases c <;>
    simp_all [labels, margin, tokens, (point_logits r _ _ _).1, (point_logits r _ _ _).2] <;>
    norm_num

/-- The observed data admit an actual parameter point, CM87. -/
theorem feasible : ∃ p : SequenceParameters Bool 3, FitsObserved tokens labels p :=
  ⟨point 0, point_fits 0⟩

/-- Every fitting learned model gives the unseen input a margin at least
three, CM87. Its coefficient certificate used only the observed labels. -/
theorem forced_query_margin (p : SequenceParameters Bool 3) (hp : FitsObserved tokens labels p) :
    3 ≤ margin p query true false :=
  certificate_sound tokens labels query true false 3 weights checked_certificate p hp

/-- The forced unseen result is the original object itself under every
tie-selection witness, CM87. The certified margin excludes a tie. -/
theorem forced_query_answer (p : SequenceParameters Bool 3)
    (hp : FitsObserved tokens labels p) (witness : Bool) :
    nextObject p query witness = true := by
  apply nextObject_of_unique
  intro c hc
  cases c
  · have h := forced_query_margin p hp
    unfold margin at h
    linarith
  · exact False.elim (hc rfl)

/-- A certificate transfers to an unknown task only under explicit
structural realizability, CM87. Its parameters need not be known to the checker. -/
theorem realizable_target (target : LearnedHinge.Prefix Bool (3 + 1) → Bool)
    (hreal : ∃ p : SequenceParameters Bool 3, FitsObserved tokens labels p ∧
      ∀ x, target x = nextObject p x false) : target query = true := by
  obtain ⟨p, hp, ht⟩ := hreal
  rw [ht, forced_query_answer p hp false]

example : 3 ≤ margin (point 0) query true false :=
  forced_query_margin (point 0) (point_fits 0)

example : nextObject (point 0) query false = true :=
  forced_query_answer (point 0) (point_fits 0) false

example : (fun x => nextObject (point 0) x false) query = true :=
  realizable_target (fun x => nextObject (point 0) x false)
    ⟨point 0, point_fits 0, fun _ => rfl⟩

example : (∀ p : SequenceParameters Bool 3, FitsObserved tokens labels p →
    3 ≤ margin p query true false) ↔
      ∃ w : Fin 3 → Bool → ℝ, Certificate tokens labels query true false 3 w :=
  forced_margin_iff_certificate tokens labels query true false 3 feasible

example : margin (point 0) query true false =
    ∑ i, ∑ c, weights i c * margin (point 0) (tokens i) (labels i) c :=
  certificate_identity tokens labels query true false weights checked_certificate.2.2.2 (point 0)

example : 3 - (∑ i, ∑ c, weights i c * (0 : ℝ)) ≤ margin (point 0) query true false :=
  certificate_robust tokens labels query true false 3 weights checked_certificate (point 0)
    (fun _ _ => 0) (by simpa only [sub_zero, FitsObserved] using point_fits 0)

example : 3 - (0 : ℝ) * (∑ i, ∑ c, weights i c) ≤ margin (point 0) query true false :=
  certificate_uniform_error tokens labels query true false 3 weights checked_certificate (point 0)
    0 (by simpa only [sub_zero, FitsObserved] using point_fits 0)

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Witness
