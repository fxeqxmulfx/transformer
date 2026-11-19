module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Witness.Refusal

/-!
# Unobserved coordinates can change unseen answers without changing training

Source: `review/convex/moments.md`, CM90. A raw coordinate silent on every
training prefix can freely change a responding query margin. In particular,
each unseen ordered two-object input has such a coordinate in the complete
CM86 family. This is nonidentification from labels, not a claim that no
algorithm or additional structural bias can select a correct completion.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates

open LearnedHinge (Prefix)
open Identification

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- Changing a silent coordinate preserves every actual training logit,
CM90. This covers arbitrary fitted and unfitted parameter points. -/
theorem silent_logits (p : SequenceParameters O T)
    (tokens : Fin N → Prefix O (T + 1)) (z : RealCoordinate O T)
    (hz : ∀ i, response (tokens i) z = 0) (t : ℝ) (i : Fin N) :
    sequenceLogits (decode (shift (encode p) z t)) (tokens i) = sequenceLogits p (tokens i) := by
  funext c
  rw [logit_shift, decode_encode, hz i]
  simp

/-- A silent coordinate preserves all observed unit margins, CM90.
The full original classes and record indices are unchanged. -/
theorem silent_fits (p : SequenceParameters O T)
    (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (z : RealCoordinate O T) (hz : ∀ i, response (tokens i) z = 0)
    (hp : FitsObserved tokens labels p) (t : ℝ) :
    FitsObserved tokens labels (decode (shift (encode p) z t)) := by
  intro i c hc
  simpa only [margin, silent_logits p tokens z hz t i] using hp i c hc

/-- A responding query margin can take any real value while all observed
margins still fit, CM90. Feasibility and a nonzero actual response are explicit. -/
theorem silent_margin_arbitrary (p : SequenceParameters O T)
    (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (query : Prefix O (T + 1)) (z : RealCoordinate O T) (a b : O)
    (hz : ∀ i, response (tokens i) z = 0) (hp : FitsObserved tokens labels p)
    (hc : z.2.1 = a) (hab : a ≠ b) (hs : response query z ≠ 0) (value : ℝ) :
    ∃ q : SequenceParameters O T, FitsObserved tokens labels q ∧ margin q query a b = value := by
  let t := (value - margin p query a b) / response query z
  refine ⟨decode (shift (encode p) z t), silent_fits p tokens labels z hz hp t, ?_⟩
  simp only [margin, logit_shift, decode_encode, hc, ite_eq_right (Ne.symm hab), add_zero]
  dsimp [t]
  rw [div_mul_cancel₀ _ hs]
  unfold margin
  ring

/-- No finite lower-margin certificate survives a silent responding
coordinate, CM90. This is stronger than failure of one positive threshold. -/
theorem silent_no_certificate (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (query : Prefix O (T + 1)) (z : RealCoordinate O T) (a b : O)
    (hz : ∀ i, response (tokens i) z = 0)
    (hf : ∃ p : SequenceParameters O T, FitsObserved tokens labels p)
    (hc : z.2.1 = a) (hab : a ≠ b) (hs : response query z ≠ 0) (bound : ℝ) :
    ¬ ∃ w, Certificate tokens labels query a b bound w := by
  rintro ⟨w, hw⟩
  obtain ⟨p, hp⟩ := hf
  obtain ⟨q, hq, hm⟩ := silent_margin_arbitrary p tokens labels query z a b hz hp hc hab hs (bound - 1)
  have h := certificate_sound tokens labels query a b bound w hw q hq
  rw [hm] at h
  linarith

/-- An ordered sequence of two original immutable objects, CM90. -/
def pairInput (a b : O) : Prefix O (1 + 1) := ⟨[a, b], by simp⟩

/-- A complete raw joint coordinate reading the two input addresses,
CM90. No arithmetic operation or target answer is built into this address. -/
def pairCoordinate (c a b : O) : RealCoordinate O 1 :=
  (false, c, some b, some (.inl (0, a)))

omit [Fintype O] in
/-- The actual response of this coordinate distinguishes the complete
ordered two-object input, CM90. -/
theorem pair_response (a b c u v : O) :
    response (pairInput a b) (pairCoordinate c u v) = if a = u ∧ b = v then 1 else 0 := by
  by_cases ha : a = u <;> by_cases hb : b = v <;>
    simp [response, pairInput, pairCoordinate, queryRow, contextRow, past, ha, hb]

/-- Every missing ordered two-object input has a silent responding
coordinate, so its answer is not forced by the training labels in this full
family, CM90. This statement concerns any object identities, not arithmetic. -/
theorem missing_pair_no_certificate (inputs : Fin N → O × O) (labels : Fin N → O)
    (u v a b : O) (hab : a ≠ b) (hmissing : ∀ i, inputs i ≠ (u, v))
    (hf : ∃ p : SequenceParameters O 1,
      FitsObserved (fun i => pairInput (inputs i).1 (inputs i).2) labels p) (bound : ℝ) :
    ¬ ∃ w, Certificate (fun i => pairInput (inputs i).1 (inputs i).2) labels
      (pairInput u v) a b bound w := by
  apply silent_no_certificate _ _ _ (pairCoordinate a u v) _ _ _ hf rfl hab _ bound
  · intro i
    rw [pair_response]
    have hn : ¬ ((inputs i).1 = u ∧ (inputs i).2 = v) := by
      rintro ⟨ha, hb⟩
      exact hmissing i (Prod.ext ha hb)
    simp [hn]
  · simp [pair_response]

private def freeCoordinate : RealCoordinate Bool 3 :=
  (false, true, none, some (.inl (1, true)))

private theorem free_silent (i : Fin 3) : response (Witness.tokens i) freeCoordinate = 0 := by
  fin_cases i <;> simp [response, Witness.tokens, freeCoordinate, Restriction.cube,
    queryRow, contextRow, past]

example : sequenceLogits (decode (shift (encode (Witness.point 0)) freeCoordinate 2))
    (Witness.tokens 0) = sequenceLogits (Witness.point 0) (Witness.tokens 0) :=
  silent_logits _ _ _ free_silent 2 0

example : FitsObserved Witness.tokens Witness.labels
    (decode (shift (encode (Witness.point 0)) freeCoordinate 2)) :=
  silent_fits _ _ _ _ free_silent (Witness.point_fits 0) 2

example (value : ℝ) : ∃ p : SequenceParameters Bool 3,
    FitsObserved Witness.tokens Witness.labels p ∧ margin p Witness.ambiguous true false = value := by
  apply silent_margin_arbitrary (Witness.point 0) _ _ _ freeCoordinate _ _ free_silent
    (Witness.point_fits 0) rfl (by decide) _ value
  simp [response, freeCoordinate, Witness.ambiguous, Restriction.cube, queryRow, contextRow, past]

example (bound : ℝ) : ¬ ∃ w, Certificate Witness.tokens Witness.labels
    Witness.ambiguous true false bound w := by
  apply silent_no_certificate _ _ _ freeCoordinate _ _ free_silent Witness.feasible rfl (by decide) _ bound
  simp [response, freeCoordinate, Witness.ambiguous, Restriction.cube, queryRow, contextRow, past]

private def pairPoint : SequenceParameters Bool 1 := fun c q d =>
  if c = false ∧ q = none ∧ d = none then 1 else 0

private theorem pair_feasible : ∃ p : SequenceParameters Bool 1,
    FitsObserved (fun _ : Fin 1 => pairInput false false) (fun _ => false) p := by
  refine ⟨pairPoint, ?_⟩
  intro i c hc
  cases c
  · exact False.elim (hc rfl)
  · norm_num [margin, sequenceLogits_eq, logits_eq, pairPoint, queryRow,
      contextRow, past, pairInput, Matrix.mulVec, dotProduct, Fintype.sum_option,
      Fintype.sum_sum_type, Fintype.sum_prod_type, Fin.sum_univ_succ]

example (bound : ℝ) : ¬ ∃ w, Certificate (fun _ : Fin 1 => pairInput false false)
    (fun _ => false) (pairInput false true) true false bound w := by
  exact missing_pair_no_certificate (fun _ : Fin 1 => (false, false)) (fun _ => false)
    false true true false (by decide) (by intro i; decide) pair_feasible bound

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates
