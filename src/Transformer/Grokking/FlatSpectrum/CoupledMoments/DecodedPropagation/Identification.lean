module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Basic

/-!
# Sound unseen answers from the union of surviving phase domains

Source: `review/flat_spectrum.md`, CM19. Normalize the unknown target at one
observed row. Its anchor phase is among the enumerated branches and survives
every sweep. Taking the union over branches and decoder choices is essential:
the theorem never assumes that a preferred fitted decoder is the true one.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation

open Finset DecodedRecovery

variable {q : ℕ} [NeZero q] {n S C : Type*} [Fintype n] [DecidableEq n]
  [Fintype S] [DecidableEq C] {d : ZMod q → C} {cell : S → n × n} {labels : S → C}

omit [NeZero q] [DecidableEq n] in
/-- The output of any retained assignment occurs in the computed output set. -/
theorem mem_outputs {D : Domains n q} {r s : n → ZMod q} (hD : Contains D r s)
    (i j : n) : d (r i + s j) ∈ outputs d D i j := by
  rw [outputs, ite_eq_left (live_of_contains hD)]
  exact mem_biUnion.mpr ⟨r i, hD.1 i, mem_image.mpr ⟨s j, hD.2 j, rfl⟩⟩

/-- **No compatible answer is removed** (CM19). All phases are unknown to the
procedure; the only initial information is the observed anchor and its label. -/
theorem mem_answers {r s : n → ZMod q} (hfit : Fits d cell labels r s)
    (anchor : S) (steps : ℕ) (i j : n) :
    d (r i + s j) ∈ answers d cell labels anchor steps i j := by
  let a := (cell anchor).1
  let b := (cell anchor).2
  have hstart := initial_contains r s a b
  have hnorm := normalize_fits hfit a
  have hrun := run_preserves hnorm hstart steps
  have hout := mem_outputs (d := d) hrun i j
  rw [normalize_sum] at hout
  apply mem_biUnion.mpr
  refine ⟨r a + s b, mem_filter.mpr ⟨mem_univ _, ?_⟩, hout⟩
  exact hfit anchor

/-- A singleton output set identifies the decoded answer, regardless of which
compatible latent assignment generated the observations. -/
theorem identifies {r s : n → ZMod q} (hfit : Fits d cell labels r s)
    (anchor : S) (steps : ℕ) {i j : n} {label : C}
    (hanswer : answers d cell labels anchor steps i j = {label}) :
    d (r i + s j) = label := by
  have hmem := mem_answers hfit anchor steps i j
  rw [hanswer] at hmem
  exact mem_singleton.mp hmem

/-- An empty computed answer set proves that no global phase assignment fits.
A nonempty or nonsingleton set need not correspond to a global completion. -/
theorem empty_refutes (anchor : S) (steps : ℕ) {i j : n}
    (hempty : answers d cell labels anchor steps i j = ∅) :
    ¬ ∃ r s, Fits d cell labels r s := by
  rintro ⟨r, s, hfit⟩
  have hmem := mem_answers hfit anchor steps i j
  rw [hempty] at hmem
  exact notMem_empty _ hmem

/-- Two actual completions with different answers force refusal at every
iteration budget. A large computed set alone does not establish ambiguity. -/
theorem not_singleton_of_disagree {r s r' s' : n → ZMod q}
    (hfit : Fits d cell labels r s) (hfit' : Fits d cell labels r' s')
    (anchor : S) (steps : ℕ) {i j : n}
    (hne : d (r i + s j) ≠ d (r' i + s' j)) (label : C) :
    answers d cell labels anchor steps i j ≠ {label} := by
  intro hanswer
  exact hne ((identifies hfit anchor steps hanswer).trans
    (identifies hfit' anchor steps hanswer).symm)

variable {K : Type*} [Fintype K] {decoder : K → ZMod q → C}

/-- The union retains each compatible answer from every member of the bank. -/
theorem mem_bankAnswers (k : K) {r s : n → ZMod q}
    (hfit : Fits (decoder k) cell labels r s) (anchor : S) (steps : ℕ) (i j : n) :
    decoder k (r i + s j) ∈ bankAnswers decoder cell labels anchor steps i j := by
  apply mem_biUnion.mpr
  exact ⟨k, mem_univ _, mem_answers hfit anchor steps i j⟩

/-- **Identification over the whole declared bank** (CM19). The true decoder
index is arbitrary; it is not chosen by a training or held-out score. -/
theorem bank_identifies (k : K) {r s : n → ZMod q}
    (hfit : Fits (decoder k) cell labels r s) (anchor : S) (steps : ℕ) {i j : n} {label : C}
    (hanswer : bankAnswers decoder cell labels anchor steps i j = {label}) :
    decoder k (r i + s j) = label := by
  have hmem := mem_bankAnswers k hfit anchor steps i j
  rw [hanswer] at hmem
  exact mem_singleton.mp hmem

/-- An empty bank answer set refutes the entire declared family, not only the
last candidate tried by a native search. -/
theorem empty_bank_refutes (anchor : S) (steps : ℕ) {i j : n}
    (hempty : bankAnswers decoder cell labels anchor steps i j = ∅) :
    ¬ ∃ k r s, Fits (decoder k) cell labels r s := by
  rintro ⟨k, r, s, hfit⟩
  have hmem := mem_bankAnswers k hfit anchor steps i j
  rw [hempty] at hmem
  exact notMem_empty _ hmem

/-- Disagreement between two fitting decoders also forbids a singleton. Merely
choosing the one with a preferred training score cannot certify the answer. -/
theorem bank_not_singleton_of_disagree (k k' : K) {r s r' s' : n → ZMod q}
    (hfit : Fits (decoder k) cell labels r s)
    (hfit' : Fits (decoder k') cell labels r' s') (anchor : S) (steps : ℕ) {i j : n}
    (hne : decoder k (r i + s j) ≠ decoder k' (r' i + s' j)) (label : C) :
    bankAnswers decoder cell labels anchor steps i j ≠ {label} := by
  intro hanswer
  exact hne ((bank_identifies k hfit anchor steps hanswer).trans
    (bank_identifies k' hfit' anchor steps hanswer).symm)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation
