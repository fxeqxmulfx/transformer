/-
# Closed `MAJ²` formulas for piecewise-testable languages

arXiv:2506.16055v3, Appendix E, proof of `thm:ltc0_hierarchy`.
A nonempty pattern is tested by quantifying its middle position. Boolean
combinations of these closed tests handle the separating languages, including
`D_1`, which is not a single existential depth-zero test.
-/

import Transformer.CRASP.MajTwoEquiv

namespace Transformer.CRASP

universe u
variable {σ : Type u} [DecidableEq σ]

/-- A pattern of at most `2k - 1` symbols has a closed depth-`k` test.
This is the middle-position construction used in Appendix E's proof of
`thm:ltc0_hierarchy`, with strict counts as required by Appendix A. -/
theorem exists_closed_majTwo_of_sublist (k : ℕ) (hk : 0 < k) (s : List σ)
    (hs : s.length ≤ 2 * k - 1) :
    ∃ φ ∈ MajTwo σ k, φ.Closed ∧ φ.lang = {w : List σ | s.Sublist w} := by
  by_cases hnil : s = []
  · subst s
    refine ⟨Maj2.closedTop, by simpa [MajTwo] using (show 1 ≤ k by omega),
      Maj2.closed_closedTop, ?_⟩
    ext w
    simp [Maj2.lang_closedTop]
  · let mid := s.length / 2
    cases hd : s.drop mid with
    | nil =>
        have hlen := congrArg List.length hd
        simp only [List.length_drop, List.length_nil] at hlen
        have hpos := List.length_pos_iff.mpr hnil
        dsimp [mid] at hlen
        omega
    | cons c r =>
        have hlen := congrArg List.length hd
        simp only [List.length_drop, List.length_cons] at hlen
        have hleft : (s.take mid).length ≤ k - 1 := by
          rw [List.length_take]
          dsimp [mid]
          omega
        have hright : r.length ≤ k - 1 := by dsimp [mid] at hlen; omega
        let body := Form.and (.and (subseqStrict (s.take mid).reverse) (.sym c)) (subseqAfter r)
        have hb : body ∈ TLC σ (k - 1) := by
          refine ⟨by simp [body, Form.pnpFree], ?_⟩
          simp only [body, Form.depth, depth_subseqStrict, List.length_reverse,
            depth_subseqAfter, max_le_iff]
          exact ⟨⟨hleft, Nat.zero_le _⟩, hright⟩
        obtain ⟨φ, hφ, hc, hlang⟩ := exists_closed_majTwo (k - 1) body hb
        refine ⟨φ, by simpa only [Nat.sub_add_cancel hk] using hφ, hc, ?_⟩
        rw [hlang]
        ext w
        have hsplit : s = s.take mid ++ c :: r := by
          rw [← hd, List.take_append_drop]
        rw [hsplit]
        simp only [Set.mem_ofPred_eq, body, Form.sat, Bool.and_eq_true,
          sat_subseqStrict, List.reverse_reverse, sat_subseqAfter, decide_eq_true_eq, and_assoc]
        exact (append_cons_sublist_iff w (s.take mid) r c).symm

/-- Negation takes the complement language (Appendix E, `def:MAJtwo`). -/
theorem Maj2.lang_neg (φ : Maj2 σ) : φ.neg.lang = φ.langᶜ := by
  ext w
  simp [Maj2.lang, Maj2.models, Maj2.sat]

/-- Conjunction takes the intersection language (Appendix E, `def:MAJtwo`). -/
theorem Maj2.lang_and (φ ψ : Maj2 σ) : (φ.and ψ).lang = φ.lang ∩ ψ.lang := by
  ext w
  simp [Maj2.lang, Maj2.models, Maj2.sat]

/-- Boolean combinations of the closed pattern tests preserve their depth.
Source: arXiv:2506.16055v3, Appendix E, proof of `thm:ltc0_hierarchy`. -/
theorem PT.exists_closed_majTwo (e : PT σ) (k : ℕ) (hk : 0 < k)
    (he : e.width ≤ 2 * k - 1) : ∃ φ ∈ MajTwo σ k, φ.Closed ∧ φ.lang = e.lang := by
  induction e with
  | jexpr s => exact exists_closed_majTwo_of_sublist k hk s he
  | neg e ih =>
      obtain ⟨φ, hφ, hc, hlang⟩ := ih he
      exact ⟨φ.neg, hφ, hc, by rw [Maj2.lang_neg, PT.lang, hlang]⟩
  | and e f ihe ihf =>
      obtain ⟨φ, hφ, hcφ, hlφ⟩ := ihe (le_trans (le_max_left _ _) he)
      obtain ⟨ψ, hψ, hcψ, hlψ⟩ := ihf (le_trans (le_max_right _ _) he)
      refine ⟨φ.and ψ, by change max φ.depth ψ.depth ≤ k; exact max_le hφ hψ, ?_, ?_⟩
      · intro v
        simp [Maj2.freeIn, hcφ v, hcψ v]
      · rw [Maj2.lang_and, PT.lang, hlφ, hlψ]

/-- Piecewise testability gives a closed `MAJ²` definition at half the width.
Source: arXiv:2506.16055v3, Appendix E, proof of `thm:ltc0_hierarchy`. -/
theorem exists_closed_majTwo_of_piecewise_testable (k : ℕ) (hk : 0 < k)
    (L : Set (List σ)) (hL : KPiecewiseTestable (2 * k - 1) L) :
    ∃ φ ∈ MajTwo σ k, φ.Closed ∧ φ.lang = L := by
  obtain ⟨e, he, rfl⟩ := hL
  exact e.exists_closed_majTwo k hk he

/-- The pattern-width and piecewise-testability hypotheses have witnesses. -/
example : 0 < 1 ∧ [true].length ≤ 2 * 1 - 1 ∧
    (PT.jexpr [true]).width ≤ 2 * 1 - 1 ∧
    KPiecewiseTestable (2 * 1 - 1) {w : List Bool | [true].Sublist w} :=
  ⟨one_pos, le_rfl, le_rfl, .intro (.jexpr [true]) ⟨le_rfl, rfl⟩⟩

end Transformer.CRASP
