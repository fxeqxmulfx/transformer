module

public import Transformer.MRCR.PhraseKeyPrev

/-!
# The formula of `TL[◁#, Y]` answers spelled asks

In no paper; the formula `phraseAnswerForm` of `Transformer.MRCR.PhraseKeyPrev` accepts a
conversation iff `v` answers the query that closes it (`models_phraseAnswerForm`), so `TL[◁#, Y]`
of arXiv:2506.16055v3, Appendix F, answers spelled asks at depth two
(`definableY_phraseAnswer`). The count of `askEnd` up to the position `j` is the number of asks
that start before the index `j - 2` (`val_countL_askEnd`); at the answer after an ask at the
index `i` it counts that ask too, as `about` stands at `i + 1` (`count_add_two_of_askAt`).
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP

variable {F N V R : ℕ} {w : List (PhraseTok F N V R)} {f : Fin F} {t : Fin N} {v : Fin V}

/-- **The count of `askEnd` is the number of asks**: at the position `j` it counts the asks
"`f` about `t`" that start before the index `j - 2`. -/
theorem val_countL_askEnd (w : List (PhraseTok F N V R)) (f : Fin F) (t : Fin N) :
    ∀ j, (TermP.countL (askEnd f t)).val w j = Nat.count (AskAt w f t) (j - 2)
  | 0 => by simp [TermP.val]
  | j + 1 => by
      rw [TermP.val_countL_succ, val_countL_askEnd w f t j]
      have hs := sat_askEnd (w := w) (f := f) (t := t) (Nat.succ_pos j)
      rcases Nat.lt_or_ge j 2 with h | h
      · rw [ite_eq_right fun hj => absurd (hs.1 hj).1 (by omega), show j + 1 - 2 = 0 by omega,
          show j - 2 = 0 by omega, Nat.add_zero]
      · rw [show j + 1 - 2 = j - 2 + 1 by omega, Nat.count_succ, show j + 1 - 3 = j - 2 by omega]
          at *
        by_cases hA : AskAt w f t (j - 2)
        · rw [ite_eq_left (hs.2 ⟨by omega, hA⟩), ite_eq_left hA]
        · rw [ite_eq_right fun hj => hA (hs.1 hj).2, ite_eq_right hA]

/-- After an ask "`f` about `t`" at the index `i`, the asks before the index `i + 2` are those
before `i` and that ask: `about` stands at `i + 1`. -/
theorem count_add_two_of_askAt {i : ℕ} (hA : AskAt w f t i) :
    Nat.count (AskAt w f t) (i + 2) = Nat.count (AskAt w f t) i + 1 := by
  rw [Nat.count_succ, Nat.count_succ, ite_eq_left hA, ite_eq_right]
  rintro ⟨h, -, -⟩
  exact absurd (h.symm.trans hA.2.1) (by simp)

/-- `needle f t r v` holds at the position `j` iff `v` answers there the ask "`f` about `t`" with
`r` such asks before it, which starts at the index `j - 4`. -/
theorem sat_needle {j : ℕ} (hj : 0 < j) (r : ℕ) :
    (needle f t r v).sat w j = true ↔ 3 < j ∧ AskAt w f t (j - 4) ∧
      w[j - 1]? = some (.val v) ∧ Nat.count (AskAt w f t) (j - 4) = r := by
  simp only [needle, FormP.sat, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    not_lt, sat_answerHere hj, val_countL_askEnd, TermP.val_addConst,
    show TermP.val w j (.one : TermP (PhraseTok F N V R)) = 1 from rfl]
  constructor
  · rintro ⟨⟨h, hA, hv⟩, h1, h2⟩
    rw [show j - 2 = j - 4 + 2 by omega, count_add_two_of_askAt hA] at h1 h2
    exact ⟨h, hA, hv, by omega⟩
  · rintro ⟨h, hA, hv, hc⟩
    rw [show j - 2 = j - 4 + 2 by omega, count_add_two_of_askAt hA]
    exact ⟨⟨h, hA, hv⟩, by omega, by omega⟩

/-- `queryForm f t r v` holds at the position `n` iff the query for the ask "`f` about `t`" with
`r` such asks before it stands there, and `v` answers that ask before it. -/
theorem sat_queryForm {n : ℕ} (r : Fin R) :
    (queryForm f t r v).sat w n = true ↔ w[n - 1]? = some (.query f t r) ∧
      ∃ i, i + 4 ≤ n ∧ AskAt w f t i ∧ w[i + 3]? = some (.val v) ∧
        Nat.count (AskAt w f t) i = r := by
  simp only [queryForm, FormP.sat, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true',
    decide_eq_false_iff_not, not_lt]
  refine and_congr_right fun _ => ?_
  rw [show TermP.val w n (.one : TermP (PhraseTok F N V R)) = 1 from rfl, TermP.val,
    Nat.one_le_iff_ne_zero, ← Nat.pos_iff_ne_zero, List.length_filter_pos_iff]
  constructor
  · rintro ⟨j, hj, hs⟩
    rw [List.mem_range'_1] at hj
    obtain ⟨h, hA, hv, hc⟩ := (sat_needle (by omega) r).1 hs
    exact ⟨j - 4, by omega, hA, by rwa [show j - 4 + 3 = j - 1 by omega], hc⟩
  · rintro ⟨i, hi, hA, hv, hc⟩
    refine ⟨i + 4, List.mem_range'_1.2 ⟨by omega, by omega⟩, (sat_needle (by omega) r).2 ?_⟩
    simp only [Nat.add_sub_cancel]
    exact ⟨by omega, hA, hv, hc⟩

/-- **The formula answers the query**: it accepts `w` iff `v` answers the query that closes
`w`. -/
theorem models_phraseAnswerForm (v : Fin V) (w : List (PhraseTok F N V R)) :
    (phraseAnswerForm v).models w ↔ PhraseAnswer w v := by
  rw [FormP.models, phraseAnswerForm, FormP.sat_any, PhraseAnswer, List.getLast?_eq_getElem?]
  constructor
  · rintro ⟨φ, hφ, h⟩
    obtain ⟨⟨f, t, r⟩, -, rfl⟩ := List.mem_map.1 hφ
    obtain ⟨hq, i, -, hA, hv, hc⟩ := (sat_queryForm r).1 h
    exact ⟨f, t, r, hq, i, hA, hc, hv⟩
  · rintro ⟨f, t, r, hq, i, hA, hc, hv⟩
    have hi : i + 3 < w.length := (List.getElem?_eq_some_iff.1 hv).1
    refine ⟨_, List.mem_map.2 ⟨(f, t, r), List.mem_product.2 ⟨List.mem_finRange f,
      List.mem_product.2 ⟨List.mem_finRange t, List.mem_finRange r⟩⟩, rfl⟩,
      (sat_queryForm r).2 ⟨hq, i, by omega, hA, hv, hc⟩⟩

/-- **`TL[◁#, Y]` answers spelled asks at depth two**: the conversations that `v` answers are
the language of a formula of depth two without `MOD`. -/
theorem definableY_phraseAnswer (v : Fin V) :
    DefinableY {w : List (PhraseTok F N V R) | PhraseAnswer w v} 2 :=
  ⟨phraseAnswerForm v, phraseAnswerForm_mem v, Set.ext (models_phraseAnswerForm v)⟩

/-- The hypotheses of `count_add_two_of_askAt` and `sat_needle` hold in the conversation
"`0` about `0`", answer `0`, query: at the ask and at its answer. -/
example := count_add_two_of_askAt (w := [PhraseTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .val (0 : Fin 1), .query (R := 1) 0 0 0]) (f := 0) (t := 0) (i := 0) ⟨rfl, rfl, rfl⟩

example := sat_needle (w := [PhraseTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .val (0 : Fin 1), .query (R := 1) 0 0 0]) (f := 0) (t := 0) (v := 0) (j := 4) (by norm_num) 0

end Transformer.MRCR
