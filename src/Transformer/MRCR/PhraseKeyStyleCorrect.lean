module

public import Transformer.MRCR.PhraseKeyStyle

/-!
# The formula of `TL[◁#, Y]` answers asks with a style

In no paper. The formula `styleAnswerForm` of `Transformer.MRCR.PhraseKeyStyle` accepts a
conversation iff `v` answers the query that closes it (`models_styleAnswerForm`), so `TL[◁#, Y]`
of arXiv:2506.16055v3, Appendix F, answers asks with keys of depth three at depth two
(`definableY_styleAnswer`), as `Transformer.MRCR.PhraseKeyPrevCorrect` shows for depth two. The
count of `styleAskEnd` up to the position `j` is the number of asks that start before the index
`j - 3` (`val_countL_styleAskEnd`). At the answer after an ask at the index `i` it counts that
ask too, as `about` stands at `i + 1` (`count_add_two_of_styleAskAt`).
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP

variable {F N S V R : ℕ} {w : List (StyleTok F N S V R)} {f : Fin F} {t : Fin N} {σ : Fin S}
  {v : Fin V}

/-- **The count of `styleAskEnd` is the number of asks**: at the position `j` it counts the asks
"`f` about `t` in the style `σ`" that start before the index `j - 3`. -/
theorem val_countL_styleAskEnd (w : List (StyleTok F N S V R)) (f : Fin F) (t : Fin N)
    (σ : Fin S) :
    ∀ j, (TermP.countL (styleAskEnd f t σ)).val w j = Nat.count (StyleAskAt w f t σ) (j - 3)
  | 0 => by simp [TermP.val]
  | j + 1 => by
      rw [TermP.val_countL_succ, val_countL_styleAskEnd w f t σ j]
      have hs := sat_styleAskEnd (w := w) (f := f) (t := t) (σ := σ) (Nat.succ_pos j)
      rcases Nat.lt_or_ge j 3 with h | h
      · rw [ite_eq_right fun hj => absurd (hs.1 hj).1 (by omega), show j + 1 - 3 = 0 by omega,
          show j - 3 = 0 by omega, Nat.add_zero]
      · rw [show j + 1 - 3 = j - 3 + 1 by omega, Nat.count_succ, show j + 1 - 4 = j - 3 by omega]
          at *
        by_cases hA : StyleAskAt w f t σ (j - 3)
        · rw [ite_eq_left (hs.2 ⟨by omega, hA⟩), ite_eq_left hA]
        · rw [ite_eq_right fun hj => hA (hs.1 hj).2, ite_eq_right hA]

/-- After an ask "`f` about `t` in the style `σ`" at the index `i`, the asks before the index
`i + 2` are those before `i` and that ask: `about` stands at `i + 1`. -/
theorem count_add_two_of_styleAskAt {i : ℕ} (hA : StyleAskAt w f t σ i) :
    Nat.count (StyleAskAt w f t σ) (i + 2) = Nat.count (StyleAskAt w f t σ) i + 1 := by
  rw [Nat.count_succ, Nat.count_succ, ite_eq_left hA, ite_eq_right]
  rintro ⟨h, -, -, -⟩
  exact absurd (h.symm.trans hA.2.1) (by simp)

/-- `styleNeedle f t σ r v` holds at the position `j` iff `v` answers there the ask "`f` about `t`
in the style `σ`" with `r` such asks before it, which starts at the index `j - 5`. -/
theorem sat_styleNeedle {j : ℕ} (hj : 0 < j) (r : ℕ) :
    (styleNeedle f t σ r v).sat w j = true ↔ 4 < j ∧ StyleAskAt w f t σ (j - 5) ∧
      w[j - 1]? = some (.val v) ∧ Nat.count (StyleAskAt w f t σ) (j - 5) = r := by
  simp only [styleNeedle, FormP.sat, Bool.and_eq_true, Bool.not_eq_true',
    decide_eq_false_iff_not, not_lt, sat_styleAnswerHere hj, val_countL_styleAskEnd,
    TermP.val_addConst, show TermP.val w j (.one : TermP (StyleTok F N S V R)) = 1 from rfl]
  constructor
  · rintro ⟨⟨h, hA, hv⟩, h1, h2⟩
    rw [show j - 3 = j - 5 + 2 by omega, count_add_two_of_styleAskAt hA] at h1 h2
    exact ⟨h, hA, hv, by omega⟩
  · rintro ⟨h, hA, hv, hc⟩
    rw [show j - 3 = j - 5 + 2 by omega, count_add_two_of_styleAskAt hA]
    exact ⟨⟨h, hA, hv⟩, by omega, by omega⟩

/-- `styleQueryForm f t σ r v` holds at the position `n` iff the query for the ask "`f` about `t`
in the style `σ`" with `r` such asks before it stands there, and `v` answers that ask before
it. -/
theorem sat_styleQueryForm {n : ℕ} (r : Fin R) :
    (styleQueryForm f t σ r v).sat w n = true ↔ w[n - 1]? = some (.query f t σ r) ∧
      ∃ i, i + 5 ≤ n ∧ StyleAskAt w f t σ i ∧ w[i + 4]? = some (.val v) ∧
        Nat.count (StyleAskAt w f t σ) i = r := by
  simp only [styleQueryForm, FormP.sat, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true',
    decide_eq_false_iff_not, not_lt]
  refine and_congr_right fun _ => ?_
  rw [show TermP.val w n (.one : TermP (StyleTok F N S V R)) = 1 from rfl, TermP.val,
    Nat.one_le_iff_ne_zero, ← Nat.pos_iff_ne_zero, List.length_filter_pos_iff]
  constructor
  · rintro ⟨j, hj, hs⟩
    rw [List.mem_range'_1] at hj
    obtain ⟨h, hA, hv, hc⟩ := (sat_styleNeedle (by omega) r).1 hs
    exact ⟨j - 5, by omega, hA, by rwa [show j - 5 + 4 = j - 1 by omega], hc⟩
  · rintro ⟨i, hi, hA, hv, hc⟩
    refine ⟨i + 5, List.mem_range'_1.2 ⟨by omega, by omega⟩,
      (sat_styleNeedle (by omega) r).2 ?_⟩
    simp only [Nat.add_sub_cancel]
    exact ⟨by omega, hA, hv, hc⟩

/-- **The formula answers the query**: it accepts `w` iff `v` answers the query that closes
`w`. -/
theorem models_styleAnswerForm (v : Fin V) (w : List (StyleTok F N S V R)) :
    (styleAnswerForm v).models w ↔ StyleAnswer w v := by
  rw [FormP.models, styleAnswerForm, FormP.sat_any, StyleAnswer, List.getLast?_eq_getElem?]
  constructor
  · rintro ⟨φ, hφ, h⟩
    obtain ⟨⟨f, t, σ, r⟩, -, rfl⟩ := List.mem_map.1 hφ
    obtain ⟨hq, i, -, hA, hv, hc⟩ := (sat_styleQueryForm r).1 h
    exact ⟨f, t, σ, r, hq, i, hA, hc, hv⟩
  · rintro ⟨f, t, σ, r, hq, i, hA, hc, hv⟩
    have hi : i + 4 < w.length := (List.getElem?_eq_some_iff.1 hv).1
    refine ⟨_, List.mem_map.2 ⟨(f, t, σ, r), List.mem_product.2 ⟨List.mem_finRange f,
      List.mem_product.2 ⟨List.mem_finRange t, List.mem_product.2 ⟨List.mem_finRange σ,
        List.mem_finRange r⟩⟩⟩, rfl⟩, (sat_styleQueryForm r).2 ⟨hq, i, by omega, hA, hv, hc⟩⟩

/-- **`TL[◁#, Y]` answers asks with a style at depth two**: the conversations that `v` answers
are the language of a formula of depth two without `MOD`. -/
theorem definableY_styleAnswer (v : Fin V) :
    DefinableY {w : List (StyleTok F N S V R) | StyleAnswer w v} 2 :=
  ⟨styleAnswerForm v, styleAnswerForm_mem v, Set.ext (models_styleAnswerForm v)⟩

/-- The hypotheses of `count_add_two_of_styleAskAt` and `sat_styleNeedle` hold in the
conversation "`0` about `0` in the style `0`", answer `0`, query: at the ask and at its
answer. -/
example := count_add_two_of_styleAskAt (w := [StyleTok.fmt (0 : Fin 1), .about,
  .topic (0 : Fin 1), .style (0 : Fin 1), .val (0 : Fin 1), .query (R := 1) 0 0 0 0]) (f := 0)
  (t := 0) (σ := 0) (i := 0) ⟨rfl, rfl, rfl, rfl⟩

example := sat_styleNeedle (w := [StyleTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .style (0 : Fin 1), .val (0 : Fin 1), .query (R := 1) 0 0 0 0]) (f := 0) (t := 0) (σ := 0)
  (v := 0) (j := 5) (by norm_num) 0

end Transformer.MRCR
