module

public import Transformer.CRASP.GramsForms
public import Transformer.MRCR.PhraseKeyPrevCorrect

/-!
# The answer formula over grams

In no paper. The formula `Transformer.MRCR.phraseAnswerForm` of `TL[◁#, Y]` reads with `Y` the
tokens at most three positions back. Over the grams of four tokens
(`Transformer.CRASP.Grams.grams 3 id w`), the same formula with `Y^m Q_a` read off the gram
(`Transformer.CRASP.Grams.symAt`) is one of `TL[◁#]` of depth two (`phraseAnswerG_mem`), which
holds of the grams of `w` iff `w` asks for `v` (`mem_lang_phraseAnswerG`).
`Transformer.MRCR.PhraseKeyLogits` compiles it into transformers with ALiBi, KERPLE and SSMax.
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP Transformer.CRASP.Grams

variable {F N V R : ℕ}

/-- The grams of four tokens of a conversation. -/
abbrev PGram (F N V R : ℕ) := Gram (PhraseTok F N V R) 3

/-- `askEnd` over grams. -/
noncomputable def askEndG (f : Fin F) (t : Fin N) : Form (PGram F N V R) :=
  Form.all [symAt 0 (.topic t), symAt 1 .about, symAt 2 (.fmt f)]

/-- `answerHere` over grams. -/
noncomputable def answerHereG (f : Fin F) (t : Fin N) (v : Fin V) : Form (PGram F N V R) :=
  Form.all [symAt 0 (.val v), symAt 1 (.topic t), symAt 2 .about, symAt 3 (.fmt f)]

/-- `needle` over grams. -/
noncomputable def needleG (f : Fin F) (t : Fin N) (r : ℕ) (v : Fin V) : Form (PGram F N V R) :=
  .and (answerHereG f t v)
    (.and (.neg (.lt (.countL (askEndG f t)) (constTerm r)))
      (.neg (.lt (constTerm r) (.countL (askEndG f t)))))

/-- `queryForm` over grams. -/
noncomputable def queryFormG (f : Fin F) (t : Fin N) (r : Fin R) (v : Fin V) :
    Form (PGram F N V R) :=
  .and (symAt 0 (.query f t r)) (.neg (.lt (.countL (needleG f t r v)) .one))

/-- `phraseAnswerForm` over grams. -/
noncomputable def phraseAnswerG (v : Fin V) : Form (PGram F N V R) :=
  Form.any ((List.finRange F ×ˢ (List.finRange N ×ˢ List.finRange R)).map
    fun x => queryFormG x.1 x.2.1 x.2.2 v)

section Fragment

theorem askEndG_mem (f : Fin F) (t : Fin N) : askEndG (V := V) (R := R) f t ∈ TLCl _ 0 :=
  Form.all_mem _ fun φ hφ => by
    simp only [List.mem_cons, List.mem_nil_iff, or_false] at hφ
    rcases hφ with rfl | rfl | rfl <;> exact symAt_mem _ _

theorem answerHereG_mem (f : Fin F) (t : Fin N) (v : Fin V) :
    answerHereG (R := R) f t v ∈ TLCl _ 0 :=
  Form.all_mem _ fun φ hφ => by
    simp only [List.mem_cons, List.mem_nil_iff, or_false] at hφ
    rcases hφ with rfl | rfl | rfl | rfl <;> exact symAt_mem _ _

theorem needleG_mem (f : Fin F) (t : Fin N) (r : ℕ) (v : Fin V) :
    needleG (R := R) f t r v ∈ TLCl _ 1 := by
  obtain ⟨h1, h2, h3⟩ := askEndG_mem (V := V) (R := R) f t
  obtain ⟨c1, c2, c3, -⟩ := constTerm_props (Γ := PGram F N V R) r
  have hd : (askEndG (V := V) (R := R) f t).depth = 0 := by omega
  exact Form.and_mem (Form.mem_mono (answerHereG_mem f t v) (Nat.zero_le 1))
    (Form.and_mem (Form.neg_mem ⟨by simp [Form.past, Term.past, h1, c1],
      by simp [Form.pnpFree, Term.pnpFree, h2, c2], by simp [Form.depth, Term.depth, hd, c3]⟩)
      (Form.neg_mem ⟨by simp [Form.past, Term.past, h1, c1],
        by simp [Form.pnpFree, Term.pnpFree, h2, c2], by simp [Form.depth, Term.depth, hd, c3]⟩))

theorem queryFormG_mem (f : Fin F) (t : Fin N) (r : Fin R) (v : Fin V) :
    queryFormG f t r v ∈ TLCl _ 2 := by
  obtain ⟨h1, h2, h3⟩ := needleG_mem (R := R) f t r v
  exact Form.and_mem (Form.mem_mono (symAt_mem _ _) (Nat.zero_le 2)) (Form.neg_mem
    ⟨by simp [Form.past, Term.past, h1], by simp [Form.pnpFree, Term.pnpFree, h2],
      by simp only [Form.depth, Term.depth]; omega⟩)

/-- **The answer formula over grams is one of `TL[◁#]` of depth two.** -/
theorem phraseAnswerG_mem (v : Fin V) :
    phraseAnswerG (F := F) (N := N) (R := R) v ∈ TLCl _ 2 :=
  Form.any_mem _ fun φ hφ => by
    obtain ⟨x, -, rfl⟩ := List.mem_map.mp hφ
    exact queryFormG_mem _ _ _ _

end Fragment

section Sat

variable {w : List (PhraseTok F N V R)} {j : ℕ}

theorem sat_askEndG (f : Fin F) (t : Fin N) (hj : 0 < j) (hjw : j ≤ w.length) :
    (askEndG f t).sat (grams 3 id w) j = (askEnd f t).sat w j := by
  apply Bool.eq_iff_iff.mpr
  simp only [askEndG, askEnd, Form.sat_all, FormP.sat_all, List.mem_cons, List.mem_nil_iff,
    or_false, forall_eq_or_imp, forall_eq, sat_symAt _ _ hj hjw, FormP.sat_shift_sym _ _ w hj,
    FormP.sat, decide_eq_true_eq]
  simp [hj]

/-- The hypotheses of `sat_askEndG`, `sat_answerHereG`, `sat_needleG` and `sat_queryFormG` hold in
the conversation "`0` about `0`", answer `0`, query. -/
example := sat_askEndG (w := [PhraseTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .val (0 : Fin 1), .query (R := 1) 0 0 0]) (j := 3) 0 0 (by norm_num) (by norm_num)

theorem sat_answerHereG (f : Fin F) (t : Fin N) (v : Fin V) (hj : 0 < j) (hjw : j ≤ w.length) :
    (answerHereG f t v).sat (grams 3 id w) j = (answerHere f t v).sat w j := by
  apply Bool.eq_iff_iff.mpr
  simp only [answerHereG, answerHere, Form.sat_all, FormP.sat_all, List.mem_cons,
    List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq, sat_symAt _ _ hj hjw,
    FormP.sat_shift_sym _ _ w hj, FormP.sat, decide_eq_true_eq]
  simp [hj]

example := sat_answerHereG (w := [PhraseTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .val (0 : Fin 1), .query (R := 1) 0 0 0]) (j := 4) 0 0 0 (by norm_num) (by norm_num)

theorem sat_needleG (f : Fin F) (t : Fin N) (r : ℕ) (v : Fin V) (hj : 0 < j)
    (hjw : j ≤ w.length) :
    (needleG f t r v).sat (grams 3 id w) j = (needle f t r v).sat w j := by
  have hc : (Term.countL (askEndG f t)).val (grams 3 id w) j =
      (TermP.countL (askEnd f t)).val w j :=
    val_countL_grams hjw fun i hi hiw => sat_askEndG f t hi hiw
  have hk : (constTerm r : Term (PGram F N V R)).val (grams 3 id w) j =
      (TermP.addConst r .one).val w j := by
    rw [val_constTerm, TermP.val_addConst]
    rfl
  simp only [needleG, needle, Form.sat, FormP.sat, sat_answerHereG f t v hj hjw, hc, hk]

example := sat_needleG (w := [PhraseTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .val (0 : Fin 1), .query (R := 1) 0 0 0]) (j := 4) 0 0 0 0 (by norm_num) (by norm_num)

theorem sat_queryFormG (f : Fin F) (t : Fin N) (r : Fin R) (v : Fin V) (hj : 0 < j)
    (hjw : j ≤ w.length) :
    (queryFormG f t r v).sat (grams 3 id w) j = (queryForm f t r v).sat w j := by
  have hc : (Term.countL (needleG f t r v)).val (grams 3 id w) j =
      (TermP.countL (needle f t r v)).val w j :=
    val_countL_grams hjw fun i hi hiw => sat_needleG f t r v hi hiw
  have h0 : (symAt 0 (.query f t r) : Form (PGram F N V R)).sat (grams 3 id w) j =
      (FormP.sym (.query f t r)).sat w j := by
    apply Bool.eq_iff_iff.mpr
    rw [sat_symAt _ _ hj hjw]
    simp [FormP.sat, hj]
  have h1 : (Term.one : Term (PGram F N V R)).val (grams 3 id w) j = 1 := rfl
  have h1' : (TermP.one : TermP (PhraseTok F N V R)).val w j = 1 := rfl
  simp only [queryFormG, queryForm, Form.sat, FormP.sat, h0, hc, h1, h1']

example := sat_queryFormG (w := [PhraseTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .val (0 : Fin 1), .query (R := 1) 0 0 0]) (j := 5) 0 0 0 0 (by norm_num) le_rfl

theorem sat_phraseAnswerG (v : Fin V) (hj : 0 < j) (hjw : j ≤ w.length) :
    (phraseAnswerG (R := R) v).sat (grams 3 id w) j = (phraseAnswerForm v).sat w j := by
  apply Bool.eq_iff_iff.mpr
  simp only [phraseAnswerG, phraseAnswerForm, Form.sat_any, FormP.sat_any, List.mem_map,
    exists_exists_and_eq_and, sat_queryFormG _ _ _ _ hj hjw]

/-- **The answer formula holds of the grams of `w` iff `w` asks for `v`.** -/
theorem mem_lang_phraseAnswerG (v : Fin V) (w : List (PhraseTok F N V R)) :
    grams 3 id w ∈ (phraseAnswerG v).lang ↔ PhraseAnswer w v := by
  rcases Nat.eq_zero_or_pos w.length with h0 | hpos
  · obtain rfl : w = [] := List.eq_nil_of_length_eq_zero h0
    have hg : grams 3 id ([] : List (PhraseTok F N V R)) = [] := by simp [grams]
    refine ⟨fun h => ?_, fun ⟨f, t, r, hl, _⟩ => by simp at hl⟩
    change (phraseAnswerG v).sat (grams 3 id []) (grams 3 id []).length = true at h
    rw [hg, phraseAnswerG, Form.sat_any] at h
    obtain ⟨φ, hφ, hs⟩ := h
    obtain ⟨x, -, rfl⟩ := List.mem_map.mp hφ
    simp [queryFormG, Form.sat, sat_symAt_nil] at hs
  · rw [← models_phraseAnswerForm v w]
    change (phraseAnswerG v).sat (grams 3 id w) (grams 3 id w).length = true ↔
      (phraseAnswerForm v).sat w w.length = true
    rw [length_grams, sat_phraseAnswerG v hpos le_rfl]

/-- The hypotheses of `sat_phraseAnswerG` hold at the query of the conversation "`0` about `0`",
answer `0`, query. -/
example := sat_phraseAnswerG (w := [PhraseTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .val (0 : Fin 1), .query (R := 1) 0 0 0]) (j := 5) 0 (by norm_num) le_rfl

end Sat

end Transformer.MRCR
