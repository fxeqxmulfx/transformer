module

public import Transformer.CRASP.GramsForms
public import Transformer.MRCR.PhraseKeyStyleCorrect

/-!
# The answer formula for asks with a style, over grams

In no paper. The formula `Transformer.MRCR.styleAnswerForm` of `TL[◁#, Y]` reads with `Y` the
tokens at most four positions back. Over the grams of five tokens
(`Transformer.CRASP.Grams.grams 4 id w`), the same formula with `Y^m Q_a` read off the gram
(`Transformer.CRASP.Grams.symAt`) is one of `TL[◁#]` of depth two (`styleAnswerG_mem`), which
holds of the grams of `w` iff `w` asks for `v` (`mem_lang_styleAnswerG`), as
`Transformer.MRCR.PhraseKeyGrams` shows for keys of depth two.
`Transformer.MRCR.PhraseKeyStyleLogits` compiles it into rounded transformers.
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP Transformer.CRASP.Grams

variable {F N S V R : ℕ}

/-- The grams of five tokens of a conversation with styles. -/
abbrev SGram (F N S V R : ℕ) := Gram (StyleTok F N S V R) 4

/-- `styleAskEnd` over grams. -/
noncomputable def styleAskEndG (f : Fin F) (t : Fin N) (σ : Fin S) : Form (SGram F N S V R) :=
  Form.all [symAt 0 (.style σ), symAt 1 (.topic t), symAt 2 .about, symAt 3 (.fmt f)]

/-- `styleAnswerHere` over grams. -/
noncomputable def styleAnswerHereG (f : Fin F) (t : Fin N) (σ : Fin S) (v : Fin V) :
    Form (SGram F N S V R) :=
  Form.all [symAt 0 (.val v), symAt 1 (.style σ), symAt 2 (.topic t), symAt 3 .about,
    symAt 4 (.fmt f)]

/-- `styleNeedle` over grams. -/
noncomputable def styleNeedleG (f : Fin F) (t : Fin N) (σ : Fin S) (r : ℕ) (v : Fin V) :
    Form (SGram F N S V R) :=
  .and (styleAnswerHereG f t σ v)
    (.and (.neg (.lt (.countL (styleAskEndG f t σ)) (constTerm r)))
      (.neg (.lt (constTerm r) (.countL (styleAskEndG f t σ)))))

/-- `styleQueryForm` over grams. -/
noncomputable def styleQueryFormG (f : Fin F) (t : Fin N) (σ : Fin S) (r : Fin R) (v : Fin V) :
    Form (SGram F N S V R) :=
  .and (symAt 0 (.query f t σ r)) (.neg (.lt (.countL (styleNeedleG f t σ r v)) .one))

/-- `styleAnswerForm` over grams. -/
noncomputable def styleAnswerG (v : Fin V) : Form (SGram F N S V R) :=
  Form.any ((List.finRange F ×ˢ (List.finRange N ×ˢ (List.finRange S ×ˢ List.finRange R))).map
    fun x => styleQueryFormG x.1 x.2.1 x.2.2.1 x.2.2.2 v)

section Fragment

theorem styleAskEndG_mem (f : Fin F) (t : Fin N) (σ : Fin S) :
    styleAskEndG (V := V) (R := R) f t σ ∈ TLCl _ 0 :=
  Form.all_mem _ fun φ hφ => by
    simp only [List.mem_cons, List.mem_nil_iff, or_false] at hφ
    rcases hφ with rfl | rfl | rfl | rfl <;> exact symAt_mem _ _

theorem styleAnswerHereG_mem (f : Fin F) (t : Fin N) (σ : Fin S) (v : Fin V) :
    styleAnswerHereG (R := R) f t σ v ∈ TLCl _ 0 :=
  Form.all_mem _ fun φ hφ => by
    simp only [List.mem_cons, List.mem_nil_iff, or_false] at hφ
    rcases hφ with rfl | rfl | rfl | rfl | rfl <;> exact symAt_mem _ _

theorem styleNeedleG_mem (f : Fin F) (t : Fin N) (σ : Fin S) (r : ℕ) (v : Fin V) :
    styleNeedleG (R := R) f t σ r v ∈ TLCl _ 1 := by
  obtain ⟨h1, h2, h3⟩ := styleAskEndG_mem (V := V) (R := R) f t σ
  obtain ⟨c1, c2, c3, -⟩ := constTerm_props (Γ := SGram F N S V R) r
  have hd : (styleAskEndG (V := V) (R := R) f t σ).depth = 0 := by omega
  exact Form.and_mem (Form.mem_mono (styleAnswerHereG_mem f t σ v) (Nat.zero_le 1))
    (Form.and_mem (Form.neg_mem ⟨by simp [Form.past, Term.past, h1, c1],
      by simp [Form.pnpFree, Term.pnpFree, h2, c2], by simp [Form.depth, Term.depth, hd, c3]⟩)
      (Form.neg_mem ⟨by simp [Form.past, Term.past, h1, c1],
        by simp [Form.pnpFree, Term.pnpFree, h2, c2], by simp [Form.depth, Term.depth, hd, c3]⟩))

theorem styleQueryFormG_mem (f : Fin F) (t : Fin N) (σ : Fin S) (r : Fin R) (v : Fin V) :
    styleQueryFormG f t σ r v ∈ TLCl _ 2 := by
  obtain ⟨h1, h2, h3⟩ := styleNeedleG_mem (R := R) f t σ r v
  exact Form.and_mem (Form.mem_mono (symAt_mem _ _) (Nat.zero_le 2)) (Form.neg_mem
    ⟨by simp [Form.past, Term.past, h1], by simp [Form.pnpFree, Term.pnpFree, h2],
      by simp only [Form.depth, Term.depth]; omega⟩)

/-- **The answer formula over grams is one of `TL[◁#]` of depth two.** -/
theorem styleAnswerG_mem (v : Fin V) :
    styleAnswerG (F := F) (N := N) (S := S) (R := R) v ∈ TLCl _ 2 :=
  Form.any_mem _ fun φ hφ => by
    obtain ⟨x, -, rfl⟩ := List.mem_map.mp hφ
    exact styleQueryFormG_mem _ _ _ _ _

end Fragment

section Sat

variable {w : List (StyleTok F N S V R)} {j : ℕ}

theorem sat_styleAskEndG (f : Fin F) (t : Fin N) (σ : Fin S) (hj : 0 < j) (hjw : j ≤ w.length) :
    (styleAskEndG f t σ).sat (grams 4 id w) j = (styleAskEnd f t σ).sat w j := by
  apply Bool.eq_iff_iff.mpr
  simp only [styleAskEndG, styleAskEnd, Form.sat_all, FormP.sat_all, List.mem_cons,
    List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq, sat_symAt _ _ hj hjw,
    FormP.sat_shift_sym _ _ w hj, FormP.sat, decide_eq_true_eq]
  simp [hj]

theorem sat_styleAnswerHereG (f : Fin F) (t : Fin N) (σ : Fin S) (v : Fin V) (hj : 0 < j)
    (hjw : j ≤ w.length) :
    (styleAnswerHereG f t σ v).sat (grams 4 id w) j = (styleAnswerHere f t σ v).sat w j := by
  apply Bool.eq_iff_iff.mpr
  simp only [styleAnswerHereG, styleAnswerHere, Form.sat_all, FormP.sat_all, List.mem_cons,
    List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq, sat_symAt _ _ hj hjw,
    FormP.sat_shift_sym _ _ w hj, FormP.sat, decide_eq_true_eq]
  simp [hj]

theorem sat_styleNeedleG (f : Fin F) (t : Fin N) (σ : Fin S) (r : ℕ) (v : Fin V) (hj : 0 < j)
    (hjw : j ≤ w.length) :
    (styleNeedleG f t σ r v).sat (grams 4 id w) j = (styleNeedle f t σ r v).sat w j := by
  have hc : (Term.countL (styleAskEndG f t σ)).val (grams 4 id w) j =
      (TermP.countL (styleAskEnd f t σ)).val w j :=
    val_countL_grams hjw fun i hi hiw => sat_styleAskEndG f t σ hi hiw
  have hk : (constTerm r : Term (SGram F N S V R)).val (grams 4 id w) j =
      (TermP.addConst r .one).val w j := by
    rw [val_constTerm, TermP.val_addConst]
    rfl
  simp only [styleNeedleG, styleNeedle, Form.sat, FormP.sat, sat_styleAnswerHereG f t σ v hj hjw,
    hc, hk]

theorem sat_styleQueryFormG (f : Fin F) (t : Fin N) (σ : Fin S) (r : Fin R) (v : Fin V)
    (hj : 0 < j) (hjw : j ≤ w.length) :
    (styleQueryFormG f t σ r v).sat (grams 4 id w) j = (styleQueryForm f t σ r v).sat w j := by
  have hc : (Term.countL (styleNeedleG f t σ r v)).val (grams 4 id w) j =
      (TermP.countL (styleNeedle f t σ r v)).val w j :=
    val_countL_grams hjw fun i hi hiw => sat_styleNeedleG f t σ r v hi hiw
  have h0 : (symAt 0 (.query f t σ r) : Form (SGram F N S V R)).sat (grams 4 id w) j =
      (FormP.sym (.query f t σ r)).sat w j := by
    apply Bool.eq_iff_iff.mpr
    rw [sat_symAt _ _ hj hjw]
    simp [FormP.sat, hj]
  have h1 : (Term.one : Term (SGram F N S V R)).val (grams 4 id w) j = 1 := rfl
  have h1' : (TermP.one : TermP (StyleTok F N S V R)).val w j = 1 := rfl
  simp only [styleQueryFormG, styleQueryForm, Form.sat, FormP.sat, h0, hc, h1, h1']

theorem sat_styleAnswerG (v : Fin V) (hj : 0 < j) (hjw : j ≤ w.length) :
    (styleAnswerG (R := R) v).sat (grams 4 id w) j = (styleAnswerForm v).sat w j := by
  apply Bool.eq_iff_iff.mpr
  simp only [styleAnswerG, styleAnswerForm, Form.sat_any, FormP.sat_any, List.mem_map,
    exists_exists_and_eq_and, sat_styleQueryFormG _ _ _ _ _ hj hjw]

/-- The hypotheses of the `sat_…G` theorems hold in the conversation "`0` about `0` in the style
`0`", answer `0`, query: at the style, at the answer and at the query. -/
example := sat_styleAskEndG (w := [StyleTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .style (0 : Fin 1), .val (0 : Fin 1), .query (R := 1) 0 0 0 0]) (j := 4) 0 0 0 (by norm_num)
  (by norm_num)

example := sat_styleAnswerHereG (w := [StyleTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .style (0 : Fin 1), .val (0 : Fin 1), .query (R := 1) 0 0 0 0]) (j := 5) 0 0 0 0
  (by norm_num) (by norm_num)

example := sat_styleNeedleG (w := [StyleTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .style (0 : Fin 1), .val (0 : Fin 1), .query (R := 1) 0 0 0 0]) (j := 5) 0 0 0 0 0
  (by norm_num) (by norm_num)

example := sat_styleQueryFormG (w := [StyleTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .style (0 : Fin 1), .val (0 : Fin 1), .query (R := 1) 0 0 0 0]) (j := 6) 0 0 0 0 0
  (by norm_num) le_rfl

example := sat_styleAnswerG (w := [StyleTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .style (0 : Fin 1), .val (0 : Fin 1), .query (R := 1) 0 0 0 0]) (j := 6) 0 (by norm_num) le_rfl

/-- **The answer formula holds of the grams of `w` iff `w` asks for `v`.** -/
theorem mem_lang_styleAnswerG (v : Fin V) (w : List (StyleTok F N S V R)) :
    grams 4 id w ∈ (styleAnswerG v).lang ↔ StyleAnswer w v := by
  rcases Nat.eq_zero_or_pos w.length with h0 | hpos
  · obtain rfl : w = [] := List.eq_nil_of_length_eq_zero h0
    have hg : grams 4 id ([] : List (StyleTok F N S V R)) = [] := by simp [grams]
    refine ⟨fun h => ?_, fun ⟨f, t, σ, r, hl, _⟩ => by simp at hl⟩
    change (styleAnswerG v).sat (grams 4 id []) (grams 4 id []).length = true at h
    rw [hg, styleAnswerG, Form.sat_any] at h
    obtain ⟨φ, hφ, hs⟩ := h
    obtain ⟨x, -, rfl⟩ := List.mem_map.mp hφ
    simp [styleQueryFormG, Form.sat, sat_symAt_nil] at hs
  · rw [← models_styleAnswerForm v w]
    change (styleAnswerG v).sat (grams 4 id w) (grams 4 id w).length = true ↔
      (styleAnswerForm v).sat w w.length = true
    rw [length_grams, sat_styleAnswerG v hpos le_rfl]

end Sat

end Transformer.MRCR
