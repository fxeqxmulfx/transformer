module

public import Mathlib.Data.List.ProdSigma
public import Transformer.CRASP.LeftQuotientCorrect
public import Transformer.CRASP.PreviousPredicates
public import Transformer.MRCR.PhraseKey

/-!
# A formula of `TL[◁#, Y]` answers spelled asks

In no paper. Source: MRCR of arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37, and MRCR-V2
of arXiv:2507.06261v6, line 75 of `8_appendix/1_evaluation_additional_details.tex`, as in
`Transformer.MRCR.PhraseKey`. `TL[◁#, Y]` is `TL[◁#]` with the operator `Y`, which reads the
previous position: arXiv:2506.16055v3, Appendix F (`app:tlclpos`), where every rounded transformer
with ALiBi rounds into it (`thm:rtfr_to_TLCly`, `Transformer.CRASP.exists_mem_TLClY_of_alibi`).

No formula of `TL[◁#, MOD]` answers the query of a conversation whose asks are spelled by words
(`Transformer.MRCR.PhraseKey`, `Transformer.MRCR.PhraseKeyRope`,
`Transformer.MRCR.PhraseKeyTemplatedLimits`). One of `TL[◁#, Y]` of depth two does, on every
string (`definableY_phraseAnswer`). `Y` reads the words of an ask at its topic (`askEnd`) and at
the answer after it (`answerHere`); a count of depth one tells that `r` such asks stand before
that ask (`needle`); a count of depth two finds the needle from the query (`queryForm`). Over
grams of four tokens it is a formula of `TL[◁#]`, which rounded transformers with ALiBi, KERPLE
or SSMax run (`Transformer.MRCR.PhraseKeyGrams`, `Transformer.MRCR.PhraseKeyLogits`).
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP

variable {F N V R : ℕ}

/-- `Q_t ∧ Y Q_about ∧ Y² Q_f`: an ask "`f` about `t`" ends here, at its topic. -/
def askEnd (f : Fin F) (t : Fin N) : FormP (PhraseTok F N V R) :=
  FormP.all [.sym (.topic t), FormP.shift 1 (.sym .about), FormP.shift 2 (.sym (.fmt f))]

/-- `Q_v ∧ Y Q_t ∧ Y² Q_about ∧ Y³ Q_f`: `v` stands here, after an ask "`f` about `t`". -/
def answerHere (f : Fin F) (t : Fin N) (v : Fin V) : FormP (PhraseTok F N V R) :=
  FormP.all [.sym (.val v), FormP.shift 1 (.sym (.topic t)), FormP.shift 2 (.sym .about),
    FormP.shift 3 (.sym (.fmt f))]

/-- **The needle** `answerHere f t v ∧ #[askEnd f t] = r + 1`: `v` answers here the ask
"`f` about `t`" with `r` such asks before it; the count up to here counts that ask too. -/
def needle (f : Fin F) (t : Fin N) (r : ℕ) (v : Fin V) : FormP (PhraseTok F N V R) :=
  .and (answerHere f t v)
    (.and (.neg (.lt (.countL (askEnd f t)) (TermP.addConst r .one)))
      (.neg (.lt (TermP.addConst r .one) (.countL (askEnd f t)))))

/-- `Q_{f,t,r} ∧ #[needle f t r v] ≥ 1`: the query for the ask "`f` about `t`" with `r` such asks
before it stands here, and `v` answers that ask. -/
def queryForm (f : Fin F) (t : Fin N) (r : Fin R) (v : Fin V) : FormP (PhraseTok F N V R) :=
  .and (.sym (.query f t r)) (.neg (.lt (.countL (needle f t r v)) .one))

/-- **The formula of depth two**: one of the `queryForm`. -/
def phraseAnswerForm (v : Fin V) : FormP (PhraseTok F N V R) :=
  FormP.any ((List.finRange F ×ˢ (List.finRange N ×ˢ List.finRange R)).map
    fun x => queryForm x.1 x.2.1 x.2.2 v)

section Fragment

theorem askEnd_mem (f : Fin F) (t : Fin N) : askEnd (V := V) (R := R) f t ∈ TLClY _ 0 := by
  refine FormP.all_mem_Y _ fun φ hφ => ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hφ
  rcases hφ with rfl | rfl | rfl
  · exact ⟨rfl, le_rfl⟩
  all_goals exact FormP.shift_mem_Y ⟨rfl, le_rfl⟩ _

theorem answerHere_mem (f : Fin F) (t : Fin N) (v : Fin V) :
    answerHere (R := R) f t v ∈ TLClY _ 0 := by
  refine FormP.all_mem_Y _ fun φ hφ => ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hφ
  rcases hφ with rfl | rfl | rfl | rfl
  · exact ⟨rfl, le_rfl⟩
  all_goals exact FormP.shift_mem_Y ⟨rfl, le_rfl⟩ _

theorem needle_mem (f : Fin F) (t : Fin N) (r : ℕ) (v : Fin V) :
    needle (R := R) f t r v ∈ TLClY _ 1 := by
  obtain ⟨hm, hd⟩ := askEnd_mem (V := V) (R := R) f t
  have ho : (TermP.addConst r (.one : TermP (PhraseTok F N V R))).modFree = true := by
    rw [TermP.modFree_addConst]; rfl
  have ho' : (TermP.addConst r (.one : TermP (PhraseTok F N V R))).depth ≤ 1 := by
    rw [TermP.depth_addConst]; exact Nat.zero_le _
  have hc' : (TermP.countL (askEnd (V := V) (R := R) f t)).depth ≤ 1 := by
    simp only [TermP.depth]; omega
  exact FormP.and_mem_Y (FormP.mem_mono_Y (answerHere_mem f t v) (Nat.zero_le 1))
    (FormP.and_mem_Y (FormP.neg_mem_Y ⟨by simp [FormP.modFree, TermP.modFree, hm, ho],
      max_le hc' ho'⟩) (FormP.neg_mem_Y ⟨by simp [FormP.modFree, TermP.modFree, hm, ho],
      max_le ho' hc'⟩))

theorem queryForm_mem (f : Fin F) (t : Fin N) (r : Fin R) (v : Fin V) :
    queryForm f t r v ∈ TLClY _ 2 := by
  obtain ⟨hm, hd⟩ := needle_mem (R := R) f t r v
  refine FormP.and_mem_Y ⟨rfl, Nat.zero_le _⟩ (FormP.neg_mem_Y ⟨?_, ?_⟩)
  · simp [FormP.modFree, TermP.modFree, hm]
  · change max ((needle (R := R) f t r v).depth + 1) 0 ≤ 2
    omega

/-- **The answer formula has depth two and no `MOD`.** -/
theorem phraseAnswerForm_mem (v : Fin V) : phraseAnswerForm (F := F) (N := N) (R := R) v ∈
    TLClY _ 2 :=
  FormP.any_mem_Y _ fun φ hφ => by
    obtain ⟨x, -, rfl⟩ := List.mem_map.mp hφ
    exact queryForm_mem _ _ _ _

end Fragment


section Phrase

variable {w : List (PhraseTok F N V R)} {f : Fin F} {t : Fin N} {v : Fin V}

/-- `askEnd f t` holds at the position `j` iff an ask "`f` about `t`" starts at the index
`j - 3`. -/
theorem sat_askEnd {j : ℕ} (hj : 0 < j) :
    (askEnd f t).sat w j = true ↔ 2 < j ∧ AskAt w f t (j - 3) := by
  simp only [askEnd, FormP.sat_all, List.mem_cons, List.mem_nil_iff, or_false,
    forall_eq_or_imp, forall_eq, FormP.sat_shift_sym _ _ w hj, FormP.sat, decide_eq_true_eq,
    AskAt]
  constructor
  · rintro ⟨h0, ⟨-, h1⟩, ⟨h, h2⟩⟩
    refine ⟨h, ?_, ?_, ?_⟩
    · rwa [show j - 3 = j - 2 - 1 by omega]
    · rwa [show j - 3 + 1 = j - 1 - 1 by omega]
    · rwa [show j - 3 + 2 = j - 1 by omega]
  · rintro ⟨h, h2, h1, h0⟩
    refine ⟨?_, ⟨by omega, ?_⟩, ⟨h, ?_⟩⟩
    · rwa [show j - 1 = j - 3 + 2 by omega]
    · rwa [show j - 1 - 1 = j - 3 + 1 by omega]
    · rwa [show j - 2 - 1 = j - 3 by omega]

/-- `answerHere f t v` holds at the position `j` iff an ask "`f` about `t`" starts at the index
`j - 4` and `v` stands after it. -/
theorem sat_answerHere {j : ℕ} (hj : 0 < j) :
    (answerHere f t v).sat w j = true ↔
      3 < j ∧ AskAt w f t (j - 4) ∧ w[j - 1]? = some (.val v) := by
  simp only [answerHere, FormP.sat_all, List.mem_cons, List.mem_nil_iff, or_false,
    forall_eq_or_imp, forall_eq, FormP.sat_shift_sym _ _ w hj, FormP.sat, decide_eq_true_eq,
    AskAt]
  constructor
  · rintro ⟨h0, ⟨-, h1⟩, ⟨-, h2⟩, ⟨h, h3⟩⟩
    refine ⟨h, ⟨?_, ?_, ?_⟩, h0⟩
    · rwa [show j - 4 = j - 3 - 1 by omega]
    · rwa [show j - 4 + 1 = j - 2 - 1 by omega]
    · rwa [show j - 4 + 2 = j - 1 - 1 by omega]
  · rintro ⟨h, ⟨h3, h2, h1⟩, h0⟩
    refine ⟨h0, ⟨by omega, ?_⟩, ⟨by omega, ?_⟩, ⟨h, ?_⟩⟩
    · rwa [show j - 1 - 1 = j - 4 + 2 by omega]
    · rwa [show j - 2 - 1 = j - 4 + 1 by omega]
    · rwa [show j - 3 - 1 = j - 4 by omega]

/-- The hypotheses of `sat_askEnd` and `sat_answerHere` hold at the topic and at the answer of
the ask "`0` about `0`" answered by `0`. -/
example := sat_askEnd (w := [PhraseTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .val (0 : Fin 1), .query (R := 1) 0 0 0]) (f := 0) (t := 0) (j := 3) (by norm_num)

example := sat_answerHere (w := [PhraseTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .val (0 : Fin 1), .query (R := 1) 0 0 0]) (f := 0) (t := 0) (v := 0) (j := 4) (by norm_num)

end Phrase

end Transformer.MRCR
