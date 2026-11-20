module

public import Transformer.MRCR.PhraseKeyPrev

/-!
# Asks with a style: keys of depth three spelled by words

In no paper. Source: MRCR-V2 of arXiv:2507.06261v6, line 75 of
`8_appendix/1_evaluation_additional_details.tex`, which nests the dictionary of MRCR
(arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37) to depth three by a style: "write a poem
about penguins in an archaic style". `Transformer.MRCR.PhraseKey` spells keys of depth two,
"`f` about `t`". Here the style word `σ` follows the topic, and the query asks for the answer to
the ask "`f` about `t` in the style `σ`" with `r` such asks before it (`StyleAnswer`).

A formula of `TL[◁#, Y]` of depth two answers the query as `Transformer.MRCR.PhraseKeyPrev` does
for depth two, with `Y` reading one token further back: the ask ends at its style word
(`styleAskEnd`), the answer stands after it (`styleAnswerHere`), a count of depth one finds the
needle (`styleNeedle`) and a count of depth two the query (`styleQueryForm`).
`Transformer.MRCR.PhraseKeyStyleCorrect` proves that it answers, and
`Transformer.MRCR.PhraseKeyStyleLogits` compiles it into rounded transformers of six layers.
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP

variable {F N S V R : ℕ}

/-- The tokens of a conversation whose asks carry a style, as in MRCR-V2: the format word `f`,
`about`, the topic word `t`, the style word `σ`, the answer token `v`, and the query for the ask
"`f` about `t` in the style `σ`" with `r` such asks before it. -/
inductive StyleTok (F N S V R : ℕ)
  | fmt (f : Fin F)
  | about
  | topic (t : Fin N)
  | style (σ : Fin S)
  | val (v : Fin V)
  | query (f : Fin F) (t : Fin N) (σ : Fin S) (r : Fin R)
  deriving DecidableEq, Fintype

/-- The ask "`f` about `t` in the style `σ`" is spelled at the indices `i`, …, `i + 3` of `w`. -/
def StyleAskAt (w : List (StyleTok F N S V R)) (f : Fin F) (t : Fin N) (σ : Fin S) (i : ℕ) :
    Prop :=
  w[i]? = some (.fmt f) ∧ w[i + 1]? = some .about ∧ w[i + 2]? = some (.topic t) ∧
    w[i + 3]? = some (.style σ)

instance (w : List (StyleTok F N S V R)) (f : Fin F) (t : Fin N) (σ : Fin S) :
    DecidablePred (StyleAskAt w f t σ) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

/-- **The answer to the query that closes `w`** (MRCR-V2, keys of depth three spelled by words):
`w` ends with the query for the ask "`f` about `t` in the style `σ`" with `r` such asks before
it, `w` spells that ask, and `v` answers it, as the token after it. -/
def StyleAnswer (w : List (StyleTok F N S V R)) (v : Fin V) : Prop :=
  ∃ f t σ r, w.getLast? = some (.query f t σ r) ∧
    ∃ i, StyleAskAt w f t σ i ∧ Nat.count (StyleAskAt w f t σ) i = r ∧ w[i + 4]? = some (.val v)

/-- `Q_σ ∧ Y Q_t ∧ Y² Q_about ∧ Y³ Q_f`: an ask "`f` about `t` in the style `σ`" ends here. -/
def styleAskEnd (f : Fin F) (t : Fin N) (σ : Fin S) : FormP (StyleTok F N S V R) :=
  FormP.all [.sym (.style σ), FormP.shift 1 (.sym (.topic t)), FormP.shift 2 (.sym .about),
    FormP.shift 3 (.sym (.fmt f))]

/-- `Q_v ∧ Y Q_σ ∧ Y² Q_t ∧ Y³ Q_about ∧ Y⁴ Q_f`: `v` stands here, after such an ask. -/
def styleAnswerHere (f : Fin F) (t : Fin N) (σ : Fin S) (v : Fin V) :
    FormP (StyleTok F N S V R) :=
  FormP.all [.sym (.val v), FormP.shift 1 (.sym (.style σ)), FormP.shift 2 (.sym (.topic t)),
    FormP.shift 3 (.sym .about), FormP.shift 4 (.sym (.fmt f))]

/-- **The needle** `styleAnswerHere f t σ v ∧ #[styleAskEnd f t σ] = r + 1`: `v` answers here the
ask "`f` about `t` in the style `σ`" with `r` such asks before it. -/
def styleNeedle (f : Fin F) (t : Fin N) (σ : Fin S) (r : ℕ) (v : Fin V) :
    FormP (StyleTok F N S V R) :=
  .and (styleAnswerHere f t σ v)
    (.and (.neg (.lt (.countL (styleAskEnd f t σ)) (TermP.addConst r .one)))
      (.neg (.lt (TermP.addConst r .one) (.countL (styleAskEnd f t σ)))))

/-- `Q_{f,t,σ,r} ∧ #[styleNeedle f t σ r v] ≥ 1`: the query stands here, and `v` answers its
ask. -/
def styleQueryForm (f : Fin F) (t : Fin N) (σ : Fin S) (r : Fin R) (v : Fin V) :
    FormP (StyleTok F N S V R) :=
  .and (.sym (.query f t σ r)) (.neg (.lt (.countL (styleNeedle f t σ r v)) .one))

/-- **The formula of depth two**: one of the `styleQueryForm`. -/
def styleAnswerForm (v : Fin V) : FormP (StyleTok F N S V R) :=
  FormP.any ((List.finRange F ×ˢ (List.finRange N ×ˢ (List.finRange S ×ˢ List.finRange R))).map
    fun x => styleQueryForm x.1 x.2.1 x.2.2.1 x.2.2.2 v)

section Fragment

theorem styleAskEnd_mem (f : Fin F) (t : Fin N) (σ : Fin S) :
    styleAskEnd (V := V) (R := R) f t σ ∈ TLClY _ 0 := by
  refine FormP.all_mem_Y _ fun φ hφ => ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hφ
  rcases hφ with rfl | rfl | rfl | rfl
  · exact ⟨rfl, le_rfl⟩
  all_goals exact FormP.shift_mem_Y ⟨rfl, le_rfl⟩ _

theorem styleAnswerHere_mem (f : Fin F) (t : Fin N) (σ : Fin S) (v : Fin V) :
    styleAnswerHere (R := R) f t σ v ∈ TLClY _ 0 := by
  refine FormP.all_mem_Y _ fun φ hφ => ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hφ
  rcases hφ with rfl | rfl | rfl | rfl | rfl
  · exact ⟨rfl, le_rfl⟩
  all_goals exact FormP.shift_mem_Y ⟨rfl, le_rfl⟩ _

theorem styleNeedle_mem (f : Fin F) (t : Fin N) (σ : Fin S) (r : ℕ) (v : Fin V) :
    styleNeedle (R := R) f t σ r v ∈ TLClY _ 1 := by
  obtain ⟨hm, hd⟩ := styleAskEnd_mem (V := V) (R := R) f t σ
  have ho : (TermP.addConst r (.one : TermP (StyleTok F N S V R))).modFree = true := by
    rw [TermP.modFree_addConst]; rfl
  have ho' : (TermP.addConst r (.one : TermP (StyleTok F N S V R))).depth ≤ 1 := by
    rw [TermP.depth_addConst]; exact Nat.zero_le _
  have hc' : (TermP.countL (styleAskEnd (V := V) (R := R) f t σ)).depth ≤ 1 := by
    simp only [TermP.depth]; omega
  exact FormP.and_mem_Y (FormP.mem_mono_Y (styleAnswerHere_mem f t σ v) (Nat.zero_le 1))
    (FormP.and_mem_Y (FormP.neg_mem_Y ⟨by simp [FormP.modFree, TermP.modFree, hm, ho],
      max_le hc' ho'⟩) (FormP.neg_mem_Y ⟨by simp [FormP.modFree, TermP.modFree, hm, ho],
      max_le ho' hc'⟩))

theorem styleQueryForm_mem (f : Fin F) (t : Fin N) (σ : Fin S) (r : Fin R) (v : Fin V) :
    styleQueryForm f t σ r v ∈ TLClY _ 2 := by
  obtain ⟨hm, hd⟩ := styleNeedle_mem (R := R) f t σ r v
  refine FormP.and_mem_Y ⟨rfl, Nat.zero_le _⟩ (FormP.neg_mem_Y ⟨?_, ?_⟩)
  · simp [FormP.modFree, TermP.modFree, hm]
  · change max ((styleNeedle (R := R) f t σ r v).depth + 1) 0 ≤ 2
    omega

/-- **The answer formula has depth two and no `MOD`.** -/
theorem styleAnswerForm_mem (v : Fin V) :
    styleAnswerForm (F := F) (N := N) (S := S) (R := R) v ∈ TLClY _ 2 :=
  FormP.any_mem_Y _ fun φ hφ => by
    obtain ⟨x, -, rfl⟩ := List.mem_map.mp hφ
    exact styleQueryForm_mem _ _ _ _ _

end Fragment

section Phrase

variable {w : List (StyleTok F N S V R)} {f : Fin F} {t : Fin N} {σ : Fin S} {v : Fin V}

/-- `styleAskEnd f t σ` holds at the position `j` iff such an ask starts at the index `j - 4`. -/
theorem sat_styleAskEnd {j : ℕ} (hj : 0 < j) :
    (styleAskEnd f t σ).sat w j = true ↔ 3 < j ∧ StyleAskAt w f t σ (j - 4) := by
  simp only [styleAskEnd, FormP.sat_all, List.mem_cons, List.mem_nil_iff, or_false,
    forall_eq_or_imp, forall_eq, FormP.sat_shift_sym _ _ w hj, FormP.sat, decide_eq_true_eq,
    StyleAskAt]
  constructor
  · rintro ⟨h0, ⟨-, h1⟩, ⟨-, h2⟩, ⟨h, h3⟩⟩
    refine ⟨h, ?_, ?_, ?_, ?_⟩
    · rwa [show j - 4 = j - 3 - 1 by omega]
    · rwa [show j - 4 + 1 = j - 2 - 1 by omega]
    · rwa [show j - 4 + 2 = j - 1 - 1 by omega]
    · rwa [show j - 4 + 3 = j - 1 by omega]
  · rintro ⟨h, h3, h2, h1, h0⟩
    refine ⟨?_, ⟨by omega, ?_⟩, ⟨by omega, ?_⟩, ⟨h, ?_⟩⟩
    · rwa [show j - 1 = j - 4 + 3 by omega]
    · rwa [show j - 1 - 1 = j - 4 + 2 by omega]
    · rwa [show j - 2 - 1 = j - 4 + 1 by omega]
    · rwa [show j - 3 - 1 = j - 4 by omega]

/-- `styleAnswerHere f t σ v` holds at the position `j` iff such an ask starts at the index
`j - 5` and `v` stands after it. -/
theorem sat_styleAnswerHere {j : ℕ} (hj : 0 < j) :
    (styleAnswerHere f t σ v).sat w j = true ↔
      4 < j ∧ StyleAskAt w f t σ (j - 5) ∧ w[j - 1]? = some (.val v) := by
  simp only [styleAnswerHere, FormP.sat_all, List.mem_cons, List.mem_nil_iff, or_false,
    forall_eq_or_imp, forall_eq, FormP.sat_shift_sym _ _ w hj, FormP.sat, decide_eq_true_eq,
    StyleAskAt]
  constructor
  · rintro ⟨h0, ⟨-, h1⟩, ⟨-, h2⟩, ⟨-, h3⟩, ⟨h, h4⟩⟩
    refine ⟨h, ⟨?_, ?_, ?_, ?_⟩, h0⟩
    · rwa [show j - 5 = j - 4 - 1 by omega]
    · rwa [show j - 5 + 1 = j - 3 - 1 by omega]
    · rwa [show j - 5 + 2 = j - 2 - 1 by omega]
    · rwa [show j - 5 + 3 = j - 1 - 1 by omega]
  · rintro ⟨h, ⟨h4, h3, h2, h1⟩, h0⟩
    refine ⟨h0, ⟨by omega, ?_⟩, ⟨by omega, ?_⟩, ⟨by omega, ?_⟩, ⟨h, ?_⟩⟩
    · rwa [show j - 1 - 1 = j - 5 + 3 by omega]
    · rwa [show j - 2 - 1 = j - 5 + 2 by omega]
    · rwa [show j - 3 - 1 = j - 5 + 1 by omega]
    · rwa [show j - 4 - 1 = j - 5 by omega]

/-- The hypotheses of `sat_styleAskEnd` and `sat_styleAnswerHere` hold at the style and at the
answer of the ask "`0` about `0` in the style `0`" answered by `0`. -/
example := sat_styleAskEnd (w := [StyleTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .style (0 : Fin 1), .val (0 : Fin 1), .query (R := 1) 0 0 0 0]) (f := 0) (t := 0) (σ := 0)
  (j := 4) (by norm_num)

example := sat_styleAnswerHere (w := [StyleTok.fmt (0 : Fin 1), .about, .topic (0 : Fin 1),
  .style (0 : Fin 1), .val (0 : Fin 1), .query (R := 1) 0 0 0 0]) (f := 0) (t := 0) (σ := 0)
  (v := 0) (j := 5) (by norm_num)

end Phrase

end Transformer.MRCR
