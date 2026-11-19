module

public import Mathlib.Data.Nat.Count
public import Mathlib.Tactic.DeriveFintype
public import Transformer.CRASP.LatestRecall
public import Transformer.CRASP.Suffix

/-!
# Asks spelled by words need positions

Source: MRCR of arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37, whose keys are a topic and
a writing format; MRCR-V2 of arXiv:2507.06261v6, Appendix, line 75 of
`8_appendix/1_evaluation_additional_details.tex`, spells its keys in words: "write a poem about
penguins in an archaic style", "write a poem about penguins". The rounded transformers and the
logic `TL[◁#]` are those of arXiv:2506.16055v3.

`Transformer.MRCR.Ordinal` gives an ask and its answer one token, and two layers then return the
needle with the desired index at every length. Here an ask is the phrase "`f` about `t`" of three
tokens (`PhraseTok`, `AskAt`), the token after it answers it, and the conversation closes with a
query for the answer to the ask of a format and a topic with `r` such asks before it
(`PhraseAnswer`). No future-masked rounded transformer, which has no positional encoding, accepts
exactly the conversations that `v` answers, at every length, whatever its depth, width and
precision (`not_phraseAnswers`).

Take the words of a string of bits, `b` the format word and `a` the word `about`, and close them
by a topic, an answer and the query for the first ask of that format and topic (`bitInstance`).
The only ask can end at the topic, and it is there iff the bits end in `ba`
(`phraseAnswer_bitInstance_iff`). A transformer that answered every conversation would, with the
fixed suffix moved into its formula (`CRASP.DefinableL.append`) and the words read through the
bits (`CRASP.RTfr.comap`), give a formula of `TL[◁#]` over the bits that tells `u·aba` from
`u·baa`, against `CRASP.exists_models_iff_secondLast`.

The conversations that defeat it are not templated: the text before the ask is a run of format
words and `about`, where MRCR alternates templated asks with answers. That templated conversations
defeat it as well is not proved here. That RoPE and sinusoidal encodings at rational angles, which
round to `TL[◁#, MOD]`, fail too is `Transformer.MRCR.PhraseKeyRope`.
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP

variable {F N V R : ℕ}

/-- The tokens of a conversation whose asks are spelled by words: a format word (`poem`), the
word `about`, a topic word (`penguins`), an answer, and the query that closes the conversation,
for the answer to the ask "`f` about `t`" with `r` such asks before it. -/
inductive PhraseTok (F N V R : ℕ)
  | fmt (f : Fin F)
  | about
  | topic (t : Fin N)
  | val (v : Fin V)
  | query (f : Fin F) (t : Fin N) (r : Fin R)
  deriving DecidableEq, Fintype

/-- The ask "`f` about `t`" is spelled at the indices `i`, `i + 1`, `i + 2` of `w`. -/
def AskAt (w : List (PhraseTok F N V R)) (f : Fin F) (t : Fin N) (i : ℕ) : Prop :=
  w[i]? = some (.fmt f) ∧ w[i + 1]? = some .about ∧ w[i + 2]? = some (.topic t)

instance (w : List (PhraseTok F N V R)) (f : Fin F) (t : Fin N) : DecidablePred (AskAt w f t) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _ ∧ _))

/-- **The answer to the query that closes `w`** (MRCR with asks spelled by words): `w` ends with
the query for the ask "`f` about `t`" with `r` such asks before it, `w` spells that ask, and `v`
answers it, as the token after it. -/
def PhraseAnswer (w : List (PhraseTok F N V R)) (v : Fin V) : Prop :=
  ∃ f t r, w.getLast? = some (.query f t r) ∧
    ∃ i, AskAt w f t i ∧ Nat.count (AskAt w f t) i = r ∧ w[i + 3]? = some (.val v)

/-- The word of a bit: `b` is the format word `f`, `a` is `about`. -/
def bitWord (f : Fin F) : Bool → PhraseTok F N V R
  | true => .fmt f
  | false => .about

/-- The conversation of a string of bits: its words, the topic `t`, the answer `v`, and the query
for the ask "`f` about `t`" with `r` such asks before it. -/
def bitInstance (f : Fin F) (t : Fin N) (v : Fin V) (r : Fin R) (w : List Bool) :
    List (PhraseTok F N V R) :=
  w.map (bitWord f) ++ [.topic t, .val v, .query f t r]

section Instance

variable {f : Fin F} {t : Fin N} {v : Fin V} {r : Fin R} {w : List Bool} {j : ℕ}

theorem getElem?_bitInstance_of_lt (h : j < w.length) :
    (bitInstance f t v r w)[j]? = some (bitWord f w[j]) := by
  rw [bitInstance, List.getElem?_append_left (by simpa using h), List.getElem?_map,
    List.getElem?_eq_getElem h, Option.map_some]

theorem getElem?_bitInstance_of_le (h : w.length ≤ j) :
    (bitInstance f t v r w)[j]? = [PhraseTok.topic t, .val v, .query f t r][j - w.length]? := by
  rw [bitInstance, List.getElem?_append_right (by simpa using h), List.length_map]

/-- A topic stands only after the bits. -/
theorem eq_length_of_getElem?_bitInstance {t' : Fin N}
    (h : (bitInstance f t v r w)[j]? = some (.topic t')) : j = w.length := by
  rcases lt_or_ge j w.length with hj | hj
  · rw [getElem?_bitInstance_of_lt hj, Option.some.injEq] at h
    exact absurd h (by cases w[j] <;> simp [bitWord])
  · rw [getElem?_bitInstance_of_le hj] at h
    rcases hm : j - w.length with _ | _ | _ | m <;> simp [hm] at h
    omega

end Instance

/-- The hypotheses of `getElem?_bitInstance_of_lt`, `getElem?_bitInstance_of_le` and
`eq_length_of_getElem?_bitInstance` are satisfiable: the conversation of `b` holds the format
word at `0` and the topic at `1`. -/
example : (bitInstance (0 : Fin 1) (0 : Fin 1) (0 : Fin 1) (0 : Fin 1) [true])[0]? =
      some (bitWord 0 [true][0]) ∧
    (bitInstance (0 : Fin 1) (0 : Fin 1) (0 : Fin 1) (0 : Fin 1) [true])[1]? =
      [PhraseTok.topic 0, .val 0, .query 0 0 0][1 - [true].length]? ∧ 1 = [true].length :=
  ⟨getElem?_bitInstance_of_lt (by decide), getElem?_bitInstance_of_le le_rfl,
    eq_length_of_getElem?_bitInstance (f := (0 : Fin 1)) (t := (0 : Fin 1)) (v := (0 : Fin 1))
      (r := (0 : Fin 1)) (w := [true]) (j := 1) (t' := 0) rfl⟩

/-- **Over the conversations of bits the answer is `Σ*ba`.**  The only ask can end at the topic,
so the conversation spells it iff the bits end in `ba`; it is then the first, answered by `v`. -/
theorem phraseAnswer_bitInstance_iff [NeZero R] (f : Fin F) (t : Fin N) (v : Fin V)
    (w : List Bool) :
    PhraseAnswer (bitInstance f t v (0 : Fin R) w) v ↔ ∃ u, w = u ++ [true, false] := by
  have hl : (bitInstance f t v (0 : Fin R) w).getLast? = some (.query f t 0) := by
    simp [bitInstance]
  constructor
  · rintro ⟨f', t', r', hq, i, ⟨h₀, h₁, h₂⟩, -, -⟩
    rw [hl, Option.some.injEq, PhraseTok.query.injEq] at hq
    obtain ⟨rfl, rfl, -⟩ := hq
    have hi := eq_length_of_getElem?_bitInstance h₂
    rw [getElem?_bitInstance_of_lt (by omega)] at h₀ h₁
    refine ⟨w.take i, (eq_take_append_pair (m := i) hi.symm).trans ?_⟩
    revert h₀ h₁
    cases w[i] <;> cases w[i + 1] <;> simp [bitWord]
  · rintro ⟨u, rfl⟩
    refine ⟨f, t, 0, hl, u.length, ⟨?_, ?_, ?_⟩, ?_, ?_⟩
    · rw [getElem?_bitInstance_of_lt (by simp)]
      simp [bitWord]
    · rw [getElem?_bitInstance_of_lt (by simp)]
      simp [bitWord]
    · rw [getElem?_bitInstance_of_le (by simp)]
      simp
    · rw [Fin.val_zero]
      refine Nat.count_of_forall_not fun j hj ⟨_, _, h₂⟩ => ?_
      have := eq_length_of_getElem?_bitInstance h₂
      simp at this
      omega
    · rw [getElem?_bitInstance_of_le (by simp)]
      simp

/-- The hypothesis of `phraseAnswer_bitInstance_iff` is satisfiable: "`f` about `t`", answered,
then queried. -/
example : PhraseAnswer (bitInstance (0 : Fin 1) (0 : Fin 1) (0 : Fin 1) (0 : Fin 1) [true, false])
    0 :=
  (phraseAnswer_bitInstance_iff 0 0 0 [true, false]).2 ⟨[], rfl⟩

/-- **No future-masked rounded transformer answers MRCR with asks spelled by words**: none
accepts exactly the conversations that `v` answers, at every length, whatever its depth, width
and precision. The strings of bits whose conversations (`bitInstance`) it accepts would form a
language of `TL[◁#]` (`thm:rtfr_to_TLCl` and `thm:transformer_equivalence` of
arXiv:2506.16055v3, with `CRASP.DefinableL.append` and `CRASP.RTfr.comap`). They are those
ending in `ba` (`phraseAnswer_bitInstance_iff`), which holds of `u·aba` and fails of `u·baa` for
every `u`, against `CRASP.exists_models_iff_secondLast`. -/
theorem not_phraseAnswers [NeZero F] [NeZero N] [NeZero R] {p s d k : ℕ} (v : Fin V)
    (T : RTfr (Option (PhraseTok F N V R)) p s d k) :
    ¬ ∀ w, T.Accepts (bos w) ↔ PhraseAnswer w v := by
  intro hT
  obtain ⟨φ, hφ, hL⟩ := exists_mem_TLCl_of_rtfr T
  obtain ⟨p', s', d', T', hT'⟩ := (definableL_iff_recognizes _ k).mp
    (DefinableL.append [.topic 0, .val v, .query 0 0 0] ⟨φ, hφ, hL⟩)
  obtain ⟨χ, hχ, hχL⟩ := exists_mem_TLCl_of_rtfr (T'.comap (Option.map (bitWord 0)))
  have key (w : List Bool) : χ.models w ↔ ∃ u, w = u ++ [true, false] := by
    rw [← phraseAnswer_bitInstance_iff (R := R) (0 : Fin F) (0 : Fin N) v w, ← hT]
    exact (Set.ext_iff.1 hχL w).trans ((RTfr.accepts_comap_bos T' _ w).trans (hT' _))
  obtain ⟨u, hu⟩ := exists_models_iff_secondLast k χ hχ
  rw [key, key] at hu
  obtain ⟨u', hu'⟩ := hu.1 ⟨u ++ [false], by simp⟩
  exact append_not_mem_secondLast u ⟨u', false, hu'⟩

/-- The hypotheses of `not_phraseAnswers` are satisfiable: one format, topic, answer and
ordinal. -/
example {p s d k : ℕ} (T : RTfr (Option (PhraseTok 1 1 1 1)) p s d k) :
    ¬ ∀ w, T.Accepts (bos w) ↔ PhraseAnswer w 0 :=
  not_phraseAnswers 0 T

end Transformer.MRCR
