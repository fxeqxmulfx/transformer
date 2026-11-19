module

public import Transformer.CRASP.LeftQuotientCorrect
public import Transformer.CRASP.PositionalEmbedding
public import Transformer.MRCR.PhraseKeyRope
public import Transformer.MRCR.PhraseKeyTemplated

/-!
# Templated conversations need more than RoPE

In no paper. Source: MRCR of arXiv:2409.12640v2, `evals.tex`, line 127, and MRCR-V2 of
arXiv:2507.06261v6, line 75 of `8_appendix/1_evaluation_additional_details.tex`, as in
`Transformer.MRCR.PhraseKeyTemplated`. The rounded transformers are those of arXiv:2506.16055v3,
which rounds them without positions to `TL[◁#]` (`thm:rtfr_to_TLCl`) and with RoPE or sinusoidal
encodings at rational angles to `TL[◁#, MOD]` (Appendix F, `thm:rtfr_to_TLClmod`,
`thm:rtfr_eq_tlclmod`).

`Transformer.MRCR.PhraseKey` and `Transformer.MRCR.PhraseKeyRope` defeat these transformers with
conversations that are not templated. Templated conversations whose query is answered defeat them
as well, given two formats, two topics and two answers: no formula of `TL[◁#, MOD]` accepts
exactly those whose query `v` answers (`not_phraseAnswers_templated_mod`), and so no rounded
transformer without positions or with RoPE or sinusoidal encodings at rational angles does,
whatever its depth, width and precision (`not_phraseAnswers_templated`,
`not_phraseAnswers_templated_rope`, `not_phraseAnswers_templated_sinusoidal`).

A templated conversation starts with an ask, so the blocks of `Transformer.MRCR.PhraseKeyTemplated`
come after "`f₁` about", which moves into the formula (`CRASP.FormP.leftQuot`, which keeps it free
of `Y`). For some `u` the formula then accepts both or neither of the conversations of `u·ab` and
`u·ba` (`CRASP.FormP.exists_models_iff_blocks`); both are templated and answered, by `v` and by
`v'` (`flatMap_turnBlock_append_pair`, `phraseAnswer_closing_iff`).
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP

variable {F N V R : ℕ}

/-- After "`f₁` about", the blocks of `u·c'c` and the close are a word without the topic `t₀`,
then "`f_c` about" and the close. -/
theorem flatMap_turnBlock_append_pair [NeZero R] {f₀ f₁ : Fin F} {t₀ t₁ : Fin N} (ht : t₀ ≠ t₁)
    (v v' : Fin V) (M : ℕ) (u : List Bool) (c' c : Bool) :
    ∃ A : List (PhraseTok F N V R), .topic t₀ ∉ A ∧
      [.fmt f₁, .about] ++ ((u ++ [c', c]).flatMap (turnBlock f₀ f₁ t₁ v M) ++
          closing f₀ t₀ v v') =
        A ++ [.fmt (if c then f₀ else f₁), .about] ++ closing f₀ t₀ v v' := by
  refine ⟨[.fmt f₁, .about] ++ (u ++ [c']).flatMap (turnBlock f₀ f₁ t₁ v M) ++
    (.topic t₁ :: List.replicate (M - 3) (.val v)), ?_, ?_⟩
  · simp [turnBlock, ht]
  · simp [turnBlock]

/-- The hypothesis of `flatMap_turnBlock_append_pair` is satisfiable: two topics. -/
example : ∃ A : List (PhraseTok 2 2 2 1), .topic 0 ∉ A ∧
    [.fmt 1, .about] ++ (([] ++ [false, true]).flatMap (turnBlock 0 1 1 0 4) ++
        closing 0 0 0 1) =
      A ++ [.fmt (if true then 0 else 1), .about] ++ closing 0 0 0 1 :=
  flatMap_turnBlock_append_pair (by decide) 0 1 4 [] false true

/-- **No formula of `TL[◁#, MOD]` answers templated MRCR with asks spelled by words**: none
accepts exactly the templated conversations, among those whose query is answered, that `v`
answers. After "`f₁` about" it is a formula free of `Y` (`CRASP.FormP.leftQuot`); for `M` a
multiple of its positive moduli it confuses the blocks of `u·ab` and `u·ba`, closed
(`CRASP.FormP.exists_models_iff_blocks`), whose conversations are answered by `v` and `v'`. -/
theorem not_phraseAnswers_templated_mod [NeZero R] (hF : 1 < F) (hN : 1 < N) (hV : 1 < V)
    (v : Fin V) {φ : FormP (PhraseTok F N V R)} (hp : φ.prevFree = true) :
    ¬ ∀ w, Templated w → (∃ u, PhraseAnswer w u) → (φ.models w ↔ PhraseAnswer w v) := by
  intro hφ
  have : Nontrivial (Fin F) := Fin.nontrivial_iff_two_le.2 hF
  have : Nontrivial (Fin N) := Fin.nontrivial_iff_two_le.2 hN
  have : Nontrivial (Fin V) := Fin.nontrivial_iff_two_le.2 hV
  obtain ⟨f₀, f₁, hf⟩ := exists_pair_ne (Fin F)
  obtain ⟨t₀, t₁, ht⟩ := exists_pair_ne (Fin N)
  obtain ⟨v', hv⟩ := exists_ne v
  have hψ := φ.prevFree_leftQuot [.fmt f₁, .about]
  rw [hp] at hψ
  obtain ⟨M, hM, hd⟩ : ∃ M, 4 ≤ M ∧ (φ.leftQuot [.fmt f₁, .about]).modLcm ∣ M :=
    ⟨4 * (φ.leftQuot [.fmt f₁, .about]).modLcm,
      by have := (φ.leftQuot [.fmt f₁, .about]).modLcm_pos; omega, dvd_mul_left _ _⟩
  obtain ⟨u, hu⟩ := FormP.exists_models_iff_blocks hψ (by omega) hd
    (length_turnBlock (f₀ := f₀) (f₁ := f₁) (t₁ := t₁) (v := v) (by omega))
    (s := closing f₀ t₀ v v') (List.cons_ne_nil _ _)
  have hs : ∀ x : List Bool,
      x.flatMap (turnBlock f₀ f₁ t₁ v M) ++ closing (R := R) f₀ t₀ v v' ≠ [] :=
    fun x => by simp [closing]
  rw [FormP.models_leftQuot _ (hs _), FormP.models_leftQuot _ (hs _)] at hu
  have key (c' c : Bool) :
      φ.models ([.fmt f₁, .about] ++ ((u ++ [c', c]).flatMap (turnBlock f₀ f₁ t₁ v M) ++
        closing f₀ t₀ v v')) ↔ v = if c then v else v' := by
    obtain ⟨A, hA, he⟩ := flatMap_turnBlock_append_pair (R := R) ht v v' M u c' c (f₀ := f₀)
      (f₁ := f₁)
    have htemp := templated_turnBlocks (R := R) (f₀ := f₀) (f₁ := f₁) (t₀ := t₀) (t₁ := t₁)
      (v := v) (v' := v') hM (u ++ [c', c])
    rw [List.append_assoc, he] at htemp
    rw [he, hφ _ htemp ⟨_, (phraseAnswer_closing_iff hA hf c _).2 rfl⟩,
      phraseAnswer_closing_iff hA hf c v]
  rw [key, key] at hu
  simp only [Bool.false_eq_true, ↓reduceIte, true_iff] at hu
  exact hv hu.symm

/-- The hypothesis of `not_phraseAnswers_templated_mod` is satisfiable: the test `Q_about`. -/
example : ¬ ∀ w, Templated w → (∃ u, PhraseAnswer w u) →
    ((FormP.sym (.about : PhraseTok 2 2 2 1)).models w ↔ PhraseAnswer w 0) :=
  not_phraseAnswers_templated_mod one_lt_two one_lt_two one_lt_two 0 rfl

/-- **No future-masked rounded transformer answers templated MRCR with asks spelled by words**:
none accepts exactly the templated conversations, among those whose query is answered, that `v`
answers, whatever its depth, width and precision. It rounds to a formula of `TL[◁#]`
(`thm:rtfr_to_TLCl` of arXiv:2506.16055v3, `CRASP.exists_mem_TLCl_of_rtfr`), which is a formula
of `TL[◁#, MOD]` (`CRASP.DefinableL.exists_formP`), against `not_phraseAnswers_templated_mod`. -/
theorem not_phraseAnswers_templated [NeZero R] (hF : 1 < F) (hN : 1 < N) (hV : 1 < V)
    {p s d k : ℕ} (v : Fin V) (T : RTfr (Option (PhraseTok F N V R)) p s d k) :
    ¬ ∀ w, Templated w → (∃ u, PhraseAnswer w u) → (T.Accepts (bos w) ↔ PhraseAnswer w v) :=
  fun h => by
    obtain ⟨φ, hφ, hL⟩ := exists_mem_TLCl_of_rtfr T
    obtain ⟨ψ, -, -, hp, hψ⟩ := DefinableL.exists_formP ⟨φ, hφ, hL⟩
    exact not_phraseAnswers_templated_mod hF hN hV v hp fun w hw ha =>
      (Set.ext_iff.1 hψ w).trans (h w hw ha)

/-- **No rounded transformer with RoPE at rational angles answers templated MRCR with asks
spelled by words**, whatever its depth, width and precision (`thm:rtfr_to_TLClmod` of
arXiv:2506.16055v3, `CRASP.exists_mem_TLClMod_of_rope`, and `not_phraseAnswers_templated_mod`). -/
theorem not_phraseAnswers_templated_rope [NeZero R] (hF : 1 < F) (hN : 1 < N) (hV : 1 < V)
    {p s d k : ℕ} (v : Fin V) (θ : ℕ → ℝ) (hθ : (PosEnc.rope θ).RationalAngles)
    (T : PTfr (Option (PhraseTok F N V R)) p s d k) (hT : T.pe = .rope θ) :
    ¬ ∀ w, Templated w → (∃ u, PhraseAnswer w u) → (T.Accepts (bos w) ↔ PhraseAnswer w v) :=
  fun h => by
    obtain ⟨φ, ⟨hp, -⟩, hL⟩ := exists_mem_TLClMod_of_rope θ hθ T hT
    exact not_phraseAnswers_templated_mod hF hN hV v hp fun w hw ha =>
      (Set.ext_iff.1 hL w).trans (h w hw ha)

/-- **No rounded transformer with sinusoidal encodings at rational angles answers templated MRCR
with asks spelled by words**, whatever its depth, width and precision (`thm:rtfr_eq_tlclmod` of
arXiv:2506.16055v3, `CRASP.exists_mem_TLClMod_of_sinusoidal`, and
`not_phraseAnswers_templated_mod`). -/
theorem not_phraseAnswers_templated_sinusoidal [NeZero R] (hF : 1 < F) (hN : 1 < N)
    (hV : 1 < V) {p s d k : ℕ} (v : Fin V) (θ : ℕ → ℝ)
    (hθ : (PosEnc.sinusoidal θ).RationalAngles) (T : PTfr (Option (PhraseTok F N V R)) p s d k)
    (hT : T.pe = .sinusoidal θ) :
    ¬ ∀ w, Templated w → (∃ u, PhraseAnswer w u) → (T.Accepts (bos w) ↔ PhraseAnswer w v) :=
  fun h => by
    obtain ⟨φ, ⟨hp, -⟩, hL⟩ := exists_mem_TLClMod_of_sinusoidal θ hθ T hT
    exact not_phraseAnswers_templated_mod hF hN hV v hp fun w hw ha =>
      (Set.ext_iff.1 hL w).trans (h w hw ha)

/-- The hypotheses of the three theorems above are satisfiable: the zero transformers, without
positions or at the angles `0`, over two formats, topics and answers. -/
example (pe : PosEnc) (hpe : pe = .rope (fun _ => 0) ∨ pe = .sinusoidal (fun _ => 0))
    (T : RTfr (Option (PhraseTok 2 2 2 1)) 2 0 0 0) :
    (¬ ∀ w, Templated w → (∃ u, PhraseAnswer w u) → (T.Accepts (bos w) ↔ PhraseAnswer w 0)) ∧
    ∃ T : PTfr (Option (PhraseTok 2 2 2 1)) 2 0 0 0, T.pe = pe ∧
      ¬ ∀ w, Templated w → (∃ u, PhraseAnswer w u) → (T.Accepts (bos w) ↔ PhraseAnswer w 0) := by
  refine ⟨not_phraseAnswers_templated one_lt_two one_lt_two one_lt_two 0 T, {
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := pe }, rfl, ?_⟩
  rcases hpe with rfl | rfl
  · exact not_phraseAnswers_templated_rope one_lt_two one_lt_two one_lt_two 0 _
      ⟨fun _ => 0, fun _ => by simp⟩ _ rfl
  · exact not_phraseAnswers_templated_sinusoidal one_lt_two one_lt_two one_lt_two 0 _
      ⟨fun _ => 0, fun _ => by simp⟩ _ rfl

end Transformer.MRCR
