module

public import Transformer.CRASP.SecondLastMod
public import Transformer.MRCR.PhraseKey

/-!
# Asks spelled by words need more than RoPE

In no paper. Source: MRCR of arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37, and
MRCR-V2 of arXiv:2507.06261v6, line 75 of `8_appendix/1_evaluation_additional_details.tex`, as in
`Transformer.MRCR.PhraseKey`. The rounded transformers with RoPE or sinusoidal encodings at
rational angles are those of arXiv:2506.16055v3, Appendix F, which rounds them to `TL[◁#, MOD]`
(`thm:rtfr_to_TLClmod`, `thm:rtfr_eq_tlclmod`).

`Transformer.MRCR.PhraseKey` proves that no rounded transformer without positions answers MRCR
with asks spelled by words. Positions read modulo a fixed number do not help: no formula of
`TL[◁#, MOD]` accepts exactly the conversations that `v` answers (`not_phraseAnswers_mod`), and so
no rounded transformer with RoPE or sinusoidal encodings at rational angles does, whatever its
depth, width and precision (`not_phraseAnswers_rope`, `not_phraseAnswers_sinusoidal`).

Let `M` be a multiple of the positive moduli of the formula. Write each bit `c` as the words of
its block `a^{M−1} c` (`CRASP.padBlock`), `b` the format word and `a` the word `about`, and close
them by `a`, a topic, an answer and the query: this is the conversation of the bits of the blocks
closed by `a` (`bitInstance_flatMap_padBlock`). For some `u` the formula accepts both or neither
of the conversations of `u·ab` and of `u·ba` (`CRASP.FormP.exists_models_iff_blocks`). Their bits
end in `ba` and in `aa`, so `v` answers the first only (`phraseAnswer_bitInstance_iff`).

As in `Transformer.MRCR.PhraseKey`, the conversations that defeat it are not templated; templated
ones whose query is answered are `Transformer.MRCR.PhraseKeyTemplatedLimits`. ALiBi, which rounds
to `TL[◁#, Y]`, is not covered: `Y` reads the previous token, and `Σ*bΣ`, the core of the
argument, is definable with it (`CRASP.exists_definableY_not_definableMod`).
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP

variable {F N V R : ℕ}

/-- The words of the blocks of `x`, closed by `a`, a topic, an answer and the query, are the
conversation of the bits of the blocks of `x` closed by `a`. -/
theorem bitInstance_flatMap_padBlock (f : Fin F) (t : Fin N) (v : Fin V) (r : Fin R) (M : ℕ)
    (x : List Bool) :
    bitInstance f t v r (x.flatMap (padBlock M) ++ [false]) =
      x.flatMap (fun c => (padBlock M c).map (bitWord f)) ++
        [.about, .topic t, .val v, .query f t r] := by
  simp [bitInstance, List.map_flatMap, bitWord]

/-- **No formula of `TL[◁#, MOD]` answers MRCR with asks spelled by words**: none accepts exactly
the conversations that `v` answers. With `M` the least common multiple of its positive moduli,
it confuses the conversations of the blocks of `u·ab` and of `u·ba`, closed by `a`
(`CRASP.FormP.exists_models_iff_blocks`, `bitInstance_flatMap_padBlock`), whose bits end in `ba`
and in `aa` (`phraseAnswer_bitInstance_iff`). -/
theorem not_phraseAnswers_mod [NeZero F] [NeZero N] [NeZero R] (v : Fin V)
    {φ : FormP (PhraseTok F N V R)} (hp : φ.prevFree = true) :
    ¬ ∀ w, φ.models w ↔ PhraseAnswer w v := by
  intro hφ
  have hM := φ.modLcm_pos
  obtain ⟨u, hu⟩ := FormP.exists_models_iff_blocks hp hM dvd_rfl
    (g := fun c => (padBlock φ.modLcm c).map (bitWord 0))
    (fun c => by rw [List.length_map, length_padBlock hM c])
    (s := [.about, .topic 0, .val v, .query 0 0 0]) (List.cons_ne_nil _ _)
  have key (x : List Bool) :
      φ.models (x.flatMap (fun c => (padBlock φ.modLcm c).map (bitWord 0)) ++
          [.about, .topic 0, .val v, .query 0 0 0]) ↔
        ∃ u', x.flatMap (padBlock φ.modLcm) ++ [false] = u' ++ [true, false] := by
    rw [hφ, ← bitInstance_flatMap_padBlock, phraseAnswer_bitInstance_iff]
  have h₁ : ∃ u', (u ++ [false, true]).flatMap (padBlock φ.modLcm) ++ [false] =
      u' ++ [true, false] :=
    ⟨u.flatMap (padBlock φ.modLcm) ++ padBlock φ.modLcm false ++
      List.replicate (φ.modLcm - 1) false, by simp [padBlock]⟩
  obtain ⟨u', hu'⟩ := (key _).1 (hu.1 ((key _).2 h₁))
  rw [show (u ++ [true, false]).flatMap (padBlock φ.modLcm) ++ [false] =
      u.flatMap (padBlock φ.modLcm) ++ padBlock φ.modLcm true ++
        List.replicate (φ.modLcm - 1) false ++ [false, false] by simp [padBlock]] at hu'
  exact append_pair_not_mem_secondLast _ ⟨u', false, hu'⟩

/-- The hypothesis of `not_phraseAnswers_mod` is satisfiable: the test `Q_about`. -/
example : ¬ ∀ w, (FormP.sym (.about : PhraseTok 1 1 1 1)).models w ↔ PhraseAnswer w 0 :=
  not_phraseAnswers_mod 0 rfl

/-- **No rounded transformer with RoPE at rational angles answers MRCR with asks spelled by
words**: none accepts exactly the conversations that `v` answers, at every length, whatever its
depth, width and precision. It would round to a formula of `TL[◁#, MOD]` (`thm:rtfr_to_TLClmod`
of arXiv:2506.16055v3, `CRASP.exists_mem_TLClMod_of_rope`), against `not_phraseAnswers_mod`. -/
theorem not_phraseAnswers_rope [NeZero F] [NeZero N] [NeZero R] {p s d k : ℕ} (v : Fin V)
    (θ : ℕ → ℝ) (hθ : (PosEnc.rope θ).RationalAngles)
    (T : PTfr (Option (PhraseTok F N V R)) p s d k) (hT : T.pe = .rope θ) :
    ¬ ∀ w, T.Accepts (bos w) ↔ PhraseAnswer w v := fun h => by
  obtain ⟨φ, ⟨hp, -⟩, hL⟩ := exists_mem_TLClMod_of_rope θ hθ T hT
  exact not_phraseAnswers_mod v hp fun w => (Set.ext_iff.1 hL w).trans (h w)

/-- **No rounded transformer with sinusoidal encodings at rational angles answers MRCR with asks
spelled by words**, whatever its depth, width and precision (`thm:rtfr_eq_tlclmod` of
arXiv:2506.16055v3, `CRASP.exists_mem_TLClMod_of_sinusoidal`, and `not_phraseAnswers_mod`). -/
theorem not_phraseAnswers_sinusoidal [NeZero F] [NeZero N] [NeZero R] {p s d k : ℕ} (v : Fin V)
    (θ : ℕ → ℝ) (hθ : (PosEnc.sinusoidal θ).RationalAngles)
    (T : PTfr (Option (PhraseTok F N V R)) p s d k) (hT : T.pe = .sinusoidal θ) :
    ¬ ∀ w, T.Accepts (bos w) ↔ PhraseAnswer w v := fun h => by
  obtain ⟨φ, ⟨hp, -⟩, hL⟩ := exists_mem_TLClMod_of_sinusoidal θ hθ T hT
  exact not_phraseAnswers_mod v hp fun w => (Set.ext_iff.1 hL w).trans (h w)

/-- The hypotheses of `not_phraseAnswers_rope` and `not_phraseAnswers_sinusoidal` are
satisfiable: the zero transformers at the angles `0`, over one format, topic, answer and
ordinal. -/
example (pe : PosEnc) (hpe : pe = .rope (fun _ => 0) ∨ pe = .sinusoidal (fun _ => 0)) :
    ∃ T : PTfr (Option (PhraseTok 1 1 1 1)) 2 0 0 0, T.pe = pe ∧
      ¬ ∀ w, T.Accepts (bos w) ↔ PhraseAnswer w 0 := by
  refine ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := pe }, rfl, ?_⟩
  rcases hpe with rfl | rfl
  · exact not_phraseAnswers_rope 0 _ ⟨fun _ => 0, fun _ => by simp⟩ _ rfl
  · exact not_phraseAnswers_sinusoidal 0 _ ⟨fun _ => 0, fun _ => by simp⟩ _ rfl

end Transformer.MRCR
