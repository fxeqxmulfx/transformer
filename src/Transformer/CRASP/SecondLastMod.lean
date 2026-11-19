module

public import Transformer.CRASP.ModBlocksCorrect
public import Transformer.CRASP.ModBlocksDepth
public import Transformer.CRASP.PeriodicEncodingsToLogic
public import Transformer.CRASP.SecondLast

/-!
# `Σ*bΣ` is not definable in `TL[◁#, MOD]`

In no paper. arXiv:2506.16055v3 rounds transformers with RoPE or sinusoidal encodings at rational
angles to `TL[◁#, MOD]` and with ALiBi to `TL[◁#, Y]` (Appendix F, `thm:rtfr_to_TLClmod`,
`thm:rtfr_eq_tlclmod`, `thm:rtfr_to_TLCly`), proves a depth hierarchy in each, and compares
neither fragment with the other. `Σ*bΣ` lies in `TL[◁#, Y]` at depth `0`
(`definableY_secondLast`); here it lies in `TL[◁#, MOD]` at no depth.

Write each bit `c` as a block of `M` symbols, `M` a multiple of the positive moduli of a formula
`φ` of `TL[◁#, MOD]`, and close the blocks by a fixed nonempty string `s`. The translation of
`Transformer.CRASP.ModBlocks` lies in `TL[◁#]` (`FormP.lastTr_mem`) and accepts `x c` iff `φ`
accepts the blocks of `x` closed by `s` (`FormP.models_lastTr`). For some `u` it accepts both or
neither of `u·aba` and `u·baa` (`exists_models_iff_secondLast`), so `φ` accepts both or neither
of the blocks of `u·ab` and of `u·ba`, closed by `s` (`FormP.exists_models_iff_blocks`). With the
block `a^{M−1} c` of `c` (`padBlock`) and `s = a`, the first ends in `ba` and the second in `aa`
(`not_definableMod_secondLast`). So no rounded transformer with RoPE or sinusoidal encodings at
rational angles recognizes `Σ*bΣ` (`not_rope_recognizes_secondLast`,
`not_sinusoidal_recognizes_secondLast`).
-/

@[expose] public section

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

/-- **A formula of `TL[◁#, MOD]` confuses the blocks of `u·ab` with those of `u·ba`.**  For `φ`
free of `Y`, `M > 0` a multiple of its positive moduli, blocks `g c` of `M` symbols and a
nonempty `s`, some `u` makes `φ` accept both or neither of `g u · g a · g b · s` and
`g u · g b · g a · s`. -/
theorem FormP.exists_models_iff_blocks {φ : FormP σ} (hp : φ.prevFree = true) {M : ℕ}
    (hM : 0 < M) (hd : φ.modLcm ∣ M) {g : Bool → List σ} (hg : ∀ c, (g c).length = M)
    {s : List σ} (hs : s ≠ []) :
    ∃ u : List Bool, φ.models ((u ++ [false, true]).flatMap g ++ s) ↔
      φ.models ((u ++ [true, false]).flatMap g ++ s) := by
  obtain ⟨u, hu⟩ := exists_models_iff_secondLast _ _ (FormP.lastTr_mem g M s s.length φ le_rfl)
  refine ⟨u, ?_⟩
  rw [← FormP.models_lastTr hM hg hp hd _ false hs, ← FormP.models_lastTr hM hg hp hd _ false hs]
  simpa using hu

/-- The hypotheses of `FormP.exists_models_iff_blocks` are satisfiable: `Q_b`, blocks of one
bit, and `a` after them. -/
example : ∃ u : List Bool, (FormP.sym true).models ((u ++ [false, true]).flatMap (fun c => [c]) ++
      [false]) ↔ (FormP.sym true).models ((u ++ [true, false]).flatMap (fun c => [c]) ++ [false]) :=
  FormP.exists_models_iff_blocks rfl Nat.one_pos (one_dvd _) (fun _ => rfl)
    (List.cons_ne_nil _ _)

/-- The block of the bit `c`: `M − 1` letters `a`, then `c`. -/
def padBlock (M : ℕ) (c : Bool) : List Bool := List.replicate (M - 1) false ++ [c]

theorem length_padBlock {M : ℕ} (hM : 0 < M) (c : Bool) : (padBlock M c).length = M := by
  simp only [padBlock, List.length_append, List.length_replicate, List.length_singleton]
  omega

/-- The hypothesis of `length_padBlock` is satisfiable: the block `ab` of `b`. -/
example : (padBlock 2 true).length = 2 := length_padBlock two_pos true

theorem append_pair_not_mem_secondLast (w : List Bool) : w ++ [false, false] ∉ secondLast := by
  rintro ⟨v, x, h⟩
  have h' := (List.append_inj' h rfl).2
  simp at h'

/-- **`Σ*bΣ` is not definable in `TL[◁#, MOD]`**, at any depth. The blocks `a^{M−1} c` of
`u·ab` and of `u·ba`, closed by `a`, end in `ba` and in `aa`, and `FormP.exists_models_iff_blocks`
confuses them for some `u`. -/
theorem not_definableMod_secondLast (k : ℕ) : ¬ DefinableMod secondLast k := by
  rintro ⟨φ, ⟨hp, -⟩, hL⟩
  have hM := φ.modLcm_pos
  obtain ⟨u, hu⟩ := FormP.exists_models_iff_blocks hp hM dvd_rfl (length_padBlock hM)
    (s := [false]) (List.cons_ne_nil _ _)
  have key (w : List Bool) : φ.models w ↔ w ∈ secondLast := Set.ext_iff.1 hL w
  rw [key, key] at hu
  have h₁ : (u ++ [false, true]).flatMap (padBlock φ.modLcm) ++ [false] ∈ secondLast :=
    ⟨u.flatMap (padBlock φ.modLcm) ++ padBlock φ.modLcm false ++
      List.replicate (φ.modLcm - 1) false, false, by simp [padBlock]⟩
  have e : (u ++ [true, false]).flatMap (padBlock φ.modLcm) ++ [false] =
      u.flatMap (padBlock φ.modLcm) ++ padBlock φ.modLcm true ++
        List.replicate (φ.modLcm - 1) false ++ [false, false] := by
    simp [padBlock]
  exact append_pair_not_mem_secondLast _ (e ▸ hu.1 h₁)

/-- **`TL[◁#, Y]` is not contained in `TL[◁#, MOD]`**: `Σ*bΣ` is definable with `Y` at depth `0`
and with `MOD` at no depth. -/
theorem exists_definableY_not_definableMod :
    ∃ L : Set (List Bool), DefinableY L 0 ∧ ∀ k, ¬ DefinableMod L k :=
  ⟨secondLast, definableY_secondLast, not_definableMod_secondLast⟩

/-- **No rounded transformer with RoPE at rational angles recognizes `Σ*bΣ`**, whatever its
depth, width and precision: it would round to a formula of `TL[◁#, MOD]` (`thm:rtfr_to_TLClmod`
of arXiv:2506.16055v3, `exists_mem_TLClMod_of_rope`), against `not_definableMod_secondLast`. -/
theorem not_rope_recognizes_secondLast {p s d k : ℕ} (θ : ℕ → ℝ)
    (hθ : (PosEnc.rope θ).RationalAngles) (T : PTfr (Option Bool) p s d k)
    (hT : T.pe = .rope θ) : ¬ T.Recognizes secondLast := fun h => by
  obtain ⟨φ, hφ, hL⟩ := exists_mem_TLClMod_of_rope θ hθ T hT
  exact not_definableMod_secondLast k ⟨φ, hφ, hL.trans (Set.ext fun w => h w)⟩

/-- **No rounded transformer with sinusoidal encodings at rational angles recognizes `Σ*bΣ`**,
whatever its depth, width and precision (`thm:rtfr_eq_tlclmod` of arXiv:2506.16055v3,
`exists_mem_TLClMod_of_sinusoidal`, and `not_definableMod_secondLast`). -/
theorem not_sinusoidal_recognizes_secondLast {p s d k : ℕ} (θ : ℕ → ℝ)
    (hθ : (PosEnc.sinusoidal θ).RationalAngles) (T : PTfr (Option Bool) p s d k)
    (hT : T.pe = .sinusoidal θ) : ¬ T.Recognizes secondLast := fun h => by
  obtain ⟨φ, hφ, hL⟩ := exists_mem_TLClMod_of_sinusoidal θ hθ T hT
  exact not_definableMod_secondLast k ⟨φ, hφ, hL.trans (Set.ext fun w => h w)⟩

/-- The hypotheses of `not_rope_recognizes_secondLast` and `not_sinusoidal_recognizes_secondLast`
are satisfiable: the zero transformers at the angles `0`. -/
example (pe : PosEnc) (hpe : pe = .rope (fun _ => 0) ∨ pe = .sinusoidal (fun _ => 0)) :
    ∃ T : PTfr (Option Bool) 2 0 0 0, T.pe = pe ∧ ¬ T.Recognizes secondLast := by
  refine ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := pe }, rfl, ?_⟩
  rcases hpe with rfl | rfl
  · exact not_rope_recognizes_secondLast _ ⟨fun _ => 0, fun _ => by simp⟩ _ rfl
  · exact not_sinusoidal_recognizes_secondLast _ ⟨fun _ => 0, fun _ => by simp⟩ _ rfl

end CRASP
end Transformer
