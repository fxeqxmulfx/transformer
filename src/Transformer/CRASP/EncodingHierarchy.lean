/-
# The depth hierarchy under position encodings

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.5, `thm:rtfr_pes_depth_hierarchy`: neither sinusoidal
encodings, nor RoPE, nor ALiBi collapse the transformer depth hierarchy.  The
statement is on the books unproved; its negative half for `k > 0` is proved
from the three simulations of Appendix F (`CRASP.PositionalHierarchy`),
carried as hypotheses.
-/

import Transformer.CRASP.PositionalHierarchy

namespace Transformer
namespace CRASP

/-- **The negative half of `thm:rtfr_pes_depth_hierarchy`, from the three
simulations.**  For `k > 0`, no depth-`k` transformer with sinusoidal, RoPE or
ALiBi encoding (at rational angles) recognizes `E_{k+1}`.

This is the paper's derivation: `thm:rtfr_eq_tlclmod`, `thm:rtfr_to_TLClmod`
or `thm:rtfr_to_TLCly` puts the language in `TL[◁#, MOD]_k` or `TL[◁#, Y]_k`,
and `thm:tlclpos_depth_hierarchy` denies it to both.  None of the three
simulations is proved, so none is used: they are the hypotheses `hsin`,
`hrope`, `halibi`, stated exactly as `definableMod_iff_recognizes_sinusoidal`
(its `←` direction), `exists_mem_TLClMod_of_rope` and
`exists_mem_TLClY_of_alibi` state them.  `k > 0` is what
`thm:tlclpos_depth_hierarchy` asks for.

Source: arXiv:2506.16055v3, §4.5, `thm:rtfr_pes_depth_hierarchy`, and the
three unnamed theorems of Appendix F. -/
theorem not_recognizes_altPlusNeutral_of_simulations (k : ℕ) (hk : 0 < k)
    (hsin : ∀ (L : Set (List (Option Bool))) (j : ℕ), 1 ≤ j →
      (∃ (p s d : ℕ) (θ : ℕ → ℝ) (T : PTfr (Option (Option Bool)) p s d j),
        T.pe = .sinusoidal θ ∧ (PosEnc.sinusoidal θ).RationalAngles ∧ T.Recognizes L) →
      DefinableMod L j)
    (hrope : ∀ {p s d j : ℕ} (θ : ℕ → ℝ), (PosEnc.rope θ).RationalAngles →
      ∀ T : PTfr (Option (Option Bool)) p s d j, T.pe = .rope θ →
        ∃ φ ∈ TLClMod (Option Bool) j,
          φ.lang = {w : List (Option Bool) | T.Accepts (bos w)})
    (halibi : ∀ {p s d j : ℕ} (a : ℝ) (T : PTfr (Option (Option Bool)) p s d j),
      T.pe = .alibi a →
        ∃ φ ∈ TLClY (Option Bool) j,
          φ.lang = {w : List (Option Bool) | T.Accepts (bos w)})
    (pe : PosEnc) (hpe : pe.IsStandard) (hrat : pe.RationalAngles) :
    ∀ (p s d : ℕ) (T : PTfr (Option (Option Bool)) p s d k),
      T.pe = pe → ¬ T.Recognizes (altPlusNeutral (k + 1)) := by
  intro p s d T hTpe hrec
  refine (definablePos_altPlusNeutral k hk).2 ?_
  have hlang : {w : List (Option Bool) | T.Accepts (bos w)} = altPlusNeutral (k + 1) :=
    Set.ext hrec
  cases pe with
  | plain => exact hpe.elim
  | sinusoidal θ =>
      obtain ⟨φ, hφ, hφlang⟩ :=
        hsin (altPlusNeutral (k + 1)) k hk ⟨p, s, d, θ, T, hTpe, hrat, hrec⟩
      exact ⟨φ, TLClMod_subset_TLClPos _ _ hφ, hφlang⟩
  | rope θ =>
      obtain ⟨φ, hφ, hφlang⟩ := hrope θ hrat T hTpe
      exact ⟨φ, TLClMod_subset_TLClPos _ _ hφ, hφlang.trans hlang⟩
  | alibi a =>
      obtain ⟨φ, hφ, hφlang⟩ := halibi a T hTpe
      exact ⟨φ, TLClY_subset_TLClPos _ _ hφ, hφlang.trans hlang⟩

/-- The hypotheses of `not_recognizes_altPlusNeutral_of_simulations` other
than the three simulations are satisfiable: `k = 1`, ALiBi with slope `1`.
The simulations are the paper's theorems of Appendix F, sorried as
`definableMod_iff_recognizes_sinusoidal`, `exists_mem_TLClMod_of_rope` and
`exists_mem_TLClY_of_alibi`; they are witnessed when those are proved. -/
example : 0 < 1 ∧ (PosEnc.alibi 1).IsStandard ∧ (PosEnc.alibi 1).RationalAngles :=
  ⟨one_pos, trivial, trivial⟩

/-- **Theorem `thm:rtfr_pes_depth_hierarchy`.**  *The depth hierarchy survives
the position encodings.*

A depth-`(k+1)` transformer can recognize `E_{k+1}` but no depth-`k` one can,
whether the transformers use sinusoidal position embeddings, RoPE or ALiBi.
The rational-angle assumption is the one under which the periodic encodings
are simulated at all; it is vacuous for ALiBi.

Not proved.  The negative half for `k > 0` is
`not_recognizes_altPlusNeutral_of_simulations`, on the three simulations of
Appendix F, none of which is proved.  The paper states the theorem for every
`k`, but its derivation goes through `thm:tlclpos_depth_hierarchy`, stated for
`k > 0`, and `thm:rtfr_eq_tlclmod`, stated for `k ≥ 1`; the case `k = 0` is
not argued.  The positive half asks for a depth-`(k+1)` transformer carrying
the *given* encoding, which needs the construction of `thm:rtfr_to_TLCl`.

Source: arXiv:2506.16055v3, §4.5, `thm:rtfr_pes_depth_hierarchy`. -/
theorem rtfr_pes_depth_hierarchy (k : ℕ)
    (pe : PosEnc) (hpe : pe.IsStandard) (hrat : pe.RationalAngles) :
    (∃ (p s d : ℕ) (T : PTfr (Option (Option Bool)) p s d (k + 1)),
        T.pe = pe ∧ T.Recognizes (altPlusNeutral (k + 1))) ∧
      ∀ (p s d : ℕ) (T : PTfr (Option (Option Bool)) p s d k),
        T.pe = pe → ¬ T.Recognizes (altPlusNeutral (k + 1)) := by
  sorry

/-- The hypotheses of `rtfr_pes_depth_hierarchy` are satisfiable: ALiBi with
slope `1` is one of the three encodings and carries no angles to be
rational. -/
example : (PosEnc.alibi 1).IsStandard ∧ (PosEnc.alibi 1).RationalAngles :=
  ⟨trivial, trivial⟩

end CRASP
end Transformer
