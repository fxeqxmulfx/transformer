/-
# The depth hierarchy under position encodings

arXiv:2506.16055v3, §4.5, theorem `thm:rtfr_pes_depth_hierarchy`,
and the three encoding-specific hierarchies of Appendix F.
Both halves are proved, including the depth-zero lower bound.
-/

import Transformer.CRASP.PositionalHierarchy
import Transformer.CRASP.EncodingLowerBounds
import Transformer.CRASP.EncodingUpperBounds

namespace Transformer.CRASP

/-- No standard positional transformer of depth `k` recognizes `E_{k+1}`.
The periodic encodings require rational angles; ALiBi allows every slope.

Source: arXiv:2506.16055v3, §4.5, `thm:rtfr_pes_depth_hierarchy`, and
Appendix F, the sinusoidal, RoPE and ALiBi reverse simulations. -/
theorem not_recognizes_altPlusNeutral_standard (k : ℕ)
    (pe : PosEnc) (hpe : pe.IsStandard) (hrat : pe.RationalAngles)
    {p s d : ℕ} (T : PTfr (Option (Option Bool)) p s d k) (hT : T.pe = pe) :
    ¬ T.Recognizes (altPlusNeutral (k + 1)) := by
  cases pe with
  | plain => exact hpe.elim
  | sinusoidal θ => exact not_recognizes_altPlusNeutral_sinusoidal k θ hrat T hT
  | rope θ => exact not_recognizes_altPlusNeutral_rope k θ hrat T hT
  | alibi a => exact not_recognizes_altPlusNeutral_alibi k a T hT

/-- Standard-encoding, rationality and model hypotheses have witnesses (F). -/
example : ∃ T : PTfr (Option (Option Bool)) 2 0 0 0,
    T.pe.IsStandard ∧ T.pe.RationalAngles := by
  refine ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := .alibi 1 }, trivial, trivial⟩

/-- **Theorem `thm:rtfr_pes_depth_hierarchy`.** In each of the sinusoidal,
RoPE and ALiBi families, some depth-`(k+1)` model recognizes `E_{k+1}`;
every depth-`k` model in that family fails to recognize it.

The source wording is "if the transformers can use sinusoidal positional
embeddings, RoPE, or ALiBi". Its positive half quantifies over a model in
the chosen family, including the encoding parameters. The former Lean
signature instead fixed those parameters before constructing the model,
which demanded a stronger claim than the manuscript. This statement
restores the source quantifiers: `T.pe.family = some f` selects the family,
and the positive construction chooses zero angles or zero slope. The
negative half still covers every parameter choice at rational angles.
The separate depth-zero lower bound also fills the case omitted by the
manuscript's positive-depth logic hierarchy.

Source: arXiv:2506.16055v3, §4.5, `thm:rtfr_pes_depth_hierarchy`, and
Appendix F, the three unnamed encoding-specific hierarchy theorems. -/
theorem rtfr_pes_depth_hierarchy (k : ℕ) (f : EncodingFamily) :
    (∃ (p s d : ℕ) (T : PTfr (Option (Option Bool)) p s d (k + 1)),
      T.pe.family = some f ∧ T.pe.RationalAngles ∧
        T.Recognizes (altPlusNeutral (k + 1))) ∧
    ∀ (p s d : ℕ) (T : PTfr (Option (Option Bool)) p s d k),
      T.pe.family = some f → T.pe.RationalAngles →
        ¬ T.Recognizes (altPlusNeutral (k + 1)) := by
  have hpos : DefinableL (altPlusNeutral (k + 1)) (k + 1) :=
    definableL_of_kPiecewiseTestable (k + 1) _
      (kPiecewiseTestable_altPlus (k + 1) k.succ_pos).preimage_reduceOption
  obtain ⟨φ, hφ, hlang⟩ := hpos
  refine ⟨?_, ?_⟩
  · obtain ⟨p, s, d, T, hf, hrat, hrec⟩ := exists_encoding_of_mem_TLCl (k + 1) φ hφ f
    refine ⟨p, s, d, T, hf, hrat, ?_⟩
    simpa only [hlang] using hrec
  · intro p s d T hf hrat
    have hstd : T.pe.IsStandard := by
      cases hpe : T.pe with
      | plain => simp [hpe, PosEnc.family] at hf
      | sinusoidal θ => trivial
      | rope θ => trivial
      | alibi a => trivial
    exact not_recognizes_altPlusNeutral_standard k T.pe hstd hrat T rfl

end Transformer.CRASP
