/-
# Same-depth recognizers in all three position-encoding families

arXiv:2506.16055v3, §4.5, `thm:rtfr_pes_depth_hierarchy`, and Appendix F.
The parameters belong to the model being constructed: zero-angle RoPE,
zero-slope ALiBi, and a zero-angle sinusoidal model with reserved coordinates.
-/

import Transformer.CRASP.NeutralEncodings
import Transformer.CRASP.SinusoidalLiftCorrect
import Transformer.CRASP.LogicToTransformer

namespace Transformer.CRASP

universe u
variable {σ : Type u} [DecidableEq σ]

/-- The three families named in the manuscript's positional hierarchy (F). -/
inductive EncodingFamily where
  | sinusoidal
  | rope
  | alibi

/-- The family of a position encoding, with no family for the plain model (F). -/
def PosEnc.family : PosEnc → Option EncodingFamily
  | .plain => none
  | .sinusoidal _ => some .sinusoidal
  | .rope _ => some .rope
  | .alibi _ => some .alibi

/-- Every plain temporal formula has a same-depth recognizer in each family.
Source: arXiv:2506.16055v3, Appendix F, positive positional hierarchies,
using the uniform program of Appendix B.2, `thm:TLCl_to_rtfr`. -/
theorem exists_encoding_of_mem_TLCl (k : ℕ) (φ : Form σ) (hφ : φ ∈ TLCl σ k)
    (f : EncodingFamily) :
    ∃ (p s d : ℕ) (T : PTfr (Option σ) p s d k),
      T.pe.family = some f ∧ T.pe.RationalAngles ∧ T.Recognizes φ.lang := by
  let T := TemporalProgram.model φ k
  have hT : T.Recognizes φ.lang := TemporalProgram.model_recognizes φ k hφ
  cases f with
  | sinusoidal =>
      refine ⟨φ.capacity + 2, 0, 2 * TemporalProgram.dimension φ,
        SinusoidalLift.model T, rfl, ?_, ?_⟩
      · exact ⟨fun _ => 0, fun _ => by simp⟩
      · exact SinusoidalLift.model_recognizes T (fun _ _ => rfl) _ hT
  | rope =>
      refine ⟨φ.capacity + 2, 0, TemporalProgram.dimension φ,
        T.withEncoding (.rope (fun _ => 0)), rfl, ?_, ?_⟩
      · exact ⟨fun _ => 0, fun _ => by simp⟩
      · exact T.withEncoding_recognizes _ (fun _ => rfl) PosEnc.rope_zero_logit _ hT
  | alibi =>
      refine ⟨φ.capacity + 2, 0, TemporalProgram.dimension φ,
        T.withEncoding (.alibi 0), rfl, trivial, ?_⟩
      exact T.withEncoding_recognizes _ (fun _ => rfl) PosEnc.alibi_zero_logit _ hT

/-- The formula hypothesis has an ordinary letter witness (Appendix B.2/F). -/
example : (Form.sym true : Form Bool) ∈ TLCl Bool 0 := ⟨rfl, rfl, le_rfl⟩

end Transformer.CRASP
