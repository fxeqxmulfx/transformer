module

public import Transformer.SequenceModel.Attention.Defs

/-!
# The normalized attention model loses no trainable coordinates

Source: `review/flat_spectrum.md`, *General positional models*, G4. Encoding
and decoding are inverse for every matrix entry. The optimizer bridge thus
does not freeze attention, tie independent matrices, or hide a rank limit.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini

/-- G4: every ordinary full model has an exact scalar-coordinate encoding. -/
theorem decode_encode (cfg : Config) (p : ModelParams cfg) :
    decode cfg (encode cfg p) = p := by
  cases p with
  | mk E B =>
      simp only [decode, encode, LinearEquiv.apply_symm_apply]

/-- G4: no distinct scalar parameters are silently identified by the decoder. -/
theorem encode_decode (cfg : Config) (w : Parameters cfg) :
    encode cfg (decode cfg w) = w := by
  ext i
  rcases i with ⟨v, j⟩ | ⟨d, i⟩
  · rfl
  rcases i with ⟨i, j⟩ | (⟨i, j⟩ | (⟨i, j⟩ | (⟨i, j⟩ | ⟨i, j⟩))) <;>
    exact congrFun (congrFun ((matrixEquiv _ _).symm_apply_apply _) i) j

/-- G4: the parameter count includes all source matrices and counts the tied
embedding once: V*W + D*(4*W^2 + 3*W*H). Normalizers add no parameters. -/
theorem finrank_parameters (cfg : Config) :
    Module.finrank ℝ (Parameters cfg) =
      cfg.vocab_size * cfg.d_model + cfg.n_layers *
        (4 * cfg.d_model ^ 2 + 3 * cfg.d_model * cfg.d_ff) := by
  change Module.finrank ℝ (EuclideanSpace ℝ (Index cfg)) = _
  rw [finrank_euclideanSpace]
  simp only [Index, BlockIndex, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
  ring

example : Module.finrank ℝ (Parameters witnessConfig) = 404 := by
  rw [finrank_parameters]
  norm_num [witnessConfig]

example := decode_encode witnessConfig
  (decode witnessConfig (WithLp.toLp 2 (fun i => if i.isLeft then 1 else 2)))

end Transformer.SequenceModel.Attention
