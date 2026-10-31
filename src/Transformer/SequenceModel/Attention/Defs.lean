module

public import Transformer.SequenceModel.Rotary
public import Transformer.GPTMini.Reshape
public import Transformer.GPTMini.QKNorm

/-!
# Complete scalar coordinates of the supplied Peri-RMSNorm transformer

Source: `review/flat_spectrum.md`, *General positional models*, G4; the user's
`diff_gpt/domain/model` at commit `b98c665bd8faa30729fd368d257a13b6d58712a1`.
All embedding, fused-QKV, attention-output, SwiGLU gate/value/output weights
vary independently. There are no biases, learned normalization gains or
learned head temperatures. `cfg.d_ff` equals `(8*cfg.d_model+1)/3` in the
source; allowing other finite hidden sizes does not change the operators.
The existing Config/reshape types supply dimensions, not the older GPTMini
forward pass. The forward in Model below retains the supplied normalizers.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini

/-- Output-row/input-column coordinates of a complete linear matrix. -/
def matrixEquiv (m n : ℕ) :
    Matrix (Fin m) (Fin n) ℝ ≃ₗ[ℝ] (EucSpace n →L[ℝ] EucSpace m) :=
  Matrix.toEuclideanLin.trans LinearMap.toContinuousLinearMap

/-- The actual five matrices; gate and value are halves of c_fc. -/
structure BlockParams (cfg : Config) where
  qkv : EucSpace cfg.d_model →L[ℝ] EucSpace (3 * cfg.d_model)
  projection : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_model
  gate : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_ff
  value : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_ff
  output : EucSpace cfg.d_ff →L[ℝ] EucSpace cfg.d_model

/-- Embedding and tied readout share the same table, as in GPT.__init__. -/
structure ModelParams (cfg : Config) where
  embedding : Fin cfg.vocab_size → EucSpace cfg.d_model
  blocks : Fin cfg.n_layers → BlockParams cfg

abbrev BlockIndex (cfg : Config) :=
  (Fin (3 * cfg.d_model) × Fin cfg.d_model) ⊕
    ((Fin cfg.d_model × Fin cfg.d_model) ⊕
      ((Fin cfg.d_ff × Fin cfg.d_model) ⊕
        ((Fin cfg.d_ff × Fin cfg.d_model) ⊕ (Fin cfg.d_model × Fin cfg.d_ff))))

abbrev Index (cfg : Config) := (Fin cfg.vocab_size × Fin cfg.d_model) ⊕
  (Fin cfg.n_layers × BlockIndex cfg)

abbrev Parameters (cfg : Config) := Weights (Index cfg)

/-- All source weights, with no restricted-rank or frozen-embedding premise. -/
def decode (cfg : Config) (w : Parameters cfg) : ModelParams cfg where
  embedding v := WithLp.toLp 2 (fun i => w.ofLp (.inl (v, i)))
  blocks d := {
    qkv := matrixEquiv _ _ (fun i j => w.ofLp (.inr (d, .inl (i, j))))
    projection := matrixEquiv _ _ (fun i j => w.ofLp (.inr (d, .inr (.inl (i, j)))))
    gate := matrixEquiv _ _ (fun i j => w.ofLp (.inr (d, .inr (.inr (.inl (i, j))))))
    value := matrixEquiv _ _ (fun i j => w.ofLp (.inr (d, .inr (.inr (.inr (.inl (i, j)))))))
    output := matrixEquiv _ _ (fun i j => w.ofLp (.inr (d, .inr (.inr (.inr (.inr (i, j))))))) }

/-- Recover the scalar coordinates of any source model. -/
def encode (cfg : Config) (p : ModelParams cfg) : Parameters cfg :=
  WithLp.toLp 2 fun
    | .inl (v, i) => p.embedding v i
    | .inr (d, .inl (i, j)) => (matrixEquiv _ _).symm (p.blocks d).qkv i j
    | .inr (d, .inr (.inl (i, j))) => (matrixEquiv _ _).symm (p.blocks d).projection i j
    | .inr (d, .inr (.inr (.inl (i, j)))) => (matrixEquiv _ _).symm (p.blocks d).gate i j
    | .inr (d, .inr (.inr (.inr (.inl (i, j))))) => (matrixEquiv _ _).symm (p.blocks d).value i j
    | .inr (d, .inr (.inr (.inr (.inr (i, j))))) => (matrixEquiv _ _).symm (p.blocks d).output i j

/-- A small positive source-shaped architecture for concrete theorem witnesses. -/
abbrev witnessConfig : Config where
  vocab_size := 3
  n_layers := 2
  n_heads := 1
  d_model := 4
  d_ff := 11
  max_seq_len := 5
  rope_theta := 10000
  divides := by decide
  head_even := by decide
  n_heads_pos := by decide
  n_layers_pos := by decide
  d_model_pos := by decide
  d_ff_pos := by decide
  vocab_pos := by decide
  max_seq_len_pos := by decide
  theta_pos := by norm_num

end Transformer.SequenceModel.Attention
