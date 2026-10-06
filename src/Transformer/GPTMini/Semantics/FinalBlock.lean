import Transformer.GPTMini.Semantics.Block
import Transformer.GPTMini.Semantics.Readout

/-!
# Internal semantic certificates imply an actual List Int answer

Source: GPTMini.forward/Block.forward at f11b6e2 and the checked integer
adapter at cbafbe9. The given model's last block transports head and FFN
codes into the semantic answer direction. The transport error must be
smaller than half the scaled embedding separation. Then the actual
RMSNorm, tied unembedding, output softmax and greedy decoder emit that
answer. Correct final logits and SolvesTask are not premises.

Earlier layers are included in the actual hidden state supplied to this
last block. Their responsibility is to produce the local representation
and routing invariants. This conditional verification theorem neither
constructs task-solving weights nor proves that an optimizer finds them.
The semantic target can be a recall value, a depth/parity label, or EOS.
For EOS, the earlier layers must also retain the completion phase; an
ones-count feature alone does not establish this condition. In recall,
the target value must be the last raw binding identified by
Basis.RecallAnswer, not an unordered value-code mixture.

It uses the existing mathematical shared-epsilon model, as documented in
TokenInterface.Basic, and does not certify PyTorch rounding.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface

/-- The full stack's final hidden row is the original last residual block evaluated on its real input.
Source: GPTMini.hidden's block loop at f11b6e2, with the positive layer count made explicit. -/
theorem hidden_last_block (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ) (tokens : Fin T → Fin cfg.vocab_size)
    (L : ℕ) (hL : cfg.n_layers = L + 1) (i : Fin T) :
    hidden cfg params eps positions tokens cfg.n_layers i =
      blockForward cfg (params.blocks ⟨L, by omega⟩) eps positions
        (hidden cfg params eps positions tokens L) i := by
  calc
    _ = hidden cfg params eps positions tokens (L + 1) i :=
      congrArg (fun n => hidden cfg params eps positions tokens n i) hL
    _ = _ := by rw [hidden, dite_eq_left (by omega : L < cfg.n_layers)]

example : controlConfig.n_layers = 1 + 1 := by decide

/-- Given local semantic codes and an error budget, the actual model appends their answer code's token.
Source: the complete original GPTMini stack at f11b6e2 and its checked List Int continuation.
The only semantic target is an internal residual code; actual logits are derived from it. -/
theorem tokenFunction_of_final_block (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (heps : 0 < eps)
    (head : Fin cfg.vocab_size)
    (tail : List (Fin cfg.vocab_size))
    (answer : Fin cfg.vocab_size)
    (L : ℕ)
    (hL : cfg.n_layers = L + 1)
    (codes : Fin cfg.n_heads → EucSpace cfg.head_dim)
    (ffnCode : EucSpace cfg.d_model)
    (headError : ℝ)
    (ffnError : ℝ)
    (scale : ℝ)
    (delta : ℝ)
    (hscale : 0 < scale)
    (hdelta : 0 < delta)
    (hlen : tail.length + 1 ≤ cfg.max_seq_len)
    -- The codebook geometry is checked independently of this input's actual logits.
    (hgeometry : ∀ v, v ≠ answer →
      ‖params.embedding v‖ ≤
        ‖params.embedding answer‖ ∧
          delta ≤
            ‖params.embedding answer - params.embedding v‖)
    -- These heads are extracted from the actual last block, after its own RMSNorm.
    (hheads : ∀ h,
      ‖headAt cfg (params.blocks ⟨L, by omega⟩).attn eps
        (fun i => (i.val : ℝ))
        (hidden cfg params eps (fun i => (i.val : ℝ)) (head :: tail).get L)
        h ⟨tail.length, by simp⟩ - codes h‖ ≤ headError)
    -- The FFN receives the actual stream after the attention residual addition.
    (hffn :
      ‖ffnSubLayer cfg (params.blocks ⟨L, by omega⟩).ffn eps
        (fun i => hidden cfg params eps (fun j => (j.val : ℝ)) (head :: tail).get L i +
          attnSubLayer cfg (params.blocks ⟨L, by omega⟩).attn eps (fun j => (j.val : ℝ))
            (hidden cfg params eps (fun j => (j.val : ℝ)) (head :: tail).get L) i)
          ⟨tail.length, by simp⟩ - ffnCode‖ ≤ ffnError)
    -- Ideal local codes, the real input residual, and W_o form the semantic target.
    (hcode :
      hidden cfg params eps (fun i => (i.val : ℝ)) (head :: tail).get L
        ⟨tail.length, by simp⟩ +
          (params.blocks ⟨L, by omega⟩).attn.W_o (headMerge cfg codes) +
          ffnCode = scale • params.embedding answer)
    -- All head and FFN errors together must fit inside the decoding neighborhood.
    (hbudget :
      ‖(params.blocks ⟨L, by omega⟩).attn.W_o‖ *
        (Real.sqrt (cfg.n_heads : ℝ) * headError) + ffnError <
          scale * delta / 2) :
    tokenFunction cfg params eps (Transformer.Basis.decodeTokens (head :: tail)) =
      Transformer.Basis.decodeTokens (head :: tail) ++ [(answer.val : ℤ)] := by
  -- Prove the internal state certificate, then use the original model's complete readout.
  apply tokenFunction_of_code cfg params eps heps head tail answer scale delta hscale hdelta
    hlen hgeometry
  rw [hidden_last_block cfg params eps _ _ L hL]
  -- Actual sublayer errors propagate through merging, W_o, and both residuals.
  have h := block_error_le cfg (params.blocks ⟨L, by omega⟩) eps (fun i => (i.val : ℝ))
    (hidden cfg params eps (fun i => (i.val : ℝ)) (head :: tail).get L)
    ⟨tail.length, by simp⟩ codes ffnCode headError ffnError hheads hffn
  rw [hcode] at h
  exact h.trans_lt hbudget

/-- The certificate's code-transport equation is attained on a supervised Basis depth prefix.
Source: hidden_control and the original inactive residual blocks of the concrete small model. -/
theorem control_final_block_code :
    hidden controlConfig controlParams (1 / 100000) (fun i => (i.val : ℝ))
        ([⟨1, by decide⟩, ⟨9, by decide⟩] : List (Fin controlConfig.vocab_size)).get 1
        ⟨1, by decide⟩ + (controlParams.blocks ⟨1, by decide⟩).attn.W_o
          (headMerge controlConfig (fun _ => 0)) + 0 =
      (1 / 2 : ℝ) • controlParams.embedding ⟨15, by decide⟩ := by
  rw [hidden_control]
  simp [controlParams, controlEmbedding]

/-- All internal certificate premises are simultaneously realized by an actual small GPTMini context.
Source: the BOS,a control, checked through its real final block rather than its final logits. -/
example : tokenFunction controlConfig controlParams (1 / 100000) [1, 9] = [1, 9, 15] := by
  have h := tokenFunction_of_final_block controlConfig controlParams (1 / 100000) (by norm_num)
    ⟨1, by decide⟩
    [⟨9, by decide⟩]
    ⟨15, by decide⟩
    1
    (by decide)
    (fun _ => 0)
    0
    0
    0
    (1 / 2)
    1
    (by norm_num)
    (by norm_num)
    (by decide)
    control_code_geometry
    (by
      intro h
      rw [show (controlParams.blocks ⟨1, by decide⟩).attn =
        { W_qkv := 0, W_o := 0, log_alpha := fun _ => 0 } from rfl, headAt_zero_qkv]
      simp)
    (by simp [controlParams, ffnSubLayer, relu2FFN])
    control_final_block_code
    (by
      simp [controlParams])
  simpa [Transformer.Basis.decodeTokens] using h

end Transformer.GPTMini.Semantics
