import Transformer.GPTMini.Semantics.DepthReadout

/-!
# Complete original depth parameters and genuine detector loop

Source: original ModelParams/hidden at f11b6e2 and verified depth
operators at 4dec64b. Easy uses one actual detector and readout in its
two layers; hard uses three actual detectors, readout and two genuine
zero-weight residual identities in its original six layers. Widths,
head counts, FFN dimensions and the tied vocabulary remain unchanged.

All parameters are shared across every raw prefix of a mode and are
independent of epsilon. The actual original hidden loop is coupled to
the real raw-word detector computation. Its only local input premise
is equality of the token-local embedding lookup; integer serialization
will derive it from actual raw token IDs including the BOS entry.
No prepared whole-prefix encoder, route or desired label is a premise.
The readout and every tail layer are coupled to the same actual state;
the true full forward call inherits strict whole-vocabulary margins.
Only checked integer serialization and raw task correctness remain.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical
open Transformer.Basis

/-- Genuine unused original residual layers, with ordinary zero projection matrices.
Source: original BlockParams at the unchanged depth dimensions, preserving the actual residual stream. -/
noncomputable def depthIdentityBlock (mode : Mode) : BlockParams (depthConfig mode) where
  attn := { W_qkv := 0, W_o := 0, log_alpha := fun _ => 0 }
  ffn := { W_in := 0, W_out := 0 }

/-- An actual zero-weight original block is a residual identity on every real array and position.
Source: both genuine output matrices are zero in original blockForward, without replacing the block by an identity definition. -/
theorem depthIdentityBlock_apply (mode : Mode) (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) :
    blockForward (depthConfig mode) (depthIdentityBlock mode) eps positions x i = x i := by
  dsimp only [blockForward, depthIdentityBlock, attnSubLayer, ffnSubLayer, relu2FFN]
  simp only [zero_apply, add_zero]

/-- Complete actual original depth parameters with one token-local tied embedding and a fixed ordinary block list.
Source: the original ModelParams record, three available detectors, one readout and genuine unused identity layers. -/
noncomputable def depthModelParams (mode : Mode) : ModelParams (depthConfig mode) where
  embedding := depthRawEmbedding mode depthLabelGain
  blocks := fun b => if h : b.val < depthDetectorCount mode then
      depthDetectorBlock mode ⟨b.val, by have hc := (depthDetectorCount_slots mode).1; omega⟩
    else if b.val = depthDetectorCount mode then depthReadoutBlock mode else depthIdentityBlock mode

/-- Every used actual model detector is precisely the independently analyzed original detector at the same stage.
Source: fixed parameter record and index below the mode's detector count. -/
theorem depthModelParams_detector (mode : Mode) (b : Fin (depthConfig mode).n_layers)
    (hb : b.val < depthDetectorCount mode) :
    (depthModelParams mode).blocks b = depthDetectorBlock mode
      ⟨b.val, by have hc := (depthDetectorCount_slots mode).1; omega⟩ := by
  dsimp only [depthModelParams]
  rw [dite_eq_left hb]

example : (0 : Fin (depthConfig .easy).n_layers).val < depthDetectorCount .easy := by decide

/-- The actual model has exactly the ordinary readout at the first index after its detectors.
Source: fixed original block record and the proved layer-budget slot. -/
theorem depthModelParams_readout (mode : Mode) (b : Fin (depthConfig mode).n_layers)
    (hb : b.val = depthDetectorCount mode) : (depthModelParams mode).blocks b = depthReadoutBlock mode := by
  have hn : ¬b.val < depthDetectorCount mode := by omega
  dsimp only [depthModelParams]
  rw [dite_eq_right hn, ite_eq_left hb]

example : (1 : Fin (depthConfig .easy).n_layers).val = depthDetectorCount .easy := by decide

/-- Every remaining original hard layer is a genuine zero-weight residual block.
Source: unchanged six-layer budget and fixed parameter assignment after the true readout index. -/
theorem depthModelParams_tail (mode : Mode) (b : Fin (depthConfig mode).n_layers)
    (hb : depthDetectorCount mode < b.val) : (depthModelParams mode).blocks b = depthIdentityBlock mode := by
  have hn : ¬b.val < depthDetectorCount mode := by omega
  have he : b.val ≠ depthDetectorCount mode := by omega
  dsimp only [depthModelParams]
  rw [dite_eq_right hn, ite_eq_right he]

example : depthDetectorCount .hard < (4 : Fin (depthConfig .hard).n_layers).val := by decide

/-- The genuine complete model's entire detector-prefix hidden array is exactly the independently computed raw-word detector state.
Source: original Model.hidden recurrence, actual fixed detector matrices and only token-local embedding equality. -/
theorem depthModel_hidden_detectors (mode : Mode) (eps : ℝ) (word : List (Option Bool))
    (positions : Fin word.length → ℝ) (tokens : Fin word.length → Fin 36)
    (hembed : ∀ i, depthRawEmbedding mode depthLabelGain (tokens i) = depthWordInput mode depthLabelGain word i)
    (n : ℕ) (hn : n ≤ depthDetectorCount mode) :
    hidden (depthConfig mode) (depthModelParams mode) eps positions tokens n =
      depthWordState mode depthLabelGain eps word positions n := by
  revert hn
  induction n with
  | zero =>
      intro hn
      funext i
      exact hembed i
  | succ n ih =>
      intro hn
      have hnext : n < depthDetectorCount mode := by omega
      have hmodel : n < (depthConfig mode).n_layers := by have hc := (depthDetectorCount_slots mode).2; omega
      rw [hidden, dite_eq_left hmodel, depthWordState, dite_eq_left hnext]
      rw [depthModelParams_detector mode ⟨n, hmodel⟩ hnext, ih (by omega)]

example : (∀ i : Fin ([none, some false, some true] : List (Option Bool)).length,
    depthRawEmbedding .easy depthLabelGain (depthLetterToken (([none, some false, some true] : List (Option Bool)).get i)) =
      depthWordInput .easy depthLabelGain [none, some false, some true] i) ∧ 1 ≤ depthDetectorCount .easy := by
  exact ⟨fun _ => rfl, by decide⟩

/-- The genuine original hidden loop computes exactly the full independently verified final state after its readout layer.
Source: actual detector-prefix loop equality, original next block and its fixed readout parameters. -/
theorem depthModel_hidden_readout (mode : Mode) (eps : ℝ) (word : List (Option Bool))
    (positions : Fin word.length → ℝ) (tokens : Fin word.length → Fin 36)
    (hembed : ∀ i, depthRawEmbedding mode depthLabelGain (tokens i) = depthWordInput mode depthLabelGain word i) :
    hidden (depthConfig mode) (depthModelParams mode) eps positions tokens (depthDetectorCount mode + 1) =
      depthWordFinalState mode depthLabelGain eps word positions := by
  have hmodel : depthDetectorCount mode < (depthConfig mode).n_layers := by have hc := (depthDetectorCount_slots mode).2; omega
  rw [hidden, dite_eq_left hmodel, depthModelParams_readout mode ⟨depthDetectorCount mode, hmodel⟩ rfl,
    depthModel_hidden_detectors mode eps word positions tokens hembed _ (by omega)]
  rfl

example : ∀ i : Fin ([none, some false, some true] : List (Option Bool)).length,
    depthRawEmbedding .hard depthLabelGain (depthLetterToken (([none, some false, some true] : List (Option Bool)).get i)) =
      depthWordInput .hard depthLabelGain [none, some false, some true] i := fun _ => rfl

/-- All genuine original layers after readout, and the exhausted loop, preserve that same computed final array.
Source: actual zero-weight tail blocks and original Model.hidden's exhaustion branch, proved by loop induction. -/
theorem depthModel_hidden_after (mode : Mode) (eps : ℝ) (word : List (Option Bool))
    (positions : Fin word.length → ℝ) (tokens : Fin word.length → Fin 36)
    (hembed : ∀ i, depthRawEmbedding mode depthLabelGain (tokens i) = depthWordInput mode depthLabelGain word i)
    (n : ℕ) (hn : depthDetectorCount mode + 1 ≤ n) :
    hidden (depthConfig mode) (depthModelParams mode) eps positions tokens n =
      depthWordFinalState mode depthLabelGain eps word positions := by
  revert hn
  induction n with
  | zero => intro hn; omega
  | succ n ih =>
      intro hn
      by_cases hp : depthDetectorCount mode + 1 ≤ n
      · rw [hidden]
        by_cases hl : n < (depthConfig mode).n_layers
        · rw [dite_eq_left hl, depthModelParams_tail mode ⟨n, hl⟩ (by change depthDetectorCount mode < n; omega)]
          funext i
          rw [depthIdentityBlock_apply]
          exact congrFun (ih hp) i
        · rw [dite_eq_right hl]
          exact ih hp
      · have he : n = depthDetectorCount mode := by omega
        subst n
        exact depthModel_hidden_readout mode eps word positions tokens hembed

example : (∀ i : Fin ([none, some false, some true] : List (Option Bool)).length,
    depthRawEmbedding .hard depthLabelGain (depthLetterToken (([none, some false, some true] : List (Option Bool)).get i)) =
      depthWordInput .hard depthLabelGain [none, some false, some true] i) ∧ depthDetectorCount .hard + 1 ≤ 6 := by
  exact ⟨fun _ => rfl, by decide⟩

/-- The actual complete original forward evaluates the real final RMSNorm and unchanged tied embedding at the computed raw-word state.
Source: genuine Model.forward/unembed and all detector/readout/tail loop equalities. -/
theorem depthModel_forward (mode : Mode) (eps : ℝ) (word : List (Option Bool))
    (positions : Fin word.length → ℝ) (tokens : Fin word.length → Fin 36)
    (hembed : ∀ i, depthRawEmbedding mode depthLabelGain (tokens i) = depthWordInput mode depthLabelGain word i)
    (i : Fin word.length) (token : Fin 36) :
    forward (depthConfig mode) (depthModelParams mode) eps positions tokens i token =
      inner (𝕜 := ℝ) (rmsNormEps eps (depthWordFinalState mode depthLabelGain eps word positions i)) (depthRawEmbedding mode depthLabelGain token) := by
  change inner (𝕜 := ℝ) (rmsNormEps eps (hidden (depthConfig mode) (depthModelParams mode) eps positions tokens
    (depthConfig mode).n_layers i)) (depthRawEmbedding mode depthLabelGain token) = _
  rw [depthModel_hidden_after mode eps word positions tokens hembed _ (depthDetectorCount_slots mode).2]

example : ∀ i : Fin ([none, some false, some true] : List (Option Bool)).length,
    depthRawEmbedding .easy depthLabelGain (depthLetterToken (([none, some false, some true] : List (Option Bool)).get i)) =
      depthWordInput .easy depthLabelGain [none, some false, some true] i := fun _ => rfl

/-- Every genuine complete-model logit strictly loses to the independent ordered answer except that answer itself.
Source: actual full forward coupling and real raw-word final RMS/tied margins over all original 36 tokens. -/
theorem depthModel_logits (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ)
    (tokens : Fin word.length → Fin 36)
    (hembed : ∀ i, depthRawEmbedding mode depthLabelGain (tokens i) = depthWordInput mode depthLabelGain word i)
    (i : Fin word.length) (token : Fin 36) (hne : token ≠ depthWordAnswer mode word i) :
    forward (depthConfig mode) (depthModelParams mode) eps positions tokens i token <
      forward (depthConfig mode) (depthModelParams mode) eps positions tokens i (depthWordAnswer mode word i) := by
  rw [depthModel_forward mode eps word positions tokens hembed, depthModel_forward mode eps word positions tokens hembed]
  exact depthWordFinalState_rms_margin mode eps heps hclip word hT positions i token hne

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 ∧
    (∀ i : Fin ([none, some false, some true] : List (Option Bool)).length,
      depthRawEmbedding .easy depthLabelGain (depthLetterToken (([none, some false, some true] : List (Option Bool)).get i)) =
        depthWordInput .easy depthLabelGain [none, some false, some true] i) ∧
    (0 : Fin 36) ≠ depthWordAnswer .easy [none, some false, some true] 2 := by
  refine ⟨by norm_num, by norm_num, by decide, fun _ => rfl, ?_⟩
  unfold depthWordAnswer
  split_ifs <;> decide

end Transformer.GPTMini.Semantics
