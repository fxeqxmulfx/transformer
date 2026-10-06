import Transformer.GPTMini.Semantics.ResidualFeature
import Transformer.GPTMini.Semantics.CountBlock

/-!
# Raw count and completion phase coexist in one original block

Source: the small Basis parity GPTMini at cbafbe9 and the unchanged
Block.forward/CausalMHA.forward at f11b6e2. The phase control protects an
independent residual coordinate while computing the same ONE feature as
the count control. Its nonzero output projection retains the count in
the first coordinate. Thus the earlier count and phase results concern
simultaneously attainable features of a single concrete model, rather
than mutually incompatible value-encoding hypotheses.

Every ONE count is taken from the actual integer IDs. No oracle label,
correct final logits or prepared semantic value array is a premise.
The protected phase survives the full stack; the exact count formula
here is for its first block. Later blocks and the parity/EOS readout
still need their own invariants. This is verification of internal task
features, not a proof of full Basis accuracy or optimization convergence.
The existing shared-epsilon, real-arithmetic deviation is unchanged.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface
open Transformer.Basis

/-- Additional phase embeddings leave the actual prenorm ONE projection unchanged.
Source: the independent first and second coordinates in the original fused QKV input. -/
theorem phase_projected_eq_count (eps : ℝ) (token : Fin countConfig.vocab_size) :
    countQKV (rmsNormEps eps (phaseEmbedding token)) =
      countQKV (rmsNormEps eps (countEmbedding token)) := by
  by_cases hone : token.val = 22
  · simp [phaseEmbedding, countEmbedding, hone]
  · simp only [phaseEmbedding, countEmbedding, ite_eq_right hone]
    split_ifs <;>
      simp [countQKV, ContinuousLinearMap.smulRight_apply, innerSL_apply_apply,
        rmsNormEps, controlUnit, phaseUnit, EuclideanSpace.inner_single_left]

/-- The phase model's actual first head is the same raw counting computation on all finite inputs.
Source: CausalMHA.forward at f11b6e2, with its true RMSNorm and fused projection. -/
theorem phase_first_head_eq_count (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (i : Fin T) :
    headAt countConfig (phaseParams.blocks ⟨0, by decide⟩).attn eps positions
        (embed countConfig phaseParams tokens) 0 i =
      headAt countConfig countAttn eps positions (embed countConfig countParams tokens) 0 i := by
  dsimp only [headAt, embed, phaseParams, countAttn, countParams]
  have hp := funext fun j => phase_projected_eq_count eps (tokens j)
  exact congrArg (fun projected : Fin T → EucSpace 192 =>
    attentionHead countConfig 0 eps
      (fun j => headSlice countConfig (qkvSlice countConfig (qkvQ countConfig) (projected j)) 0)
      (fun j => headSlice countConfig (qkvSlice countConfig (qkvK countConfig) (projected j)) 0)
      (fun j => headSlice countConfig (qkvSlice countConfig (qkvV countConfig) (projected j)) 0)
      positions i) hp

/-- The nonzero output projection keeps the ONE coordinate exactly.
Source: the rank-one W_o in phaseParams, with a unit first embedding coordinate. -/
theorem phaseOutput_probe (x : EucSpace 64) :
    inner (𝕜 := ℝ) controlUnit (phaseOutput x) = inner (𝕜 := ℝ) controlUnit x := by
  rw [phaseOutput, ContinuousLinearMap.smulRight_apply, innerSL_apply_apply,
    real_inner_smul_right, real_inner_self_eq_norm_sq, controlUnit_norm]
  ring

/-- The first model coordinate after merging is the first head's first coordinate.
Source: the original contiguous view/transpose/reshape at f11b6e2; this is a coordinate identity. -/
theorem phase_headMerge_probe (heads : Fin countConfig.n_heads → EucSpace countConfig.head_dim) :
    inner (𝕜 := ℝ) controlUnit (headMerge countConfig heads) =
      inner (𝕜 := ℝ) countDirection (heads 0) := by
  have hs : headSplit countConfig 0 = (0, ⟨0, by decide⟩) := by decide
  simp only [controlUnit, countDirection, EuclideanSpace.inner_single_left,
    headMerge_apply, hs]

/-- The actual phase model's first residual block retains the exact normalized raw ONE count.
Source: its true head, nonzero W_o and residual addition at f11b6e2; phase encoding is simultaneously preserved. -/
theorem phase_block_one_signal (eps : ℝ) (head last : Fin countConfig.vocab_size)
    (body : List (Fin countConfig.vocab_size)) (hlast : last.val ≠ 22) :
    inner (𝕜 := ℝ) controlUnit
        (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
          (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩ -
          phaseParams.embedding last) =
      ((decodeTokens (head :: (body ++ [last]))).count oneBit : ℝ) /
        (body.length + 2 : ℕ) * countScale eps := by
  have hlastToken : (head :: (body ++ [last])).get ⟨body.length + 1, by simp⟩ = last := by
    simp only [List.get_eq_getElem, List.getElem_cons_succ,
      List.getElem_append_right (le_refl body.length), Nat.sub_self, List.getElem_cons_zero]
  have hstate : hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
        (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩ - phaseParams.embedding last =
      attnSubLayer countConfig (phaseParams.blocks ⟨0, by decide⟩).attn eps
        (fun j => (j.val : ℝ)) (embed countConfig phaseParams (head :: (body ++ [last])).get)
        ⟨body.length + 1, by simp⟩ := by
    rw [hidden, dite_eq_left (by decide : 0 < countConfig.n_layers)]
    simp only [hidden]
    unfold blockForward
    simp only [phaseParams, ffnSubLayer, relu2FFN, zero_apply, add_zero]
    rw [embed, hlastToken]
    module
  rw [hstate, attnSubLayer_eq_heads]
  change inner (𝕜 := ℝ) controlUnit (phaseOutput (headMerge countConfig _)) = _
  rw [phaseOutput_probe, phase_headMerge_probe, phase_first_head_eq_count,
    count_head_raw_count eps head last body hlast]
  rw [real_inner_smul_right, real_inner_smul_right,
    real_inner_self_eq_norm_sq, countDirection_norm]
  ring

example : (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 := by decide

/-- The protected phase and normalized count both occur in the actual first hidden state of a raw parity prompt.
Source: the two-ONE no-scratchpad prompt at cbafbe9, retaining BOS/ZERO/SEP rather than a precomputed count. -/
theorem phase_prompt_features (eps : ℝ) :
    phaseProbe (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
        ([⟨1, by decide⟩, ⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩,
          ⟨21, by decide⟩, ⟨18, by decide⟩] : List (Fin countConfig.vocab_size)).get 1 ⟨5, by decide⟩) = 1 ∧
      inner (𝕜 := ℝ) controlUnit
        (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
          ([⟨1, by decide⟩, ⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩,
            ⟨21, by decide⟩, ⟨18, by decide⟩] : List (Fin countConfig.vocab_size)).get 1
          ⟨5, by decide⟩) = (1 / 3 : ℝ) * countScale eps := by
  constructor
  · rw [hidden_probe_preserved countConfig phaseParams eps phaseProbe
      phase_output_kernel.1 phase_output_kernel.2]
    have hu : ‖phaseUnit‖ = 1 := by simp [phaseUnit, PiLp.norm_single]
    simp [phaseParams, phaseEmbedding, phaseProbe, innerSL_apply_apply, hu]
  · have h := phase_block_one_signal eps ⟨1, by decide⟩ ⟨18, by decide⟩
      [⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩, ⟨21, by decide⟩] (by decide)
    have horth : inner (𝕜 := ℝ) controlUnit (phaseParams.embedding ⟨18, by decide⟩) = 0 := by
      simp [phaseParams, phaseEmbedding, controlUnit, phaseUnit, EuclideanSpace.inner_single_left]
    rw [inner_sub_right, horth, sub_zero] at h
    norm_num [decodeTokens, oneBit] at h
    simpa only [List.cons_append, List.nil_append, List.length_cons, List.length_nil] using h

/-- The correct answer continuation has the same count signal and the opposite protected phase.
Source: the valid equal-count six-token prefix requiring EOS in Parity.solve at cbafbe9. -/
theorem phase_answer_features (eps : ℝ) :
    phaseProbe (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
        ([⟨1, by decide⟩, ⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩,
          ⟨18, by decide⟩, ⟨24, by decide⟩] : List (Fin countConfig.vocab_size)).get 1 ⟨5, by decide⟩) = -1 ∧
      inner (𝕜 := ℝ) controlUnit
        (hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
          ([⟨1, by decide⟩, ⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩,
            ⟨18, by decide⟩, ⟨24, by decide⟩] : List (Fin countConfig.vocab_size)).get 1
          ⟨5, by decide⟩) = (1 / 3 : ℝ) * countScale eps := by
  constructor
  · rw [hidden_probe_preserved countConfig phaseParams eps phaseProbe
      phase_output_kernel.1 phase_output_kernel.2]
    have hu : ‖phaseUnit‖ = 1 := by simp [phaseUnit, PiLp.norm_single]
    simp [phaseParams, phaseEmbedding, phaseProbe, innerSL_apply_apply, hu]
  · have h := phase_block_one_signal eps ⟨1, by decide⟩ ⟨24, by decide⟩
      [⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩, ⟨18, by decide⟩] (by decide)
    have horth : inner (𝕜 := ℝ) controlUnit (phaseParams.embedding ⟨24, by decide⟩) = 0 := by
      simp [phaseParams, phaseEmbedding, controlUnit, phaseUnit, EuclideanSpace.inner_single_left]
    rw [inner_sub_right, horth, sub_zero] at h
    norm_num [decodeTokens, oneBit] at h
    simpa only [List.cons_append, List.nil_append, List.length_cons, List.length_nil] using h

/-- The ONE channel is active: its output projection sends the actual unit direction to itself.
Source: phaseParams' nonzero W_o; phase preservation does not require disabling all attention updates. -/
theorem phaseOutput_active : phaseOutput controlUnit = controlUnit := by
  rw [phaseOutput, ContinuousLinearMap.smulRight_apply, innerSL_apply_apply,
    real_inner_self_eq_norm_sq, controlUnit_norm]
  norm_num

end Transformer.GPTMini.Semantics
