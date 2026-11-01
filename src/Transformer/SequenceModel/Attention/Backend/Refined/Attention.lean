module

public import Transformer.SequenceModel.Attention.Backend.Refined.Scores
public import Transformer.SequenceModel.Attention.Backend.Refined.Softmax
public import Transformer.SequenceModel.Attention.Backend.Refined.Linear
public import Transformer.SequenceModel.Attention.Backend.Refined.NormOutputs

/-!
# Local bounds through SDPA, XSA and the first residual

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M4.
The first attention output is structurally zero. Its small RMS denominator
therefore cannot amplify an error in another position or operation.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Backend.Refined
open GPTMini Stability Execution Complete Adaptive
open scoped BigOperators

/-- M4: score and value perturbations, plus the captured SDPA residual. -/
theorem error_average (i : Fin 2) :
    ‖raw_average i - (sourceAverage i)‖ ≤ 31 / 10000000 := by
  have hl : ‖raw_average i - ∑ j, causalWeight (fun a b => raw_scores a b) i j • vPart j‖ ≤
      3 / 10000000 := by fin_cases i; exact local_average_0; exact local_average_1
  have h := average_step (fun a b => raw_scores a b) (sourceAttention.scores 0)
    vPart (fun j => (sourceV j)) (raw_average i) i
    (11 / 10) (25 / 100000000) (6 / 10000000) (3 / 10000000)
    (fun j hj => error_scores i j) error_vpart bound_value hl
  exact h.trans (by norm_num)

/-- M4: own-value normalization retains its recorded norm, rather than 1/xeps. -/
theorem error_direction (i : Fin 2) :
    ‖raw_direction i - (sourceDirection i)‖ ≤ 135 / 100000000 := by
  have h := l2_step (eps := 1 / 1000000000000) (by norm_num) (vPart i)
    ((sourceV i)) (raw_direction i)
    2 (6 / 10000000) (15 / 100000000) (by norm_num) (gain_direction i)
    (error_vpart i) (local_direction i)
  exact h.trans (by norm_num)

/-- M4: XSA keeps both moving inputs and the non-unit recorded direction. -/
theorem error_heads (i : Fin 2) :
    ‖raw_heads i - (sourceHeads i)‖ ≤ 93 / 10000000 := by
  fin_cases i
  · convert (show ‖(0 : EucSpace 4)‖ ≤ (93 / 10000000 : ℝ) by norm_num) using 1
    congr 1
    ext c
    change raw_heads 0 c - sourceHeads 0 c = 0
    fin_cases c <;> norm_num [raw_heads, sourceHeads, sourceAttention, exactAttention] <;> rfl
  · have hn : ‖(sourceAverage 1)‖ ≤ 111 / 100 := by
      have h := norm_of_error ((sourceAverage 1)) (raw_average 1)
        (31 / 10000000) (11 / 10) (by rw [norm_sub_rev]; exact error_average 1) (bound_average 1)
      exact h.trans (by norm_num)
    have hu : ‖(sourceDirection 1)‖ ≤ 1 :=
      normL2_norm_le (1 / 1000000000000) (by norm_num) _
    have h := xsa_pair_error (raw_average 1) ((sourceAverage 1))
      (raw_direction 1) ((sourceDirection 1)) (31 / 10000000)
      (135 / 100000000) (111 / 100) (error_average 1) (error_direction 1) hn hu
    have ht := (norm_sub_le_norm_sub_add_norm_sub (raw_heads 1)
      (raw_average 1 - inner ℝ (raw_average 1) (raw_direction 1) • raw_direction 1)
      ((sourceHeads 1))).trans (add_le_add (local_heads 1) h)
    exact ht.trans (by norm_num)

/-- M4: source one-head merge is the exact inverse reshape. -/
theorem source_merge (i : Fin 2) : headMerge cfg (sourceAttention.heads i) =
    (sourceHeads i) := by
  ext c
  fin_cases c <;> rfl

/-- M4: actual output projection after XSA. -/
theorem error_projected (i : Fin 2) :
    ‖raw_projected i - sourceAttention.projected i‖ ≤ 2424 / 1000000000 := by
  have hl := local_projected i
  rw [merge_heads] at hl
  have h := linear_step blockWeights.projection (raw_heads i) ((sourceHeads i))
    (raw_projected i) (13 / 50) (93 / 10000000) (6 / 1000000000)
    projection_norm (error_heads i) hl
  change ‖raw_projected i - blockWeights.projection (headMerge cfg (sourceAttention.heads i))‖ ≤ _
  rw [source_merge]
  exact h.trans (by norm_num)

/-- M4: branch post-RMS uses its recorded denominator only where the branch is active. -/
theorem error_attention (i : Fin 2) :
    ‖raw_attn_output i - sourceAttention.output i‖ ≤ 146 / 1000000 := by
  fin_cases i
  · have hz : sourceAttention.projected 0 = 0 := by
      change blockWeights.projection (headMerge cfg (sourceAttention.heads 0)) = 0
      rw [source_merge]
      change blockWeights.projection 0 = 0
      exact map_zero _
    change ‖raw_attn_output 0 - rmsNormEps (1 / 8388608) (sourceAttention.projected 0)‖ ≤ _
    rw [hz]
    norm_num [raw_attn_output, rmsNormEps, EuclideanSpace.norm_eq, Fin.sum_univ_succ]
  · have h := rms_step (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
      (raw_projected 1) (sourceAttention.projected 1) (raw_attn_output 1)
      60 (2424 / 1000000000) (21 / 100000000) (by norm_num) gain_attn
      (error_projected 1) (local_attn_output 1)
    exact h.trans (by norm_num)

/-- M4: the first residual sum has an exactly zero local arithmetic residual. -/
theorem error_mid (i : Fin 2) : ‖raw_mid i - sourceMid i‖ ≤ 147 / 1000000 := by
  have h := add_step (raw_initial i) (raw_attn_output i) (sourceInitial i)
    (sourceAttention.output i) (raw_mid i) (1 / 4000000) (146 / 1000000) 0
    (error_initial i) (error_attention i) (local_mid i)
  exact h.trans (by norm_num)

end Transformer.SequenceModel.Attention.Backend.Refined
