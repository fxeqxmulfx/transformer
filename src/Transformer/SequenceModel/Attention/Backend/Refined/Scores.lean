module

public import Transformer.SequenceModel.Attention.Backend.Refined.Source
public import Transformer.SequenceModel.Attention.Backend.Refined.Arithmetic
public import Transformer.SequenceModel.Attention.Backend.Refined.Scalars

/-!
# Propagating the captured input through Q/K/V and exact source RoPE

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M4.
Every radius below follows from local operation checks and exact stored
weights. No whole-state error or final classification margin is assumed.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Backend.Refined
open GPTMini Stability Execution Complete Adaptive

/-- M4: normalized embedding error, with no stored-embedding perturbation. -/
theorem error_initial (i : Fin 2) : ‖raw_initial i - sourceInitial i‖ ≤ 1 / 4000000 :=
  local_initial i

/-- M4: attention pre-RMS uses the recorded residual norm. -/
theorem error_pre (i : Fin 2) : ‖raw_pre i - sourceAttention.pre i‖ ≤ 53 / 100000000 := by
  have h := rms_step (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
    (raw_initial i) (sourceInitial i) (raw_pre i) (21 / 10) (1 / 4000000)
    (3 / 1000000000000) (by norm_num) (gain_pre i) (error_initial i) (local_pre i)
  exact h.trans (by norm_num)

/-- M4: the Q slice retains its own matrix and arithmetic scale. -/
theorem error_qpart (i : Fin 2) :
    ‖qPart i - (sourceQ i)‖ ≤ 71 / 1000000000000 := by
  have h := linear_step qMap (raw_pre i) (sourceAttention.pre i) (qPart i)
    (1 / 8000) (53 / 100000000) (4 / 1000000000000) qMap_norm (error_pre i) (local_qpart i)
  rw [qMap_apply] at h
  exact h.trans (by norm_num)

/-- M4: the K slice retains its own matrix and arithmetic scale. -/
theorem error_kpart (i : Fin 2) :
    ‖kPart i - (sourceK i)‖ ≤ 71 / 1000000000000 := by
  have h := linear_step kMap (raw_pre i) (sourceAttention.pre i) (kPart i)
    (1 / 8000) (53 / 100000000) (4 / 1000000000000) kMap_norm (error_pre i) (local_kpart i)
  rw [kMap_apply] at h
  exact h.trans (by norm_num)

/-- M4: the V slice retains its own matrix and arithmetic scale. -/
theorem error_vpart (i : Fin 2) :
    ‖vPart i - (sourceV i)‖ ≤ 6 / 10000000 := by
  have h := linear_step vMap (raw_pre i) (sourceAttention.pre i) (vPart i)
    (101 / 100) (53 / 100000000) (4 / 100000000) vMap_norm (error_pre i) (local_vpart i)
  rw [vMap_apply] at h
  exact h.trans (by norm_num)

/-- M4: Q normalization uses its own verified recorded denominator. -/
theorem error_qnorm (i : Fin 2) : ‖raw_qnorm i - (sourceQNorm i)‖ ≤ 48 / 100000000 := by
  have h := rms_step (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
    (qPart i) ((sourceQ i)) (raw_qnorm i)
    6000 (71 / 1000000000000) (5 / 100000000) (by norm_num) (gain_qnorm i)
    (error_qpart i) (local_qnorm i)
  exact h.trans (by norm_num)

/-- M4: K normalization uses its own verified recorded denominator. -/
theorem error_knorm (i : Fin 2) : ‖raw_knorm i - (sourceKNorm i)‖ ≤ 48 / 100000000 := by
  have h := rms_step (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
    (kPart i) ((sourceK i)) (raw_knorm i)
    6000 (71 / 1000000000000) (5 / 100000000) (by norm_num) (gain_knorm i)
    (error_kpart i) (local_knorm i)
  exact h.trans (by norm_num)

/-- M4: query movement includes exact-phase table error and application rounding. -/
theorem error_query (i : Fin 2) : ‖raw_query i - (sourceQuery i)‖ ≤ 6 / 10000000 := by
  have h := rotate_step (angles 0 i) (raw_cos i) (raw_sin i) (raw_qnorm i)
    ((sourceQNorm i)) (raw_query i) (2 / 5) (48 / 100000000)
    (2 / 100000000) (1 / 10000000) (by norm_num) (bound_qnorm i)
    (error_qnorm i) (local_query i) (fun c => (local_tables i c).1) (fun c => (local_tables i c).2)
  exact h.trans (by norm_num)

/-- M4: key movement includes exact-phase table error and application rounding. -/
theorem error_key (i : Fin 2) : ‖raw_key i - (sourceKey i)‖ ≤ 6 / 10000000 := by
  have h := rotate_step (angles 0 i) (raw_cos i) (raw_sin i) (raw_knorm i)
    ((sourceKNorm i)) (raw_key i) (2 / 5) (48 / 100000000)
    (2 / 100000000) (1 / 10000000) (by norm_num) (bound_knorm i)
    (error_knorm i) (local_key i) (fun c => (local_tables i c).1) (fun c => (local_tables i c).2)
  exact h.trans (by norm_num)

/-- M4: actual source scores follow from both independently perturbed Q/K vectors. -/
theorem error_scores (i j : Fin 2) :
    |raw_scores i j - sourceAttention.scores 0 i j| ≤ 25 / 100000000 := by
  have h := score_step (raw_query i) (raw_key j) ((sourceQuery i))
    ((sourceKey j)) (raw_scores i j) (2 / 5) (6 / 10000000) (2 / 1000000000)
    (error_query i) (error_key j) (bound_query i) (bound_key j) (local_scores i j)
  exact h.trans (by norm_num)

end Transformer.SequenceModel.Attention.Backend.Refined
