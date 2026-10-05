import Transformer.GPTMini.Sparsemax.ClippedKeySegment

/-!
# Two different row targets corrected by one shared key matrix

Derived finite instance for arXiv:1602.02068v2, §2.2 and §2.5,
and shared Q/K projections and QKNorm at `73f8a0b`. Two causal queries
read the same four-position context and value array. Their ordinary output
targets are `11/16` and `5/16`; both initial outputs are `1/2`, with
opposite errors. One shared key correction realizes both targets.

Epsilon is explicitly one: projected keys lie in its closed ball, so
normalization is linear on the shared matrix segment. Two anchor positions
remain active at both endpoints, with all ordinary weights exactly zero.
This is an inhabited witness for the multi-row hypotheses; it is not
an experiment, a floating-point certificate or a theorem of full-model
convergence. There are no private row parameters or attention labels.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Two actual query columns for the two ordinary causal rows.
Source: shared projections at `73f8a0b`, in the derived §2.5 example. -/
def sharedExampleQueries : Fin 4 → EucSpace 2 :=
  fun n => if n = 2 then qkQueryExample else if n = 3 then qkTransverseExample else 0

/-- A common low ordinary key, used by both query rows.
Source: the derived clipped-key construction for §2.5, with gain three. -/
def sharedExampleOrdinaryKey : EucSpace 2 :=
  (-(1 / 2) : ℝ) • qkQueryExample + (-(1 / 2) : ℝ) • qkTransverseExample

/-- One anchor-key change influencing the two query rows with opposite signs.
Source: the actual shared key directions at `73f8a0b`, in the derived example. -/
def sharedExampleChange : EucSpace 2 :=
  (-(1 / 16) : ℝ) • qkQueryExample + (1 / 16 : ℝ) • qkTransverseExample

/-- Initial actual shared key columns, with zero anchor scores.
Source: the derived §2.2 and §2.5 example at `73f8a0b`. -/
def sharedExampleStart : Fin 4 → EucSpace 2 :=
  fun n => if n = 0 ∨ n = 1 then 0 else sharedExampleOrdinaryKey

/-- A single corrected shared matrix for both ordinary targets.
Source: the derived §2.5 construction; ordinary key columns stay fixed. -/
def sharedExampleStop : Fin 4 → EucSpace 2 :=
  fun n => if n = 0 then sharedExampleChange else
    if n = 1 then -sharedExampleChange else sharedExampleOrdinaryKey

/-- Frozen values shared by both rows, with arbitrary values at inactive positions.
Source: the value sum at `73f8a0b`, after the actual causal sparsemax. -/
def sharedExampleValues : Fin 4 → ℝ :=
  fun n => if n = 0 then 0 else if n = 1 then 1 else if n = 2 then 7 else 9

/-- Two ordinary task targets; these are readout values, not attention routes.
Source context: the derived summed squared-output example for §2.5. -/
def sharedExampleTargets : Fin 2 → ℝ := fun r => if r = 0 then 11 / 16 else 5 / 16

/-- All initial keys fit the actual epsilon-one clipping regime.
Source: the derived upstream condition for §2.5 of arXiv:1602.02068v2. -/
theorem sharedExampleStart_bound (n : Fin 4) : ‖sharedExampleStart n‖ ≤ 1 := by
  have ho := norm_add_le ((-(1 / 2) : ℝ) • qkQueryExample)
    ((-(1 / 2) : ℝ) • qkTransverseExample)
  norm_num [norm_smul, qkFrame_example.1, qkFrame_example.2.1] at ho
  fin_cases n <;> norm_num [sharedExampleStart, sharedExampleOrdinaryKey, ho]

/-- All corrected keys satisfy the same clipping bound.
Source: the derived shared key segment condition for §2.5. -/
theorem sharedExampleStop_bound (n : Fin 4) : ‖sharedExampleStop n‖ ≤ 1 := by
  have hc := norm_add_le ((-(1 / 16) : ℝ) • qkQueryExample) ((1 / 16 : ℝ) • qkTransverseExample)
  norm_num [norm_smul, qkFrame_example.1, qkFrame_example.2.1] at hc
  have ho := sharedExampleStart_bound 2
  norm_num [sharedExampleStart] at ho
  have hb : ‖sharedExampleChange‖ ≤ 1 := by
    change ‖(-(1 / 16) : ℝ) • qkQueryExample + (1 / 16 : ℝ) • qkTransverseExample‖ ≤ 1
    norm_num
    linarith
  fin_cases n <;> norm_num [sharedExampleStop, norm_neg, hb, ho]

/-- Both actual initial projected score rows have zero anchors and low ordinary scores.
Source: shared projections and QKNorm at `73f8a0b`, with epsilon one. -/
theorem sharedExampleScores_start (r : Fin 2) :
    projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStart
      Transformer.ConvexRecall.basis (Fin.natAdd 2 r) =
      (fun n : Fin 4 => if n = 0 ∨ n = 1 then 0 else -(3 / 2)) := by
  rw [projectedQKScores_basis]
  funext n
  have hk := normL2_of_norm_le 1 (sharedExampleStart n) (sharedExampleStart_bound n)
  simp only [score]
  rw [hk]
  fin_cases r <;> fin_cases n <;>
    norm_num [sharedExampleQueries, sharedExampleStart, sharedExampleOrdinaryKey,
      normL2_unit 1 qkQueryExample (by norm_num) qkFrame_example.1,
      normL2_unit 1 qkTransverseExample (by norm_num) qkFrame_example.2.1,
      Real.exp_log (by norm_num : (0 : ℝ) < 3), inner_add_right, inner_smul_right,
      real_inner_self_eq_norm_sq, qkFrame_example.1, qkFrame_example.2.1,
      qkFrame_example.2.2, real_inner_comm qkQueryExample qkTransverseExample]

/-- One actual corrected key matrix produces opposite anchor-score transfers.
Source: the derived §2.5 shared direction, through QKNorm at `73f8a0b`. -/
theorem sharedExampleScores_stop (r : Fin 2) :
    projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStop
      Transformer.ConvexRecall.basis (Fin.natAdd 2 r) =
      (fun n : Fin 4 => if n = 0 then (if r = 0 then -(3 / 16) else 3 / 16)
        else if n = 1 then (if r = 0 then 3 / 16 else -(3 / 16)) else -(3 / 2)) := by
  rw [projectedQKScores_basis]
  funext n
  have hk := normL2_of_norm_le 1 (sharedExampleStop n) (sharedExampleStop_bound n)
  simp only [score]
  rw [hk]
  fin_cases r <;> fin_cases n <;>
    norm_num [sharedExampleQueries, sharedExampleStop, sharedExampleChange,
      sharedExampleOrdinaryKey, normL2_unit 1 qkQueryExample (by norm_num) qkFrame_example.1,
      normL2_unit 1 qkTransverseExample (by norm_num) qkFrame_example.2.1,
      Real.exp_log (by norm_num : (0 : ℝ) < 3), inner_add_right, inner_smul_right, inner_neg_right,
      real_inner_self_eq_norm_sq, qkFrame_example.1, qkFrame_example.2.1,
      qkFrame_example.2.2, real_inner_comm qkQueryExample qkTransverseExample]

/-- Both initial actual sparse rows use precisely the two anchors.
Source: arXiv:1602.02068v2, §2.2, certified by the normalized threshold. -/
theorem sharedExampleWeights_start (r : Fin 2) : sparseWeights
    (projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStart
      Transformer.ConvexRecall.basis (Fin.natAdd 2 r)) (Fin.natAdd 2 r) =
      (fun n : Fin 4 => if n = 0 ∨ n = 1 then 1 / 2 else 0) := by
  rw [sharedExampleScores_start]
  let scores : Fin 4 → ℝ := fun n => if n = 0 ∨ n = 1 then 0 else -(3 / 2)
  have he : thresholdWeights scores (Fin.natAdd 2 r) (-(1 / 2)) =
      (fun n : Fin 4 => if n = 0 ∨ n = 1 then 1 / 2 else 0) := by
    funext n
    fin_cases r <;> fin_cases n <;> norm_num [thresholdWeights, scores]
  have hm : ∑ n, thresholdWeights scores (Fin.natAdd 2 r) (-(1 / 2)) n = 1 := by
    rw [he]
    norm_num [Fin.sum_univ_four]
  rw [← thresholdWeights_eq_sparseWeights scores _ _ hm]
  exact he

/-- The corrected actual sparse rows retain the same support and exact ordinary zeros.
Source: arXiv:1602.02068v2, §2.2, applied to the shared corrected matrix. -/
theorem sharedExampleWeights_stop (r : Fin 2) : sparseWeights
    (projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStop
      Transformer.ConvexRecall.basis (Fin.natAdd 2 r)) (Fin.natAdd 2 r) =
      (fun n : Fin 4 => if n = 0 then (if r = 0 then 5 / 16 else 11 / 16)
        else if n = 1 then sharedExampleTargets r else 0) := by
  rw [sharedExampleScores_stop]
  let scores : Fin 4 → ℝ := fun n => if n = 0 then (if r = 0 then -(3 / 16) else 3 / 16)
    else if n = 1 then (if r = 0 then 3 / 16 else -(3 / 16)) else -(3 / 2)
  have he : thresholdWeights scores (Fin.natAdd 2 r) (-(1 / 2)) =
      (fun n : Fin 4 => if n = 0 then (if r = 0 then 5 / 16 else 11 / 16)
        else if n = 1 then sharedExampleTargets r else 0) := by
    funext n
    fin_cases r <;> fin_cases n <;> norm_num [thresholdWeights, scores, sharedExampleTargets]
  have hm : ∑ n, thresholdWeights scores (Fin.natAdd 2 r) (-(1 / 2)) n = 1 := by
    rw [he]
    fin_cases r <;> norm_num [Fin.sum_univ_four, sharedExampleTargets]
  rw [← thresholdWeights_eq_sparseWeights scores _ _ hm]
  exact he

end Transformer.GPTMini.Sparsemax
