import Transformer.GPTMini.Sparsemax.QKChart
import Transformer.GPTMini.Sparsemax.AnchorTransfer

/-!
# Persistent anchors implemented by actual normalized query/key scores

Derived construction for arXiv:1602.02068v2, §2.2 and §2.5, composed
with the actual `GPTMini.score` of `QKNormScores.forward` at `73f8a0b`,
without rotary positional maps, XSA or the output projection.
The gain `cap + 1 + sum exp ordinary` dominates every finite ordinary
score as well as every bounded anchor score. The unit-key chart therefore
realizes the previously proved anchor architecture at all finite anchor
parameters, using a fixed orthogonal unit query frame.

This changes the parameterization of keys and the per-head temperature;
it keeps the existing normalization, inner product, causal mask and
variational sparsemax. The result concerns one query row. The temperature
depends on this row's ordinary parameters, and independently parameterized
keys are not yet a theorem about shared token projections across rows.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex
open scoped BigOperators

/-- A finite gain strictly larger than every absolute score in the row.
Derived temperature restriction for §2.5 of arXiv:1602.02068v2 and
the normalized dot product at `73f8a0b`; ordinary parameters remain finite. -/
def qkAnchorGain {N : ℕ} (cap : ℝ) (ordinary : Fin N → ℝ) : ℝ :=
  cap + 1 + ∑ n : Fin N, Real.exp (ordinary n)

/-- A positive cap gives a positive gain with room above the anchor cap.
Source context: the derived Q/K realization of §2.5's score directions. -/
theorem qkAnchorGain_gt_cap {N : ℕ} (cap : ℝ) (ordinary : Fin N → ℝ) :
    cap < qkAnchorGain cap ordinary := by
  have hs : 0 ≤ ∑ n : Fin N, Real.exp (ordinary n) :=
    Finset.sum_nonneg (fun n _ => (Real.exp_pos (ordinary n)).le)
  unfold qkAnchorGain
  linarith

/-- Every score lies strictly in the gain's realizable interval.
Source: the derived architecture for §2.2 and §2.5 of arXiv:1602.02068v2;
the exponential sum controls ordinary scores without clipping them. -/
theorem qkAnchorGain_score_bounds {A N : ℕ} (cap : ℝ) (parameters : Fin A → ℝ)
    (ordinary : Fin N → ℝ) (hc : 0 < cap) (n : Fin (A + N)) :
    -qkAnchorGain cap ordinary < anchoredScores cap parameters ordinary n ∧
      anchoredScores cap parameters ordinary n < qkAnchorGain cap ordinary := by
  have hg := qkAnchorGain_gt_cap cap ordinary
  refine Fin.addCases ?_ ?_ n
  · intro a
    have hb := anchoredScores_anchor_bounds cap parameters ordinary hc a
    constructor <;> linarith
  · intro a
    have hs : Real.exp (ordinary a) ≤ ∑ n : Fin N, Real.exp (ordinary n) :=
      Finset.single_le_sum (fun n _ => (Real.exp_pos (ordinary n)).le) (Finset.mem_univ a)
    simp only [anchoredScores, Fin.addCases_right]
    unfold qkAnchorGain at *
    constructor <;> linarith [Real.exp_pos (ordinary a)]

/-- Gain and score-bound premises hold for a row with an exact sparse zero.
Source context: arXiv:1602.02068v2, §2.2 and §2.5, derived Q/K row. -/
example : ∀ n : Fin 3,
    -qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0) <
      anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) n ∧
    anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) n <
      qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0) :=
  qkAnchorGain_score_bounds (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) (by norm_num)

/-- Actual key vectors, parameterized through a fixed unit frame.
Derived Q/K restriction for arXiv:1602.02068v2, §2.5; the ordinary
QKNorm operator still computes their scores by normalized inner products. -/
def qkAnchoredKeys {h A N : ℕ} (query transverse : EucSpace h) (cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) : Fin (A + N) → EucSpace h :=
  fun n => qkKeyChart query transverse (qkAnchorGain cap ordinary)
    (anchoredScores cap parameters ordinary n)

/-- The QKNorm score row evaluated on the constructed key vectors.
Source: `QKNormScores.forward` at `73f8a0b`, with the derived key and
temperature restrictions for §2.5 of arXiv:1602.02068v2. -/
def qkAnchoredScores {h A N : ℕ} (query transverse : EucSpace h) (eps cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) : Fin (A + N) → ℝ :=
  fun n => score (Real.log (qkAnchorGain cap ordinary)) eps query
    (qkAnchoredKeys query transverse cap parameters ordinary n)

/-- The existing normalized dot product realizes the anchor scores exactly.
Source: `QKNormScores.forward` at `73f8a0b` and the derived score restrictions
for §2.2 and §2.5 of arXiv:1602.02068v2. -/
theorem qkAnchoredScores_eq {h A N : ℕ} (query transverse : EucSpace h) (eps cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ)
    (hq : ‖query‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) query transverse = 0) (he : eps ≤ 1) (hc : 0 < cap) :
    qkAnchoredScores query transverse eps cap parameters ordinary =
      anchoredScores cap parameters ordinary := by
  funext n
  have hb := qkAnchorGain_score_bounds cap parameters ordinary hc n
  exact qkKeyChart_score query transverse _ eps _ hq ht ho
    (lt_trans hc (qkAnchorGain_gt_cap cap ordinary)) he hb.1 hb.2

/-- Every normalization and score premise is inhabited at standard epsilon.
Source context: §2.5's derived unit-key realization at `73f8a0b`. -/
example : qkAnchoredScores qkQueryExample qkTransverseExample (1 / 1000000) (1 / 8)
    (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) =
    anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) :=
  qkAnchoredScores_eq _ _ _ _ _ _ qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num)

/-- All actual keys have unit norm at every finite learned assignment.
Source: the derived restriction for §2.5 of arXiv:1602.02068v2,
realized through the normalized dot product at `73f8a0b`. -/
theorem qkAnchoredKeys_norm {h A N : ℕ} (query transverse : EucSpace h) (cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ)
    (hq : ‖query‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) query transverse = 0) (hc : 0 < cap) (n : Fin (A + N)) :
    ‖qkAnchoredKeys query transverse cap parameters ordinary n‖ = 1 := by
  have hb := qkAnchorGain_score_bounds cap parameters ordinary hc n
  exact qkKeyChart_norm query transverse _ _ hq ht ho
    (lt_trans hc (qkAnchorGain_gt_cap cap ordinary)) hb.1 hb.2

/-- The unit-key hypotheses have a finite row with an inactive ordinary slot.
Source context: arXiv:1602.02068v2, §2.5, derived Q/K row. -/
example : ∀ n : Fin 3, ‖qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8)
    (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) n‖ = 1 :=
  qkAnchoredKeys_norm qkQueryExample qkTransverseExample (1 / 8)
    (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num)

/-- Every learned anchor assignment gives a differentiable array of actual keys.
Source: the derived upstream Q/K restriction for arXiv:1602.02068v2,
§2.5; the ordinary scores and temperature are fixed on this parameter path. -/
theorem qkAnchoredKeys_differentiableAt {h A N : ℕ} (query transverse : EucSpace h)
    (cap : ℝ) (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) (hc : 0 < cap) :
    DifferentiableAt ℝ (fun p => qkAnchoredKeys query transverse cap p ordinary) parameters := by
  apply differentiableAt_pi.mpr
  intro n
  have hb := qkAnchorGain_score_bounds cap parameters ordinary hc n
  have hd := qkKeyChart_differentiableAt query transverse _ _
    (lt_trans hc (qkAnchorGain_gt_cap cap ordinary)) hb.1 hb.2
  apply hd.comp parameters
  refine Fin.addCases ?_ ?_ n
  · intro a
    simpa only [anchoredScores, Fin.addCases_left, Function.comp_def,
      ContinuousLinearMap.proj_apply] using
      (boundedCoordinate_hasDerivAt cap (parameters a)).differentiableAt.comp parameters
        (ContinuousLinearMap.proj a : (Fin A → ℝ) →L[ℝ] ℝ).differentiableAt
  · intro a
    simp only [anchoredScores, Fin.addCases_right]
    exact differentiableAt_const _

/-- Actual key differentiability has finite sparse-row premises.
Source context: arXiv:1602.02068v2, §2.5, derived unit-key row. -/
example : DifferentiableAt ℝ
    (fun p : Fin 2 → ℝ => qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8) p
      (fun _ : Fin 1 => 0)) (fun _ => 0) :=
  qkAnchoredKeys_differentiableAt _ _ _ _ _ (by norm_num)

/-- Visible exact zeros persist after replacing free scores by actual Q/K scores.
Source: §2.2 of arXiv:1602.02068v2, the derived anchor architecture,
evaluated through `QKNormScores.forward`'s normalization at `73f8a0b`. -/
theorem qkAnchoredScores_sparse_example : sparseWeights
    (qkAnchoredScores qkQueryExample qkTransverseExample (1 / 1000000) (1 / 8)
      (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2 =
    (fun n : Fin 3 => if n = 2 then 0 else 1 / 2) := by
  rw [qkAnchoredScores_eq _ _ _ _ _ _ qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num)]
  exact anchoredScores_sparse_example

end Transformer.GPTMini.Sparsemax
