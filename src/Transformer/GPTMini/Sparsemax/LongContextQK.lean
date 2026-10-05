import Transformer.GPTMini.Sparsemax.PrefixInputs
import Transformer.GPTMini.Sparsemax.QKProjectedError

/-!
# Actual shared Q/K matrices with arbitrarily many dependent tokens

Derived instance for arXiv:1602.02068v2, §2.2 and §2.5, and
shared projections and QKNorm in `Attention.forward` at `73f8a0b`.
Two dedicated anchor channels and one ordinary embedding channel suffice
for any number of ordinary tokens. All ordinary inputs equal the same
nonzero vector, so full input independence fails for long contexts.
The explicit shared matrices still realize every finite anchor-score
assignment, keeping all ordinary projected keys fixed on that path.

This is one unrotated row before XSA and output projection. Repeated
ordinary embeddings are a concrete witness of the weaker accessibility
condition, not a claim that this example solves token-specific tasks.
The general partial-decoder result permits arbitrary ordinary embeddings
whenever the given matrix satisfies the stated score-family restriction.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Two anchor inputs and an arbitrary number of repeated nonzero ordinary inputs.
Source: the derived partial input accessibility example for §2.5 of
arXiv:1602.02068v2, using three actual input coordinates. -/
def longQKInputs (N : ℕ) : Fin (2 + N) → (Fin 3 → ℝ) :=
  prefixInputs (fun _ : Fin N => fun _ : Fin 1 => 1)

/-- A shared query matrix selecting the ordinary channel.
Source: the linear query projection at `73f8a0b`, in the derived example. -/
def longQKQueries : Fin 3 → EucSpace 2 := fun f => basis 2 f • qkQueryExample

/-- Shared key columns for any finite anchor parameters and context length.
Source: the derived unit-key chart for §2.5 of arXiv:1602.02068v2;
the third column is independent of anchor parameters and serves all ordinary tokens. -/
def longQKKeys (N : ℕ) (parameters : Fin 2 → ℝ) : Fin 3 → EucSpace 2 :=
  fun f => Fin.addCases
    (fun a : Fin 2 => qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8)
      parameters (fun _ : Fin N => 0) (Fin.castAdd N a))
    (fun _ : Fin 1 => qkKeyChart qkQueryExample qkTransverseExample
      (qkAnchorGain (1 / 8) (fun _ : Fin N => 0)) (-(1 / 8) - 1)) f

/-- The decoder identifies both anchors for every context length.
Source: the partial-channel restriction derived from §2.5 of arXiv:1602.02068v2. -/
theorem longQKInput_anchor (N : ℕ) (a : Fin 2) :
    prefixInputDecoder 2 1 (longQKInputs N (Fin.castAdd N a)) = basis a :=
  prefixInputDecoder_anchor _ _

/-- The decoder kills every repeated ordinary input, with no length bound.
Source: the same partial input accessibility construction for §2.5. -/
theorem longQKInput_ordinary (N : ℕ) (n : Fin N) :
    prefixInputDecoder 2 1 (longQKInputs N (Fin.natAdd 2 n)) = 0 :=
  prefixInputDecoder_ordinary _ _

/-- Contexts longer than width three genuinely fail full input independence.
Source: the derived rank obstruction for §2.5 of arXiv:1602.02068v2;
the partial decoder above still exists for these same inputs. -/
theorem longQKInputs_not_independent (N : ℕ) (hlong : 1 < N) :
    ¬ LinearIndependent ℝ (longQKInputs N) :=
  prefixInputs_not_independent_of_long _ hlong

/-- One hundred ordinary tokens satisfy the long-context obstruction premise.
Source context: arXiv:1602.02068v2, §2.5, derived partial decoder example. -/
example : ¬ LinearIndependent ℝ (longQKInputs 100) :=
  longQKInputs_not_independent 100 (by norm_num)

/-- Every ordinary row gets the certified unit query from the shared matrix.
Source: the actual query projection at `73f8a0b`, with the derived channels. -/
theorem longQKQueries_row (N : ℕ) (n : Fin N) :
    projectionEvaluation (longQKInputs N) longQKQueries (Fin.natAdd 2 n) = qkQueryExample := by
  rw [longQKInputs, projectionEvaluation_prefix_ordinary, frozenValueReadout_apply, Fin.sum_univ_one]
  norm_num [longQKQueries, basis]

/-- The one shared key matrix realizes the entire anchored family at every length.
Source: the derived partial-channel realization for §2.5 of arXiv:1602.02068v2,
evaluated by the linear key projection at `73f8a0b`. -/
theorem longQKKeys_row (N : ℕ) (parameters : Fin 2 → ℝ) :
    projectionEvaluation (longQKInputs N) (longQKKeys N parameters) =
      qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8) parameters (fun _ : Fin N => 0) := by
  funext j
  refine Fin.addCases ?_ ?_ j
  · intro a
    rw [longQKInputs, projectionEvaluation_prefix_anchor]
    simp only [longQKKeys, Fin.addCases_left]
  · intro n
    rw [longQKInputs, projectionEvaluation_prefix_ordinary, frozenValueReadout_apply, Fin.sum_univ_one]
    simp only [longQKKeys, Fin.addCases_right, one_smul, qkAnchoredKeys, anchoredScores, Real.exp_zero]

/-- Actual projected and normalized scores realize the anchors at every length.
Source: shared matrices and QKNorm at `73f8a0b`, realizing the derived
§2.2 and §2.5 anchor restriction on a dependent input family. -/
theorem longQKScores_eq (N : ℕ) (parameters : Fin 2 → ℝ) (n : Fin N) :
    projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin N => 0))) (1 / 1000000)
      longQKQueries (longQKKeys N parameters) (longQKInputs N) (Fin.natAdd 2 n) =
        anchoredScores (1 / 8) parameters (fun _ : Fin N => 0) := by
  rw [projectedQKScores_evaluation, longQKQueries_row, longQKKeys_row]
  change qkAnchoredScores qkQueryExample qkTransverseExample (1 / 1000000) (1 / 8)
    parameters (fun _ : Fin N => 0) = _
  exact qkAnchoredScores_eq _ _ (1 / 1000000) (1 / 8) parameters (fun _ : Fin N => 0)
    qkFrame_example.1 qkFrame_example.2.1 qkFrame_example.2.2 (by norm_num) (by norm_num)

/-- Both anchors precede every ordinary query, independently of context length.
Source: the visible-prefix requirement of the derived construction for §2.2. -/
theorem longQK_anchor_visible (N : ℕ) (n : Fin N) (a : Fin 2) :
    Fin.castAdd N a ≤ Fin.natAdd 2 n := by
  change (a : ℕ) ≤ 2 + (n : ℕ)
  omega

/-- The actual long row starts with two half weights and exact zeros on every ordinary token.
Source: arXiv:1602.02068v2, §2.2, applied to the derived scores and
the actual shared QKNorm matrices at `73f8a0b`. -/
theorem longQKProjection_initial (N : ℕ) (n : Fin N) : sparseWeights
    (projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin N => 0))) (1 / 1000000)
      longQKQueries (longQKKeys N (fun _ => 0)) (longQKInputs N) (Fin.natAdd 2 n)) (Fin.natAdd 2 n) =
      Fin.addCases (fun _ : Fin 2 => (1 / 2 : ℝ)) (fun _ : Fin N => 0) := by
  rw [longQKScores_eq]
  let scores := anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin N => 0)
  have hclosed : thresholdWeights scores (Fin.natAdd 2 n) (-(1 / 2)) =
      Fin.addCases (fun _ : Fin 2 => (1 / 2 : ℝ)) (fun _ : Fin N => 0) := by
    funext j
    refine Fin.addCases ?_ ?_ j
    · intro a
      norm_num [thresholdWeights, longQK_anchor_visible N n a, scores, anchoredScores,
        Fin.addCases_left, boundedCoordinate]
    · intro k
      norm_num [thresholdWeights, scores, anchoredScores, Fin.addCases_right]
  have hsum : ∑ j, thresholdWeights scores (Fin.natAdd 2 n) (-(1 / 2)) j = 1 := by
    rw [hclosed, Fin.sum_univ_add]
    norm_num [Fin.addCases_left, Fin.addCases_right]
  rw [← thresholdWeights_eq_sparseWeights scores (Fin.natAdd 2 n) (-(1 / 2)) hsum]
  exact hclosed

/-- The actual long row has a nonzero ordinary scalar output at initialization.
Source: the derived §2.5 value-anchor readout with ordinary values seven,
using the actual sparse weights and matrix projections at `73f8a0b`. -/
theorem longQKReadout_initial (N : ℕ) (n : Fin N) : frozenValueReadout
    (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin N => 7))
    (sparseWeights (projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin N => 0)))
      (1 / 1000000) longQKQueries (longQKKeys N (fun _ => 0)) (longQKInputs N) (Fin.natAdd 2 n))
      (Fin.natAdd 2 n)) = (1 / 2 : ℝ) := by
  rw [longQKProjection_initial, frozenValueReadout_apply, Fin.sum_univ_add]
  norm_num [anchoredValues, Fin.addCases_left, Fin.addCases_right, Fin.sum_univ_two,
    Fin.cases_zero, Fin.cases_succ]
  rw [Fin.sum_univ_two]
  norm_num
  rw [show (1 : Fin 2) = (0 : Fin 1).succ from rfl, Fin.cases_succ]

end Transformer.GPTMini.Sparsemax
