import Transformer.GPTMini.Sparsemax.GramSupport

/-!
# Limits of the convex embedding and routing foundation

Derived counterexamples for arXiv:1602.02068v2, §2.2, Proposition 1.
The PSD embedding domain and the joint projection energy are convex, but
the equation `weights = sparsemax(scores)` has a nonconvex graph when
supports change. The witness uses genuine shared embedding Grams in the
bounded domain and never has a singleton route.

Keeping all positions active removes that particular projection boundary,
but trainable value mixing still multiplies two learned quantities. An
actual two-position sparsemax readout violates midpoint affinity while
both endpoint supports remain full. These are architectural boundaries,
independent of any chosen task loss or FFN. The new energy formulation
therefore must not be advertised as an exact convex reformulation of
ordinary end-to-end sparsemax transformer training.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- The midpoint of two feasible, genuinely different embedding Grams.
Source: the derived bounded-Gram witness for sparsemax Proposition 1. -/
def gramExampleMid : EmbeddingGram 3 :=
  (1 / 2 : ℝ) • (0 : EmbeddingGram 3) + (1 / 2 : ℝ) • gramSparseExample

/-- Its embedding constraints remain valid along the convex segment.
Source: the bounded PSD domain in the derived architecture. -/
theorem gramExampleMid_mem_domain : gramExampleMid ∈ embeddingGramDomain 3 (3 / 8) := by
  convert gramExampleSegment_feasible (1 / 2) (by norm_num) (by norm_num) using 1
  norm_num [gramExampleMid]

/-- The true middle sparsemax row differs from the mean endpoint row.
Source: sparsemax Proposition 1, with threshold `-41/150` at the midpoint. -/
theorem gramExampleMid_weights :
    embeddingRoutes gramExampleMid gramExampleTokens gramExampleRows 0 =
      fun j : Fin 3 => if j = 2 then 7 / 75 else 34 / 75 := by
  change sparseWeights (embeddingScores gramExampleMid (fun j => j) 2) 2 = _
  rw [gramExampleMid, embeddingScores_affine, gramSparseExample_scores]
  let scores : Fin 3 → ℝ := fun j => if j = 2 then -(9 / 50) else 9 / 50
  have hs : (1 / 2 : ℝ) • embeddingScores (0 : EmbeddingGram 3) (fun j => j) 2 +
      (1 / 2 : ℝ) • (fun j : Fin 3 => if j = 2 then -(9 / 25) else 9 / 25) = scores := by
    funext j
    fin_cases j <;> norm_num [embeddingScores, scores]
  rw [hs]
  have hsum : ∑ j, thresholdWeights scores 2 (-(41 / 150)) j = 1 := by
    norm_num [thresholdWeights, scores, Fin.sum_univ_three]
  rw [← thresholdWeights_eq_sparseWeights scores _ _ hsum]
  funext j
  fin_cases j <;> norm_num [thresholdWeights, scores]

/-- Exact optimizer equality as a constraint on learned Grams and rows.
Source: sparsemax Eq. (1). This is a predicate of its parameter arguments,
not a convexity assumption or a substitute definition of sparsemax. -/
def exactGramRoutingGraph (V : ℕ) {R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) :
    Set (EmbeddingGram V × (Fin R → Fin T → ℝ)) :=
  {p | p.1 ∈ embeddingGramDomain V cap ∧ p.2 = embeddingRoutes p.1 tokens rows}

/-- Even bounded PSD embeddings with two guaranteed active positions have
a nonconvex exact sparsemax graph. Source: a derived counterexample to
unconditional convexification of sparsemax Eq. (1), not a paper claim. -/
theorem exactGramRoutingGraph_not_convex :
    ¬ Convex ℝ (exactGramRoutingGraph 3 (3 / 8) gramExampleTokens gramExampleRows) := by
  intro hconv
  let p : EmbeddingGram 3 × (Fin 1 → Fin 3 → ℝ) :=
    (0, embeddingRoutes 0 gramExampleTokens gramExampleRows)
  let q : EmbeddingGram 3 × (Fin 1 → Fin 3 → ℝ) :=
    (gramSparseExample, embeddingRoutes gramSparseExample gramExampleTokens gramExampleRows)
  have hp : p ∈ exactGramRoutingGraph 3 (3 / 8) gramExampleTokens gramExampleRows :=
    ⟨zero_mem_embeddingGramDomain _ _ (by norm_num), rfl⟩
  have hq : q ∈ exactGramRoutingGraph 3 (3 / 8) gramExampleTokens gramExampleRows :=
    ⟨gramSparseExample_mem_domain, rfl⟩
  have hm := hconv hp hq (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  have hr := congrFun (congrFun hm.2 0) 2
  change (1 / 2 : ℝ) * embeddingRoutes (0 : EmbeddingGram 3)
      gramExampleTokens gramExampleRows 0 2 +
      (1 / 2 : ℝ) * embeddingRoutes gramSparseExample gramExampleTokens gramExampleRows 0 2 =
    embeddingRoutes gramExampleMid gramExampleTokens gramExampleRows 0 2 at hr
  rw [gramZeroExample_weights, gramSparseExample_weights, gramExampleMid_weights] at hr
  norm_num at hr

/-- Dropping the score-square term gives an actual Jensen violation in
the lifted row objective. Source: sparsemax Eq. (1) versus its fixed-score
expansion; both endpoints use feasible bounded Grams and their causal rows. -/
theorem routingObjective_gram_midpoint_violation :
    (routingObjective (fun j => -embeddingScores (0 : EmbeddingGram 3) (fun j => j) 2 j / 2)
      (embeddingRoutes 0 gramExampleTokens gramExampleRows 0) +
      routingObjective (fun j => -embeddingScores gramSparseExample (fun j => j) 2 j / 2)
        (embeddingRoutes gramSparseExample gramExampleTokens gramExampleRows 0)) / 2 <
      routingObjective (fun j => -embeddingScores gramExampleMid (fun j => j) 2 j / 2)
        ((1 / 2 : ℝ) • embeddingRoutes (0 : EmbeddingGram 3) gramExampleTokens gramExampleRows 0 +
          (1 / 2 : ℝ) • embeddingRoutes gramSparseExample gramExampleTokens gramExampleRows 0) := by
  rw [gramZeroExample_weights, gramSparseExample_weights]
  simp only [gramExampleMid, embeddingScores_affine, gramSparseExample_scores]
  norm_num [routingObjective, embeddingScores, Fin.sum_univ_three]

/-- An actual two-position sparsemax value sum with both learned factors.
Source: sparsemax Eq. (1) followed by `attn @ v` at `73f8a0b`. The first
coordinate controls a score difference; the second is a learned value. -/
def twoSlotReadout (a v : ℝ) : ℝ :=
  ∑ j : Fin 2, sparseWeights (fun k => if k = 0 then -a else a) 1 j *
    (if j = 0 then 0 else v)

/-- Full-support bounded scores still give a bilinear learned-value readout.
Source: sparsemax Proposition 1 with threshold `-1/2`, followed by the
original value mixture. The interval keeps each weight at least `1/4`. -/
theorem twoSlotReadout_formula (a v : ℝ) (hl : -(1 / 4) ≤ a) (hu : a ≤ 1 / 4) :
    twoSlotReadout a v = (1 / 2 + a) * v := by
  let scores : Fin 2 → ℝ := fun j => if j = 0 then -a else a
  have ht (j : Fin 2) : thresholdWeights scores 1 (-(1 / 2)) j =
      if j = 0 then 1 / 2 - a else 1 / 2 + a := by
    fin_cases j
    · change max (-a - (-(1 / 2) : ℝ)) 0 = 1 / 2 - a
      rw [max_eq_left (by linarith)]
      ring
    · change max (a - (-(1 / 2) : ℝ)) 0 = 1 / 2 + a
      rw [max_eq_left (by linarith)]
      ring
  have hsum : ∑ j, thresholdWeights scores 1 (-(1 / 2)) j = 1 := by
    simp_rw [ht]
    norm_num [Fin.sum_univ_two]
  unfold twoSlotReadout
  rw [← thresholdWeights_eq_sparseWeights scores _ _ hsum]
  simp only [Fin.sum_univ_two, ht]
  norm_num

/-- Interior scores and a nonzero learned value inhabit both hypotheses. -/
example : twoSlotReadout 0 2 = (1 / 2 + (0 : ℝ)) * 2 :=
  twoSlotReadout_formula _ _ (by norm_num) (by norm_num)

/-- Ordinary value mixing is not affine even on a fixed full sparsemax
support. Source: the derived two-position witness for `attn @ v` at
`73f8a0b`; no task loss or FFN is used in the counterexample. -/
theorem twoSlotReadout_midpoint_failure :
    twoSlotReadout (((1 / 4 : ℝ) + (-(1 / 4))) / 2) ((1 + (-1 : ℝ)) / 2) ≠
      (twoSlotReadout (1 / 4) 1 + twoSlotReadout (-(1 / 4)) (-1)) / 2 := by
  have hmid := twoSlotReadout_formula 0 0 (by norm_num) (by norm_num)
  have hleft := twoSlotReadout_formula (1 / 4) 1 (by norm_num) (by norm_num)
  have hright := twoSlotReadout_formula (-(1 / 4)) (-1) (by norm_num) (by norm_num)
  norm_num at hmid hleft hright ⊢
  rw [hmid, hleft, hright]
  norm_num

/-- The exact graph of bounded full-support attention with learned values.
Source: the two-position sparsemax/value mixture above, independent of a
task loss. Its score and value intervals are convex parameter constraints. -/
def twoSlotValueGraph : Set ((ℝ × ℝ) × ℝ) :=
  {p | -(1 / 4) ≤ p.1.1 ∧ p.1.1 ≤ 1 / 4 ∧ -1 ≤ p.1.2 ∧ p.1.2 ≤ 1 ∧
    p.2 = twoSlotReadout p.1.1 p.1.2}

/-- Even with bounded values and unchanged full support, the exact output
graph is nonconvex. Source: the derived counterexample for the ordinary
`attn @ v` at `73f8a0b`; narrowing the FFN or task loss cannot remove it. -/
theorem twoSlotValueGraph_not_convex : ¬ Convex ℝ twoSlotValueGraph := by
  intro hconv
  have hl := twoSlotReadout_formula (1 / 4) 1 (by norm_num) (by norm_num)
  have hr := twoSlotReadout_formula (-(1 / 4)) (-1) (by norm_num) (by norm_num)
  have hz := twoSlotReadout_formula 0 0 (by norm_num) (by norm_num)
  have hp : (((1 / 4 : ℝ), (1 : ℝ)), (3 / 4 : ℝ)) ∈ twoSlotValueGraph := by
    norm_num [twoSlotValueGraph, hl]
  have hq : (((-(1 / 4) : ℝ), (-1 : ℝ)), (-(1 / 4) : ℝ)) ∈ twoSlotValueGraph := by
    norm_num [twoSlotValueGraph, hr]
  have hm := hconv hp hq (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  norm_num [twoSlotValueGraph, hz] at hm

end Transformer.GPTMini.Sparsemax
