import Transformer.GPTMini.Sparsemax.GramRouting

/-!
# Jointly convex embedding and attention projection energy

Derived from arXiv:1602.02068v2, Eq. (1). The squared projection distance
is jointly convex in free row weights and the learned Gram matrix, because
their difference is linear in these lifted variables. All examples share
one Gram matrix, and no sparsemax support is fixed.

The score-square term is essential when scores become trainable. Dropping
it is harmless for inference at fixed scores, but loses joint convexity.
For any fixed Gram matrix the exact causal sparsemax rows minimize this
energy. Adding a future task objective depending on free row weights can
change that conditional optimizer; this is not a theorem about ordinary
backpropagation through the sparsemax output. No task loss is chosen here.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- The ordinary squared projection distance, summed across contexts.
Source: sparsemax Eq. (1), with learned Gram scores. The full score-square
term remains part of the energy when the embeddings are trained. -/
def gramRoutingEnergy {V R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (p : EmbeddingGram V × (Fin R → Fin T → ℝ)) : ℝ :=
  ∑ r, ∑ j, (p.2 r j - embeddingScores p.1 (tokens r) (rows r) j) ^ 2

/-- The learned embedding/routing energy is nonnegative.
Source: the squared-distance objective in sparsemax Eq. (1). -/
theorem gramRoutingEnergy_nonneg {V R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (p : EmbeddingGram V × (Fin R → Fin T → ℝ)) :
    0 ≤ gramRoutingEnergy tokens rows p := by
  exact Finset.sum_nonneg fun r _ => Finset.sum_nonneg fun j _ => sq_nonneg _

/-- Expansion retains precisely the term constant only at fixed scores.
Source: sparsemax Eq. (1) and the repository's row objective. Its extra
score-square term is variable when the embedding Gram is learned. -/
theorem gramRoutingEnergy_expansion {V R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (p : EmbeddingGram V × (Fin R → Fin T → ℝ)) :
    gramRoutingEnergy tokens rows p =
      4 * (∑ r, routingObjective
        (fun j => -embeddingScores p.1 (tokens r) (rows r) j / 2) (p.2 r)) +
      ∑ r, ∑ j, (embeddingScores p.1 (tokens r) (rows r) j) ^ 2 := by
  unfold gramRoutingEnergy
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun r _ => ?_
  have h := routing_projection_identity (embeddingScores p.1 (tokens r) (rows r)) (p.2 r)
  linarith

/-- Exact Jensen gap while both embedding blocks and all rows move.
Source: the derived joint Gram formulation of sparsemax Eq. (1).
The residual-difference square records the curvature without assuming
that the endpoint supports agree. -/
theorem gramRoutingEnergy_segment_identity {V R T : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (p q : EmbeddingGram V × (Fin R → Fin T → ℝ)) (a b : ℝ) (hab : a + b = 1) :
    gramRoutingEnergy tokens rows (a • p + b • q) =
      a * gramRoutingEnergy tokens rows p + b * gramRoutingEnergy tokens rows q -
        a * b * (∑ r, ∑ j,
          ((p.2 r j - embeddingScores p.1 (tokens r) (rows r) j) -
            (q.2 r j - embeddingScores q.1 (tokens r) (rows r) j)) ^ 2) := by
  have hpoly (x y : ℝ) : (a * x + b * y) ^ 2 =
      a * x ^ 2 + b * y ^ 2 - a * b * (x - y) ^ 2 := by
    have hb : b = 1 - a := by linarith
    rw [hb]
    ring
  have hrow (r : Fin R) (j : Fin T) :
      ((a • p + b • q).2 r j -
        embeddingScores (a • p + b • q).1 (tokens r) (rows r) j) ^ 2 =
      a * (p.2 r j - embeddingScores p.1 (tokens r) (rows r) j) ^ 2 +
        b * (q.2 r j - embeddingScores q.1 (tokens r) (rows r) j) ^ 2 -
          a * b * ((p.2 r j - embeddingScores p.1 (tokens r) (rows r) j) -
            (q.2 r j - embeddingScores q.1 (tokens r) (rows r) j)) ^ 2 := by
    simp only [Prod.snd_add, Prod.fst_add, Prod.smul_fst, Prod.smul_snd,
      embeddingScores_affine, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    have hl : a * p.2 r j + b * q.2 r j -
        (a * embeddingScores p.1 (tokens r) (rows r) j +
          b * embeddingScores q.1 (tokens r) (rows r) j) =
        a * (p.2 r j - embeddingScores p.1 (tokens r) (rows r) j) +
          b * (q.2 r j - embeddingScores q.1 (tokens r) (rows r) j) := by ring
    rw [hl, hpoly]
  unfold gramRoutingEnergy
  simp_rw [hrow]
  simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]

/-- Distinct feasible causal rows inhabit the Jensen-identity hypothesis. -/
example : gramRoutingEnergy (fun _ : Fin 1 => fun j : Fin 2 => j) (fun _ => 1)
    ((1 / 2 : ℝ) • ((0 : EmbeddingGram 2), fun _ : Fin 1 => basis (0 : Fin 2)) +
      (1 / 2 : ℝ) • ((0 : EmbeddingGram 2), fun _ : Fin 1 => basis (1 : Fin 2))) =
      (1 / 2 : ℝ) := by
  norm_num [gramRoutingEnergy, embeddingScores, basis, Fin.sum_univ_one, Fin.sum_univ_two,
    Prod.snd_add, Prod.fst_add, Prod.smul_fst, Prod.smul_snd, Pi.add_apply, Pi.smul_apply]

/-- Convexity holds jointly in learned embeddings and attention variables.
Source: the derived lifted projection energy of sparsemax Eq. (1).
The theorem does not substitute the optimizer back into a task loss. -/
theorem gramRoutingEnergy_convex {V R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) :
    ConvexOn ℝ (gramRoutingDomain V cap rows) (gramRoutingEnergy tokens rows) := by
  refine ⟨gramRoutingDomain_convex V cap rows, ?_⟩
  intro p _ q _ a b ha hb hab
  have hg : 0 ≤ ∑ r, ∑ j,
      ((p.2 r j - embeddingScores p.1 (tokens r) (rows r) j) -
        (q.2 r j - embeddingScores q.1 (tokens r) (rows r) j)) ^ 2 :=
    Finset.sum_nonneg fun r _ => Finset.sum_nonneg fun j _ => sq_nonneg _
  have hp := mul_nonneg (mul_nonneg ha hb) hg
  rw [gramRoutingEnergy_segment_identity _ _ _ _ _ _ hab]
  simp only [smul_eq_mul]
  linarith

/-- For a fixed learned Gram, each exact sparsemax row minimizes distance.
Source: sparsemax Eq. (1), retaining the causal mask. -/
theorem embeddingRoutes_row_min {V R T : ℕ} (G : EmbeddingGram V)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (r : Fin R)
    (weights : Fin T → ℝ) (hw : weights ∈ simplexOn {j : Fin T | j ≤ rows r}) :
    (∑ j, (embeddingRoutes G tokens rows r j -
      embeddingScores G (tokens r) (rows r) j) ^ 2) ≤
        ∑ j, (weights j - embeddingScores G (tokens r) (rows r) j) ^ 2 := by
  have hmin := (sparseWeights_spec (embeddingScores G (tokens r) (rows r)) (rows r)).2
    weights hw
  have hs := routing_projection_identity (embeddingScores G (tokens r) (rows r))
    (embeddingRoutes G tokens rows r)
  have hwid := routing_projection_identity (embeddingScores G (tokens r) (rows r)) weights
  change routingObjective _ (embeddingRoutes G tokens rows r) ≤ _ at hmin
  linarith

/-- A feasible basis comparator inhabits the conditional row theorem. -/
example : (∑ j : Fin 2, (embeddingRoutes (0 : EmbeddingGram 2)
    (fun _ : Fin 1 => fun j : Fin 2 => j) (fun _ => 1) 0 j) ^ 2) ≤
      ∑ j : Fin 2, (basis (0 : Fin 2) j) ^ 2 := by
  simpa only [embeddingScores, Matrix.zero_apply, sub_zero] using
    embeddingRoutes_row_min (0 : EmbeddingGram 2)
      (fun _ : Fin 1 => fun j : Fin 2 => j) (fun _ => 1) 0 (basis 0)
      (basis_mem_simplex _ _ (by decide))

/-- Exact sparsemax minimizes all free rows conditionally on the Gram.
Source: the summed causal specialization of sparsemax Eq. (1).
This remains valid as supports change between embedding assignments. -/
theorem gramRoutingEnergy_conditional_min {V R T : ℕ} (G : EmbeddingGram V)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (weights : Fin R → Fin T → ℝ)
    (hw : ∀ r, weights r ∈ simplexOn {j : Fin T | j ≤ rows r}) :
    gramRoutingEnergy tokens rows (G, embeddingRoutes G tokens rows) ≤
      gramRoutingEnergy tokens rows (G, weights) := by
  exact Finset.sum_le_sum fun r _ => embeddingRoutes_row_min G tokens rows r _ (hw r)

/-- Actual inferred rows inhabit the conditional joint minimization premises. -/
example : gramRoutingEnergy (fun _ : Fin 1 => fun j : Fin 2 => j) (fun _ => 1)
    ((0 : EmbeddingGram 2), embeddingRoutes (0 : EmbeddingGram 2)
      (fun _ : Fin 1 => fun j : Fin 2 => j) (fun _ => 1)) ≤
    gramRoutingEnergy (fun _ : Fin 1 => fun j : Fin 2 => j) (fun _ => 1)
      ((0 : EmbeddingGram 2), fun _ : Fin 1 => basis (0 : Fin 2)) :=
  gramRoutingEnergy_conditional_min _ _ _ _
    (fun _ => basis_mem_simplex _ _ (by decide))

end Transformer.GPTMini.Sparsemax
