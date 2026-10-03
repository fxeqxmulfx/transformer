/-
# A solved convex replacement for the attention weight row

New modification of `experiments/archive/gpt_mini/gpt_mini.py`,
`CausalMHA.forward`, restored from commit
f11b6e27d3cfe6813a2876bdeec38565fc54258c. The simplex idea is
from arXiv:2211.11052v1, §3.1; this is not that paper's shared matrix.
Scores are inputs to inference. Their dependence on trainable Q/K matrices
is retained by the next module and is not assumed jointly convex.
-/

import Transformer.ConvexRecall.Simplex

open scoped BigOperators

noncomputable section

namespace Transformer.GPTMini.Convex

open Transformer.ConvexRecall

variable {ι : Type*} [Fintype ι]

/-- The masked simplex is closed. Source context: `CausalMHA.forward`,
causal mask; replacement motivated by arXiv:2211.11052v1, §3.1. -/
theorem simplexOn_closed (allowed : Set ι) : IsClosed (simplexOn allowed) := by
  have hn : IsClosed {a : ι → ℝ | ∀ j, 0 ≤ a j} := by
    simp only [Set.ofPred_forall]
    exact isClosed_iInter fun j => isClosed_le continuous_const (continuous_apply j)
  have hs : IsClosed {a : ι → ℝ | ∑ j, a j = 1} :=
    isClosed_eq (by fun_prop) continuous_const
  have hz : IsClosed {a : ι → ℝ | ∀ j, j ∉ allowed → a j = 0} := by
    simp only [Set.ofPred_forall]
    exact isClosed_iInter fun j => isClosed_iInter fun _ =>
      isClosed_eq (continuous_apply j) continuous_const
  exact hn.inter (hs.inter hz)

/-- The masked simplex is compact, including when it is empty. Source
context: `CausalMHA.forward`; simplex replacement of §3.1. -/
theorem simplexOn_compact (allowed : Set ι) : IsCompact (simplexOn allowed) := by
  refine IsCompact.of_isClosed_subset
    (isCompact_Icc : IsCompact (Set.Icc (0 : ι → ℝ) (fun _ => 1)))
    (simplexOn_closed allowed) ?_
  intro a ha
  refine ⟨ha.1, fun j => ?_⟩
  have h := Finset.single_le_sum (fun k (_ : k ∈ Finset.univ) => ha.1 k)
    (Finset.mem_univ j)
  simpa [ha.2.1] using h

/-- The quadratic row objective is continuous. Source context:
arXiv:2211.11052v1, §3.1, new quadratic simplex replacement. -/
theorem routingObjective_continuous (cost : ι → ℝ) :
    Continuous (routingObjective cost) := by
  unfold routingObjective
  fun_prop

/-- A continuous convex row objective attains its optimum for every cost
vector whenever there is an allowed slot. No cost-gap premise is needed.
Source: new replacement of `CausalMHA.forward`, §3.1 simplex motivation. -/
theorem routing_minimum_exists (allowed : Set ι) (winner : ι)
    (hw : winner ∈ allowed) (cost : ι → ℝ) :
    ∃ a ∈ simplexOn allowed, ∀ b ∈ simplexOn allowed,
      routingObjective cost a ≤ routingObjective cost b := by
  exact (simplexOn_compact allowed).exists_isMinOn
    ⟨basis winner, basis_mem_simplex allowed winner hw⟩
    (routingObjective_continuous cost).continuousOn

/-- Nonempty inference domains occur in the source: a causal row always
allows its own position. Source: `CausalMHA.forward`, diagonal mask. -/
example : (0 : Fin 1) ∈ {j : Fin 1 | j ≤ 0} := by
  change (0 : Fin 1) ≤ 0
  exact le_rfl

/-- Exact midpoint identity, with a positive squared-distance term.
Source: new quadratic simplex replacement of §3.1. -/
theorem routing_midpoint_identity (cost x y : ι → ℝ) :
    routingObjective cost (fun j => (x j + y j) / 2) =
      (routingObjective cost x + routingObjective cost y) / 2 -
        (∑ j, (x j - y j) ^ 2) / 16 := by
  have hs (j : ι) : ((x j + y j) / 2) ^ 2 =
      ((x j) ^ 2 + (y j) ^ 2) / 2 - (x j - y j) ^ 2 / 4 := by ring
  have hl (j : ι) : ((x j + y j) / 2) * cost j =
      (x j * cost j + y j * cost j) / 2 := by ring
  have hsq : (∑ j, ((x j + y j) / 2) ^ 2) =
      ((∑ j, (x j) ^ 2) + ∑ j, (y j) ^ 2) / 2 -
        (∑ j, (x j - y j) ^ 2) / 4 := by
    simp_rw [hs]
    rw [Finset.sum_sub_distrib, ← Finset.sum_div, ← Finset.sum_div,
      Finset.sum_add_distrib]
  have hlin : (∑ j, ((x j + y j) / 2) * cost j) =
      ((∑ j, x j * cost j) + ∑ j, y j * cost j) / 2 := by
    simp_rw [hl]
    rw [← Finset.sum_div, Finset.sum_add_distrib]
  simp only [routingObjective, hsq, hlin]
  ring

/-- Any two global minimizers of the row program are equal. Source:
new replacement of `CausalMHA.forward`, §3.1 simplex motivation. -/
theorem routing_minimizers_eq (allowed : Set ι) (cost x y : ι → ℝ)
    (hx : x ∈ simplexOn allowed) (hy : y ∈ simplexOn allowed)
    (hminx : ∀ b ∈ simplexOn allowed, routingObjective cost x ≤ routingObjective cost b)
    (hminy : ∀ b ∈ simplexOn allowed, routingObjective cost y ≤ routingObjective cost b) :
    x = y := by
  have hm : (fun j => (x j + y j) / 2) ∈ simplexOn allowed := by
    have h := simplexOn_convex allowed hx hy
      (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
      (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
    have heq : (1 / 2 : ℝ) • x + (1 / 2 : ℝ) • y =
        (fun j => (x j + y j) / 2) := by
      funext j
      simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      ring
    rwa [heq] at h
  have h₁ := hminx _ hm
  have h₂ := hminy _ hm
  rw [routing_midpoint_identity] at h₁ h₂
  have hn : ∀ j : ι, 0 ≤ (x j - y j) ^ 2 := fun _ => sq_nonneg _
  have hz : (∑ j, (x j - y j) ^ 2) = 0 := by
    have hsum := Finset.sum_nonneg (fun j (_ : j ∈ Finset.univ) => hn j)
    linarith
  have hall := (Finset.sum_eq_zero_iff_of_nonneg
    (fun j (_ : j ∈ Finset.univ) => hn j)).mp hz
  funext j
  have hj := hall j (Finset.mem_univ j)
  nlinarith

/-- The minimizer premises are satisfiable, including a causal domain
with a single position. Source context: `CausalMHA.forward`. -/
example : ∃ a ∈ simplexOn ({j : Fin 1 | j ≤ 0}),
    ∀ b ∈ simplexOn ({j : Fin 1 | j ≤ 0}),
      routingObjective (fun _ => 0) a ≤ routingObjective (fun _ => 0) b :=
  routing_minimum_exists {j : Fin 1 | j ≤ 0} (0 : Fin 1)
    (by change (0 : Fin 1) ≤ 0; exact le_rfl) (fun _ => 0)

/-- Every masked row has exactly one optimum once it has an allowed slot.
Source: new replacement of `CausalMHA.forward`, §3.1 simplex motivation. -/
theorem routing_minimum_unique (allowed : Set ι) (winner : ι)
    (hw : winner ∈ allowed) (cost : ι → ℝ) :
    ∃! a, a ∈ simplexOn allowed ∧ ∀ b ∈ simplexOn allowed,
      routingObjective cost a ≤ routingObjective cost b := by
  obtain ⟨a, ha, hmina⟩ := routing_minimum_exists allowed winner hw cost
  exact ⟨a, ⟨ha, hmina⟩, fun b hb =>
    routing_minimizers_eq allowed cost b a hb.1 ha hb.2 hmina⟩

/-- A causal sparsemax row: the unique minimizer of
`sum a²/4 - sum a*score/2`, equivalently projection of `scores` onto the
causal simplex. Source: new replacement of `CausalMHA.forward`. -/
def sparseWeights {T : ℕ} (scores : Fin T → ℝ) (i : Fin T) : Fin T → ℝ :=
  Classical.choose (routing_minimum_unique {j : Fin T | j ≤ i} i
    (by change i ≤ i; exact le_rfl)
    (fun j => -scores j / 2)).exists

/-- Sparse weights solve the stated inference program for every score
assignment. Source: new replacement of `CausalMHA.forward`. -/
theorem sparseWeights_spec {T : ℕ} (scores : Fin T → ℝ) (i : Fin T) :
    sparseWeights scores i ∈ simplexOn {j : Fin T | j ≤ i} ∧
      ∀ b ∈ simplexOn {j : Fin T | j ≤ i},
        routingObjective (fun j => -scores j / 2) (sparseWeights scores i) ≤
          routingObjective (fun j => -scores j / 2) b :=
  Classical.choose_spec (routing_minimum_unique {j : Fin T | j ≤ i} i
    (by change i ≤ i; exact le_rfl)
    (fun j => -scores j / 2)).exists

/-- The row objective and squared projection distance differ by a
constant independent of the chosen weights. Source: new replacement
of `CausalMHA.forward`, §3.1 simplex motivation. -/
theorem routing_projection_identity (scores a : ι → ℝ) :
    4 * routingObjective (fun j => -scores j / 2) a =
      (∑ j, (a j - scores j) ^ 2) - ∑ j, (scores j) ^ 2 := by
  have hl : (∑ j, a j * (-scores j / 2)) = -(∑ j, a j * scores j) / 2 := by
    calc
      _ = ∑ j, -(a j * scores j) / 2 :=
        Finset.sum_congr rfl (fun _ _ => by ring)
      _ = _ := by rw [← Finset.sum_div, Finset.sum_neg_distrib]
  have hs : (∑ j, (a j - scores j) ^ 2) =
      (∑ j, (a j) ^ 2) - 2 * (∑ j, a j * scores j) + ∑ j, (scores j) ^ 2 := by
    have h (j : ι) : (a j - scores j) ^ 2 =
        (a j) ^ 2 - 2 * (a j * scores j) + (scores j) ^ 2 := by ring
    simp_rw [h]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  simp only [routingObjective, hl, hs]
  ring

end Transformer.GPTMini.Convex
