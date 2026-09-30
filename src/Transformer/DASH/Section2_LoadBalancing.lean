/-
# DASH — greedy layer allocation to workers

arXiv:2602.02016v2, §2.3, “Load Balancing”. Layers are sorted by
parameter count, then assigned to a worker of minimum current load.
The allocation neither drops nor duplicates layer occurrences.
-/

import Mathlib.Data.List.Sort
import Mathlib.Data.Finset.Max
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic

open scoped BigOperators

noncomputable section

namespace Transformer.DASH

variable {α : Type*} {w : ℕ} [Nonempty (Fin w)]

/-- Layer order used by the source's greedy allocation,
arXiv:2602.02016v2, §2.3, descending parameter count. -/
def descendingLayers (size : α → ℕ) (layers : List α) : List α :=
  layers.mergeSort (fun a b => decide (size b ≤ size a))

/-- Choose a worker with the minimum current parameter load.
Source: arXiv:2602.02016v2, §2.3, the least-loaded GPU. -/
def leastLoaded (loads : Fin w → ℕ) : Fin w :=
  Classical.choose (Finset.exists_min_image Finset.univ loads Finset.univ_nonempty)

/-- Add a layer's parameter count only to its assigned worker.
Source: arXiv:2602.02016v2, §2.3, greedy load update. -/
def addLayerLoad (loads : Fin w → ℕ) (worker : Fin w) (size : ℕ) : Fin w → ℕ :=
  fun j => loads j + if j = worker then size else 0

/-- The actual sequential greedy assignment, with input order preserved.
Source: arXiv:2602.02016v2, §2.3, layer allocation after sorting. -/
def greedyAssignments (size : α → ℕ) (loads : Fin w → ℕ) : List α → List (α × Fin w)
  | [] => []
  | layer :: layers =>
      let worker := leastLoaded loads
      (layer, worker) :: greedyAssignments size
        (addLayerLoad loads worker (size layer)) layers

/-- Final worker loads after the same sequence of greedy choices,
arXiv:2602.02016v2, §2.3. -/
def greedyLoads (size : α → ℕ) (loads : Fin w → ℕ) : List α → Fin w → ℕ
  | [] => loads
  | layer :: layers =>
      greedyLoads size (addLayerLoad loads (leastLoaded loads) (size layer)) layers

/-- Every chosen worker has no larger load than any other worker.
Source: arXiv:2602.02016v2, §2.3, the greedy minimum rule. -/
theorem leastLoaded_le (loads : Fin w → ℕ) (j : Fin w) :
    loads (leastLoaded loads) ≤ loads j :=
  (Classical.choose_spec
    (Finset.exists_min_image Finset.univ loads Finset.univ_nonempty)).2 j (Finset.mem_univ j)

/-- Sorting obeys the source's descending parameter-count rule,
arXiv:2602.02016v2, §2.3. -/
theorem descendingLayers_sorted (size : α → ℕ) (layers : List α) :
    (descendingLayers size layers).Pairwise (fun a b => size b ≤ size a) := by
  have h := List.pairwise_mergeSort
    (le := fun a b : α => decide (size b ≤ size a))
    (by intro a b c hab hbc; simpa using le_trans (of_decide_eq_true hbc) (of_decide_eq_true hab))
    (by intro a b; simp only [Bool.or_eq_true, decide_eq_true_eq]; exact le_total _ _) layers
  simpa only [descendingLayers, decide_eq_true_eq] using h

/-- Sorting preserves the list of layer occurrences,
arXiv:2602.02016v2, §2.3. -/
theorem descendingLayers_perm (size : α → ℕ) (layers : List α) :
    (descendingLayers size layers).Perm layers := List.mergeSort_perm _ _

omit [Nonempty (Fin w)] in
/-- One load update accounts for exactly the layer's parameter count.
Source: arXiv:2602.02016v2, §2.3, disjoint optimizer-state allocation. -/
theorem addLayerLoad_sum (loads : Fin w → ℕ) (worker : Fin w) (size : ℕ) :
    (∑ j, addLayerLoad loads worker size j) = (∑ j, loads j) + size := by
  simp [addLayerLoad, Finset.sum_add_distrib]

/-- Greedy allocation assigns every layer occurrence once and keeps its
identity. Source: arXiv:2602.02016v2, §2.3, partitioning layers among workers. -/
theorem greedyAssignments_layers (size : α → ℕ) (loads : Fin w → ℕ) (layers : List α) :
    (greedyAssignments size loads layers).map Prod.fst = layers := by
  induction layers generalizing loads with
  | nil => rfl
  | cons layer layers ih => simp only [greedyAssignments, List.map_cons, ih]

/-- The sorted greedy procedure partitions the original layer occurrences.
Source: arXiv:2602.02016v2, §2.3, the full load-balancing procedure. -/
theorem sortedGreedy_preserves_layers (size : α → ℕ) (loads : Fin w → ℕ) (layers : List α) :
    ((greedyAssignments size loads (descendingLayers size layers)).map Prod.fst).Perm layers := by
  rw [greedyAssignments_layers]
  exact descendingLayers_perm size layers

/-- Total worker load is exactly the initial load plus all assigned parameters.
Source: arXiv:2602.02016v2, §2.3, scattered optimizer state. -/
theorem greedyLoads_total (size : α → ℕ) (loads : Fin w → ℕ) (layers : List α) :
    (∑ j, greedyLoads size loads layers j) = (∑ j, loads j) + (layers.map size).sum := by
  induction layers generalizing loads with
  | nil => simp [greedyLoads]
  | cons layer layers ih =>
    rw [greedyLoads, ih, addLayerLoad_sum]
    simp [Nat.add_assoc]

/-- A single worker witnesses the nonempty-worker hypothesis used by the
greedy choice, arXiv:2602.02016v2, §2.3. -/
example : Nonempty (Fin 1) := inferInstance

end Transformer.DASH
