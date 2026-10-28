module

public import Mathlib.Combinatorics.SimpleGraph.Acyclic
public import Transformer.Grokking.FlatSpectrum.PermutationRecovery

/-!
# Rigidity is connectivity

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Setting: "`Ω` is
rigid when `r_i + s_j = 0` on `Ω` forces `r_i + s_j = 0` everywhere. This holds exactly when
the bipartite graph on rows and columns, with an edge for each cell of `Ω`, is connected"; and
*Scope*: "A rigid pattern is connected, so it has at least `2p − 1` cells".

The graph is `patternGraph Ω` on `m ⊕ n`: row `i` and column `j` are adjacent when
`(i, j) ∈ Ω`.

* On a connected pattern `(r_i, −s_j)` agrees across every edge, hence everywhere
  (`rigid_of_preconnected`).
* On a rigid pattern, `r = 1` and `s = −1` on the component of a row, and `0` elsewhere, vanish
  on `Ω`; rigidity spreads the component to every row and column (`connected_of_rigid`).

So rigidity is connectivity (`rigid_iff_connected`). Both need a row and a column, as the
source's `N × N` tables with `N ≥ 1` have: without rows every pattern is rigid. A connected graph
on `#m + #n` vertices has at least `#m + #n − 1` edges, one per cell, so a rigid `N × N` pattern
has at least `2N − 1` cells (`card_add_card_le_ncard_add_one_of_rigid`). `ConnectedRecovery`
restates Theorems E and G with graphs.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open SimpleGraph

variable {m n : Type*}

/-- The bipartite graph of a pattern `Ω`, `review/flat_spectrum.md`, Setting: the rows and the
columns are the vertices, and row `i` is adjacent to column `j` when `(i, j) ∈ Ω`. -/
def patternGraph (Ω : Set (m × n)) : SimpleGraph (m ⊕ n) :=
  fromRel fun x y => ∃ i j, x = .inl i ∧ y = .inr j ∧ (i, j) ∈ Ω

@[simp] lemma patternGraph_adj_inl_inr {Ω : Set (m × n)} {i : m} {j : n} :
    (patternGraph Ω).Adj (.inl i) (.inr j) ↔ (i, j) ∈ Ω := by
  simp [patternGraph]

@[simp] lemma patternGraph_adj_inr_inl {Ω : Set (m × n)} {i : m} {j : n} :
    (patternGraph Ω).Adj (.inr j) (.inl i) ↔ (i, j) ∈ Ω := by
  simp [patternGraph]

@[simp] lemma not_patternGraph_adj_inl_inl {Ω : Set (m × n)} {i i' : m} :
    ¬ (patternGraph Ω).Adj (.inl i) (.inl i') := by
  simp [patternGraph]

@[simp] lemma not_patternGraph_adj_inr_inr {Ω : Set (m × n)} {j j' : n} :
    ¬ (patternGraph Ω).Adj (.inr j) (.inr j') := by
  simp [patternGraph]

/-- On a connected pattern, `r_i + s_j = 0` on `Ω` forces `r_i + s_j = 0` everywhere. -/
lemma rigid_of_preconnected {Ω : Set (m × n)} (hΩ : (patternGraph Ω).Preconnected)
    (r : m → ℂ) (s : n → ℂ) (h : ∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) (i : m) (j : n) :
    r i + s j = 0 := by
  have hφ (x y : m ⊕ n) (hxy : (patternGraph Ω).Adj x y) :
      Sum.elim r (fun j => -s j) x = Sum.elim r (fun j => -s j) y := by
    rcases x with i | j <;> rcases y with i' | j'
    · exact absurd hxy not_patternGraph_adj_inl_inl
    · exact eq_neg_of_add_eq_zero_left (h _ (patternGraph_adj_inl_inr.mp hxy))
    · exact (eq_neg_of_add_eq_zero_left (h _ (patternGraph_adj_inr_inl.mp hxy))).symm
    · exact absurd hxy not_patternGraph_adj_inr_inr
  have key (x y : m ⊕ n) (hxy : Relation.ReflTransGen (patternGraph Ω).Adj x y) :
      Sum.elim r (fun j => -s j) x = Sum.elim r (fun j => -s j) y := by
    induction hxy with
    | refl => rfl
    | tail _ hab ih => exact ih.trans (hφ _ _ hab)
  have := key _ _ ((reachable_iff_reflTransGen _ _).mp (hΩ (.inl i) (.inr j)))
  rw [Sum.elim_inl, Sum.elim_inr] at this
  rw [this, neg_add_cancel]

example := rigid_of_preconnected (Ω := (Set.univ : Set (Fin 1 × Fin 1)))
  (fun x y => by
    rcases x with i | j <;> rcases y with i' | j'
    · rw [Subsingleton.elim i i']
    · exact Adj.reachable (by simp)
    · exact Adj.reachable (by simp)
    · rw [Subsingleton.elim j j'])
  1 (-1) (fun _ _ => by norm_num) 0 0

/-- A rigid pattern with a row and a column is connected. -/
lemma connected_of_rigid [Nonempty m] [Nonempty n] {Ω : Set (m × n)}
    (hΩ : ∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) :
    (patternGraph Ω).Connected := by
  classical
  obtain ⟨i₀⟩ := ‹Nonempty m›
  obtain ⟨j₀⟩ := ‹Nonempty n›
  have h := hΩ (fun i => if (patternGraph Ω).Reachable (.inl i₀) (.inl i) then 1 else 0)
    (fun j => if (patternGraph Ω).Reachable (.inl i₀) (.inr j) then -1 else 0) fun ij hij => by
      have hadj := patternGraph_adj_inl_inr.mpr hij
      by_cases hi : (patternGraph Ω).Reachable (.inl i₀) (.inl ij.1)
      · rw [ite_eq_left hi, ite_eq_left (hi.trans hadj.reachable), add_neg_cancel]
      · rw [ite_eq_right hi, ite_eq_right fun hj => hi (hj.trans hadj.symm.reachable), add_zero]
  have hcol (j : n) : (patternGraph Ω).Reachable (.inl i₀) (.inr j) := by
    by_contra hj
    have := h i₀ j
    rw [ite_eq_left (Reachable.refl _), ite_eq_right hj, add_zero] at this
    exact one_ne_zero this
  have hrow (i : m) : (patternGraph Ω).Reachable (.inl i₀) (.inl i) := by
    by_contra hi
    have := h i j₀
    rw [ite_eq_right hi, ite_eq_left (hcol j₀), zero_add, neg_eq_zero] at this
    exact one_ne_zero this
  have hall : ∀ x, (patternGraph Ω).Reachable (.inl i₀) x
    | .inl i => hrow i
    | .inr j => hcol j
  exact connected_iff _ |>.mpr ⟨fun u v => (hall u).symm.trans (hall v), ⟨.inl i₀⟩⟩

/-- **Rigidity is connectivity** (`review/flat_spectrum.md`, Setting): with a row and a column,
`Ω` is rigid exactly when its bipartite graph is connected. -/
theorem rigid_iff_connected [Nonempty m] [Nonempty n] (Ω : Set (m × n)) :
    (∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) ↔
      (patternGraph Ω).Connected :=
  ⟨connected_of_rigid, fun h => rigid_of_preconnected h.preconnected⟩

example := (rigid_iff_connected {ij : Fin 2 × Fin 2 | ij ≠ (1, 0)}).mp rigid_ne_one_zero

/-- **A rigid pattern has at least `#m + #n − 1` cells** (`review/flat_spectrum.md`, Scope: "A
rigid pattern is connected, so it has at least `2p − 1` cells"). -/
theorem card_add_card_le_ncard_add_one_of_rigid [Finite m] [Finite n] [Nonempty m]
    [Nonempty n] {Ω : Set (m × n)}
    (hΩ : ∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0) :
    Nat.card m + Nat.card n ≤ Ω.ncard + 1 := by
  have h := (connected_of_rigid hΩ).card_vert_le_card_edgeSet_add_one
  rw [Nat.card_sum, Nat.card_coe_set_eq] at h
  refine h.trans (Nat.add_le_add_right ?_ 1)
  refine (Set.ncard_le_ncard (t := (fun ij : m × n => s(Sum.inl ij.1, Sum.inr ij.2)) '' Ω)
    (fun e he => ?_) ((Set.toFinite Ω).image _)).trans (Set.ncard_image_le (Set.toFinite Ω))
  induction e using Sym2.ind with
  | h x y =>
    rw [mem_edgeSet] at he
    rcases x with i | j <;> rcases y with i' | j'
    · exact absurd he not_patternGraph_adj_inl_inl
    · exact ⟨(i, j'), patternGraph_adj_inl_inr.mp he, rfl⟩
    · exact ⟨(i', j), patternGraph_adj_inr_inl.mp he, Sym2.eq_swap⟩
    · exact absurd he not_patternGraph_adj_inr_inr

example : Nat.card (Fin 2) + Nat.card (Fin 2) ≤ {ij : Fin 2 × Fin 2 | ij ≠ (1, 0)}.ncard + 1 :=
  card_add_card_le_ncard_add_one_of_rigid rigid_ne_one_zero

/-- Rigidity of a pattern given cell by cell is connectivity of its graph. -/
lemma rigid_iff_connected' [Nonempty m] [Nonempty n] (P : m → n → Prop) :
    (∀ (r : m → ℂ) (s : n → ℂ), (∀ i j, P i j → r i + s j = 0) → ∀ i j, r i + s j = 0) ↔
      (patternGraph {ij | P ij.1 ij.2}).Connected :=
  (forall₂_congr fun _ _ => imp_congr_left
    ⟨fun h (ij : m × n) hij => h ij.1 ij.2 hij, fun h i j hij => h (i, j) hij⟩).trans
    (rigid_iff_connected _)

example := (rigid_iff_connected' (m := Fin 2) (n := Fin 2) fun i j => (i, j) ≠ (1, 0)).mp
  fun r s h => rigid_ne_one_zero r s fun ij hij => h ij.1 ij.2 hij

end Transformer.Grokking.FlatSpectrum
