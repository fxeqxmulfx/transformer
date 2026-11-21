module

public import Mathlib.Combinatorics.Matroid.Circuit
public import Mathlib.Combinatorics.Matroid.IndepAxioms
public import Mathlib.Combinatorics.Matroid.Rank.ENat
public import Transformer.GainMatroid.Section2_Classes

/-!
# Frame and lift matroids of a biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2. A set `C` of edges is a circuit of the frame
matroid `F(G, B)` when `C ∈ B`, or when `C` induces a tight handcuff, a loose handcuff or a theta
containing no balanced cycle. For the lift matroid `L(G, B)`, two vertex-disjoint cycles replace the
loose handcuff. The source draws these subgraphs in `Circuits.pdf` (pages 1, 2, 4 and 1, 3, 4) and
cites Zaslavsky for the fact that they are the circuits of a matroid. Here `IsFrameMatroid` and
`IsLiftMatroid` are predicates on a given matroid, so that fact is not assumed.

Matroid connectivity follows Oxley, whom the source cites at the start of §2. A partition `(X, Y)`
is a `k`-separation when both sides have at least `k` elements and `r(X) + r(Y) - r(M) < k`, and `M`
is `n`-connected when it has no `k`-separation for `k < n`. Mathlib has neither this notion nor a
uniform matroid at this version, so both are defined here.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

namespace OrientedGraph

variable (G : OrientedGraph V E)

/-- A tight handcuff (`Circuits.pdf`, page 1): two edge-disjoint cycles meeting in one vertex. -/
def IsTightHandcuff (X : Set E) : Prop :=
  ∃ (C D : Set E) (v : V), G.IsCycle C ∧ G.IsCycle D ∧ Disjoint C D ∧
    G.verts C ∩ G.verts D = {v} ∧ X = C ∪ D

/-- A loose handcuff (`Circuits.pdf`, page 2): two vertex-disjoint cycles and a path joining them
that meets each cycle only at its end. -/
def IsLooseHandcuff (X : Set E) : Prop :=
  ∃ (C D : Set E) (x y : V) (P : G.Walk x y), G.IsCycle C ∧ G.IsCycle D ∧
    Disjoint (G.verts C) (G.verts D) ∧ P.IsPath ∧ x ∈ G.verts C ∧ y ∈ G.verts D ∧
    (∀ z ∈ P.support, z ∈ G.verts C → z = x) ∧ (∀ z ∈ P.support, z ∈ G.verts D → z = y) ∧
    X = C ∪ D ∪ P.edgeSet

/-- Two vertex-disjoint cycles (`Circuits.pdf`, page 3). -/
def IsDisjointCycles (X : Set E) : Prop :=
  ∃ C D : Set E, G.IsCycle C ∧ G.IsCycle D ∧ Disjoint (G.verts C) (G.verts D) ∧ X = C ∪ D

/-- The edges of a theta subgraph (`Circuits.pdf`, page 4). -/
def IsThetaSet (X : Set E) : Prop :=
  ∃ (x y : V) (P Q R : G.Walk x y), IsTheta P Q R ∧ X = P.edgeSet ∪ Q.edgeSet ∪ R.edgeSet

/-- The contrabalanced biased graph on `G` (§2.1): no cycle is balanced. -/
def contrabalanced : BiasedGraph V E where
  graph := G
  balanced := ∅
  isCycle_of_mem := fun _ h => absurd h (Set.notMem_empty _)
  theta := fun _ h => absurd h (Set.notMem_empty _)

end OrientedGraph

namespace BiasedGraph

variable (Ω : BiasedGraph V E)

/-- **§2.2.** A circuit of the frame matroid `F(G, B)`: a balanced cycle, or a tight handcuff, loose
handcuff or theta containing no balanced cycle. -/
def IsFrameCircuit (X : Set E) : Prop :=
  X ∈ Ω.balanced ∨ ((Ω.graph.IsTightHandcuff X ∨ Ω.graph.IsLooseHandcuff X ∨
    Ω.graph.IsThetaSet X) ∧ ∀ C ⊆ X, Ω.graph.IsCycle C → C ∉ Ω.balanced)

/-- **§2.2.** A circuit of the lift matroid `L(G, B)`: a balanced cycle, or a tight handcuff, two
vertex-disjoint cycles or a theta containing no balanced cycle. -/
def IsLiftCircuit (X : Set E) : Prop :=
  X ∈ Ω.balanced ∨ ((Ω.graph.IsTightHandcuff X ∨ Ω.graph.IsDisjointCycles X ∨
    Ω.graph.IsThetaSet X) ∧ ∀ C ⊆ X, Ω.graph.IsCycle C → C ∉ Ω.balanced)

/-- `M` is the frame matroid `F(G, B)`. -/
def IsFrameMatroid (M : Matroid E) : Prop :=
  M.E = Set.univ ∧ ∀ X, M.IsCircuit X ↔ Ω.IsFrameCircuit X

/-- `M` is the lift matroid `L(G, B)`. -/
def IsLiftMatroid (M : Matroid E) : Prop :=
  M.E = Set.univ ∧ ∀ X, M.IsCircuit X ↔ Ω.IsLiftCircuit X

/-- `(G, B)` represents `M` (§1): `M = F(G, B)` or `M = L(G, B)`. -/
def Represents (M : Matroid E) : Prop :=
  Ω.IsFrameMatroid M ∨ Ω.IsLiftMatroid M

end BiasedGraph

/-- A `k`-separation of `M` (Oxley): a partition `(X, Y)` of the ground set with at least `k`
elements on each side and `r(X) + r(Y) - r(M) < k`. -/
structure IsKSeparation (M : Matroid E) (k : ℕ) (X Y : Set E) : Prop where
  disjoint : Disjoint X Y
  union : X ∪ Y = M.E
  left : (k : ℕ∞) ≤ X.encard
  right : (k : ℕ∞) ≤ Y.encard
  rank : M.eRk X + M.eRk Y + 1 ≤ M.eRank + k

/-- `M` is `n`-connected (Oxley): it has no `k`-separation for `1 ≤ k < n`. -/
def IsNConnected (M : Matroid E) (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k → k < n → ∀ X Y, ¬IsKSeparation M k X Y

section Uniform

/-- The uniform matroid on a finite type: a set is independent when it has at most `r` elements. -/
def uniformMatroid (α : Type*) [Finite α] (r : ℕ) : Matroid α :=
  (IndepMatroid.ofFinite (E := Set.univ) Set.finite_univ (fun I => I.ncard ≤ r) (by simp)
    (fun _ J hJ hIJ => (Set.ncard_le_ncard hIJ (Set.toFinite J)).trans hJ)
    (fun I J _ hJ hlt => by
      obtain ⟨e, heJ, heI⟩ := Set.exists_mem_notMem_of_ncard_lt_ncard hlt (Set.toFinite I)
      exact ⟨e, heJ, heI, (Set.ncard_insert_le e I).trans (hlt.trans_le hJ)⟩)
    (fun I _ => Set.subset_univ I)).matroid

variable {α : Type*} [Finite α] {r : ℕ}

theorem uniformMatroid_E : (uniformMatroid α r).E = Set.univ := rfl

theorem uniformMatroid_indep {I : Set α} : (uniformMatroid α r).Indep I ↔ I.ncard ≤ r := by
  simp [uniformMatroid]

/-- The circuits of `U_{r,n}` are the sets of `r + 1` elements. -/
theorem uniformMatroid_isCircuit_iff {X : Set α} :
    (uniformMatroid α r).IsCircuit X ↔ X.ncard = r + 1 := by
  rw [Matroid.isCircuit_iff_dep_forall_sdiff_singleton_indep]
  constructor
  · rintro ⟨hdep, hdel⟩
    have hlt : r < X.ncard := not_le.1 fun h => hdep.1 (uniformMatroid_indep.2 h)
    obtain ⟨e, he⟩ : X.Nonempty := (Set.ncard_pos (Set.toFinite X)).1 (by omega)
    have h := uniformMatroid_indep.1 (hdel e he)
    rw [Set.ncard_sdiff_singleton_of_mem he] at h
    omega
  · intro h
    refine ⟨⟨fun hI => ?_, Set.subset_univ X⟩, fun e he => ?_⟩
    · have := uniformMatroid_indep.1 hI
      omega
    · rw [uniformMatroid_indep, Set.ncard_sdiff_singleton_of_mem he]
      omega

/-- The rank of a set in `U_{r,n}`. -/
theorem uniformMatroid_eRk (X : Set α) :
    (uniformMatroid α r).eRk X = min X.encard (r : ℕ∞) := by
  apply le_antisymm
  · refine le_min (Matroid.eRk_le_encard _ X) ?_
    obtain ⟨I, hI⟩ := (uniformMatroid α r).exists_isBasis' X
    rw [← hI.encard_eq_eRk, ← (Set.toFinite I).cast_ncard_eq]
    exact_mod_cast uniformMatroid_indep.1 hI.indep
  · obtain ⟨t, htX, ht⟩ := Set.exists_subset_encard_eq (min_le_left X.encard (r : ℕ∞))
    have hindep : (uniformMatroid α r).Indep t := by
      rw [uniformMatroid_indep]
      have h : (t.ncard : ℕ∞) ≤ r := by
        rw [(Set.toFinite t).cast_ncard_eq, ht]
        exact min_le_right _ _
      exact_mod_cast h
    calc min X.encard (r : ℕ∞) = t.encard := ht.symm
      _ = (uniformMatroid α r).eRk t := hindep.eRk_eq_encard.symm
      _ ≤ (uniformMatroid α r).eRk X := (uniformMatroid α r).eRk_mono htX

/-- `U_{3,5}` has rank 3. -/
theorem uniformMatroid_three_five_eRank : (uniformMatroid (Fin 5) 3).eRank = 3 := by
  rw [Matroid.eRank_def, uniformMatroid_E, uniformMatroid_eRk, Set.encard_univ]
  simp only [ENat.card_eq_coe_fintype_card, Fintype.card_fin]
  exact min_eq_right (by exact_mod_cast (by omega : (3 : ℕ) ≤ 5))

/-- `U_{3,5}` is 3-connected. -/
theorem uniformMatroid_three_five_isNConnected : IsNConnected (uniformMatroid (Fin 5) 3) 3 := by
  intro k hk1 hk3 X Y ⟨hd, hu, hX, hY, hsep⟩
  rw [uniformMatroid_E] at hu
  have hXY : X.ncard + Y.ncard = 5 := by
    rw [← Set.ncard_union_eq hd (Set.toFinite X) (Set.toFinite Y), hu]
    simp [Set.ncard_univ]
  rw [uniformMatroid_three_five_eRank, uniformMatroid_eRk, uniformMatroid_eRk,
    ← (Set.toFinite X).cast_ncard_eq, ← (Set.toFinite Y).cast_ncard_eq] at hsep
  rw [← (Set.toFinite X).cast_ncard_eq] at hX
  rw [← (Set.toFinite Y).cast_ncard_eq] at hY
  have hX' : k ≤ X.ncard := by exact_mod_cast hX
  have hY' : k ≤ Y.ncard := by exact_mod_cast hY
  have hsep' : min X.ncard 3 + min Y.ncard 3 + 1 ≤ 3 + k := by
    have hmin : ∀ a : ℕ, min (a : ℕ∞) ((3 : ℕ) : ℕ∞) = ((min a 3 : ℕ) : ℕ∞) := fun a =>
      ((Nat.mono_cast (α := ℕ∞)).map_min (a := a) (b := 3)).symm
    rw [hmin, hmin] at hsep
    exact_mod_cast hsep
  omega

end Uniform

end Transformer.GainMatroid
