module

public import Mathlib.Combinatorics.Matroid.Map
public import Transformer.GainMatroid.Section2_CycleDegrees
public import Transformer.GainMatroid.Section2_DeltaY
public import Transformer.GainMatroid.Section2_FiniteMatroids

/-!
# Δ-Y and Y-Δ exchanges of matroids

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4: "To perform a Δ-Y exchange in a matroid `M`,
take a copy of `M(K₄)` on ground set `{a, b, c, a', b', c'}` labelled so that `{a, b, c}` is a
triangle, `{a', b', c'}` is a triad, and each of `{a, b', c'}`, `{a', b, c'}`, and `{a', b', c}` are
triangles. Let `X = {a, b, c}` be a triangle of `M`, and take the generalized parallel connection of
`M` and `M(K₄)` across `X`. Finally, delete `X` and relabel each of `a', b', c'` by `a, b, c`."

`k4Matroid` is that copy of `M(K₄)`: `k4Matroid_isCircuit_iff` identifies its circuits with the
cycles of `K4`, and `k4Matroid_triad` proves the triad. The generalized parallel connection, which
the source takes from Oxley's text, is the matroid whose flats meet the ground set of each part in
a flat of that part (`IsDeltaMatroid`). A Y-Δ exchange is defined by duality (`IsNablaMatroid`).

In a balanced biased graph the circuits of the frame and lift matroids are the cycles
(`BiasedGraph.IsBalanced.isFrameCircuit_iff`, `BiasedGraph.IsBalanced.isLiftCircuit_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

/-- The independent sets of `M(K₄)` on `{a, b, c} = inl 0, 1, 2` and `{a', b', c'} = inr 0, 1, 2`:
at most three elements, and not one of the four triangles of §2.4. -/
def k4Indep (I : Finset (Fin 3 ⊕ Fin 3)) : Prop :=
  I.card ≤ 3 ∧ ¬(.inl 0 ∈ I ∧ .inl 1 ∈ I ∧ .inl 2 ∈ I) ∧ ¬(.inl 0 ∈ I ∧ .inr 1 ∈ I ∧ .inr 2 ∈ I) ∧
    ¬(.inr 0 ∈ I ∧ .inl 1 ∈ I ∧ .inr 2 ∈ I) ∧ ¬(.inr 0 ∈ I ∧ .inr 1 ∈ I ∧ .inl 2 ∈ I)

instance : DecidablePred k4Indep := fun I => by
  unfold k4Indep
  infer_instance

/-- **`M(K₄)`** with the labels of §2.4. -/
noncomputable def k4Matroid : Matroid (Fin 3 ⊕ Fin 3) :=
  matroidOfFinset k4Indep (by decide) (by decide +kernel) (by decide +kernel)

/-- The edges of `K4` carrying the labels of §2.4, for `v = 0`, `x = 1`, `y = 2`, `z = 3`:
`a = yz`, `b = xz`, `c = xy`, `a' = vx`, `b' = vy`, `c' = vz`. -/
def k4Label : Fin 3 ⊕ Fin 3 → Fin 6 := Sum.elim ![5, 4, 3] ![0, 1, 2]

theorem k4Label_injective : Function.Injective k4Label := by decide

set_option maxRecDepth 100000 in
private theorem k4Circuit_test : ∀ C : Finset (Fin 3 ⊕ Fin 3),
    (¬k4Indep C ∧ ∀ e ∈ C, k4Indep (C.erase e)) ↔
      C.image k4Label ∈ OrientedGraph.K4Cycles := by
  decide +kernel

/-- `k4Matroid` is the cycle matroid of `K₄`: its circuits are the cycles of `K4`, read through the
labels of §2.4. -/
theorem k4Matroid_isCircuit_iff (C : Set (Fin 3 ⊕ Fin 3)) :
    k4Matroid.IsCircuit C ↔ OrientedGraph.K4.IsCycle (k4Label '' C) := by
  obtain ⟨C, rfl⟩ := (Set.toFinite C).exists_finset_coe
  have himage : k4Label '' (C : Set (Fin 3 ⊕ Fin 3)) = ((C.image k4Label : Finset _) : Set _) := by
    simp
  rw [himage, OrientedGraph.K4_isCycle_iff, ← k4Circuit_test]
  exact matroidOfFinset_isCircuit_iff C

set_option maxRecDepth 100000 in
/-- `{a', b', c'}` is a triad of `k4Matroid`, as §2.4 requires. -/
theorem k4Matroid_triad : k4Matroid✶.IsCircuit
    (({.inr 0, .inr 1, .inr 2} : Finset (Fin 3 ⊕ Fin 3)) : Set (Fin 3 ⊕ Fin 3)) :=
  (matroidOfFinset_dual_isCircuit_iff _).2 (by decide +kernel)

variable {E : Type*}

/-- The copy of `M(K₄)` inside `E ⊕ Fin 3`: its triangle is `a`, `b`, `c` of `E`, and its elements
`a'`, `b'`, `c'` are the three new elements. -/
def deltaEmbed (a b c : E) : Fin 3 ⊕ Fin 3 → E ⊕ Fin 3 :=
  Sum.elim (fun i => Sum.inl (![a, b, c] i)) Sum.inr

/-- The relabelling of §2.4: `a`, `b`, `c` name the new elements `a'`, `b'`, `c'`, and the other
elements keep their names. Its range misses `X`, so pulling back along it deletes `X`. -/
def deltaRelabel [DecidableEq E] (a b c e : E) : E ⊕ Fin 3 :=
  if e = a then .inr 0 else if e = b then .inr 1 else if e = c then .inr 2 else .inl e

/-- `deltaEmbed` is injective on a labelled triangle. -/
theorem deltaEmbed_injective {a b c : E} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    Function.Injective (deltaEmbed a b c) := by
  rintro (i | i) (j | j) h
  · have h' : ![a, b, c] i = ![a, b, c] j := Sum.inl_injective h
    fin_cases i <;> fin_cases j <;> simp_all
  · exact (Sum.inl_ne_inr h).elim
  · exact (Sum.inr_ne_inl h).elim
  · exact congrArg _ (Sum.inr_injective h)

/-- `deltaRelabel` is injective. -/
theorem deltaRelabel_injective [DecidableEq E] (a b c : E) :
    Function.Injective (deltaRelabel a b c) := by
  intro e f h
  simp only [deltaRelabel] at h
  split_ifs at h <;> simp_all

/-- **`M' = Δ_X M`** (§2.4) for a triangle `X = {a, b, c}` of `M`: `P` is the generalized parallel
connection of `M` and `M(K₄)` across `X`, and `M'` is `P \ X` relabelled. -/
structure IsDeltaMatroid [DecidableEq E] (M : Matroid E) (a b c : E) (M' : Matroid E) : Prop where
  ne_ab : a ≠ b
  ne_ac : a ≠ c
  ne_bc : b ≠ c
  /-- `X` is a triangle of `M`. -/
  triangle : M.IsCircuit {a, b, c}
  /-- The generalized parallel connection, through its flats. -/
  exists_connection : ∃ P : Matroid (E ⊕ Fin 3), P.E = Sum.inl '' M.E ∪ Set.range Sum.inr ∧
    (∀ S, P.IsFlat S ↔ M.IsFlat (Sum.inl ⁻¹' S) ∧ k4Matroid.IsFlat (deltaEmbed a b c ⁻¹' S)) ∧
    M' = P.comap (deltaRelabel a b c)

/-- **`M' = ∇_Y M`** (§2.4) for a triad `Y = {a, b, c}` of `M`: `∇_Y M = (Δ_Y M*)*`. -/
def IsNablaMatroid [DecidableEq E] (M : Matroid E) (a b c : E) (M' : Matroid E) : Prop :=
  ∃ N, IsDeltaMatroid M✶ a b c N ∧ M' = N✶

/-- A Δ-Y exchange keeps the ground set. -/
theorem IsDeltaMatroid.ground_eq [DecidableEq E] {M M' : Matroid E} {a b c : E}
    (h : IsDeltaMatroid M a b c M') : M'.E = M.E := by
  obtain ⟨P, hE, -, rfl⟩ := h.exists_connection
  have hX : ({a, b, c} : Set E) ⊆ M.E := h.triangle.subset_ground
  have ha : a ∈ M.E := hX (by simp)
  have hb : b ∈ M.E := hX (by simp)
  have hc : c ∈ M.E := hX (by simp)
  ext e
  rw [Matroid.comap_ground_eq, Set.mem_preimage, hE]
  unfold deltaRelabel
  split_ifs with h1 h2 h3
  · simpa [h1] using ha
  · simpa [h2] using hb
  · simpa [h3] using hc
  · simp

namespace BiasedGraph

variable {V : Type*} {Ω : BiasedGraph V E} {X : Set E}

/-- Every circuit of a frame matroid contains a cycle. -/
theorem IsFrameCircuit.exists_isCycle (h : Ω.IsFrameCircuit X) : ∃ C ⊆ X, Ω.graph.IsCycle C := by
  rcases h with hX | ⟨hshape, -⟩
  · exact ⟨X, subset_rfl, Ω.isCycle_of_mem _ hX⟩
  · rcases hshape with ⟨C, D, v, hC, -, -, -, rfl⟩ |
      ⟨C, D, x, y, P, hC, -, -, -, -, -, -, -, rfl⟩ | ⟨x, y, P, Q, R, hT, rfl⟩
    · exact ⟨C, Set.subset_union_left, hC⟩
    · exact ⟨C, Set.subset_union_left.trans Set.subset_union_left, hC⟩
    · exact ⟨_, Set.subset_union_left, hT.isCycle.1⟩

/-- Every circuit of a lift matroid contains a cycle. -/
theorem IsLiftCircuit.exists_isCycle (h : Ω.IsLiftCircuit X) : ∃ C ⊆ X, Ω.graph.IsCycle C := by
  rcases h with hX | ⟨hshape, -⟩
  · exact ⟨X, subset_rfl, Ω.isCycle_of_mem _ hX⟩
  · rcases hshape with ⟨C, D, v, hC, -, -, -, rfl⟩ | ⟨C, D, hC, -, -, rfl⟩ |
      ⟨x, y, P, Q, R, hT, rfl⟩
    · exact ⟨C, Set.subset_union_left, hC⟩
    · exact ⟨C, Set.subset_union_left, hC⟩
    · exact ⟨_, Set.subset_union_left, hT.isCycle.1⟩

/-- In a balanced biased graph the circuits of the frame matroid are the cycles. -/
theorem IsBalanced.isFrameCircuit_iff (h : Ω.IsBalanced) :
    Ω.IsFrameCircuit X ↔ Ω.graph.IsCycle X := by
  refine ⟨?_, fun hX => Or.inl (h X hX)⟩
  rintro (hX | ⟨hshape, hno⟩)
  · exact Ω.isCycle_of_mem _ hX
  · obtain ⟨C, hCX, hC⟩ := IsFrameCircuit.exists_isCycle (Ω := Ω) (Or.inr ⟨hshape, hno⟩)
    exact absurd (h C hC) (hno C hCX hC)

/-- In a balanced biased graph the circuits of the lift matroid are the cycles. -/
theorem IsBalanced.isLiftCircuit_iff (h : Ω.IsBalanced) :
    Ω.IsLiftCircuit X ↔ Ω.graph.IsCycle X := by
  refine ⟨?_, fun hX => Or.inl (h X hX)⟩
  rintro (hX | ⟨hshape, hno⟩)
  · exact Ω.isCycle_of_mem _ hX
  · obtain ⟨C, hCX, hC⟩ := IsLiftCircuit.exists_isCycle (Ω := Ω) (Or.inr ⟨hshape, hno⟩)
    exact absurd (h C hC) (hno C hCX hC)

end BiasedGraph

example :=
  (gainBiased_one_isBalanced (Γ := ℚˣ) OrientedGraph.K4).isFrameCircuit_iff (X := {0, 1, 3})

example :=
  (gainBiased_one_isBalanced (Γ := ℚˣ) OrientedGraph.K4).isLiftCircuit_iff (X := {0, 1, 3})

end Transformer.GainMatroid
