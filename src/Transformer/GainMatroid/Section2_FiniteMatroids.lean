module

public import Mathlib.Combinatorics.Matroid.Closure
public import Mathlib.Data.Fintype.Powerset
public import Transformer.GainMatroid.Section2_CycleDegrees
public import Transformer.GainMatroid.Section2_Matroids

/-!
# Finite matroids from their independent sets

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2 and §2.4. `matroidOfFinset` builds a matroid
on a finite type from a decidable predicate on finite sets, and `FlatTest` decides its flats, so
that small matroids can be checked by evaluation.

In a balanced biased graph every handcuff and theta contains a balanced cycle, so the circuits of
the frame and lift matroids are the cycles of the graph (`BiasedGraph.IsBalanced.isFrameCircuit_iff`
and `BiasedGraph.IsBalanced.isLiftCircuit_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

section FiniteMatroid

variable {α : Type*} [DecidableEq α]

/-- `S` passes this test when every element outside `S` extends every maximal independent subset
of `S`. For a matroid this says that `S` is a flat (`isFlat_coe_iff`). -/
def FlatTest (indep : Finset α → Prop) (S : Finset α) : Prop :=
  ∀ I, I ⊆ S → indep I → (∀ f ∈ S, indep (insert f I) → f ∈ I) →
    ∀ e, e ∉ S → indep (insert e I)

instance [Fintype α] (indep : Finset α → Prop) [DecidablePred indep] (S : Finset α) :
    Decidable (FlatTest indep S) := by
  unfold FlatTest
  infer_instance

/-- A finite set is a flat exactly when it passes `FlatTest`. -/
theorem isFlat_coe_iff [Finite α] (M : Matroid α) (hE : M.E = Set.univ) (S : Finset α) :
    M.IsFlat (S : Set α) ↔ FlatTest (fun I : Finset α => M.Indep (I : Set α)) S := by
  have hsub (X : Set α) : X ⊆ M.E := hE ▸ Set.subset_univ X
  constructor
  · intro hS I hIS hI hmax e he
    have hb : M.IsBasis (I : Set α) S := hI.isBasis_of_forall_insert (by exact_mod_cast hIS)
      fun f hf => (Matroid.not_indep_iff (hsub _)).1 fun h =>
        hf.2 (Finset.mem_coe.2 (hmax f (Finset.mem_coe.1 hf.1) (by simpa using h)))
    by_contra hdep
    have hdep' : M.Dep (insert e (I : Set α)) :=
      (Matroid.not_indep_iff (hsub _)).1 (by simpa using hdep)
    have hcl : e ∈ M.closure (I : Set α) := hI.mem_closure_iff.2 (Or.inl hdep')
    rw [hb.closure_eq_closure, hS.closure] at hcl
    exact he (Finset.mem_coe.1 hcl)
  · intro h
    obtain ⟨I, hI⟩ := M.exists_isBasis (S : Set α) (hsub _)
    obtain ⟨I', rfl⟩ := (Set.toFinite I).exists_finset_coe
    rw [Matroid.isFlat_iff_closure_eq]
    refine Set.Subset.antisymm (fun e he => ?_) (M.subset_closure _ (hsub _))
    by_contra heS
    have hins : M.Indep ((insert e I' : Finset α) : Set α) :=
      h I' (by exact_mod_cast hI.subset) hI.indep
        (fun f hf hfI => Finset.mem_coe.1 (hI.mem_of_insert_indep (Finset.mem_coe.2 hf)
          (by simpa using hfI))) e (fun hes => heS (Finset.mem_coe.2 hes))
    rw [← hI.closure_eq_closure] at he
    rcases hI.indep.mem_closure_iff.1 he with hdep | heI
    · exact hdep.not_indep (by simpa using hins)
    · exact heS (hI.subset heI)

/-- The matroid on all of `α` whose independent finite sets are those of `indep`. -/
noncomputable def matroidOfFinset (indep : Finset α → Prop) (h0 : indep ∅)
    (hsub : ∀ I J : Finset α, indep J → I ⊆ J → indep I)
    (haug : ∀ I J : Finset α, indep I → indep J → I.card < J.card →
      ∃ e ∈ J, e ∉ I ∧ indep (insert e I)) : Matroid α :=
  (IndepMatroid.ofFinset Set.univ indep h0 (fun I J hJ hIJ => hsub I J hJ hIJ)
    (fun I J hI hJ hlt => haug I J hI hJ hlt) (fun _ _ => Set.subset_univ _)).matroid

variable {indep : Finset α → Prop} {h0 : indep ∅}
  {hsub : ∀ I J : Finset α, indep J → I ⊆ J → indep I}
  {haug : ∀ I J : Finset α, indep I → indep J → I.card < J.card →
    ∃ e ∈ J, e ∉ I ∧ indep (insert e I)}

theorem matroidOfFinset_E : (matroidOfFinset indep h0 hsub haug).E = Set.univ := rfl

theorem matroidOfFinset_indep (I : Finset α) :
    (matroidOfFinset indep h0 hsub haug).Indep (I : Set α) ↔ indep I := by
  simp [matroidOfFinset]

/-- The flats of `matroidOfFinset`, by `FlatTest` on its predicate. -/
theorem matroidOfFinset_isFlat_iff [Finite α] (S : Finset α) :
    (matroidOfFinset indep h0 hsub haug).IsFlat (S : Set α) ↔ FlatTest indep S := by
  rw [isFlat_coe_iff _ matroidOfFinset_E]
  have h : (fun I : Finset α => (matroidOfFinset indep h0 hsub haug).Indep (I : Set α)) = indep :=
    funext fun I => propext (matroidOfFinset_indep I)
  rw [h]

/-- The circuits of `matroidOfFinset`: dependent sets all of whose one-point deletions are
independent. -/
theorem matroidOfFinset_isCircuit_iff (C : Finset α) :
    (matroidOfFinset indep h0 hsub haug).IsCircuit (C : Set α) ↔
      ¬indep C ∧ ∀ e ∈ C, indep (C.erase e) := by
  rw [Matroid.isCircuit_iff_dep_forall_sdiff_singleton_indep, Matroid.Dep]
  have herase (e : α) : (C : Set α) \ {e} = ((C.erase e : Finset α) : Set α) := by
    simp [Finset.coe_erase]
  simp only [matroidOfFinset_indep, herase, matroidOfFinset_E, Set.subset_univ, and_true,
    Finset.mem_coe]

end FiniteMatroid

namespace BiasedGraph

variable {V E : Type*} {Ω : BiasedGraph V E} {X : Set E}

private theorem exists_isCycle_of_theta {G : OrientedGraph V E} (h : G.IsThetaSet X) :
    ∃ C ⊆ X, G.IsCycle C := by
  obtain ⟨x, y, P, Q, R, hT, rfl⟩ := h
  exact ⟨_, Set.subset_union_left, hT.isCycle.1⟩

/-- In a balanced biased graph the circuits of the frame matroid are the cycles: a handcuff or a
theta contains a cycle, which is balanced. -/
theorem IsBalanced.isFrameCircuit_iff (h : Ω.IsBalanced) :
    Ω.IsFrameCircuit X ↔ Ω.graph.IsCycle X := by
  refine ⟨?_, fun hX => Or.inl (h X hX)⟩
  rintro (hX | ⟨hshape, hno⟩)
  · exact Ω.isCycle_of_mem _ hX
  · obtain ⟨C, hCX, hC⟩ : ∃ C ⊆ X, Ω.graph.IsCycle C := by
      rcases hshape with ⟨C, D, v, hC, -, -, -, rfl⟩ |
        ⟨C, D, x, y, P, hC, -, -, -, -, -, -, -, rfl⟩ | ht
      · exact ⟨C, Set.subset_union_left, hC⟩
      · exact ⟨C, Set.subset_union_left.trans Set.subset_union_left, hC⟩
      · exact exists_isCycle_of_theta ht
    exact absurd (h C hC) (hno C hCX hC)

/-- In a balanced biased graph the circuits of the lift matroid are the cycles. -/
theorem IsBalanced.isLiftCircuit_iff (h : Ω.IsBalanced) :
    Ω.IsLiftCircuit X ↔ Ω.graph.IsCycle X := by
  refine ⟨?_, fun hX => Or.inl (h X hX)⟩
  rintro (hX | ⟨hshape, hno⟩)
  · exact Ω.isCycle_of_mem _ hX
  · obtain ⟨C, hCX, hC⟩ : ∃ C ⊆ X, Ω.graph.IsCycle C := by
      rcases hshape with ⟨C, D, v, hC, -, -, -, rfl⟩ | ⟨C, D, hC, -, -, rfl⟩ | ht
      · exact ⟨C, Set.subset_union_left, hC⟩
      · exact ⟨C, Set.subset_union_left, hC⟩
      · exact exists_isCycle_of_theta ht
    exact absurd (h C hC) (hno C hCX hC)

end BiasedGraph

/-- `U_{2,3}`, built from its independent sets, for the examples below. -/
noncomputable def twoOfThree : Matroid (Fin 3) :=
  matroidOfFinset (fun I => I.card ≤ 2) (by decide) (by decide +kernel) (by decide +kernel)

example := isFlat_coe_iff twoOfThree matroidOfFinset_E {0}

example : twoOfThree.Indep (({0, 1} : Finset (Fin 3)) : Set (Fin 3)) :=
  (matroidOfFinset_indep _).2 (by decide)

example : twoOfThree.IsFlat (({0} : Finset (Fin 3)) : Set (Fin 3)) :=
  (matroidOfFinset_isFlat_iff _).2 (by decide +kernel)

example : twoOfThree.IsCircuit (({0, 1, 2} : Finset (Fin 3)) : Set (Fin 3)) :=
  (matroidOfFinset_isCircuit_iff _).2 (by decide +kernel)

example :=
  (gainBiased_one_isBalanced (Γ := ℚˣ) OrientedGraph.K4).isFrameCircuit_iff (X := {0, 1, 3})

example :=
  (gainBiased_one_isBalanced (Γ := ℚˣ) OrientedGraph.K4).isLiftCircuit_iff (X := {0, 1, 3})

end Transformer.GainMatroid
