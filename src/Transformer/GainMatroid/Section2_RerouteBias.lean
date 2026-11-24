module

public import Transformer.GainMatroid.Section2_Rerouting

/-!
# Cycles through two edges at a vertex have one bias

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5. Let every cycle that misses a vertex `u` be
balanced. Two paths `P` and `Q` between the same ends that miss `u` close cycles of the same bias
with a path `S` openly disjoint from both (`BiasedGraph.balanced_iff_of_openly`): rerouting `P` to
start with the first edge of `Q` (`BiasedGraph.exists_reroute`) keeps the bias, and moving that edge
onto `S` leaves a shorter `Q`. For `S` the path through `u` along two edges `e` and `f` at `u`, all
cycles through `e` and `f` whose other edges form a path missing `u` have one bias
(`BiasedGraph.insert_mem_balanced_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE}

open OrientedGraph OrientedGraph.Walk

namespace BiasedGraph

variable {Ω : BiasedGraph V E}

/-- If every cycle that misses `u` is balanced, two paths `P` and `Q` between the same ends that
miss `u` close cycles of the same bias with a path `S` openly disjoint from both. -/
theorem balanced_iff_of_openly {u x y : V}
    (hbal : ∀ {a b : V} (A B : Ω.graph.Walk a b), a ≠ b → A.IsPath → B.IsPath → Openly A B →
      u ∉ A.support → u ∉ B.support → A.edgeSet ∪ B.edgeSet ∈ Ω.balanced)
    {P Q S : Ω.graph.Walk x y} (hP : P.IsPath) (hQ : Q.IsPath) (hS : S.IsPath)
    (hPS : Openly P S) (hQS : Openly Q S) (huP : u ∉ P.support) (huQ : u ∉ Q.support) :
    P.edgeSet ∪ S.edgeSet ∈ Ω.balanced ↔ Q.edgeSet ∪ S.edgeSet ∈ Ω.balanced := by
  induction Q with
  | nil =>
    obtain rfl := hP.eq_nil
    exact Iff.rfl
  | @cons x q y e₁ h₁ Q ih =>
    obtain ⟨P', hP', huP', hP'S, hiff⟩ := exists_reroute hbal hP hQ hS hPS hQS huP huQ
    refine hiff.trans ?_
    rw [edgeSet_cons, edgeSet_cons, Set.insert_union, Set.insert_union]
    by_cases hqy : q = y
    · subst hqy
      rw [(isPath_cons.1 hP').2.eq_nil, (isPath_cons.1 hQ).2.eq_nil]
    have hqS : q ∉ S.support := fun h =>
      (hQS.verts q (List.mem_cons_of_mem _ Q.start_mem_support) h).elim
        (fun hqx => (isPath_cons.1 hQ).1 (by rw [← hqx]; exact Q.start_mem_support)) hqy
    have key := ih (P := P') (S := cons e₁ h₁.symm S) (isPath_cons.1 hP').2 (isPath_cons.1 hQ).2
      (isPath_cons.2 ⟨hqS, hS⟩) (hP'S.of_cons hP') (hQS.of_cons hQ)
      (fun h => huP' (List.mem_cons_of_mem _ h)) (fun h => huQ (List.mem_cons_of_mem _ h))
    rwa [edgeSet_cons, Set.union_insert, Set.union_insert] at key

/-- If every cycle that misses `u` is balanced, two edges `e` and `f` joining `x` and `y` to `u`
close cycles of one bias with any two paths from `x` to `y` that miss `u`. -/
theorem insert_mem_balanced_iff {u x y : V} {e f : E}
    (hbal : ∀ {a b : V} (A B : Ω.graph.Walk a b), a ≠ b → A.IsPath → B.IsPath → Openly A B →
      u ∉ A.support → u ∉ B.support → A.edgeSet ∪ B.edgeSet ∈ Ω.balanced)
    (he : Ω.graph.Joins e x u) (hf : Ω.graph.Joins f y u) {P Q : Ω.graph.Walk x y}
    (hP : P.IsPath) (hQ : Q.IsPath) (huP : u ∉ P.support) (huQ : u ∉ Q.support) :
    insert e (insert f P.edgeSet) ∈ Ω.balanced ↔ insert e (insert f Q.edgeSet) ∈ Ω.balanced := by
  by_cases hxy : x = y
  · subst hxy
    rw [hP.eq_nil, hQ.eq_nil]
  have hxu : x ≠ u := by
    rintro rfl
    exact huP P.start_mem_support
  have hyu : y ≠ u := by
    rintro rfl
    exact huP P.end_mem_support
  have hS : (cons e he (cons f hf.symm (nil y))).IsPath := by
    unfold Walk.IsPath
    simp [Walk.support, hxu, hxy, hyu.symm]
  have hE : ∀ R : Ω.graph.Walk x y,
      insert e (insert f R.edgeSet) = R.edgeSet ∪ (cons e he (cons f hf.symm (nil y))).edgeSet := by
    intro R
    ext g
    simp [edgeSet, edges]
    tauto
  rw [hE P, hE Q]
  exact balanced_iff_of_openly hbal hP hQ hS (openly_of_not_mem he hf.symm huP)
    (openly_of_not_mem he hf.symm huQ) huP huQ

end BiasedGraph

example := BiasedGraph.balanced_iff_of_openly (Ω := rerouteBiased) (u := 2)
  (fun A B hab hA hB hAB _ _ => rerouteBiased_balanced A B hab hA hB hAB) rerouteP_isPath
  rerouteQ_isPath rerouteS_isPath rerouteP_openly rerouteQ_openly (by decide) (by decide)

example := BiasedGraph.insert_mem_balanced_iff (Ω := rerouteBiased) (u := 2) (x := 0) (y := 1)
  (e := 2) (f := 3) (fun A B hab hA hB hAB _ _ => rerouteBiased_balanced A B hab hA hB hAB)
  (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) rerouteP_isPath rerouteQ_isPath (by decide) (by decide)

end Transformer.GainMatroid
