module

public import Transformer.GainMatroid.Section3_SmallExamples

/-!
# Theorem 5.1: properly unbalanced biased graphs

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §5. On a loopless, 2-connected, properly unbalanced
biased graph, canonical frame matrices of two realizations are projectively equivalent exactly when
the gain functions are switching equivalent, canonical lift matrices exactly when they are
switching-and-scaling equivalent, and a canonical frame matrix is never projectively equivalent to a
canonical lift matrix.

The source proves this through Theorem 6 and the unavoidable minors of §3. Here the proof is direct.
A 2-connected graph stays connected off each vertex (`IsKConnected.connectedOff`), and a realization
of a properly unbalanced biased graph has an unbalanced closed walk off each vertex
(`BiasedGraph.Realizes.hasNoBalancingVertex`), so the row theorems of §4 apply. The second gain
function need not realize the same biased graph, and in (iii) it is arbitrary.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {F Γ : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- Each vertex of a walk with an edge is an end of one of its edges. -/
theorem Walk.mem_verts_of_mem_support {u v : V} (p : G.Walk u v) (hp : p.edges ≠ []) {x : V}
    (hx : x ∈ p.support) : x ∈ G.verts p.edgeSet := by
  induction p with
  | nil => simp [Walk.edges] at hp
  | @cons a b c e hj p ih =>
    have hmem : e ∈ (Walk.cons e hj p).edgeSet := by simp [Walk.edgeSet, Walk.edges]
    have hmono : G.verts p.edgeSet ⊆ G.verts (Walk.cons e hj p).edgeSet :=
      verts_mono fun f (hf : f ∈ p.edges) =>
        (List.mem_cons_of_mem e hf : f ∈ (Walk.cons e hj p).edges)
    have hends : a ∈ G.verts (Walk.cons e hj p).edgeSet ∧
        b ∈ G.verts (Walk.cons e hj p).edgeSet := by
      rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
      · exact ⟨mem_verts.2 ⟨e, hmem, Or.inl h₁⟩, mem_verts.2 ⟨e, hmem, Or.inr h₂⟩⟩
      · exact ⟨mem_verts.2 ⟨e, hmem, Or.inr h₂⟩, mem_verts.2 ⟨e, hmem, Or.inl h₁⟩⟩
    rcases List.mem_cons.1 hx with rfl | hx
    · exact hends.1
    · by_cases hnil : p.edges = []
      · cases p with
        | nil =>
          have : x = b := by simpa [Walk.support] using hx
          exact this ▸ hends.2
        | cons f hf q => simp [Walk.edges] at hnil
      · exact hmono (ih hnil hx)

/-- A 2-connected graph stays connected off each vertex (§2.1). Otherwise the edges meeting the
vertices that reach `x` off `w` form one side of a vertical separation of order at most one. -/
theorem IsKConnected.connectedOff (h : G.IsKConnected 2) : G.ConnectedOff := by
  classical
  intro w x y hx hy hxw hyw
  rcases h with ⟨-, hsep⟩ | ⟨-, hcomplete⟩
  swap
  · by_cases hxy : x = y
    · subst hxy
      exact ⟨.nil x, by simp [Walk.support, hxw.symm]⟩
    · obtain ⟨e, he⟩ := hcomplete x y hxy
      exact ⟨.cons e he (.nil y), by simp [Walk.support, hxw.symm, hyw.symm]⟩
  by_contra hno
  let P : Set V := {v | ∃ p : G.Walk v x, w ∉ p.support}
  have hxP : x ∈ P := ⟨.nil x, by simp [Walk.support, hxw.symm]⟩
  have hyP : y ∉ P := fun ⟨p, hp⟩ => hno ⟨p.reverse, by
    rwa [Walk.support_reverse, List.mem_reverse]⟩
  have hext : ∀ {v z : V} {e : E}, v ∈ P → G.Joins e v z → z ≠ w → z ∈ P := by
    rintro v z e ⟨p, hp⟩ hj hz
    exact ⟨.cons e hj.symm p, by simp [Walk.support, hz.symm, hp]⟩
  let A : Set E := {e | G.tail e ∈ P ∨ G.head e ∈ P}
  have hVA : ∀ z ∈ G.verts A, z ∈ P ∨ z = w := by
    intro z hz
    obtain ⟨e, he | he, hend | hend⟩ := mem_verts.1 hz
    · exact Or.inl (hend ▸ he)
    · by_cases hzw : z = w
      · exact Or.inr hzw
      · exact Or.inl (hext he (Or.inl ⟨rfl, hend⟩) hzw)
    · by_cases hzw : z = w
      · exact Or.inr hzw
      · exact Or.inl (hext he (Or.inr ⟨hend, rfl⟩) hzw)
    · exact Or.inl (hend ▸ he)
  have hVB : ∀ z ∈ G.verts Aᶜ, z ∉ P := by
    intro z hz hzP
    obtain ⟨e, he, hend | hend⟩ := mem_verts.1 hz
    · exact he (Or.inl (hend ▸ hzP))
    · exact he (Or.inr (hend ▸ hzP))
  obtain ⟨e₁, -, he₁⟩ := mem_verts.1 hx
  obtain ⟨e₂, -, he₂⟩ := mem_verts.1 hy
  have he₁A : e₁ ∈ A := by
    rcases he₁ with h | h
    · exact Or.inl (h ▸ hxP)
    · exact Or.inr (h ▸ hxP)
  have hxA : x ∈ G.verts A := mem_verts.2 ⟨e₁, he₁A, he₁⟩
  have hyNotA : y ∉ G.verts A := fun hyA => (hVA y hyA).elim hyP hyw
  have he₂B : e₂ ∈ Aᶜ := fun he₂A => hyNotA (mem_verts.2 ⟨e₂, he₂A, he₂⟩)
  have hyB : y ∈ G.verts Aᶜ := mem_verts.2 ⟨e₂, he₂B, he₂⟩
  have hsub : G.verts A ∩ G.verts Aᶜ ⊆ {w} := fun z hz =>
    (hVA z hz.1).elim (fun hzP => absurd hzP (hVB z hz.2)) id
  have hfin : (G.verts A ∩ G.verts Aᶜ).Finite := (Set.finite_singleton w).subset hsub
  have hcard : (G.verts A ∩ G.verts Aᶜ).ncard ≤ 1 := by
    simpa using Set.ncard_le_ncard hsub (Set.finite_singleton w)
  have hone : ((G.verts A ∩ G.verts Aᶜ).ncard : ℕ∞) ≤ 1 := by exact_mod_cast hcard
  exact hsep _ (by omega) A Aᶜ ⟨⟨disjoint_compl_right, Set.union_compl_self A,
    hone.trans (Set.one_le_encard_iff_nonempty.2 ⟨e₁, he₁A⟩),
    hone.trans (Set.one_le_encard_iff_nonempty.2 ⟨e₂, he₂B⟩), hfin.cast_ncard_eq.symm⟩,
    ⟨x, hxA, fun hxB => hVB x hxB hxP⟩, ⟨y, hyB, hyNotA⟩⟩

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E}

/-- An unbalanced biased graph has a vertex. -/
theorem nonempty_of_not_isBalanced (h : ¬Ω.IsBalanced) : Nonempty V := by
  obtain ⟨C, hC⟩ := not_forall.1 h
  obtain ⟨⟨u, -⟩, -⟩ := Classical.not_imp.1 hC
  exact ⟨u⟩

variable [DecidableEq V]

/-- A realization of a properly unbalanced biased graph has a closed walk of non-identity gain off
each vertex: an unbalanced cycle missing that vertex. -/
@[to_additive AddRealizes.hasNoAddBalancingVertex
  /-- An additive realization of a properly unbalanced biased graph has a closed walk of nonzero
  gain off each vertex: an unbalanced cycle missing that vertex. -/]
theorem Realizes.hasNoBalancingVertex [Group Γ] {γ : E → Γ} (h : Ω.Realizes γ)
    (hΩ : Ω.IsProperlyUnbalanced) : Ω.graph.HasNoBalancingVertex γ := by
  intro w
  by_contra hcon
  refine hΩ.2 ⟨hΩ.1, w, fun C hC _ => ?_⟩
  by_contra hw
  obtain ⟨⟨u, W, hW, rfl⟩, hnot⟩ := hC
  refine hcon ⟨u, W, fun hs => hw (W.mem_verts_of_mem_support hW.edges_ne_nil hs),
    fun hg => hnot ?_⟩
  rw [show Ω.balanced = Ω.graph.balancedCycles γ from h.symm]
  exact (hW.mem_balancedCycles_iff γ).2 hg

variable [Fintype V] [Fintype E] [DecidableEq E] [Field F]

/-- **Theorem 5.1 (i).** On a loopless, 2-connected, properly unbalanced biased graph, the canonical
frame matrices of `φ` and `ψ` are projectively equivalent exactly when `φ` and `ψ` are switching
equivalent. The source also assumes that `ψ` realizes the biased graph. -/
theorem IsProperlyUnbalanced.frame_projEquiv_iff (hΩ : Ω.IsProperlyUnbalanced)
    (hG : Ω.graph.Loopless) (h2 : Ω.graph.IsKConnected 2) {φ ψ : E → Fˣ} (hφ : Ω.Realizes φ) :
    ProjEquiv (frameMatrix Ω.graph φ) (frameMatrix Ω.graph ψ) ↔ SwitchEquiv Ω.graph φ ψ :=
  Transformer.GainMatroid.frame_projEquiv_iff hG h2.connectedOff (hφ.hasNoBalancingVertex hΩ)

/-- **Theorem 5.1 (ii).** On a loopless, 2-connected, properly unbalanced biased graph, the
canonical lift matrices of `φ` and `ψ` are projectively equivalent exactly when `φ` and `ψ` are
switching-and-scaling equivalent. The source also assumes that `ψ` realizes the biased graph. -/
theorem IsProperlyUnbalanced.lift_projEquiv_iff (hΩ : Ω.IsProperlyUnbalanced)
    (hG : Ω.graph.Loopless) (h2 : Ω.graph.IsKConnected 2) {φ ψ : E → F} (hφ : Ω.AddRealizes φ) :
    ProjEquiv (liftMatrix Ω.graph φ) (liftMatrix Ω.graph ψ) ↔ SwitchScaleEquiv Ω.graph φ ψ :=
  Transformer.GainMatroid.lift_projEquiv_iff hG h2.connectedOff (hφ.hasNoAddBalancingVertex hΩ)

omit [Fintype E] [DecidableEq E] in
/-- **Theorem 5.1 (iii).** On a loopless, 2-connected, properly unbalanced biased graph, the
canonical frame matrix of an `Fˣ`-realization is projectively equivalent to no canonical lift
matrix. The source assumes that `ψ` is an `F⁺`-realization of the biased graph. -/
theorem IsProperlyUnbalanced.not_projEquiv_frame_lift (hΩ : Ω.IsProperlyUnbalanced)
    (hG : Ω.graph.Loopless) (h2 : Ω.graph.IsKConnected 2) {φ : E → Fˣ} (hφ : Ω.Realizes φ)
    (ψ : E → F) : ¬ProjEquiv (frameMatrix Ω.graph φ) (liftMatrix Ω.graph ψ) :=
  haveI := nonempty_of_not_isBalanced hΩ.1
  Transformer.GainMatroid.not_projEquiv_frame_lift hG h2.connectedOff
    (hφ.hasNoBalancingVertex hΩ) ψ

end BiasedGraph

example : twoC3.ConnectedOff := twoC3_isKConnected.connectedOff

example : twoC3.HasNoBalancingVertex twoC3Gain :=
  BiasedGraph.Realizes.hasNoBalancingVertex (Ω := twoC3.gainBiased twoC3Gain) rfl
    twoC3Gain_isProperlyUnbalanced

example (ψ : Fin 6 → (ZMod 3)ˣ) : ProjEquiv (frameMatrix twoC3 twoC3Gain) (frameMatrix twoC3 ψ) ↔
    SwitchEquiv twoC3 twoC3Gain ψ :=
  twoC3Gain_isProperlyUnbalanced.frame_projEquiv_iff twoC3_loopless twoC3_isKConnected rfl

example (ψ : Fin 6 → ZMod 3) : ProjEquiv (liftMatrix twoC3 twoC3AddGain) (liftMatrix twoC3 ψ) ↔
    SwitchScaleEquiv twoC3 twoC3AddGain ψ :=
  twoC3AddGain_isProperlyUnbalanced.lift_projEquiv_iff twoC3_loopless twoC3_isKConnected rfl

example (ψ : Fin 6 → ZMod 3) : ¬ProjEquiv (frameMatrix twoC3 twoC3Gain) (liftMatrix twoC3 ψ) :=
  twoC3Gain_isProperlyUnbalanced.not_projEquiv_frame_lift twoC3_loopless twoC3_isKConnected rfl ψ

end Transformer.GainMatroid
