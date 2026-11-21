module

public import Mathlib.Data.Set.Finite.Basic
public import Transformer.GainMatroid.Section2_Biased

/-!
# Forests and Proposition 2.1

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3. A forest is a set of edges containing no
cycle. Proposition 2.1 (`exists_switch_eq_one_of_isForest`) says that a switching function can make
the gain of every edge of a forest the identity. The source calls this immediate. For a finite
forest the proof adds the edges one at a time: the ends of a new edge lie in different components
of the part already normalized, so switching one component by a constant normalizes the new edge.
Hence two walks in any forest with the same ends have the same gain (`gain_eq_of_isForest`), and
the general case switches each vertex by the gain of a walk from a chosen root of its component.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {Γ : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

namespace Walk

/-- A function with equal values at the ends of every edge of a walk has equal values at the ends
of the walk. -/
theorem eq_of_forall_mem_edges {α : Type*} (η : V → α) {u v : V} (p : G.Walk u v)
    (h : ∀ e ∈ p.edges, η (G.tail e) = η (G.head e)) : η u = η v := by
  induction p with
  | nil => rfl
  | @cons a b c e hj p ih =>
    have he := h e List.mem_cons_self
    have hab : η a = η b := by
      rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
      · rw [← h₁, ← h₂]
        exact he
      · rw [← h₁, ← h₂]
        exact he.symm
    exact hab.trans (ih fun f hf => h f (List.mem_cons_of_mem _ hf))

end Walk

/-- `F` is a **forest** of `G` (§2.3): it contains no cycle. -/
def IsForest (G : OrientedGraph V E) (F : Set E) : Prop :=
  ∀ C, G.IsCycle C → ¬C ⊆ F

theorem IsForest.mono {F F' : Set E} (h : G.IsForest F) (hsub : F' ⊆ F) : G.IsForest F' :=
  fun C hC hle => h C hC (hle.trans hsub)

/-- `v` is reachable from `u` along edges of `S`. -/
def ReachableIn (G : OrientedGraph V E) (S : Set E) (u v : V) : Prop :=
  ∃ p : G.Walk u v, ∀ e ∈ p.edges, e ∈ S

theorem reachableIn_refl (S : Set E) (u : V) : G.ReachableIn S u u :=
  ⟨.nil u, by simp [Walk.edges]⟩

/-- A reachable vertex stays reachable across an edge of `S`. -/
theorem ReachableIn.step {S : Set E} {u v w : V} (h : G.ReachableIn S u v) {f : E} (hf : f ∈ S)
    (hj : G.Joins f v w) : G.ReachableIn S u w := by
  obtain ⟨p, hp⟩ := h
  refine ⟨p.append (.cons f hj (.nil w)), fun g hg => ?_⟩
  rw [Walk.edges_append] at hg
  simp only [Walk.edges, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hg
  rcases hg with hg | rfl
  · exact hp g hg
  · exact hf

/-- The ends of an edge added to a forest are not already linked. -/
theorem IsForest.not_reachableIn {e : E} {S : Set E} (hF : G.IsForest (insert e S)) (he : e ∉ S) :
    ¬G.ReachableIn S (G.head e) (G.tail e) := by
  rintro ⟨p, hp⟩
  obtain ⟨q, hq, hsub⟩ := p.exists_isPath
  have hj : G.Joins e (G.tail e) (G.head e) := Or.inl ⟨rfl, rfl⟩
  have hW : (Walk.cons e hj q).IsCycleWalk := by
    refine ⟨by simp [Walk.edges], ?_, hq⟩
    exact List.nodup_cons.2 ⟨fun hmem => he (hp e (hsub e hmem)), hq.edges_nodup⟩
  refine hF _ ⟨_, _, hW, rfl⟩ fun f hf => ?_
  have hf' : f ∈ e :: q.edges := hf
  rcases List.mem_cons.1 hf' with rfl | hf'
  · exact Set.mem_insert _ _
  · exact Set.mem_insert_of_mem _ (hp f (hsub f hf'))

theorem ReachableIn.symm {S : Set E} {u v : V} (h : G.ReachableIn S u v) : G.ReachableIn S v u := by
  obtain ⟨p, hp⟩ := h
  exact ⟨p.reverse, fun e he => hp e (by rwa [Walk.edges_reverse, List.mem_reverse] at he)⟩

theorem ReachableIn.trans {S : Set E} {u v w : V} (h₁ : G.ReachableIn S u v)
    (h₂ : G.ReachableIn S v w) : G.ReachableIn S u w := by
  obtain ⟨p, hp⟩ := h₁
  obtain ⟨q, hq⟩ := h₂
  refine ⟨p.append q, fun e he => ?_⟩
  rw [Walk.edges_append, List.mem_append] at he
  exact he.elim (hp e) (hq e)

/-- Reachability along `S` as an equivalence relation: its classes are the components of `S`. -/
def reachableSetoid (G : OrientedGraph V E) (S : Set E) : Setoid V where
  r := G.ReachableIn S
  iseqv := ⟨reachableIn_refl S, ReachableIn.symm, ReachableIn.trans⟩

/-- The finite case of Proposition 2.1: add the edges of the forest one at a time. -/
@[to_additive exists_addSwitch_eq_zero_of_finite
  /-- The finite case of Proposition 2.1 for an additive group. -/]
theorem exists_switch_eq_one_of_finite [Group Γ] {F : Set E} (hfin : F.Finite) (hF : G.IsForest F)
    (γ : E → Γ) : ∃ η : V → Γ, ∀ e ∈ F, switch G η γ e = 1 := by
  classical
  induction F, hfin using Set.Finite.induction_on with
  | empty => exact ⟨1, by simp⟩
  | @insert e S heS _ ih =>
    obtain ⟨ζ, hζ⟩ := ih (hF.mono (Set.subset_insert e S))
    let ξ : V → Γ := fun v => if G.ReachableIn S (G.head e) v then (switch G ζ γ e)⁻¹ else 1
    have hξh : ξ (G.head e) = (switch G ζ γ e)⁻¹ := by
      simp [ξ, reachableIn_refl S (G.head e)]
    have hξt : ξ (G.tail e) = 1 := by
      simp [ξ, hF.not_reachableIn heS]
    have hstep : ∀ f ∈ S, ξ (G.tail f) = ξ (G.head f) := by
      intro f hf
      have hiff : G.ReachableIn S (G.head e) (G.tail f) ↔
          G.ReachableIn S (G.head e) (G.head f) :=
        ⟨fun h => h.step hf (Or.inl ⟨rfl, rfl⟩), fun h => h.step hf (Or.inr ⟨rfl, rfl⟩)⟩
      simp only [ξ, hiff]
    refine ⟨ζ * ξ, fun f hf => ?_⟩
    rw [← switch_switch, switch_apply]
    rcases Set.mem_insert_iff.1 hf with rfl | hf
    · rw [hξt, hξh]
      simp
    · rw [hζ f hf, hstep f hf]
      simp

/-- Two walks in a forest with the same ends have the same gain. -/
@[to_additive addGain_eq_of_isForest
  /-- Two walks in a forest with the same ends have the same additive gain. -/]
theorem gain_eq_of_isForest [DecidableEq V] [Group Γ] {F : Set E} (hF : G.IsForest F) (γ : E → Γ)
    {u v : V} (p q : G.Walk u v) (hp : ∀ e ∈ p.edges, e ∈ F) (hq : ∀ e ∈ q.edges, e ∈ F) :
    p.gain γ = q.gain γ := by
  have hsub : {e | e ∈ p.edges ++ q.edges} ⊆ F := fun e he =>
    (List.mem_append.1 he).elim (hp e) (hq e)
  obtain ⟨η, hη⟩ := exists_switch_eq_one_of_finite (List.finite_toSet _) (hF.mono hsub) γ
  rw [Walk.gain_eq_of_potential η p fun e he => hη e (List.mem_append_left _ he),
    Walk.gain_eq_of_potential η q fun e he => hη e (List.mem_append_right _ he)]

/-- **Proposition 2.1** (§2.3). For a forest `F` and a gain function `γ`, some switching function
makes `γ^η` the identity on every edge of `F`. -/
@[to_additive exists_addSwitch_eq_zero_of_isForest
  /-- **Proposition 2.1** (§2.3) for an additive group. -/]
theorem exists_switch_eq_one_of_isForest [Group Γ] {F : Set E} (hF : G.IsForest F) (γ : E → Γ) :
    ∃ η : V → Γ, ∀ e ∈ F, switch G η γ e = 1 := by
  classical
  let s := G.reachableSetoid F
  let r : V → V := fun v => (Quotient.mk s v).out
  have hr : ∀ v, ∃ p : G.Walk (r v) v, ∀ e ∈ p.edges, e ∈ F := fun v => Quotient.mk_out (s := s) v
  choose P hP using hr
  refine ⟨fun v => ((P v).gain γ)⁻¹, fun f hf => ?_⟩
  have hj : G.Joins f (G.tail f) (G.head f) := Or.inl ⟨rfl, rfl⟩
  have hrr : r (G.tail f) = r (G.head f) :=
    congrArg Quotient.out (Quotient.sound (s := s) ((reachableIn_refl F _).step hf hj))
  have key : ∀ {a b : V} (_ : a = b) (A : G.Walk a (G.tail f)) (B : G.Walk b (G.head f)),
      (∀ e ∈ A.edges, e ∈ F) → (∀ e ∈ B.edges, e ∈ F) → A.gain γ * γ f = B.gain γ := by
    rintro a _ rfl A B hA hB
    have h₀ := gain_eq_of_isForest hF γ (A.append (.cons f hj (.nil _))) B (fun e he => ?_) hB
    · rw [Walk.gain_append, Walk.gain_cons, dart_of_tail γ rfl] at h₀
      simpa using h₀
    · rw [Walk.edges_append] at he
      simp only [Walk.edges, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at he
      rcases he with he | rfl
      · exact hA e he
      · exact hf
  rw [switch_apply, inv_inv, key hrr (P _) (P _) (hP _) (hP _), mul_inv_cancel]

end OrientedGraph

end Transformer.GainMatroid
