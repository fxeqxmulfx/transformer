module

public import Mathlib.Data.List.Nodup
public import Mathlib.Data.Set.Basic
public import Transformer.GainMatroid.Section2_Gains

/-!
# Walks and paths

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1 and §2.3. A walk is the source's sequence
`(v₁, e₁, v₂, …, vₙ, eₙ, vₙ₊₁)` in which `vᵢ, vᵢ₊₁` are the ends of `eᵢ`; a path repeats no vertex.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE}

namespace OrientedGraph

/-- `e` has ends `u` and `v`, in either order. -/
def Joins (G : OrientedGraph V E) (e : E) (u v : V) : Prop :=
  (G.tail e = u ∧ G.head e = v) ∨ (G.tail e = v ∧ G.head e = u)

theorem Joins.symm {G : OrientedGraph V E} {e : E} {u v : V} (h : G.Joins e u v) :
    G.Joins e v u :=
  Or.symm h

/-- A walk `(v₁, e₁, v₂, …, eₙ, vₙ₊₁)` from `v₁` to `vₙ₊₁` (§2.3). -/
inductive Walk (G : OrientedGraph V E) : V → V → Type (max uV uE)
  | nil (u : V) : Walk G u u
  | cons {u v w : V} (e : E) (h : G.Joins e u v) (p : Walk G v w) : Walk G u w

namespace Walk

variable {G : OrientedGraph V E}

/-- The edges of a walk, in order. -/
def edges : ∀ {u v : V}, G.Walk u v → List E
  | _, _, nil _ => []
  | _, _, cons e _ p => e :: p.edges

/-- The vertices of a walk, in order. -/
def support : ∀ {u v : V}, G.Walk u v → List V
  | _, _, nil u => [u]
  | u, _, cons _ _ p => u :: p.support

/-- The set of edges of a walk. -/
def edgeSet {u v : V} (p : G.Walk u v) : Set E := {e | e ∈ p.edges}

/-- Concatenation of walks. -/
def append : ∀ {u v w : V}, G.Walk u v → G.Walk v w → G.Walk u w
  | _, _, _, nil _, q => q
  | _, _, _, cons e h p, q => cons e h (p.append q)

/-- The same walk traversed backwards. -/
def reverse : ∀ {u v : V}, G.Walk u v → G.Walk v u
  | _, _, nil u => nil u
  | _, _, cons e h p => p.reverse.append (cons e h.symm (nil _))

/-- A path: a walk that repeats no vertex. -/
def IsPath {u v : V} (p : G.Walk u v) : Prop := p.support.Nodup

theorem edges_append {u v w : V} (p : G.Walk u v) (q : G.Walk v w) :
    (p.append q).edges = p.edges ++ q.edges := by
  induction p with
  | nil => rfl
  | cons e h p ih => simp [append, edges, ih]

theorem support_eq_cons {u v : V} (p : G.Walk u v) : p.support = u :: p.support.tail := by
  cases p <;> rfl

theorem support_append {u v w : V} (p : G.Walk u v) (q : G.Walk v w) :
    (p.append q).support = p.support ++ q.support.tail := by
  induction p with
  | nil => exact q.support_eq_cons
  | cons e h p ih => simp [append, support, ih]

theorem edges_reverse {u v : V} (p : G.Walk u v) : p.reverse.edges = p.edges.reverse := by
  induction p with
  | nil => rfl
  | cons e h p ih => simp [reverse, edges, edges_append, ih]

theorem support_reverse {u v : V} (p : G.Walk u v) : p.reverse.support = p.support.reverse := by
  induction p with
  | nil => rfl
  | cons e h p ih => simp [reverse, support, support_append, ih]

theorem start_mem_support {u v : V} (p : G.Walk u v) : u ∈ p.support := by
  rw [support_eq_cons]
  exact List.mem_cons_self

theorem end_mem_support {u v : V} (p : G.Walk u v) : v ∈ p.support := by
  have h := p.reverse.start_mem_support
  rwa [support_reverse, List.mem_reverse] at h

/-- Both ends of an edge of a walk lie on the walk. -/
theorem ends_mem_support {u v : V} (p : G.Walk u v) {e : E} (he : e ∈ p.edges) :
    G.tail e ∈ p.support ∧ G.head e ∈ p.support := by
  induction p with
  | nil => simp [edges] at he
  | @cons a b c f h p ih =>
    simp only [edges, List.mem_cons] at he
    rcases he with rfl | he
    · have hb : b ∈ (cons e h p).support := List.mem_cons_of_mem _ p.start_mem_support
      have ha : a ∈ (cons e h p).support := List.mem_cons_self
      rcases h with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
      · exact ⟨h₁ ▸ ha, h₂ ▸ hb⟩
      · exact ⟨h₁ ▸ hb, h₂ ▸ ha⟩
    · obtain ⟨h₁, h₂⟩ := ih he
      exact ⟨List.mem_cons_of_mem _ h₁, List.mem_cons_of_mem _ h₂⟩

theorem IsPath.reverse {u v : V} {p : G.Walk u v} (hp : p.IsPath) : p.reverse.IsPath := by
  unfold IsPath at hp ⊢
  rw [support_reverse]
  exact List.nodup_reverse.2 hp

/-- A path repeats no edge. -/
theorem IsPath.edges_nodup {u v : V} {p : G.Walk u v} (hp : p.IsPath) : p.edges.Nodup := by
  induction p with
  | nil => simp [edges]
  | @cons a b c e h p ih =>
    have hcons : a ∉ p.support ∧ p.support.Nodup := by
      simpa [IsPath, support] using hp
    refine List.nodup_cons.2 ⟨fun he => ?_, ih hcons.2⟩
    obtain ⟨h₁, h₂⟩ := p.ends_mem_support he
    rcases h with ⟨h₃, -⟩ | ⟨-, h₃⟩
    · exact hcons.1 (h₃ ▸ h₁)
    · exact hcons.1 (h₃ ▸ h₂)

end Walk

/-- One link from `0` to `1` joins `1` and `0`. -/
example : linkGraph.Joins () 1 0 := (Or.inl ⟨rfl, rfl⟩ : linkGraph.Joins () 0 1).symm

/-- The one-edge walk along the link of `linkGraph`. -/
def linkWalk : linkGraph.Walk 0 1 := .cons () (Or.inl ⟨rfl, rfl⟩) (.nil 1)

theorem linkWalk_isPath : linkWalk.IsPath := by
  simp [Walk.IsPath, linkWalk, Walk.support, linkGraph]

example : linkGraph.tail () ∈ linkWalk.support ∧ linkGraph.head () ∈ linkWalk.support :=
  Walk.ends_mem_support _ (by simp [linkWalk, Walk.edges])

example : linkWalk.reverse.IsPath := linkWalk_isPath.reverse

example : linkWalk.edges.Nodup := linkWalk_isPath.edges_nodup

end OrientedGraph

end Transformer.GainMatroid
