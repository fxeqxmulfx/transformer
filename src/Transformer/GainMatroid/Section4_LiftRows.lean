module

public import Transformer.GainMatroid.Section4_Relations

/-!
# The rows of a projective equivalence, lift case

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4. Let `T A_L(G,φ) S = B` with `B` a canonical lift
matrix. The row of a vertex `w` of `T` has an entry `c` in the column of the extra row and entries
`g(v)` in the vertex columns. It satisfies `c φ(e) + g(tail e) - g(head e) = 0` on every edge away
from `w`, and these relations add up along a walk (`Walk.add_relation`). A closed walk of nonzero
gain avoiding `w` forces `c = 0`, and then `g` is constant on `G - w`. The edges at `w` then carry
one scale, and connectivity makes `S` a scalar.

The source runs this computation on each small graph in Lemmas 4.4 and 4.10.
`switchScaleEquiv_of_projEquiv` is the forward direction of Theorem 5.1 (ii) for any loopless graph
that stays connected off each vertex and has no balancing vertex.
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

variable {V E F : Type*} [DecidableEq V] [Field F]

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- The lift row relation of an edge, read from its end `u`. -/
theorem add_relation_dart (φ : E → F) (c : F) (g : V → F) {e : E} {u v : V} (hj : G.Joins e u v)
    (h : c * φ e + g (G.tail e) - g (G.head e) = 0) : c * G.addDart φ e u + g u - g v = 0 := by
  by_cases ht : G.tail e = u
  · have hh : G.head e = v := by
      rcases hj with ⟨-, h₂⟩ | ⟨h₁, h₂⟩
      · exact h₂
      · exact h₂.trans (ht.symm.trans h₁)
    rw [addDart_of_tail φ ht, ← ht, ← hh]
    exact h
  · obtain ⟨h₁, h₂⟩ : G.tail e = v ∧ G.head e = u := hj.resolve_left fun hc => ht hc.1
    rw [addDart_of_ne φ ht, ← h₁, ← h₂]
    linear_combination -h

/-- The lift row relations along a walk combine into its additive gain. -/
theorem Walk.add_relation (φ : E → F) (c : F) (g : V → F) {u v : V} (p : G.Walk u v)
    (h : ∀ e ∈ p.edges, c * φ e + g (G.tail e) - g (G.head e) = 0) :
    c * p.addGain φ + g u - g v = 0 := by
  induction p with
  | nil => simp
  | @cons a b d e hj p ih =>
    have h₁ := add_relation_dart φ c g hj (h e List.mem_cons_self)
    have h₂ := ih fun f hf => h f (List.mem_cons_of_mem _ hf)
    rw [Walk.addGain_cons]
    linear_combination h₁ + h₂

end OrientedGraph

open OrientedGraph

variable [Fintype V]

/-- An entry of `T` times a canonical lift matrix. -/
theorem mul_liftMatrix_apply {R : Type*} (G : OrientedGraph V E) (T : Matrix R (Option V) F)
    (ψ : E → F) (r : R) (e : E) :
    (T * liftMatrix G ψ) r e =
      T r none * ψ e + T r (some (G.tail e)) - T r (some (G.head e)) := by
  rw [Matrix.mul_apply, Fintype.sum_option, liftMatrix_none,
    sum_mul_liftMatrix_some G (fun w => T r (some w)) ψ e]
  ring

variable [Fintype E] [DecidableEq E]

/-- **Theorem 5.1 (ii), forward direction, for the lift rows.** If a loopless graph stays connected
off each vertex and `φ` has no balancing vertex, then `ψ` with a projectively equivalent canonical
lift matrix is switching-and-scaling equivalent to `φ`. Unlike the source, `ψ` need not realize the
same biased graph, and no minor of the graph is used. -/
theorem switchScaleEquiv_of_projEquiv {G : OrientedGraph V E} (hG : G.Loopless)
    (hc : G.ConnectedOff) {φ ψ : E → F} (hb : G.HasNoAddBalancingVertex φ)
    (hP : ProjEquiv (liftMatrix G φ) (liftMatrix G ψ)) : SwitchScaleEquiv G φ ψ := by
  obtain ⟨T, s, hTu, hT⟩ := (projEquiv_iff_exists_mul_mul_diagonal _ _).1 hP
  have hentry : ∀ r e, (T r none * φ e + T r (some (G.tail e)) - T r (some (G.head e))) * s e =
      liftMatrix G ψ r e := by
    intro r e
    rw [← hT, mul_diagonal, mul_liftMatrix_apply]
  have hoff : ∀ w e, G.tail e ≠ w → G.head e ≠ w →
      T (some w) none * φ e + T (some w) (some (G.tail e)) - T (some w) (some (G.head e)) = 0 :=
    fun w e h₁ h₂ => (mul_eq_zero.1 ((hentry (some w) e).trans
      (liftMatrix_of_ne G ψ h₁.symm h₂.symm))).resolve_right (s e).ne_zero
  have hcol : ∀ w, T (some w) none = 0 := fun w => by
    obtain ⟨u, W, hW, hg⟩ := hb w
    have h := W.add_relation φ (T (some w) none) (fun x => T (some w) (some x)) fun e he => by
      obtain ⟨h₁, h₂⟩ := W.ends_mem_support he
      exact hoff w e (fun hc => hW (hc ▸ h₁)) (fun hc => hW (hc ▸ h₂))
    have h' : T (some w) none * W.addGain φ = 0 := by linear_combination h
    exact (mul_eq_zero.1 h').resolve_right hg
  have hconst : ∀ w x y, x ∈ G.verts Set.univ → y ∈ G.verts Set.univ → x ≠ w → y ≠ w →
      T (some w) (some x) = T (some w) (some y) := fun w x y hx hy hxw hyw => by
    obtain ⟨p, hp⟩ := hc w x y hx hy hxw hyw
    refine p.eq_of_forall_mem_edges (fun v => T (some w) (some v)) fun e he => ?_
    obtain ⟨h₁, h₂⟩ := p.ends_mem_support he
    have h := hoff w e (fun hc => hp (hc ▸ h₁)) (fun hc => hp (hc ▸ h₂))
    rw [hcol w, zero_mul, zero_add] at h
    exact sub_eq_zero.1 h
  have hjoin : ∀ w y e, G.Joins e w y →
      (T (some w) (some w) - T (some w) (some y)) * s e = 1 := by
    rintro w y e (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
    · have h := hentry (some (G.tail e)) e
      rwa [hcol, zero_mul, zero_add, liftMatrix_tail G ψ (hG e)] at h
    · have h := hentry (some (G.head e)) e
      rw [hcol, zero_mul, zero_add, liftMatrix_head G ψ (hG e)] at h
      linear_combination -h
  have hshare : ∀ w y y' e f, G.Joins e w y → G.Joins f w y' → (s e : F) = s f := by
    intro w y y' e f he hf
    have hy := hconst w y y' he.right_mem_verts hf.right_mem_verts
      (Joins.ne_of_loopless hG he).symm (Joins.ne_of_loopless hG hf).symm
    have h₁ := hjoin w y e he
    have h₂ := hjoin w y' f hf
    rw [← hy] at h₂
    have hne : T (some w) (some w) - T (some w) (some y) ≠ 0 := fun h0 => by
      simp [h0] at h₁
    exact mul_left_cancel₀ hne (h₁.trans h₂.symm)
  have hwalk : ∀ {a b : V} (p : G.Walk a b) (e f : E) (x y : V), G.Joins e a x → G.Joins f b y →
      (s e : F) = s f := by
    intro a b p
    induction p with
    | nil => exact fun e f x y he hf => hshare _ _ _ _ _ he hf
    | @cons a v b e' hj p ih =>
      exact fun e f x y he hf => (hshare _ _ _ _ _ he hj).trans (ih e' f a y hj.symm hf)
  have hsconst : ∀ e f, (s e : F) = s f := fun e f => by
    obtain ⟨p⟩ := hc.nonempty_walk hG hb.exists_edge_off (tail_mem_verts e) (tail_mem_verts f)
    exact hwalk p e f _ _ (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
  have hnone : T none none ≠ 0 := fun h0 => by
    have hdet : T.det = 0 := Matrix.det_eq_zero_of_column_eq_zero none fun i => by
      cases i with
      | none => exact h0
      | some w => exact hcol w
    exact ((Matrix.isUnit_iff_isUnit_det T).1 hTu).ne_zero hdet
  rcases isEmpty_or_nonempty E with hE | ⟨⟨e₀⟩⟩
  · exact ⟨0, 1, funext fun e => isEmptyElim e⟩
  refine ⟨fun v => -T none (some v) / T none none,
    Units.mk0 (T none none * s e₀) (mul_ne_zero hnone (s e₀).ne_zero), funext fun e => ?_⟩
  have h := hentry none e
  rw [liftMatrix_none, hsconst e e₀] at h
  simp only [Pi.smul_apply, smul_eq_mul, Units.val_mk0, addSwitch_apply]
  rw [← h]
  field_simp
  ring

/-- The frame half of Theorem 5.1 (i) for any loopless graph that stays connected off each vertex
and has no balancing vertex. -/
theorem frame_projEquiv_iff {G : OrientedGraph V E} (hG : G.Loopless) (hc : G.ConnectedOff)
    {φ ψ : E → Fˣ} (hb : G.HasNoBalancingVertex φ) :
    ProjEquiv (frameMatrix G φ) (frameMatrix G ψ) ↔ SwitchEquiv G φ ψ :=
  ⟨switchEquiv_of_projEquiv hG hc hb, projEquiv_frameMatrix_of_switchEquiv G⟩

/-- The lift half of Theorem 5.1 (ii) for any loopless graph that stays connected off each vertex
and has no balancing vertex. -/
theorem lift_projEquiv_iff {G : OrientedGraph V E} (hG : G.Loopless) (hc : G.ConnectedOff)
    {φ ψ : E → F} (hb : G.HasNoAddBalancingVertex φ) :
    ProjEquiv (liftMatrix G φ) (liftMatrix G ψ) ↔ SwitchScaleEquiv G φ ψ :=
  ⟨switchScaleEquiv_of_projEquiv hG hc hb, projEquiv_liftMatrix_of_switchScaleEquiv G⟩

end Transformer.GainMatroid
