module

public import Transformer.GainMatroid.Section4_Graphs
public import Transformer.GainMatroid.Section4_Switching

/-!
# The rows of a projective equivalence, frame case

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4. Let `T A_F(G,φ) S = B`. In the row of a vertex
`w`, the matrix `B` vanishes on every edge away from `w`, so that row `t` of `T` satisfies
`t(tail e) = t(head e) φ(e)` on these edges. Along a walk these relations combine into the gain of
the walk (`Walk.eq_mul_gain`). A closed walk of non-identity gain avoiding `w` therefore forces `t`
to vanish on its component of `G - w` (`eq_zero_of_isUnbalancedOff`).

The source runs this computation on each small graph in Lemmas 4.3, 4.5, 4.9. Here it is one
statement. `switchEquiv_of_projEquiv` is the forward direction of Theorem 5.1 (i) and
`not_projEquiv_frame_lift` is Theorem 5.1 (iii), both for any loopless graph that stays connected
off each vertex and has no balancing vertex. In the second, the lift gain function is arbitrary.
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

variable {V E F : Type*} [DecidableEq V] [Field F]

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- The frame row relation of an edge, read from its end `u`. -/
theorem eq_mul_dart (φ : E → Fˣ) (t : V → F) {e : E} {u v : V} (hj : G.Joins e u v)
    (h : t (G.tail e) = t (G.head e) * φ e) : t u = t v * (G.dart φ e u : Fˣ) := by
  by_cases ht : G.tail e = u
  · have hh : G.head e = v := by
      rcases hj with ⟨-, h₂⟩ | ⟨h₁, h₂⟩
      · exact h₂
      · exact h₂.trans (ht.symm.trans h₁)
    rw [dart_of_tail φ ht, ← ht, ← hh]
    exact h
  · obtain ⟨h₁, h₂⟩ : G.tail e = v ∧ G.head e = u := hj.resolve_left fun hc => ht hc.1
    rw [dart_of_ne φ ht, ← h₁, ← h₂, h, Units.val_inv_eq_inv_val, mul_assoc,
      mul_inv_cancel₀ (φ e).ne_zero, mul_one]

/-- The frame row relations along a walk combine into its gain. -/
theorem Walk.eq_mul_gain (φ : E → Fˣ) (t : V → F) {u v : V} (p : G.Walk u v)
    (h : ∀ e ∈ p.edges, t (G.tail e) = t (G.head e) * φ e) : t u = t v * (p.gain φ : Fˣ) := by
  induction p with
  | nil => simp
  | @cons a b c e hj p ih =>
    have h₁ := eq_mul_dart φ t hj (h e List.mem_cons_self)
    have h₂ := ih fun f hf => h f (List.mem_cons_of_mem _ hf)
    rw [Walk.gain_cons, Units.val_mul, h₁, h₂]
    ring

/-- The frame row relations around a closed walk of non-identity gain force `t = 0` at its base. -/
theorem Walk.eq_zero_of_gain_ne_one (φ : E → Fˣ) (t : V → F) {u : V} (W : G.Walk u u)
    (h : ∀ e ∈ W.edges, t (G.tail e) = t (G.head e) * φ e) (hg : W.gain φ ≠ 1) : t u = 0 := by
  by_contra h0
  refine hg (Units.ext ?_)
  have hrel := W.eq_mul_gain φ t h
  exact mul_left_cancel₀ h0 (by rw [Units.val_one, mul_one]; exact hrel.symm)

/-- A function obeying the frame row relation on all edges off `w` vanishes wherever an unbalanced
closed walk reaches off `w`. -/
theorem eq_zero_of_isUnbalancedOff {φ : E → Fˣ} {t : V → F} {w x : V}
    (hrel : ∀ e, G.tail e ≠ w → G.head e ≠ w → t (G.tail e) = t (G.head e) * φ e)
    (hx : G.IsUnbalancedOff φ w x) : t x = 0 := by
  obtain ⟨u, W, p, hW, hp, hg⟩ := hx
  have hoff : ∀ {a b : V} (q : G.Walk a b), w ∉ q.support → ∀ e ∈ q.edges,
      t (G.tail e) = t (G.head e) * φ e := fun q hq e he => by
    obtain ⟨h₁, h₂⟩ := q.ends_mem_support he
    exact hrel e (fun hc => hq (hc ▸ h₁)) (fun hc => hq (hc ▸ h₂))
  have hu : t u = 0 := W.eq_zero_of_gain_ne_one φ t (hoff W hW) hg
  have hpx := p.eq_mul_gain φ t (hoff p hp)
  rw [hu] at hpx
  exact (mul_eq_zero.1 hpx.symm).resolve_right (Units.ne_zero _)

end OrientedGraph

open OrientedGraph

variable [Fintype V]

/-- Row `τ` applied to a column of the canonical frame matrix. -/
theorem sum_mul_frameMatrix (G : OrientedGraph V E) (τ : V → F) (φ : E → Fˣ) (e : E) :
    ∑ w, τ w * frameMatrix G φ w e = τ (G.tail e) - τ (G.head e) * φ e := by
  simp [frameMatrix_apply, mul_sub, Finset.sum_sub_distrib]

theorem mul_frameMatrix_apply {R : Type*} (G : OrientedGraph V E) (T : Matrix R V F) (φ : E → Fˣ)
    (r : R) (e : E) : (T * frameMatrix G φ) r e = T r (G.tail e) - T r (G.head e) * φ e := by
  rw [Matrix.mul_apply]
  exact sum_mul_frameMatrix G (T r) φ e

/-- **Theorem 5.1 (i), forward direction, for the frame rows.** If a loopless graph stays connected
off each vertex and `φ` has no balancing vertex, then `ψ` with a projectively equivalent canonical
frame matrix is switching equivalent to `φ`. Unlike the source, `ψ` need not realize the same biased
graph, and no minor of the graph is used. -/
theorem switchEquiv_of_projEquiv [Fintype E] [DecidableEq E] {G : OrientedGraph V E}
    (hG : G.Loopless) (hc : G.ConnectedOff) {φ ψ : E → Fˣ} (hb : G.HasNoBalancingVertex φ)
    (hP : ProjEquiv (frameMatrix G φ) (frameMatrix G ψ)) : SwitchEquiv G φ ψ := by
  classical
  obtain ⟨T, s, -, hT⟩ := (projEquiv_iff_exists_mul_mul_diagonal _ _).1 hP
  have hentry : ∀ r e, (T r (G.tail e) - T r (G.head e) * φ e) * s e = frameMatrix G ψ r e := by
    intro r e
    rw [← hT, mul_diagonal, mul_frameMatrix_apply]
  have hzero : ∀ w x, x ∈ G.verts Set.univ → x ≠ w → T w x = 0 := fun w x hx hxw =>
    eq_zero_of_isUnbalancedOff (t := T w)
      (fun e h₁ h₂ => sub_eq_zero.1 ((mul_eq_zero.1 ((hentry w e).trans
        (frameMatrix_of_ne G ψ h₁.symm h₂.symm))).resolve_right (s e).ne_zero))
      (hb.isUnbalancedOff hc hx hxw)
  have htail : ∀ e, T (G.tail e) (G.tail e) * s e = 1 := fun e => by
    have h := hentry (G.tail e) e
    rwa [frameMatrix_tail G ψ (hG e), hzero _ _ (head_mem_verts e) (hG e).symm, zero_mul,
      sub_zero] at h
  have hhead : ∀ e, T (G.head e) (G.head e) * φ e * s e = ψ e := fun e => by
    have h := hentry (G.head e) e
    rw [frameMatrix_head G ψ (hG e), hzero _ _ (tail_mem_verts e) (hG e), zero_sub] at h
    linear_combination -h
  let η : V → Fˣ := fun v => if h : T v v = 0 then 1 else Units.mk0 (T v v) h
  have hη : ∀ v, T v v ≠ 0 → (η v : F) = T v v := fun v hv => by simp [η, hv]
  refine ⟨η, funext fun e => Units.ext ?_⟩
  have ht0 : T (G.tail e) (G.tail e) ≠ 0 := fun h0 => by
    have := htail e
    simp [h0] at this
  have hh0 : T (G.head e) (G.head e) ≠ 0 := fun h0 => by
    have := hhead e
    rw [h0, zero_mul, zero_mul] at this
    exact (ψ e).ne_zero this.symm
  have hs : (s e : F) = (T (G.tail e) (G.tail e))⁻¹ := by
    field_simp
    linear_combination htail e
  rw [switch_apply, Units.val_mul, Units.val_mul, Units.val_inv_eq_inv_val, hη _ ht0, hη _ hh0,
    ← hhead e, hs]
  ring

/-- **Theorem 5.1 (iii) for the frame rows.** If a loopless graph stays connected off each vertex
and `φ` has no balancing vertex, then the canonical frame matrix of `φ` is projectively equivalent
to no canonical lift matrix. The lift gain function `ψ` is arbitrary. -/
theorem not_projEquiv_frame_lift [Nonempty V] {G : OrientedGraph V E} (hG : G.Loopless)
    (hc : G.ConnectedOff) {φ : E → Fˣ} (hb : G.HasNoBalancingVertex φ) (ψ : E → F) :
    ¬ProjEquiv (frameMatrix G φ) (liftMatrix G ψ) := by
  rintro ⟨s, hs⟩
  have hrow : ∀ w : V, ∃ t : V → F, ∀ e,
      (t (G.tail e) - t (G.head e) * φ e) * s e = liftMatrix G ψ (some w) e := by
    intro w
    have hmem : liftMatrix G ψ (some w) ∈ rowSpace (scaleCols s (frameMatrix G φ)) := by
      rw [hs]
      exact row_mem_rowSpace _ _
    obtain ⟨t, ht⟩ := (Submodule.mem_span_range_iff_exists_fun F).1 hmem
    refine ⟨t, fun e => ?_⟩
    rw [← ht, ← sum_mul_frameMatrix G t φ e, Finset.sum_mul, Finset.sum_apply]
    exact Finset.sum_congr rfl fun x _ => by
      simp [Matrix.row, scaleCols, mul_assoc]
  choose t ht using hrow
  have hzero : ∀ w x, x ∈ G.verts Set.univ → x ≠ w → t w x = 0 := fun w x hx hxw =>
    eq_zero_of_isUnbalancedOff (t := t w)
      (fun e h₁ h₂ => sub_eq_zero.1 ((mul_eq_zero.1 ((ht w e).trans
        (liftMatrix_of_ne G ψ h₁.symm h₂.symm))).resolve_right (s e).ne_zero))
      (hb.isUnbalancedOff hc hx hxw)
  have htail : ∀ e, t (G.tail e) (G.tail e) * s e = 1 := fun e => by
    have h := ht (G.tail e) e
    rwa [liftMatrix_tail G ψ (hG e), hzero _ _ (head_mem_verts e) (hG e).symm, zero_mul,
      sub_zero] at h
  have hhead : ∀ e, t (G.head e) (G.head e) * φ e * s e = 1 := fun e => by
    have h := ht (G.head e) e
    rw [liftMatrix_head G ψ (hG e), hzero _ _ (tail_mem_verts e) (hG e), zero_sub] at h
    linear_combination -h
  have hrel : ∀ e, t (G.tail e) (G.tail e) = t (G.head e) (G.head e) * φ e := fun e =>
    mul_right_cancel₀ (s e).ne_zero ((htail e).trans (hhead e).symm)
  obtain ⟨w⟩ := ‹Nonempty V›
  obtain ⟨u, W, -, hg⟩ := hb w
  have hu := W.eq_zero_of_gain_ne_one φ (fun v => t v v) (fun e _ => hrel e) hg
  obtain ⟨e, -, he | he⟩ := mem_verts.1 (W.start_mem_verts hg)
  · have := htail e
    rw [he] at this
    simp [show t u u = 0 from hu] at this
  · have := hhead e
    rw [he] at this
    simp [show t u u = 0 from hu] at this

end Transformer.GainMatroid
