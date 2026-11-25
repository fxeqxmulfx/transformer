module

public import Transformer.GainMatroid.Section2_FrameColumns

/-!
# Independent columns of the canonical lift matrix

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.1: the canonical lift matrix `A_L(G, ψ)` is the
oriented incidence matrix of `G` with one more row `v₀` holding the gains, so the column of a link
from `u` to `v` with gain `g` is `g` in row `v₀`, `1` in row `u` and `-1` in row `v`, and that of a
loop is `g` in row `v₀` (`OrientedGraph.liftMatrix_col`).

Let `T ⊆ I` be a forest linking the ends of every edge of `I`, with gain zero on `T`. The columns
of `I` are independent exactly when no edge of `I \ T` has gain zero and `I \ T` has at most one
edge (`OrientedGraph.linearIndepOn_liftMatrix_iff`). The row `v₀` isolates the coefficient of the
edge of `I \ T`, and row sums over one side of an edge of `T` isolate that of the edge
(`OrientedGraph.sum_proj_liftMatrix`); the columns of `T` span the differences of the unit vectors
of the rows of the vertices of a component (`OrientedGraph.single_sub_single_mem_span_liftMatrix`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E F : Type*} [Field F] [DecidableEq V] {G : OrientedGraph V E} {ψ : E → F}

/-- The column of an edge in the canonical lift matrix: its gain times the unit vector of the row
`v₀`, plus the difference of the unit vectors of the rows of its tail and head. -/
theorem liftMatrix_col (e : E) :
    (fun r => liftMatrix G ψ r e) =
      ψ e • Pi.single none 1 + (Pi.single (some (G.tail e)) 1 - Pi.single (some (G.head e)) 1) := by
  funext r
  cases r with
  | none => simp [liftMatrix_none]
  | some w => simp [liftMatrix_some, Pi.single_apply]

/-- The sum over the rows of the vertices in `X` of the column of `e`. -/
theorem sum_proj_liftMatrix (X : Finset V) (e : E) :
    (∑ w ∈ X, LinearMap.proj (some w) : (Option V → F) →ₗ[F] F) (fun r => liftMatrix G ψ r e) =
      (if G.tail e ∈ X then 1 else 0) - if G.head e ∈ X then 1 else 0 := by
  rw [LinearMap.sum_apply]
  simp only [LinearMap.proj_apply, liftMatrix_some, Finset.sum_sub_distrib, Finset.sum_ite_eq']

/-- The row sum over `X` of an edge vanishes when `X` holds both of its ends or neither. -/
theorem sum_proj_liftMatrix_eq_zero {X : Finset V} {e : E} (hX : G.tail e ∈ X ↔ G.head e ∈ X) :
    (∑ w ∈ X, LinearMap.proj (some w) : (Option V → F) →ₗ[F] F) (fun r => liftMatrix G ψ r e) =
      0 := by
  rw [sum_proj_liftMatrix]
  by_cases ht : G.tail e ∈ X
  · rw [ite_eq_left ht, ite_eq_left (hX.1 ht), sub_self]
  · rw [ite_eq_right ht, ite_eq_right (mt hX.2 ht), sub_self]

/-- With gain zero on `T`, the columns of `T` span the difference of the unit vectors of the rows
of two vertices linked along `T`. -/
theorem single_sub_single_mem_span_liftMatrix {T : Set E} (hψ : ∀ e ∈ T, ψ e = 0) {u v : V}
    (h : G.ReachableIn T u v) :
    Pi.single (some u) (1 : F) - Pi.single (some v) 1 ∈
      Submodule.span F ((fun e r => liftMatrix G ψ r e) '' T) := by
  refine ReachableIn.rel (r := fun a b => Pi.single (some a) (1 : F) - Pi.single (some b) 1 ∈
      Submodule.span F ((fun e r => liftMatrix G ψ r e) '' T))
    (fun a => by rw [sub_self]; exact zero_mem _)
    (fun a b c h₁ h₂ => by simpa only [sub_add_sub_cancel] using add_mem h₁ h₂) (fun e he => ?_) h
  have hmem : (fun r => liftMatrix G ψ r e) ∈
      Submodule.span F ((fun e r => liftMatrix G ψ r e) '' T) :=
    Submodule.subset_span (Set.mem_image_of_mem _ he)
  rw [liftMatrix_col, hψ e he, zero_smul, zero_add] at hmem
  exact ⟨hmem, by simpa only [neg_sub] using neg_mem hmem⟩

variable [Finite V]

/-- **Independent columns of the canonical lift matrix.** Let `T ⊆ I` be a forest linking the ends
of every edge of `I`, with gain zero on `T`. The columns of `I` are linearly independent exactly
when no edge of `I \ T` has gain zero and `I \ T` has at most one edge. -/
theorem linearIndepOn_liftMatrix_iff {I T : Set E} (hTI : T ⊆ I) (hT : G.IsForest T)
    (hr : ∀ e ∈ I, G.ReachableIn T (G.tail e) (G.head e)) (hψ : ∀ e ∈ T, ψ e = 0) :
    LinearIndepOn F (fun e r => liftMatrix G ψ r e) I ↔
      (∀ e ∈ I \ T, ψ e ≠ 0) ∧ ∀ e ∈ I \ T, ∀ f ∈ I \ T, e = f := by
  constructor
  · intro hind
    have hmem := linearIndepOn_iff_notMem_span.1 hind
    have hTsub : ∀ e ∈ I \ T, T ⊆ I \ {e} := fun e he f hf =>
      ⟨hTI hf, fun hfe => he.2 ((Set.mem_singleton_iff.1 hfe) ▸ hf)⟩
    have hd : ∀ e ∈ I \ T, ∀ {x y : V}, G.ReachableIn T x y →
        Pi.single (some x) (1 : F) - Pi.single (some y) 1 ∈
          Submodule.span F ((fun e r => liftMatrix G ψ r e) '' (I \ {e})) := fun e he _ _ h =>
      Submodule.span_mono (Set.image_mono (hTsub e he))
        (single_sub_single_mem_span_liftMatrix hψ h)
    have h₁ : ∀ e ∈ I \ T, ψ e ≠ 0 := by
      intro e he he0
      apply hmem e he.1
      rw [liftMatrix_col, he0, zero_smul, zero_add]
      exact hd e he (hr e he.1)
    refine ⟨h₁, fun e he f hf => by_contra fun hfe => hmem e he.1 ?_⟩
    have hfW : (fun r => liftMatrix G ψ r f) ∈
        Submodule.span F ((fun e r => liftMatrix G ψ r e) '' (I \ {e})) :=
      Submodule.subset_span ⟨f, ⟨hf.1, fun h => hfe (Set.mem_singleton_iff.1 h).symm⟩, rfl⟩
    have hnone := sub_mem hfW (hd e he (hr f hf.1))
    rw [liftMatrix_col, add_sub_cancel_right] at hnone
    rw [liftMatrix_col]
    exact add_mem (Submodule.smul_mem _ _ ((Submodule.smul_mem_iff _ (h₁ f hf)).1 hnone))
      (hd e he (hr e he.1))
  · rintro ⟨h₁, h₂⟩
    rw [linearIndepOn_iff]
    intro l hlI hl
    have hsupp : ∀ e ∈ l.support, e ∈ I := fun e he => hlI (Finset.mem_coe.2 he)
    have hA : ∀ e ∈ I \ T, l e = 0 := fun e he =>
      coeff_eq_zero_of_apply hl (LinearMap.proj none) (h₁ e he) fun f hf hfe =>
        hψ f (by_contra fun hfT => hfe (h₂ f ⟨hsupp f hf, hfT⟩ e he))
    have hB : ∀ f ∈ T, l f = 0 := by
      intro f hf
      have hF : G.IsForest (insert f (T \ {f})) := by
        rwa [Set.insert_sdiff_singleton, Set.insert_eq_of_mem hf]
      refine coeff_eq_zero_of_apply hl
        (∑ w ∈ G.reachFinset (T \ {f}) (G.tail f), LinearMap.proj (some w)) ?_
        fun g hg hgf => ?_
      · rw [sum_proj_liftMatrix, ite_eq_left (mem_reachFinset.2 (reachableIn_refl _ _)),
          ite_eq_right fun h => hF.not_reachableIn (fun h => h.2 rfl) (mem_reachFinset.1 h).symm,
          sub_zero]
        exact one_ne_zero
      have hgT : g ∈ T := by_contra fun hgT =>
        Finsupp.mem_support_iff.1 hg (hA g ⟨hsupp g hg, hgT⟩)
      exact sum_proj_liftMatrix_eq_zero (tail_mem_reachFinset_iff ⟨hgT, hgf⟩)
    ext e
    rw [Finsupp.coe_zero, Pi.zero_apply]
    by_contra hle
    have he := hsupp e (Finsupp.mem_support_iff.2 hle)
    exact hle ((em (e ∈ T)).elim (hB e) fun heT => hA e ⟨he, heT⟩)

end OrientedGraph

open OrientedGraph

example := single_sub_single_mem_span_liftMatrix (G := pumpkin) (ψ := (0 : Fin 3 → ℚ))
  (T := {0}) (fun _ _ => rfl) (reachableIn_of_mem (Set.mem_singleton 0))

example := sum_proj_liftMatrix_eq_zero (G := pumpkin) (ψ := (0 : Fin 3 → ℚ))
  (X := {0, 1}) (e := 2) (by simp [pumpkin])

/-- Additive gain zero on the edge `0` of `pumpkin` and one on the others. -/
def pumpkinLiftGain (e : Fin 3) : ℚ := if e = 0 then 0 else 1

/-- In `pumpkin` with gain zero on `0` and one elsewhere, the columns of `0` and `1` are
independent, but those of all three edges are not. -/
example : LinearIndepOn ℚ (fun e r => liftMatrix pumpkin pumpkinLiftGain r e) {0, 1} ∧
    ¬LinearIndepOn ℚ (fun e r => liftMatrix pumpkin pumpkinLiftGain r e) Set.univ := by
  have hr (e : Fin 3) : pumpkin.ReachableIn {0} (pumpkin.tail e) (pumpkin.head e) :=
    show pumpkin.ReachableIn {0} 0 1 from reachableIn_of_mem (Set.mem_singleton 0)
  have hψ : ∀ e ∈ ({0} : Set (Fin 3)), pumpkinLiftGain e = 0 := fun e he => by
    simp [pumpkinLiftGain, Set.mem_singleton_iff.1 he]
  have h01 : ∀ e ∈ ({0, 1} : Set (Fin 3)) \ {0}, e = 1 := by
    rintro e ⟨rfl | rfl, h⟩
    · exact absurd rfl h
    · rfl
  refine ⟨(linearIndepOn_liftMatrix_iff (by simp) pumpkin_isForest (fun e _ => hr e) hψ).2
    ⟨fun e he => by simp [h01 e he, pumpkinLiftGain], fun e he f hf =>
      (h01 e he).trans (h01 f hf).symm⟩, fun h => ?_⟩
  have := ((linearIndepOn_liftMatrix_iff (Set.subset_univ _) pumpkin_isForest (fun e _ => hr e)
    hψ).1 h).2 1 ⟨Set.mem_univ _, by simp⟩ 2 ⟨Set.mem_univ _, by simp⟩
  exact absurd this (by decide)

end Transformer.GainMatroid
