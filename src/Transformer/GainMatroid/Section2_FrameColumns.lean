module

public import Transformer.GainMatroid.Section1_Canonical
public import Transformer.GainMatroid.Section2_SpanningForests

/-!
# Independent columns of the canonical frame matrix

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.1: in `A_F(G, φ)` a link from `u` to `v` with
gain `g` has `1` in row `u` and `-g` in row `v`, a loop with gain `g` has `1 - g` in its row.

Let `T ⊆ I` be a forest linking the ends of every edge of `I`, with gain one on `T`. The columns of
`I` are independent exactly when no edge of `I \ T` has gain one and no two edges of `I \ T` meet
one component of `T` (`OrientedGraph.linearIndepOn_frameMatrix_iff`). Row sums over a component of
`T`, or over one side of an edge of `T`, isolate one coefficient
(`OrientedGraph.coeff_eq_zero_of_apply`); the columns of `T` span the differences of the unit
vectors of the vertices of a component (`OrientedGraph.single_sub_single_mem_span_frameMatrix`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E F : Type*} [Field F]

/-- A functional vanishing on the vectors of a vanishing combination except `v e₀`, and not on
`v e₀`, shows that the coefficient of `e₀` vanishes. -/
theorem coeff_eq_zero_of_apply {M : Type*} [AddCommGroup M] [Module F M] {v : E → M}
    {l : E →₀ F} (hl : Finsupp.linearCombination F v l = 0) (f : M →ₗ[F] F) {e₀ : E}
    (h₀ : f (v e₀) ≠ 0) (h : ∀ e ∈ l.support, e ≠ e₀ → f (v e) = 0) : l e₀ = 0 := by
  have h' := congrArg f hl
  rw [Finsupp.apply_linearCombination, map_zero, Finsupp.linearCombination_apply, Finsupp.sum,
    Finset.sum_eq_single e₀ (fun e he hne => by rw [Function.comp_apply, h e he hne, smul_zero])
      (fun he => by rw [Finsupp.notMem_support_iff.1 he, zero_smul])] at h'
  exact (smul_eq_zero.1 h').resolve_right h₀

variable {G : OrientedGraph V E}

/-- The vertices reached from `u` along `S`, as a finite set. -/
noncomputable def reachFinset [Finite V] (G : OrientedGraph V E) (S : Set E) (u : V) : Finset V :=
  (Set.toFinite {w | G.ReachableIn S u w}).toFinset

theorem mem_reachFinset [Finite V] {S : Set E} {u w : V} :
    w ∈ G.reachFinset S u ↔ G.ReachableIn S u w :=
  Set.Finite.mem_toFinset _

/-- An edge of `S` has both ends or neither among the vertices reached from `u` along `S`. -/
theorem tail_mem_reachFinset_iff [Finite V] {S : Set E} {u : V} {e : E} (he : e ∈ S) :
    G.tail e ∈ G.reachFinset S u ↔ G.head e ∈ G.reachFinset S u := by
  rw [mem_reachFinset, mem_reachFinset]
  exact ⟨fun h => h.trans (reachableIn_of_mem he), fun h => h.trans (reachableIn_of_mem he).symm⟩

variable [DecidableEq V] {φ : E → Fˣ}

/-- The column of an edge in the canonical frame matrix: the unit vector of its tail less its gain
times the unit vector of its head. -/
theorem frameMatrix_col (e : E) :
    (fun r => frameMatrix G φ r e) =
      Pi.single (G.tail e) 1 - (φ e : F) • Pi.single (G.head e) 1 := by
  funext r
  simp [frameMatrix_apply, Pi.single_apply]

/-- The sum over the rows in `X` of the column of `e`. -/
theorem sum_proj_frameMatrix (X : Finset V) (e : E) :
    (∑ w ∈ X, LinearMap.proj w : (V → F) →ₗ[F] F) (fun r => frameMatrix G φ r e) =
      (if G.tail e ∈ X then 1 else 0) - (φ e : F) * if G.head e ∈ X then 1 else 0 := by
  rw [LinearMap.sum_apply]
  simp only [LinearMap.proj_apply, frameMatrix_apply, Finset.sum_sub_distrib, ← Finset.mul_sum,
    Finset.sum_ite_eq']

/-- With gain one on `T`, the columns of `T` span the difference of the unit vectors of two vertices
linked along `T`. -/
theorem single_sub_single_mem_span_frameMatrix {T : Set E} (hφ : ∀ e ∈ T, φ e = 1) {u v : V}
    (h : G.ReachableIn T u v) :
    Pi.single u (1 : F) - Pi.single v 1 ∈
      Submodule.span F ((fun e r => frameMatrix G φ r e) '' T) := by
  refine ReachableIn.rel (r := fun a b => Pi.single a (1 : F) - Pi.single b 1 ∈
      Submodule.span F ((fun e r => frameMatrix G φ r e) '' T))
    (fun a => by rw [sub_self]; exact zero_mem _)
    (fun a b c h₁ h₂ => by simpa only [sub_add_sub_cancel] using add_mem h₁ h₂) (fun e he => ?_) h
  have hmem : (fun r => frameMatrix G φ r e) ∈
      Submodule.span F ((fun e r => frameMatrix G φ r e) '' T) :=
    Submodule.subset_span (Set.mem_image_of_mem _ he)
  rw [frameMatrix_col, hφ e he, Units.val_one, one_smul] at hmem
  exact ⟨hmem, by simpa only [neg_sub] using neg_mem hmem⟩

/-- The terms of the row sum over `X` of an edge with gain one vanish when `X` holds both ends or
neither. -/
theorem sum_proj_frameMatrix_eq_zero {X : Finset V} {e : E} (he : φ e = 1)
    (hX : G.tail e ∈ X ↔ G.head e ∈ X) :
    (∑ w ∈ X, LinearMap.proj w : (V → F) →ₗ[F] F) (fun r => frameMatrix G φ r e) = 0 := by
  rw [sum_proj_frameMatrix, he, Units.val_one, one_mul]
  by_cases ht : G.tail e ∈ X
  · rw [ite_eq_left ht, ite_eq_left (hX.1 ht), sub_self]
  · rw [ite_eq_right ht, ite_eq_right (mt hX.2 ht), sub_self]

variable [Finite V]

/-- **Independent columns of the canonical frame matrix.** Let `T ⊆ I` be a forest linking the
ends of every edge of `I`, with gain one on `T`. The columns of `I` are linearly independent
exactly when no edge of `I \ T` has gain one and no two edges of `I \ T` meet one component of
`T`. -/
theorem linearIndepOn_frameMatrix_iff {I T : Set E} (hTI : T ⊆ I) (hT : G.IsForest T)
    (hr : ∀ e ∈ I, G.ReachableIn T (G.tail e) (G.head e)) (hφ : ∀ e ∈ T, φ e = 1) :
    LinearIndepOn F (fun e r => frameMatrix G φ r e) I ↔
      (∀ e ∈ I \ T, φ e ≠ 1) ∧
        ∀ e ∈ I \ T, ∀ f ∈ I \ T, G.ReachableIn T (G.tail e) (G.tail f) → e = f := by
  have hne : ∀ e, φ e ≠ 1 → (1 : F) - φ e ≠ 0 := fun e h =>
    sub_ne_zero.2 fun h' => h (Units.val_eq_one.1 h'.symm)
  constructor
  · intro hind
    have hmem := linearIndepOn_iff_notMem_span.1 hind
    have hTsub : ∀ e ∈ I \ T, T ⊆ I \ {e} := fun e he f hf =>
      ⟨hTI hf, fun hfe => he.2 ((Set.mem_singleton_iff.1 hfe) ▸ hf)⟩
    have hd : ∀ e ∈ I \ T, ∀ {x y : V}, G.ReachableIn T x y → Pi.single x (1 : F) - Pi.single y 1 ∈
        Submodule.span F ((fun e r => frameMatrix G φ r e) '' (I \ {e})) := fun e he _ _ h =>
      Submodule.span_mono (Set.image_mono (hTsub e he))
        (single_sub_single_mem_span_frameMatrix hφ h)
    have h₁ : ∀ e ∈ I \ T, φ e ≠ 1 := by
      intro e he he1
      apply hmem e he.1
      rw [frameMatrix_col, he1, Units.val_one, one_smul]
      exact hd e he (hr e he.1)
    refine ⟨h₁, fun e he f hf hef => by_contra fun hfe => hmem e he.1 ?_⟩
    have hfW : (fun r => frameMatrix G φ r f) ∈
        Submodule.span F ((fun e r => frameMatrix G φ r e) '' (I \ {e})) :=
      Submodule.subset_span ⟨f, ⟨hf.1, fun h => hfe (Set.mem_singleton_iff.1 h).symm⟩, rfl⟩
    have hhf := sub_mem hfW (hd e he (hr f hf.1))
    rw [frameMatrix_col, show ∀ a b : V → F, a - (φ f : F) • b - (a - b) = (1 - (φ f : F)) • b by
      intro a b; module] at hhf
    have hhe := add_mem (hd e he ((hr e he.1).symm.trans (hef.trans (hr f hf.1))))
      ((Submodule.smul_mem_iff _ (hne f (h₁ f hf))).1 hhf)
    rw [sub_add_cancel] at hhe
    have hte := add_mem (hd e he (hr e he.1)) hhe
    rw [sub_add_cancel] at hte
    rw [frameMatrix_col]
    exact sub_mem hte (Submodule.smul_mem _ _ hhe)
  · rintro ⟨h₁, h₂⟩
    rw [linearIndepOn_iff]
    intro l hlI hl
    have hsupp : ∀ e ∈ l.support, e ∈ I := fun e he => hlI (Finset.mem_coe.2 he)
    have hA : ∀ e ∈ I \ T, l e = 0 := by
      intro e he
      refine coeff_eq_zero_of_apply hl (∑ w ∈ G.reachFinset T (G.tail e), LinearMap.proj w) ?_
        fun f hf hfe => ?_
      · rw [sum_proj_frameMatrix, ite_eq_left (mem_reachFinset.2 (reachableIn_refl _ _)),
          ite_eq_left (mem_reachFinset.2 (hr e he.1)), mul_one]
        exact hne e (h₁ e he)
      by_cases hfT : f ∈ T
      · exact sum_proj_frameMatrix_eq_zero (hφ f hfT) (tail_mem_reachFinset_iff hfT)
      have hf' : f ∈ I \ T := ⟨hsupp f hf, hfT⟩
      have ht : G.tail f ∉ G.reachFinset T (G.tail e) := fun h =>
        hfe (h₂ e he f hf' (mem_reachFinset.1 h)).symm
      have hh : G.head f ∉ G.reachFinset T (G.tail e) := fun h =>
        ht (mem_reachFinset.2 ((mem_reachFinset.1 h).trans (hr f hf'.1).symm))
      rw [sum_proj_frameMatrix, ite_eq_right ht, ite_eq_right hh, mul_zero, sub_zero]
    have hB : ∀ f ∈ T, l f = 0 := by
      intro f hf
      have hF : G.IsForest (insert f (T \ {f})) := by
        rwa [Set.insert_sdiff_singleton, Set.insert_eq_of_mem hf]
      refine coeff_eq_zero_of_apply hl
        (∑ w ∈ G.reachFinset (T \ {f}) (G.tail f), LinearMap.proj w) ?_ fun g hg hgf => ?_
      · rw [sum_proj_frameMatrix, ite_eq_left (mem_reachFinset.2 (reachableIn_refl _ _)),
          ite_eq_right fun h => hF.not_reachableIn (fun h => h.2 rfl) (mem_reachFinset.1 h).symm,
          mul_zero, sub_zero]
        exact one_ne_zero
      have hgT : g ∈ T := by_contra fun hgT =>
        Finsupp.mem_support_iff.1 hg (hA g ⟨hsupp g hg, hgT⟩)
      exact sum_proj_frameMatrix_eq_zero (hφ g hgT) (tail_mem_reachFinset_iff ⟨hgT, hgf⟩)
    ext e
    rw [Finsupp.coe_zero, Pi.zero_apply]
    by_contra hle
    have he := hsupp e (Finsupp.mem_support_iff.2 hle)
    exact hle ((em (e ∈ T)).elim (hB e) fun heT => hA e ⟨he, heT⟩)

end OrientedGraph

open OrientedGraph

example := coeff_eq_zero_of_apply (F := ℚ) (v := fun _ : Fin 1 => (1 : ℚ)) (l := 0)
  (map_zero _) LinearMap.id (e₀ := 0) one_ne_zero (by simp)

example := single_sub_single_mem_span_frameMatrix (G := pumpkin)
  (φ := (1 : Fin 3 → ℚˣ)) (T := {0}) (fun _ _ => rfl) (reachableIn_of_mem (Set.mem_singleton 0))

example := sum_proj_frameMatrix_eq_zero (G := pumpkin) (φ := (1 : Fin 3 → ℚˣ))
  (X := {0, 1}) (e := 2) rfl (by simp [pumpkin])

example := tail_mem_reachFinset_iff (G := pumpkin) (u := 0)
  (Set.mem_singleton (0 : Fin 3))

/-- In `pumpkin` with all gains one, the columns of two parallel links are dependent. -/
example : ¬LinearIndepOn ℚ (fun e r => frameMatrix pumpkin (1 : Fin 3 → ℚˣ) r e) {0, 1} := fun h =>
  ((linearIndepOn_frameMatrix_iff (T := {0}) (by simp) pumpkin_isForest
    (fun _ _ => show pumpkin.ReachableIn {0} 0 1 from reachableIn_of_mem (Set.mem_singleton 0))
    fun _ _ => rfl).1 h).1 1 ⟨Set.mem_insert_of_mem _ rfl, by simp⟩ rfl

end Transformer.GainMatroid
