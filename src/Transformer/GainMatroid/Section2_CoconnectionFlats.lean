module

public import Transformer.GainMatroid.Section2_CoconnectionMaps
public import Transformer.GainMatroid.Section2_DualFlats

/-!
# The flats of the dual of `coconnectionMatrix`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, citing Brylawski. Let `A` represent `M`, with
the columns of the triad `Y = {a, b, c}` the first three columns of `I(K₄)` (`IsTriadForm`), and let
`Q = coconnectionMatrix A a b c x y z`. The dual of `M(Q)` is the generalized parallel connection of
`M*` and `M(K₄)` across `Y`: a set `S` is a flat of it exactly when its elements of `E` form a flat
of `M*` and its elements of `K₄` a flat of `M(K₄)` (`IsTriadForm.isFlat_dual_coconnection_iff`).
By `IsMatrixRepresentation.isFlat_dual_iff`, `S` is a flat of the dual of a represented matroid
when the column of every element outside `S` lies in the span of the other columns outside `S`.

* Merging the new rows into `v` carries this condition from `Q` to `A`
  (`IsTriadForm.mem_span_of_coconnection`). A functional on the new rows vanishing on the
  columns off a triangle of `M(K₄)` carries it to the triangle condition of `k4Matroid_isFlat_iff`
  (`IsTriadForm.k4_mem_of_coconnection`).
* Conversely, `liftNew v j` carries the span of columns of `A` outside `S` into the span of columns
  of `Q` outside `S` once the new columns of the third indices are outside `S`
  (`IsTriadForm.liftNew_mem_span`). With `j` a new element in `S`, if there is one, the triangles
  of `M(K₄)` through `Y` put them there (`IsTriadForm.coconnection_mem_span`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open Submodule

variable {E R F : Type*} [DecidableEq E] [DecidableEq R] [Field F] {A : Matrix R E F} {v : R}
  {a b c : E} {x y z : R}

namespace IsTriadForm

/-- `liftNew v j` sends the span of the columns of `A` on `D` into the span of the columns of `Q` on
`T` when `T` holds `D` and, with each element of `Y` in `D` other than the `j`-th, the new column of
the third index. -/
theorem liftNew_mem_span (hform : IsTriadForm A v a b c x y z) (j : Fin 3)
    {T : Set (E ⊕ Fin 3)} {D : Set E} (hD : ∀ f ∈ D, .inl f ∈ T)
    (hY : ∀ l, ![a, b, c] l ∈ D → l ≠ j → .inr (thirdIndex l j) ∈ T) {w : R → F}
    (hw : w ∈ span F ((fun f r => A r f) '' D)) :
    liftNew v j w ∈ span F (coconnectionCol A a b c x y z '' T) := by
  refine (map_span_le _ _ _).2 ?_ (mem_map_of_mem hw)
  rintro _ ⟨f, hf, rfl⟩
  rcases eq_or_eq_triad (a := a) (b := b) (c := c) f with ⟨ha, hb, hc⟩ | ⟨l, rfl⟩
  · rw [hform.liftNew_inl j ha hb hc]
    exact subset_span ⟨_, hD f hf, rfl⟩
  · rw [hform.liftNew_triad]
    exact add_mem (subset_span ⟨_, hD _ hf, rfl⟩) (newDiff_mem_span l j (hY l hf))

/-- Where the columns of `Q` outside `S` are closed, so are the columns of `A` outside `S`. -/
theorem mem_span_of_coconnection (hform : IsTriadForm A v a b c x y z) {S : Set (E ⊕ Fin 3)}
    (h : ∀ q, q ∉ S → coconnectionCol A a b c x y z q ∈
      span F (coconnectionCol A a b c x y z '' (Sᶜ \ {q}))) :
    ∀ e, e ∉ Sum.inl ⁻¹' S →
      (fun r => A r e) ∈ span F ((fun f r => A r f) '' ((Sum.inl ⁻¹' S)ᶜ \ {e})) := by
  intro e he
  rw [← hform.mergeNew_inl e]
  refine (map_span_le _ _ _).2 ?_ (mem_map_of_mem (h (.inl e) he))
  rintro _ ⟨q, ⟨hqS, hqe⟩, rfl⟩
  rcases q with f | k
  · rw [hform.mergeNew_inl f]
    exact subset_span ⟨f, ⟨hqS, fun hfe => hqe (congrArg Sum.inl hfe)⟩, rfl⟩
  · rw [mergeNew_inr]
    exact zero_mem _

/-- Where the columns of `Q` outside `S` are closed, `S` holds the third edge of every triangle of
`M(K₄)` with two edges in it. -/
theorem k4_mem_of_coconnection (hform : IsTriadForm A v a b c x y z) {S : Set (E ⊕ Fin 3)}
    (h : ∀ q, q ∉ S → coconnectionCol A a b c x y z q ∈
      span F (coconnectionCol A a b c x y z '' (Sᶜ \ {q}))) :
    ∀ C ∈ k4Triangles, ∀ g ∈ C, (∀ g' ∈ C, g' ≠ g → g' ∈ deltaEmbed a b c ⁻¹' S) →
      g ∈ deltaEmbed a b c ⁻¹' S := by
  intro C hC g hg hCg
  have hab := hform.a_ne_b
  have hac := hform.a_ne_c
  have hbc := hform.b_ne_c
  by_contra hgS
  obtain ⟨μ, hμ⟩ := exists_k4_functional (F := F) hC
  have h0 : (μ ∘ₗ newRows) (coconnectionCol A a b c x y z (deltaEmbed a b c g)) ∈
      span F (∅ : Set F) := by
    refine (map_span_le _ _ _).2 ?_ (mem_map_of_mem (h _ hgS))
    rintro _ ⟨q, ⟨hqS, hqg⟩, rfl⟩
    rcases eq_inl_or_eq_deltaEmbed (a := a) (b := b) (c := c) q with
      ⟨f, ha, hb, hc, rfl⟩ | ⟨g', rfl⟩
    · rw [LinearMap.comp_apply, newRows_inl ha hb hc, map_zero]
      exact zero_mem _
    · rw [LinearMap.comp_apply, newRows_deltaEmbed hab hac hbc]
      by_cases hg' : μ (k4NewCol g') = 0
      · rw [hg']
        exact zero_mem _
      · exact absurd (hCg g' ((hμ g').1 hg') fun hgg => hqg (congrArg (deltaEmbed a b c) hgg)) hqS
  rw [span_empty, mem_bot, LinearMap.comp_apply, newRows_deltaEmbed hab hac hbc] at h0
  exact (hμ g).2 hg h0

/-- Where the columns of `A` outside `S` are closed and `S` holds the third edge of every triangle
of `M(K₄)` with two edges in it, the columns of `Q` outside `S` are closed. -/
theorem coconnection_mem_span (hform : IsTriadForm A v a b c x y z) {S : Set (E ⊕ Fin 3)}
    (hM : ∀ e, e ∉ Sum.inl ⁻¹' S →
      (fun r => A r e) ∈ span F ((fun f r => A r f) '' ((Sum.inl ⁻¹' S)ᶜ \ {e})))
    (hK : ∀ C ∈ k4Triangles, ∀ g ∈ C, (∀ g' ∈ C, g' ≠ g → g' ∈ deltaEmbed a b c ⁻¹' S) →
      g ∈ deltaEmbed a b c ⁻¹' S) :
    ∀ q, q ∉ S → coconnectionCol A a b c x y z q ∈
      span F (coconnectionCol A a b c x y z '' (Sᶜ \ {q})) := by
  have claim (j l : Fin 3) (hj : .inr j ∈ S) (hlj : l ≠ j) (hl : .inl (![a, b, c] l) ∉ S) :
      .inr (thirdIndex l j) ∉ S := fun ht =>
    hl <| hK _ (thirdIndex_mem_k4Triangles l j hlj) (.inl l) (by simp) fun g' hg' hne => by
      simp only [Finset.mem_insert, Finset.mem_singleton] at hg'
      rcases hg' with rfl | rfl | rfl
      · exact absurd rfl hne
      · exact hj
      · exact ht
  rintro (e | k) hq
  · obtain ⟨j, hj⟩ : ∃ j : Fin 3, ∀ l, l ≠ j → .inl (![a, b, c] l) ∉ S →
        .inr (thirdIndex l j) ∉ S := by
      by_cases hS : ∃ j, Sum.inr j ∈ S
      · obtain ⟨j, hjS⟩ := hS
        exact ⟨j, fun l hlj hl => claim j l hjS hlj hl⟩
      · push Not at hS
        exact ⟨0, fun l _ _ => hS _⟩
    have hlift := hform.liftNew_mem_span j (T := Sᶜ \ {.inl e}) (D := (Sum.inl ⁻¹' S)ᶜ \ {e})
      (fun f hf => ⟨hf.1, fun h => hf.2 (Sum.inl_injective h)⟩)
      (fun l hl hlj => ⟨hj l hlj hl.1, Sum.inr_ne_inl⟩) (hM e hq)
    rcases eq_or_eq_triad (a := a) (b := b) (c := c) e with ⟨ha, hb, hc⟩ | ⟨i, rfl⟩
    · rwa [hform.liftNew_inl j ha hb hc] at hlift
    · rw [hform.liftNew_triad] at hlift
      have hd := newDiff_mem_span (A := A) (a := a) (b := b) (c := c) (x := x) (y := y) (z := z) i j
        (T := Sᶜ \ {.inl (![a, b, c] i)}) fun hij => ⟨hj i hij hq, Sum.inr_ne_inl⟩
      have := sub_mem hlift hd
      rwa [add_sub_cancel_right] at this
  · by_cases hS : ∃ j, j ≠ k ∧ Sum.inr j ∈ S
    · obtain ⟨j, hjk, hj⟩ := hS
      have hlj : thirdIndex j k ≠ j := thirdIndex_ne_left j k hjk
      have htk : thirdIndex (thirdIndex j k) j = k := thirdIndex_thirdIndex j k
      have hl : .inl (![a, b, c] (thirdIndex j k)) ∉ S := fun hlS => hq <| by
        have := hK _ (thirdIndex_mem_k4Triangles _ j hlj) (.inr (thirdIndex (thirdIndex j k) j))
          (by simp) fun g' hg' hne => by
            simp only [Finset.mem_insert, Finset.mem_singleton] at hg'
            rcases hg' with rfl | rfl | rfl
            · exact hlS
            · exact hj
            · exact absurd rfl hne
        rwa [htk] at this
      have hlift := hform.liftNew_mem_span j (T := Sᶜ \ {.inr k})
        (D := (Sum.inl ⁻¹' S)ᶜ \ {![a, b, c] (thirdIndex j k)}) (fun f hf => ⟨hf.1, Sum.inl_ne_inr⟩)
        (fun l hl hl' => ⟨claim j l hj hl' hl.1, fun h =>
          hl.2 (congrArg (![a, b, c] ·) ((thirdIndex_eq_iff l j k).1 (Sum.inr_injective h)))⟩)
        (hM _ hl)
      rw [hform.liftNew_triad] at hlift
      have hmem : coconnectionCol A a b c x y z (.inl (![a, b, c] (thirdIndex j k))) ∈
          span F (coconnectionCol A a b c x y z '' (Sᶜ \ {.inr k})) :=
        subset_span ⟨_, ⟨hl, Sum.inl_ne_inr⟩, rfl⟩
      have := coconnectionCol_thirdIndex_mem (A := A) (a := a) (b := b) (c := c) (x := x) (y := y)
        (z := z) _ _ hlj ((add_mem_cancel_left hmem).1 hlift)
      rwa [htk] at this
    · push Not at hS
      refine span_mono (Set.image_mono ?_) (coconnectionCol_inr_mem_span k)
      rintro _ ⟨j, hjk, rfl⟩
      exact ⟨hS j hjk, fun h => hjk (Sum.inr_injective h)⟩

/-- **The dual of `M(Q)` is the generalized parallel connection of `M*` and `M(K₄)` across `Y`**:
its flats are the sets whose elements of `E` form a flat of `M*` and whose elements of `K₄` form a
flat of `M(K₄)`. -/
theorem isFlat_dual_coconnection_iff [Finite E] {M : Matroid E}
    (hform : IsTriadForm A v a b c x y z) (hA : IsMatrixRepresentation M A) (S : Set (E ⊕ Fin 3)) :
    (matrixMatroid (coconnectionMatrix A a b c x y z))✶.IsFlat S ↔
      M✶.IsFlat (Sum.inl ⁻¹' S) ∧ k4Matroid.IsFlat (deltaEmbed a b c ⁻¹' S) := by
  rw [(matrixMatroid_isMatrixRepresentation _).isFlat_dual_iff, hA.isFlat_dual_iff,
    k4Matroid_isFlat_iff]
  exact ⟨fun h => ⟨hform.mem_span_of_coconnection h, hform.k4_mem_of_coconnection h⟩,
    fun ⟨hM, hK⟩ => hform.coconnection_mem_span hM hK⟩

end IsTriadForm

example := (iK4_isTriadForm (F := ℚ)).liftNew_mem_span 0 (T := Set.univ) (D := Set.univ)
  (fun _ _ => trivial) (fun _ _ _ => trivial) (subset_span ⟨3, trivial, rfl⟩)

example := (iK4_isTriadForm (F := ℚ)).mem_span_of_coconnection (S := Set.univ)
  fun _ hq => absurd trivial hq

example := (iK4_isTriadForm (F := ℚ)).k4_mem_of_coconnection (S := Set.univ)
  fun _ hq => absurd trivial hq

example := (iK4_isTriadForm (F := ℚ)).coconnection_mem_span (S := Set.univ)
  (fun _ he => absurd trivial he) fun _ _ _ _ _ => trivial

example := (iK4_isTriadForm (F := ℚ)).isFlat_dual_coconnection_iff
  (matrixMatroid_isMatrixRepresentation _)

end Transformer.GainMatroid
