module

public import Transformer.GainMatroid.Section3_Classes

/-!
# Which biased `2C₃`, tubes and `K₄` are properly unbalanced

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §3.1. The source states without proof that a biased
`K₄` is properly unbalanced if and only if it contains no balanced triangle, that a biased `2C₃` is
if and only if it has no balanced 2-cycle (proof of Proposition 3.1), and that a biased tube is if
and only if it has no balanced 2-cycle. These hold for every orientation of the graphs.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace OrientedGraph

theorem IsReorientationOf.has_end {V E : Type*} {G H : OrientedGraph V E}
    (h : G.IsReorientationOf H) {g : E} {p : V} (hp : H.tail g = p ∨ H.head g = p) :
    G.tail g = p ∨ G.head g = p := by
  rcases h g with ⟨ht, hh⟩ | ⟨ht, hh⟩
  · exact hp.imp ht.trans hh.trans
  · exact hp.symm.imp ht.trans hh.trans

end OrientedGraph

namespace BiasedGraph

private theorem twoC3_pair_off : ∀ w : Fin 3, ∃ (e f : Fin 6) (u v : Fin 3), e ≠ f ∧ u ≠ v ∧
    w ≠ u ∧ w ≠ v ∧ twoC3.Joins e u v ∧ twoC3.Joins f u v := by
  simp only [Joins, twoC3]
  decide

set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem twoC3_unique_off : ∀ (e f : Fin 6) (u v : Fin 3), e ≠ f → u ≠ v →
    twoC3.Joins e u v → twoC3.Joins f u v → ∃ w : Fin 3, w ≠ u ∧ w ≠ v ∧
      ∀ g, twoC3.tail g ≠ w → twoC3.head g ≠ w → g = e ∨ g = f := by
  simp only [Joins, twoC3]
  decide +kernel

private theorem tube_pair_off : ∀ w : Fin 4, ∃ (e f : Fin 6) (u v : Fin 4), e ≠ f ∧ u ≠ v ∧
    w ≠ u ∧ w ≠ v ∧ tube.Joins e u v ∧ tube.Joins f u v := by
  simp only [Joins, tube]
  decide

set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem tube_unique_off : ∀ (e f : Fin 6) (u v : Fin 4), e ≠ f → u ≠ v →
    tube.Joins e u v → tube.Joins f u v → ∃ (w p : Fin 4) (g : Fin 6), w ≠ u ∧ w ≠ v ∧
      (tube.tail g = p ∨ tube.head g = p) ∧
      (∀ x, tube.tail x ≠ w → tube.head x ≠ w → x = e ∨ x = f ∨ x = g) ∧
      tube.tail e ≠ p ∧ tube.head e ≠ p ∧ tube.tail f ≠ p ∧ tube.head f ≠ p := by
  simp only [Joins, tube]
  decide +kernel

private theorem K4_triangle_off (w : Fin 4) : ∃ (a b c : Fin 6) (x y z : Fin 4), a ≠ b ∧ a ≠ c ∧
    b ≠ c ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ K4.Joins a x y ∧ K4.Joins b y z ∧
    K4.Joins c z x := by
  fin_cases w
  · refine ⟨3, 5, 4, 1, 2, 3, ?_⟩
    simp only [Joins, K4]
    decide
  · refine ⟨1, 5, 2, 0, 2, 3, ?_⟩
    simp only [Joins, K4]
    decide
  · refine ⟨0, 4, 2, 0, 1, 3, ?_⟩
    simp only [Joins, K4]
    decide
  · refine ⟨0, 3, 1, 0, 1, 2, ?_⟩
    simp only [Joins, K4]
    decide

set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem K4_unique_off : ∀ (a b c : Fin 6) (x y z : Fin 4), a ≠ b → a ≠ c → b ≠ c → x ≠ y →
    x ≠ z → y ≠ z → K4.Joins a x y → K4.Joins b y z → K4.Joins c z x →
    ∃ w : Fin 4, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧
      ∀ g, K4.tail g ≠ w → K4.head g ≠ w → g = a ∨ g = b ∨ g = c := by
  simp only [Joins, K4]
  decide +kernel

/-- **§3.1, proof of Proposition 3.1.** A biased `2C₃` is properly unbalanced if and only if it has
no balanced 2-cycle. -/
theorem isProperlyUnbalanced_iff_of_twoC3 {Ω : BiasedGraph (Fin 3) (Fin 6)}
    (hG : Ω.graph.IsReorientationOf twoC3) :
    Ω.IsProperlyUnbalanced ↔ ∀ C ∈ Ω.balanced, C.encard ≠ 2 := by
  refine ⟨fun hP C hC h2 => ?_, fun h => ?_⟩
  · obtain ⟨e, f, u, v, hne, huv, he, hf, rfl⟩ := (Ω.isCycle_of_mem C hC).exists_pair h2
    obtain ⟨w, -, -, hoff⟩ := twoC3_unique_off e f u v hne huv (hG.symm.joins he) (hG.symm.joins hf)
    refine not_isProperlyUnbalanced_of_off hC (w := w) (fun D _ hlen hw => ?_) hP
    refine (Set.toFinite D).eq_of_subset_of_encard_le (fun g hg => ?_)
      (by rwa [Set.encard_pair hne])
    obtain ⟨h₁, h₂⟩ := hG.ends_ne (g := g) (w := w)
      (fun hc => hw (mem_verts.2 ⟨g, hg, Or.inl hc⟩))
      (fun hc => hw (mem_verts.2 ⟨g, hg, Or.inr hc⟩))
    simpa using hoff g h₁ h₂
  · refine isProperlyUnbalanced_of_walks le_rfl h fun w => ?_
    obtain ⟨e, f, u, v, hne, huv, hwu, hwv, he, hf⟩ := twoC3_pair_off w
    exact ⟨u, pairWalk (hG.joins he) (hG.joins hf), pairWalk_isCycleWalk _ _ hne huv,
      by simp [pairWalk, Walk.edges], by simp [pairWalk, Walk.support, hwu, hwv]⟩

/-- **§3.1.** A biased tube is properly unbalanced if and only if it has no balanced 2-cycle. -/
theorem isProperlyUnbalanced_iff_of_tube {Ω : BiasedGraph (Fin 4) (Fin 6)}
    (hG : Ω.graph.IsReorientationOf tube) :
    Ω.IsProperlyUnbalanced ↔ ∀ C ∈ Ω.balanced, C.encard ≠ 2 := by
  refine ⟨fun hP C hC h2 => ?_, fun h => ?_⟩
  · obtain ⟨e, f, u, v, hne, huv, he, hf, rfl⟩ := (Ω.isCycle_of_mem C hC).exists_pair h2
    obtain ⟨w, p, g, -, -, hgp, hoff, hep₁, hep₂, hfp₁, hfp₂⟩ :=
      tube_unique_off e f u v hne huv (hG.symm.joins he) (hG.symm.joins hf)
    refine not_isProperlyUnbalanced_of_off hC (w := w) (fun D hD hlen hw => ?_) hP
    have hends : ∀ x ∈ D, x = e ∨ x = f ∨ x = g := fun x hx => by
      obtain ⟨h₁, h₂⟩ := hG.ends_ne (g := x) (w := w)
        (fun hc => hw (mem_verts.2 ⟨x, hx, Or.inl hc⟩))
        (fun hc => hw (mem_verts.2 ⟨x, hx, Or.inr hc⟩))
      exact hoff x h₁ h₂
    have hgD : g ∉ D := hD.not_mem_of_pendant (hG.has_end hgp)
      (hG.loopless tube_loopless g) fun x hx hxg => by
        rcases hends x hx with rfl | rfl | rfl
        · exact hG.symm.ends_ne hep₁ hep₂
        · exact hG.symm.ends_ne hfp₁ hfp₂
        · exact absurd rfl hxg
    refine (Set.toFinite D).eq_of_subset_of_encard_le (fun x hx => ?_)
      (by rwa [Set.encard_pair hne])
    rcases hends x hx with rfl | rfl | rfl
    · exact Or.inl rfl
    · exact Or.inr rfl
    · exact absurd hx hgD
  · refine isProperlyUnbalanced_of_walks le_rfl h fun w => ?_
    obtain ⟨e, f, u, v, hne, huv, hwu, hwv, he, hf⟩ := tube_pair_off w
    exact ⟨u, pairWalk (hG.joins he) (hG.joins hf), pairWalk_isCycleWalk _ _ hne huv,
      by simp [pairWalk, Walk.edges], by simp [pairWalk, Walk.support, hwu, hwv]⟩

/-- **§3.1.** A biased `K₄` is properly unbalanced if and only if it has no balanced triangle. -/
theorem isProperlyUnbalanced_iff_of_K4 {Ω : BiasedGraph (Fin 4) (Fin 6)}
    (hG : Ω.graph.IsReorientationOf K4) :
    Ω.IsProperlyUnbalanced ↔ ∀ C ∈ Ω.balanced, C.encard ≠ 3 := by
  refine ⟨fun hP C hC h3 => ?_, fun h => ?_⟩
  · obtain ⟨a, b, c, x, y, z, hab, hac, hbc, hxy, hxz, hyz, ha, hb, hc', rfl⟩ :=
      (Ω.isCycle_of_mem C hC).exists_triangle h3
    obtain ⟨w, -, -, -, hoff⟩ := K4_unique_off a b c x y z hab hac hbc hxy hxz hyz
      (hG.symm.joins ha) (hG.symm.joins hb) (hG.symm.joins hc')
    refine not_isProperlyUnbalanced_of_off hC (w := w) (fun D hD _ hw => ?_) hP
    refine (Set.toFinite D).eq_of_subset_of_encard_le (fun g hg => ?_)
      (by rw [h3]; exact three_le_encard_of_K4 hG hD)
    obtain ⟨h₁, h₂⟩ := hG.ends_ne (g := g) (w := w)
      (fun hc => hw (mem_verts.2 ⟨g, hg, Or.inl hc⟩))
      (fun hc => hw (mem_verts.2 ⟨g, hg, Or.inr hc⟩))
    simpa using hoff g h₁ h₂
  · refine isProperlyUnbalanced_of_walks (k := 3) (by norm_num) (by exact_mod_cast h) fun w => ?_
    obtain ⟨a, b, c, x, y, z, hab, hac, hbc, hxy, hxz, hyz, hwx, hwy, hwz, ha, hb, hc⟩ :=
      K4_triangle_off w
    exact ⟨x, triangleWalk (hG.joins ha) (hG.joins hb) (hG.joins hc),
      triangleWalk_isCycleWalk _ _ _ hab hac hbc hxy hxz hyz,
      by simp [triangleWalk, Walk.edges], by simp [triangleWalk, Walk.support, hwx, hwy, hwz]⟩

end BiasedGraph

end Transformer.GainMatroid
