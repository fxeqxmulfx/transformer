module

public import Mathlib.Tactic.FieldSimp
public import Transformer.GainMatroid.Section2_BiasedExamples
public import Transformer.GainMatroid.Section2_Forests

/-!
# Normalized gain functions and Proposition 2.2

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3. A gain function is `F`-normalized when it is
the identity on the forest `F`. Proposition 2.2 says that for a maximal forest and an abelian group,
two normalized gain functions are switching equivalent only if they are equal
(`switchEquiv_iff_eq`), and over a field they are switching-and-scaling equivalent only if they are
proportional (`switchScaleEquiv_iff_eq_smul`).

The key step is `eq_of_maximal_isForest`: a function with equal values at the ends of each edge of
a maximal forest has equal values at the ends of every edge, because an edge outside the forest
closes a cycle with it.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {Γ : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- An edge outside a maximal forest lies on a cycle whose other edges are in the forest. -/
theorem exists_cycleWalk_of_maximal {F : Set E} (hF : Maximal G.IsForest F) {e : E} (he : e ∉ F) :
    ∃ (u : V) (W : G.Walk u u), W.IsCycleWalk ∧ e ∈ W.edges ∧ ∀ f ∈ W.edges, f ≠ e → f ∈ F := by
  have hnot : ¬G.IsForest (insert e F) := fun hins =>
    he (hF.2 hins (Set.subset_insert e F) (Set.mem_insert e F))
  simp only [IsForest, not_forall, not_not] at hnot
  obtain ⟨C, ⟨u, W, hW, rfl⟩, hsub⟩ := hnot
  have heW : e ∈ W.edges := by
    by_contra hne
    refine hF.1 _ ⟨u, W, hW, rfl⟩ fun f hf => ?_
    rcases Set.mem_insert_iff.1 (hsub hf) with rfl | hfF
    · exact absurd hf hne
    · exact hfF
  exact ⟨u, W, hW, heW, fun f hf hfe => (Set.mem_insert_iff.1 (hsub hf)).resolve_left hfe⟩

/-- A function constant across every edge of a cycle but `e` is constant across `e`. -/
theorem Walk.IsCycleWalk.eq_of_forall_ne {α : Type*} {u : V} {W : G.Walk u u} (hW : W.IsCycleWalk)
    {e : E} (he : e ∈ W.edges) (η : V → α)
    (h : ∀ f ∈ W.edges, f ≠ e → η (G.tail f) = η (G.head f)) : η (G.tail e) = η (G.head e) := by
  have hnd := hW.edges_nodup
  obtain ⟨s, t, hj, A, B, rfl⟩ := W.exists_append_cons_of_mem he
  rw [Walk.edges_append] at hnd h
  simp only [Walk.edges] at hnd h
  obtain ⟨-, hB, hAB⟩ := List.nodup_append.1 hnd
  have heA : e ∉ A.edges := fun hc => hAB e hc e List.mem_cons_self rfl
  have heB : e ∉ B.edges := (List.nodup_cons.1 hB).1
  have hA' : η u = η s := Walk.eq_of_forall_mem_edges η A fun f hf =>
    h f (List.mem_append_left _ hf) fun hc => heA (hc ▸ hf)
  have hB' : η t = η u := Walk.eq_of_forall_mem_edges η B fun f hf =>
    h f (List.mem_append_right _ (List.mem_cons_of_mem _ hf)) fun hc => heB (hc ▸ hf)
  rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · rw [h₁, h₂, ← hA', hB']
  · rw [h₁, h₂, ← hA', hB']

/-- A function constant across the edges of a maximal forest is constant across every edge. -/
theorem eq_of_maximal_isForest {α : Type*} {F : Set E} (hF : Maximal G.IsForest F) (η : V → α)
    (h : ∀ f ∈ F, η (G.tail f) = η (G.head f)) (e : E) : η (G.tail e) = η (G.head e) := by
  by_cases he : e ∈ F
  · exact h e he
  obtain ⟨u, W, hW, heW, hsub⟩ := exists_cycleWalk_of_maximal hF he
  exact hW.eq_of_forall_ne heW η fun f hf hfe => h f (hsub f hf hfe)

/-- **Proposition 2.2**, first part (§2.3). Over an abelian group, two gain functions normalized on
a maximal forest are switching equivalent exactly when they are equal. -/
@[to_additive addSwitchEquiv_iff_eq
  /-- **Proposition 2.2**, first part (§2.3), for an additive abelian group. -/]
theorem switchEquiv_iff_eq [CommGroup Γ] {F : Set E} (hF : Maximal G.IsForest F) {φ ψ : E → Γ}
    (hφ : ∀ e ∈ F, φ e = 1) (hψ : ∀ e ∈ F, ψ e = 1) : SwitchEquiv G φ ψ ↔ φ = ψ := by
  constructor
  · rintro ⟨η, rfl⟩
    have hη : ∀ f ∈ F, η (G.tail f) = η (G.head f) := by
      intro f hf
      have h₀ := hψ f hf
      rw [switch_apply, hφ f hf, mul_one, inv_mul_eq_one] at h₀
      exact h₀
    funext e
    rw [switch_apply, eq_of_maximal_isForest hF η hη e, mul_comm (η (G.head e))⁻¹,
      inv_mul_cancel_right]
  · rintro rfl
    exact (switchEquiv_equivalence G).refl _

/-- **Proposition 2.2**, second part (§2.3). Over a field, two additive gain functions normalized on
a maximal forest are switching-and-scaling equivalent exactly when `φ = a ψ` for a nonzero `a`. -/
theorem switchScaleEquiv_iff_eq_smul {K : Type*} [Field K] {F : Set E} (hF : Maximal G.IsForest F)
    {φ ψ : E → K} (hφ : ∀ e ∈ F, φ e = 0) (hψ : ∀ e ∈ F, ψ e = 0) :
    SwitchScaleEquiv G φ ψ ↔ ∃ a : Kˣ, φ = (a : K) • ψ := by
  constructor
  · rintro ⟨η, a, rfl⟩
    have hη : ∀ f ∈ F, η (G.tail f) = η (G.head f) := by
      intro f hf
      have h₀ := hψ f hf
      simp only [Pi.smul_apply, addSwitch_apply, smul_eq_mul, hφ f hf, add_zero] at h₀
      exact neg_add_eq_zero.1 ((mul_eq_zero.1 h₀).resolve_left a.ne_zero)
    refine ⟨a⁻¹, funext fun e => ?_⟩
    have h₂ : -η (G.head e) + φ e + η (G.head e) = φ e := by ring
    simp only [Pi.smul_apply, addSwitch_apply, smul_eq_mul, eq_of_maximal_isForest hF η hη e, h₂]
    rw [← mul_assoc, Units.inv_mul, one_mul]
  · rintro ⟨a, rfl⟩
    exact ⟨0, a⁻¹, by rw [addSwitch_zero, smul_smul, Units.inv_mul, one_smul]⟩

/-- The first link of `pumpkin` is a forest. -/
theorem pumpkin_isForest : pumpkin.IsForest {0} := by
  rintro C ⟨u, W, hW, rfl⟩ hsub
  cases W with
  | nil => exact hW.edges_ne_nil rfl
  | @cons _ b _ e hj p =>
    have he : e = 0 := hsub (show e ∈ e :: p.edges from List.mem_cons_self)
    cases p with
    | nil =>
      rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> exact absurd (h₁.trans h₂.symm) (by simp [pumpkin])
    | @cons _ c _ f hk q =>
      have hf : f = 0 := hsub (show f ∈ e :: f :: q.edges from
        List.mem_cons_of_mem _ List.mem_cons_self)
      have hnd : (e :: f :: q.edges).Nodup := hW.edges_nodup
      exact (List.nodup_cons.1 hnd).1 (by rw [he, hf]; exact List.mem_cons_self)

/-- The first link of `pumpkin` is a maximal forest. -/
theorem pumpkin_maximal : Maximal pumpkin.IsForest {0} := by
  refine ⟨pumpkin_isForest, fun S hS h0 e he => ?_⟩
  by_contra hne
  have hne' : (0 : Fin 3) ≠ e := fun hc => hne (hc ▸ rfl)
  refine hS _ ⟨0, _, pumpkin_cycleWalk hne', rfl⟩ fun f hf => ?_
  rw [Walk.edgeSet_append_reverse] at hf
  rcases hf with hf | hf
  · have : f = 0 := by simpa [Walk.edgeSet, Walk.edges, pumpkinWalk] using hf
    exact this ▸ h0 rfl
  · have : f = e := by simpa [Walk.edgeSet, Walk.edges, pumpkinWalk] using hf
    exact this ▸ he

example [Group Γ] (γ : Fin 3 → Γ) : ∃ η : Fin 2 → Γ, ∀ e ∈ ({0} : Set (Fin 3)),
    switch pumpkin η γ e = 1 :=
  exists_switch_eq_one_of_isForest pumpkin_isForest γ

example [Group Γ] (γ : Fin 3 → Γ) : ∃ η : Fin 2 → Γ, ∀ e ∈ ({0} : Set (Fin 3)),
    switch pumpkin η γ e = 1 :=
  exists_switch_eq_one_of_finite (Set.finite_singleton 0) pumpkin_isForest γ

example [Group Γ] (γ : Fin 3 → Γ) : (pumpkinWalk 0).gain γ = (pumpkinWalk 0).gain γ :=
  gain_eq_of_isForest pumpkin_isForest γ _ _
    (fun e he => by simpa [pumpkinWalk, Walk.edges] using he)
    (fun e he => by simpa [pumpkinWalk, Walk.edges] using he)

example : ∃ r : pumpkin.Walk 1 1, r.IsPath ∧ ∀ e ∈ r.edges, e ∈ (pumpkinWalk 0).edges :=
  (pumpkinWalk_isPath 0).exists_suffix (Walk.end_mem_support _)

example : ∃ (s t : Fin 2) (hj : pumpkin.Joins 2 s t) (A : pumpkin.Walk 0 s) (B : pumpkin.Walk t 1),
    pumpkinWalk 2 = A.append (.cons 2 hj B) :=
  (pumpkinWalk 2).exists_append_cons_of_mem (by simp [pumpkinWalk, Walk.edges])

example : pumpkin.ReachableIn {0} 1 0 :=
  ((reachableIn_refl _ 0).step (Set.mem_singleton 0) (Or.inl ⟨rfl, rfl⟩)).symm

example : pumpkin.ReachableIn {0} 0 0 :=
  ((reachableIn_refl _ 0).step (Set.mem_singleton 0) (Or.inl ⟨rfl, rfl⟩)).trans
    ((reachableIn_refl _ 1).step (Set.mem_singleton 0) (Or.inr ⟨rfl, rfl⟩))

example : (0 : Fin 2) = 0 :=
  Walk.eq_of_forall_mem_edges (fun _ : Fin 2 => (0 : Fin 2)) (pumpkinWalk 0) fun _ _ => rfl

example : ¬pumpkin.ReachableIn ∅ (pumpkin.head 0) (pumpkin.tail 0) :=
  IsForest.not_reachableIn (e := 0) (S := ∅) (by simpa using pumpkin_isForest) (by simp)

example [CommGroup Γ] (g h g' h' : Γ) :
    SwitchEquiv pumpkin ![1, g, h] ![1, g', h'] ↔ (![1, g, h] : Fin 3 → Γ) = ![1, g', h'] :=
  switchEquiv_iff_eq pumpkin_maximal (by simp) (by simp)

example (g h : ℚ) :
    SwitchScaleEquiv pumpkin ![0, g, h] ![0, 2 * g, 2 * h] ↔
      ∃ a : ℚˣ, (![0, g, h] : Fin 3 → ℚ) = (a : ℚ) • ![0, 2 * g, 2 * h] :=
  switchScaleEquiv_iff_eq_smul pumpkin_maximal (by simp) (by simp)

example (c : ℚ) : (![c, c] : Fin 2 → ℚ) (pumpkin.tail 2) = ![c, c] (pumpkin.head 2) :=
  eq_of_maximal_isForest pumpkin_maximal ![c, c] (fun _ _ => by simp [pumpkin]) 2

end OrientedGraph

end Transformer.GainMatroid
