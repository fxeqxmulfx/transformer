module

public import Transformer.GainMatroid.Section5_FanFacts

/-!
# A roll-up of the fan, and the 2-separations of the fan

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5 and §5. A cycle of a graph on three vertices
has at most three edges (`OrientedGraph.IsCycle.encard_le_three`), and a cycle of one edge is a
loop. Rolling up the unbalancing class `{1}` of the contrabalanced `fanGraph` from `2` makes `1` a
joint at `0` (`fan_rollUpGraph`), and `fanRollUpGain` realizes that roll-up. Each side of a
2-separation of `fanGraph` contains one of its 2-cycles, so no side is balanced
(`fan_not_isBalancedSet`). -/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph BiasedGraph

local instance instFactPrimeSevenFanRollUp : Fact (Nat.Prime 7) := ⟨by decide⟩

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- A cycle on three vertices has at most three edges. -/
theorem IsCycle.encard_le_three {G : OrientedGraph (Fin 3) E} {C : Set E} (hC : G.IsCycle C) :
    C.encard ≤ 3 := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  rw [hW.encard_edgeSet]
  have h := hW.support_tail_nodup.length_le_card
  have hle : W.edges.length ≤ 3 := by simpa [Walk.length_support] using h
  exact_mod_cast hle

/-- A cycle of one edge is a loop. -/
theorem IsCycle.exists_loop_of_encard_eq_one {C : Set E} (hC : G.IsCycle C) (h1 : C.encard = 1) :
    ∃ e, C = {e} ∧ G.tail e = G.head e := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  rw [hW.encard_edgeSet] at h1
  have hlen : W.edges.length = 1 := by exact_mod_cast h1
  cases W with
  | nil => simp [Walk.edges] at hlen
  | cons e he p =>
    cases p with
    | nil =>
      refine ⟨e, by ext f; simp [Walk.edgeSet, Walk.edges], ?_⟩
      rcases he with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> rw [h₁, h₂]
    | cons f hf q => simp [Walk.edges] at hlen

end OrientedGraph

/-- Each side of a 2-separation of `fanGraph` contains one of its 2-cycles `{1, 2}`, `{3, 4}`. -/
theorem fan_not_isBalancedSet {A B : Set (Fin 5)} (h : fanGraph.IsSeparation 2 A B) :
    ¬fanGraph.contrabalanced.IsBalancedSet A := by
  intro hA
  have h₁₂ : ¬({1, 2} : Set (Fin 5)) ⊆ A := fun hs => Set.notMem_empty _
    (hA _ (isCycle_pair (u := 0) (v := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
      (by decide)) hs)
  have h₃₄ : ¬({3, 4} : Set (Fin 5)) ⊆ A := fun hs => Set.notMem_empty _
    (hA _ (isCycle_pair (u := 1) (v := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
      (by decide)) hs)
  obtain ⟨s, rfl⟩ : ∃ s : Finset (Fin 5), A = ↑s := ⟨(Set.toFinite A).toFinset, by simp⟩
  have hB : B = ↑sᶜ := by
    ext e
    have hu : e ∈ (↑s ∪ B : Set (Fin 5)) := h.union ▸ Set.mem_univ e
    have hd : e ∈ (↑s : Set (Fin 5)) → e ∉ B := fun he => Set.disjoint_left.1 h.disjoint he
    by_cases he : e ∈ s
    · simpa [he] using hd (Finset.mem_coe.2 he)
    · simpa [he] using hu
  subst hB
  have hverts : ∀ t : Finset (Fin 5), fanGraph.verts ↑t = ↑(t.image fanGraph.tail ∪
      t.image fanGraph.head) := fun t => by
    ext v
    simp only [mem_verts, Finset.coe_union, Finset.coe_image, Set.mem_union, Set.mem_image,
      Finset.mem_coe]
    constructor
    · rintro ⟨e, he, h | h⟩
      exacts [Or.inl ⟨e, he, h⟩, Or.inr ⟨e, he, h⟩]
    · rintro (⟨e, he, h⟩ | ⟨e, he, h⟩)
      exacts [⟨e, he, Or.inl h⟩, ⟨e, he, Or.inr h⟩]
  have hl := h.left
  have hr := h.right
  have hi := h.inter
  rw [hverts, hverts, ← Finset.coe_inter, Set.encard_coe_eq_coe_finsetCard] at hi
  rw [Set.encard_coe_eq_coe_finsetCard] at hl hr
  have hl' : 2 ≤ s.card := by exact_mod_cast hl
  have hr' : 2 ≤ sᶜ.card := by exact_mod_cast hr
  have hi' : ((s.image fanGraph.tail ∪ s.image fanGraph.head) ∩
      (sᶜ.image fanGraph.tail ∪ sᶜ.image fanGraph.head)).card = 2 := by exact_mod_cast hi
  have h₁₂' : ¬({1, 2} : Finset (Fin 5)) ⊆ s := fun hs => h₁₂ (by
    have := Finset.coe_subset.2 hs
    simpa using this)
  have h₃₄' : ¬({3, 4} : Finset (Fin 5)) ⊆ s := fun hs => h₃₄ (by
    have := Finset.coe_subset.2 hs
    simpa using this)
  clear h h₁₂ h₃₄ hA hl hr hi
  revert s
  simp only [fanGraph]
  decide

/-- `fanGraph` with the edge `1` rolled up from `2`: a joint at `0`. -/
def fanRollUpGraph : OrientedGraph (Fin 3) (Fin 5) := ⟨![0, 0, 0, 1, 1], ![1, 0, 2, 2, 2]⟩

theorem fan_rollUpGraph : fanGraph.rollUpGraph 2 {1} = fanRollUpGraph := by
  simp only [rollUpGraph, otherEnd, Set.mem_singleton_iff, fanRollUpGraph]
  congr 1 <;> funext e <;> fin_cases e <;> simp [fanGraph]

/-- The gains `(1, 2, 1, 2, 3)` over `𝔽₇ˣ` on `fanRollUpGraph`. -/
def fanRollUpGain : Fin 5 → (ZMod 7)ˣ := ![1, fanTwo, 1, fanTwo, fanThree]

set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
/-- `fanRollUpGain` realizes the roll-up of the contrabalanced `fanGraph` on `{1}`: its joint has
gain `2`, and no 2-cycle or triangle is balanced. -/
theorem fanRollUpGain_realizes :
    (fanGraph.contrabalanced.rollUp 2 {1}).Realizes fanRollUpGain := by
  have hb : (fanGraph.contrabalanced.rollUp 2 {1}).balanced = ∅ := by
    simp [rollUp, contrabalanced]
  rw [Realizes, hb]
  show (fanGraph.rollUpGraph 2 {1}).balancedCycles fanRollUpGain = ∅
  rw [fan_rollUpGraph]
  refine Set.eq_empty_of_forall_notMem fun C hC => ?_
  have hC' := mem_balancedCycles_iff.1 hC
  have h1 : C.encard ≠ 1 := fun h1 => by
    obtain ⟨e, rfl, he⟩ := hC'.1.exists_loop_of_encard_eq_one h1
    obtain ⟨η, hη⟩ := hC'.2
    have hloop : ∀ f : Fin 5, fanRollUpGraph.tail f = fanRollUpGraph.head f → f = 1 := by
      simp only [fanRollUpGraph]
      decide
    obtain rfl := hloop e he
    have h := hη 1 rfl
    rw [switch_apply, show fanRollUpGraph.tail 1 = 0 from rfl,
      show fanRollUpGraph.head 1 = 0 from rfl, show fanRollUpGain 1 = fanTwo from rfl] at h
    have h2 : fanTwo = 1 := by
      calc fanTwo = (η 0)⁻¹ * fanTwo * η 0 := by
            rw [mul_comm (η 0)⁻¹ fanTwo, mul_assoc, inv_mul_cancel, mul_one]
        _ = 1 := h
    exact absurd h2 (by decide)
  have h2 := encard_ne_two_of_parallel (G := fanRollUpGraph) (γ := fanRollUpGain)
    (by simp only [Joins, dart, fanRollUpGraph, fanRollUpGain]; decide +kernel) C hC
  have h3 := encard_ne_three_of_triangle (G := fanRollUpGraph) (γ := fanRollUpGain)
    (by simp only [Joins, dart, fanRollUpGraph, fanRollUpGain]; decide +kernel) C hC
  have hle := hC'.1.encard_le_three
  have h0 : C.encard ≠ 0 := fun h0 => by
    obtain ⟨u, W, hW, hWC⟩ := hC'.1
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
    have heC : e ∈ C := hWC ▸ (he : e ∈ W.edgeSet)
    rw [Set.encard_eq_zero.1 h0] at heC
    exact heC
  have hfin : C.encard ≠ ⊤ := ne_top_of_le_ne_top (by simp) hle
  obtain ⟨n, hn⟩ := ENat.ne_top_iff_exists.1 hfin
  rw [← hn] at h0 h1 h2 h3 hle
  have : n ≤ 3 := by exact_mod_cast hle
  interval_cases n <;> simp_all

/-- `({0, 1, 2}, {3, 4})` is a 2-separation of `fanGraph`: the sides share the vertices `1`, `2`. -/
theorem fanGraph_isSeparation : fanGraph.IsSeparation 2 {0, 1, 2} {3, 4} where
  disjoint := Set.disjoint_left.2 fun x hx hy => by
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx hy
    rcases hx with rfl | rfl | rfl <;> rcases hy with h | h <;> exact absurd h (by decide)
  union := Set.eq_univ_of_forall fun x => by fin_cases x <;> simp
  left := by
    rw [Set.encard_insert_of_notMem (by simp), Set.encard_pair (by decide)]
    norm_num
  right := by
    rw [Set.encard_pair (by decide)]
    rfl
  inter := by
    have h : fanGraph.verts {0, 1, 2} ∩ fanGraph.verts {3, 4} = {1, 2} := by
      ext v
      simp only [Set.mem_inter_iff, mem_verts, Set.mem_insert_iff, Set.mem_singleton_iff,
        exists_eq_or_imp, exists_eq_left]
      fin_cases v <;> simp [fanGraph]
    rw [h, Set.encard_pair (by decide)]
    rfl

example := (isCycle_pair (G := fanGraph) (e := 1) (f := 2) (u := 0) (v := 2) (Or.inl ⟨rfl, rfl⟩)
  (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)).encard_le_three

example := (isCycle_singleton_of_loop (G := fanRollUpGraph) (e := 1) rfl)
  |>.exists_loop_of_encard_eq_one (Set.encard_singleton _)

example := fan_not_isBalancedSet fanGraph_isSeparation

end Transformer.GainMatroid
