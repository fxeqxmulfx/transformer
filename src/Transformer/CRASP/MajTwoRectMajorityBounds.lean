/-
# Depth and fragment bounds for the majority translation

arXiv:2506.16055v3, Appendix E, `thm:majtwo_to_tlc`: one majority
quantifier becomes one level of temporal counting.
-/

import Transformer.CRASP.MajTwoRectMajority

namespace Transformer.CRASP

universe u
variable {σ : Type u}

/-- A list of rectangle counts is PNP-free and costs one level (Appendix E). -/
theorem MajRects.countList_bounds (v : Var) (L : List (MajRect σ)) (d : ℕ)
    (hL : ∀ r ∈ L, r.Good d) :
    (countList v L).pnpFree = true ∧ (countList v L).depth ≤ d + 1 := by
  have h : ∀ t ∈ L.map (fun r => r.count v), t.pnpFree = true ∧ t.depth ≤ d + 1 := by
    rintro t ht
    obtain ⟨r, hr, rfl⟩ := List.mem_map.mp ht
    cases v
    · exact ⟨MajRect.pnpFree_countY (MajRect.good_transpose (hL r hr)),
        MajRect.depth_countY_le (MajRect.good_transpose (hL r hr))⟩
    · exact ⟨MajRect.pnpFree_countY (hL r hr), MajRect.depth_countY_le (hL r hr)⟩
  exact ⟨TermX.pnpFree_sum fun t ht => (h t ht).1,
    TermX.depth_sum_le (by omega) fun t ht => (h t ht).2⟩

/-- A finite sum of bounded terms keeps both bounds (Appendix E). -/
theorem TermX.sum_ofFn_bounds {m d : ℕ} (t : Fin (m + 1) → TermX σ)
    (h : ∀ j, (t j).pnpFree = true ∧ (t j).depth ≤ d + 1) :
    (TermX.sum (List.ofFn t)).pnpFree = true ∧
      (TermX.sum (List.ofFn t)).depth ≤ d + 1 := by
  have hL : ∀ u ∈ List.ofFn t, u.pnpFree = true ∧ u.depth ≤ d + 1 := by
    intro u hu
    obtain ⟨j, rfl⟩ := List.mem_ofFn.mp hu
    exact h j
  exact ⟨TermX.pnpFree_sum fun u hu => (hL u hu).1,
    TermX.depth_sum_le (by omega) fun u hu => (hL u hu).2⟩

/-- The total pair count is a depth-one plain-logic term (Appendix E). -/
theorem MajRects.pairCount_bounds (m d : ℕ) :
    (pairCount (σ := σ) m).pnpFree = true ∧ (pairCount (σ := σ) m).depth ≤ d + 1 := by
  have hL : ∀ t : TermX σ, t ∈ List.replicate (m + 1) (TermX.countAll (Form.topAt 0).toX) →
      t.pnpFree = true ∧ t.depth ≤ d + 1 := by
    simp only [List.mem_replicate]
    rintro t ⟨_, rfl⟩
    simp [TermX.pnpFree, Form.topAt, Form.pnpFree, Term.pnpFree, TermX.depth,
      Form.depth_toX, Form.depth, Term.depth]
  exact ⟨TermX.pnpFree_sum fun t ht => (hL t ht).1,
    TermX.depth_sum_le (by omega) fun t ht => (hL t ht).2⟩

/-- The translated majority lies in `TL[◁#,▷#]` at the original depth. -/
theorem MajRects.majority_mem {m d : ℕ} (v : Var) (R : Fin (m + 1) → MajRects σ)
    (hR : ∀ t, (R t).Good d) : majority v R ∈ TLC σ (d + 1) := by
  have hp : (positiveCount v R).pnpFree = true ∧ (positiveCount v R).depth ≤ d + 1 :=
    TermX.sum_ofFn_bounds _ fun t => countList_bounds v _ d fun r hr =>
      hR t r (List.mem_append_left _ hr)
  have hn : (negativeCount v R).pnpFree = true ∧ (negativeCount v R).depth ≤ d + 1 :=
    TermX.sum_ofFn_bounds _ fun t => countList_bounds v _ d fun r hr =>
      hR t r (List.mem_append_right _ hr)
  have hc := pairCount_bounds (σ := σ) m d
  have hf : (majorityX v R).pnpFree = true := by
    simp [majorityX, FormX.pnpFree, TermX.pnpFree, hp.1, hn.1, hc.1]
  have hd : (majorityX v R).depth ≤ d + 1 := by
    simp only [majorityX, FormX.depth, TermX.depth, max_le_iff]
    exact ⟨⟨hc.2, hn.2, hn.2⟩, hp.2, hp.2⟩
  exact ⟨FormX.pnpFree_elim _ hf, (FormX.depth_elim_le _).trans hd⟩

/-- Every bound hypothesis above has a witness (Appendix E). -/
example : (∀ r ∈ [MajRect.one], (r : MajRect Bool).Good 0) ∧
    ∃ (t : Fin 1 → TermX Bool) (R : Fin 1 → MajRects Bool),
      (∀ j, (t j).pnpFree = true ∧ (t j).depth ≤ 1) ∧ (∀ j, (R j).Good 0) := by
  refine ⟨by simp [MajRect.good_one], fun _ => TermX.one,
    fun _ => MajRects.single MajRect.one, ?_⟩
  simp [TermX.pnpFree, TermX.depth, MajRects.good_single, MajRect.good_one]

end Transformer.CRASP
