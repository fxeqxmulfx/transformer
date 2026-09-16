/-
# A neutral letter does not change piecewise testability

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E ("Depth Hierarchy"): the language
`E_k = del⁻¹(L_k)` of `thm:tlclpos_depth_hierarchy`.

The theorem asserts that `E_{k+1}` is definable in `TL[◁#]^pos_{k+1}`, but its
proof argues only the other half.  The missing half starts here: a pattern
`σ₁ ⋯ σ_k` over `Σ` is a subsequence of `del(w)` exactly when it is one of
`w`, because no letter of the pattern is `e`.  So `del⁻¹` takes a Boolean
combination of 𝒥-expressions over `Σ` to the same combination over
`Σ ∪ {e}`, and `E_k` is `k`-piecewise testable because `L_k` is.  `del` is
`List.reduceOption`, with `e` written `none`.
-/

import Mathlib.Data.List.ReduceOption
import Transformer.CRASP.PiecewiseTestable

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- A pattern over `Σ` fits into a string over `Σ ∪ {e}` exactly when it fits
into that string with its neutral letters deleted. -/
theorem map_some_sublist_iff (s : List σ) (w : List (Option σ)) :
    (s.map some).Sublist w ↔ s.Sublist w.reduceOption := by
  induction w generalizing s with
  | nil => simp
  | cons x w ih =>
      cases x with
      | none =>
          rw [List.reduceOption_cons_of_none, ← ih, List.sublist_cons_iff]
          refine ⟨?_, Or.inl⟩
          rintro (h | ⟨r, hr, -⟩)
          · exact h
          · cases s <;> simp at hr
      | some a =>
          rw [List.reduceOption_cons_of_some, List.sublist_cons_iff, List.sublist_cons_iff, ih]
          cases s with
          | nil => simp
          | cons b t => simp [ih]

/-- A Boolean combination of 𝒥-expressions over `Σ`, read over `Σ ∪ {e}`. -/
def PT.mapSome : PT σ → PT (Option σ)
  | .jexpr s => .jexpr (s.map some)
  | .neg e => .neg e.mapSome
  | .and e₁ e₂ => .and e₁.mapSome e₂.mapSome

@[simp] theorem PT.width_mapSome (e : PT σ) : e.mapSome.width = e.width := by
  induction e with
  | jexpr s => simp [PT.mapSome, PT.width]
  | neg e ih => simpa [PT.mapSome, PT.width] using ih
  | and e₁ e₂ ih₁ ih₂ => simp [PT.mapSome, PT.width, ih₁, ih₂]

/-- Read over `Σ ∪ {e}`, a Boolean combination of 𝒥-expressions defines the
preimage of its language under the deletion of `e`. -/
theorem PT.lang_mapSome (e : PT σ) : e.mapSome.lang = List.reduceOption ⁻¹' e.lang := by
  induction e with
  | jexpr s =>
      ext w
      exact map_some_sublist_iff s w
  | neg e ih => rw [PT.mapSome, PT.lang, PT.lang, ih, Set.preimage_compl]
  | and e₁ e₂ ih₁ ih₂ => rw [PT.mapSome, PT.lang, PT.lang, ih₁, ih₂, Set.preimage_inter]

/-- **A neutral letter preserves `k`-piecewise testability.**  If `L` is
`k`-piecewise testable, so is `del⁻¹(L)`; in particular `E_k` is, since `L_k`
is (`lem:piecewise_testable`).

Source: arXiv:2506.16055v3, Appendix E, the definition of `E_k` before
`thm:tlclpos_depth_hierarchy`. -/
theorem KPiecewiseTestable.preimage_reduceOption {k : ℕ} {L : Set (List σ)}
    (h : KPiecewiseTestable k L) : KPiecewiseTestable k (List.reduceOption ⁻¹' L) := by
  obtain ⟨e, hwidth, rfl⟩ := h
  exact ⟨e.mapSome, e.width_mapSome ▸ hwidth, e.lang_mapSome⟩

/-- The hypothesis of `KPiecewiseTestable.preimage_reduceOption` is
satisfiable: `L_1` is `1`-piecewise testable. -/
example : KPiecewiseTestable 1 (altPlus false 1) := kPiecewiseTestable_altPlus 1 Nat.one_pos

end CRASP
end Transformer
