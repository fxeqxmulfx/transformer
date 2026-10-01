/-
# Bounds for rectangle representations

arXiv:2506.16055v3, Appendix E, `thm:majtwo_to_tlc`: Boolean normalization
does not increase depth and does not introduce Parikh predicates.
-/

import Transformer.CRASP.MajTwoRect
import Transformer.CRASP.ExtensionsPnpFree

namespace Transformer.CRASP

universe u
variable {σ : Type u}

/-- Enlarging a depth bound preserves the rectangle conditions (Appendix E). -/
theorem MajRect.good_mono {r : MajRect σ} {d e : ℕ} (hr : r.Good d) (h : d ≤ e) :
    r.Good e := ⟨hr.1, hr.2.1, hr.2.2.1.trans h, hr.2.2.2.trans h⟩

namespace MajRects

/-- A single rectangle has its unary depth bound (Appendix E). -/
theorem good_single {r : MajRect σ} {d : ℕ} (hr : r.Good d) : (single r).Good d := by
  simpa [Good, single] using hr

/-- Enlarging a representation's depth bound is harmless (Appendix E). -/
theorem good_mono {R : MajRects σ} {d e : ℕ} (hr : R.Good d) (h : d ≤ e) : R.Good e :=
  fun r hm => MajRect.good_mono (hr r hm) h

/-- Complementation preserves the bound (Appendix E). -/
theorem good_neg {R : MajRects σ} {d : ℕ} (hr : R.Good d) : R.neg.Good d := by
  simp only [Good, neg, List.mem_append, List.mem_cons]
  rintro r ((rfl | h) | h)
  · exact MajRect.good_one d
  · exact hr r (List.mem_append_right _ h)
  · exact hr r (List.mem_append_left _ h)

/-- The pairwise product list preserves the unary bounds (Appendix E). -/
private theorem good_products {A B : List (MajRect σ)} {d : ℕ}
    (hA : ∀ r ∈ A, r.Good d) (hB : ∀ r ∈ B, r.Good d) :
    ∀ r ∈ products A B, r.Good d := by
  simp only [products, List.mem_flatMap, List.mem_map]
  rintro r ⟨a, ha, b, hb, rfl⟩
  exact MajRect.good_mul (hA a ha) (hB b hb)

/-- Conjunction preserves the bound (Appendix E). -/
theorem good_mul {R S : MajRects σ} {d : ℕ} (hR : R.Good d) (hS : S.Good d) :
    (R.mul S).Good d := by
  have Rp : ∀ r ∈ R.positive, r.Good d := fun r h => hR r (List.mem_append_left _ h)
  have Rn : ∀ r ∈ R.negative, r.Good d := fun r h => hR r (List.mem_append_right _ h)
  have Sp : ∀ r ∈ S.positive, r.Good d := fun r h => hS r (List.mem_append_left _ h)
  have Sn : ∀ r ∈ S.negative, r.Good d := fun r h => hS r (List.mem_append_right _ h)
  simp only [Good, mul, List.mem_append]
  rintro r ((h | h) | h | h)
  · exact good_products Rp Sp r h
  · exact good_products Rn Sn r h
  · exact good_products Rp Sn r h
  · exact good_products Rn Sp r h

end MajRects

/-- All bounds used above have witnesses (Appendix E). -/
example : (MajRects.single (MajRect.one : MajRect Bool)).Good 0 ∧ 0 ≤ 1 :=
  ⟨MajRects.good_single (MajRect.good_one 0), Nat.zero_le _⟩

end Transformer.CRASP
