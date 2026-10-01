/-
# Reading a binary representation on the diagonal

arXiv:2506.16055v3, Appendix E, `thm:majtwo_to_tlc`: if only one variable
is free, assigning both variables the same position leaves the answer
unchanged. Boolean normalization then gives a unary temporal formula.
-/

import Transformer.CRASP.MajTwoRectBounds
import Transformer.CRASP.BooleanFormula

namespace Transformer.CRASP

universe u
variable {σ : Type u}

/-- A rectangle on `x = y`, where its two unary tests share a position. -/
def MajRect.diagonal (r : MajRect σ) : Form σ :=
  if r.equal then .and r.left r.right else .lt .one .one

/-- The diagonal does not increase depth or introduce a PNP (Appendix E). -/
theorem MajRect.diagonal_mem {r : MajRect σ} {d : ℕ} (hr : r.Good d) :
    r.diagonal ∈ TLC σ d := by
  cases he : r.equal <;> simp [MajRect.diagonal, he, TLC, Form.pnpFree, Term.pnpFree,
    Form.depth, Term.depth, hr.1, hr.2.1, hr.2.2.1, hr.2.2.2]

/-- Integer mass of a list of Boolean indicators (Appendix E). -/
def boolMass (b : List Bool) : ℤ := (b.map fun v => if v then (1 : ℤ) else 0).sum

/-- The diagonal of a signed rectangle representation (Appendix E). -/
def MajRects.diagonal (R : MajRects σ) : Form σ :=
  Form.booleanComb (R.positive.map MajRect.diagonal ++ R.negative.map MajRect.diagonal)
    fun b => decide (0 < boolMass (b.take R.positive.length) - boolMass (b.drop R.positive.length))

/-- Boolean normalization keeps the diagonal in the same temporal fragment. -/
theorem MajRects.diagonal_mem {R : MajRects σ} {d : ℕ} (hR : R.Good d) :
    R.diagonal ∈ TLC σ d := by
  have h : ∀ φ ∈ R.positive.map MajRect.diagonal ++ R.negative.map MajRect.diagonal,
      φ ∈ TLC σ d := by
    simp only [List.mem_append, List.mem_map]
    rintro _ (⟨r, hr, rfl⟩ | ⟨r, hr, rfl⟩)
    · exact MajRect.diagonal_mem (hR r (List.mem_append_left _ hr))
    · exact MajRect.diagonal_mem (hR r (List.mem_append_right _ hr))
  exact ⟨Form.pnpFree_booleanComb _ _ fun φ hφ => (h φ hφ).1,
    Form.depth_booleanComb_le _ _ d fun φ hφ => (h φ hφ).2⟩

/-- The diagonal hypotheses have witnesses (Appendix E). -/
example : (MajRect.one : MajRect Bool).Good 0 ∧
    (MajRects.single (MajRect.one : MajRect Bool)).Good 0 :=
  ⟨MajRect.good_one 0, MajRects.good_single (MajRect.good_one 0)⟩

variable [DecidableEq σ]

/-- The unary test is exactly the rectangle's diagonal indicator. -/
theorem MajRect.value_diagonal (r : MajRect σ) (w : List σ) (i : ℕ) :
    (if r.diagonal.sat w i then (1 : ℤ) else 0) = r.value w i i := by
  cases he : r.equal <;>
    simp [MajRect.diagonal, he, MajRect.value, MajRect.region, Form.sat, Term.val]

/-- The diagonal formula tests the sign of the original signed sum. -/
theorem MajRects.sat_diagonal (R : MajRects σ) (w : List σ) (i : ℕ) :
    R.diagonal.sat w i = decide (0 < R.value w i i) := by
  rw [MajRects.diagonal, Form.sat_booleanComb]
  have hlen : ((R.positive.map MajRect.diagonal).map fun φ => φ.sat w i).length =
      R.positive.length := by simp
  simp only [List.map_append]
  rw [List.take_left' hlen, List.drop_left' hlen]
  simp only [boolMass, List.map_map, Function.comp_def, MajRect.value_diagonal, MajRects.value]
  rfl

end Transformer.CRASP
