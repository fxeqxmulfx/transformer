/-
# Rectangles for the two-variable translation

arXiv:2506.16055v3, Appendix E, proof of `thm:majtwo_to_tlc`.
A rectangle tests one temporal formula at each variable and selects some
of the three order regions. Signed sums of rectangles implement the
paper's disjunctive normal form without a disjointness side condition.
-/

import Transformer.CRASP.MajTwoOfTLC
import Transformer.CRASP.ExtensionsElim

namespace Transformer.CRASP

universe u
variable {σ : Type u}

/-- A product of unary tests restricted to order regions (Appendix E). -/
structure MajRect (σ : Type u) where
  left : Form σ
  right : Form σ
  before : Bool
  equal : Bool
  after : Bool

namespace MajRect

/-- The order region selected at the two positions (Appendix E). -/
def region (r : MajRect σ) (i j : ℕ) : Bool :=
  if i < j then r.before else if i = j then r.equal else r.after

/-- The rectangle containing every pair (Appendix E). -/
def one : MajRect σ := ⟨Form.topAt 0, Form.topAt 0, true, true, true⟩

/-- A unary test placed at its named variable (Appendix E). -/
def unary (v : Var) (φ : Form σ) : MajRect σ :=
  match v with
  | .x => ⟨φ, Form.topAt 0, true, true, true⟩
  | .y => ⟨Form.topAt 0, φ, true, true, true⟩

/-- The atom `v < u`, including the two always-false self comparisons. -/
def order (v u : Var) : MajRect σ :=
  ⟨Form.topAt 0, Form.topAt 0, decide (v = .x ∧ u = .y), false,
    decide (v = .y ∧ u = .x)⟩

/-- Intersecting rectangles multiplies their indicators (Appendix E). -/
def mul (r t : MajRect σ) : MajRect σ :=
  ⟨.and r.left t.left, .and r.right t.right,
    r.before && t.before, r.equal && t.equal, r.after && t.after⟩

/-- The unary tests are PNP-free and have depth at most `d` (Appendix E). -/
def Good (r : MajRect σ) (d : ℕ) : Prop :=
  r.left.pnpFree = true ∧ r.right.pnpFree = true ∧
    r.left.depth ≤ d ∧ r.right.depth ≤ d

/-- The universal rectangle has depth-zero unary tests (Appendix E). -/
theorem good_one (d : ℕ) : (one : MajRect σ).Good d := by
  simp [Good, one, Form.topAt, Form.pnpFree, Term.pnpFree, Form.depth, Term.depth]

/-- A unary rectangle keeps its formula's bounds (Appendix E). -/
theorem good_unary {φ : Form σ} {d : ℕ} (hp : φ.pnpFree = true) (hd : φ.depth ≤ d)
    (v : Var) : (unary v φ).Good d := by
  cases v <;>
    simp [Good, unary, Form.topAt, Form.pnpFree, Term.pnpFree, Form.depth, Term.depth, hp, hd]

/-- Order atoms require no unary counting (Appendix E). -/
theorem good_order (v u : Var) (d : ℕ) : (order (σ := σ) v u).Good d := good_one d

/-- Intersecting rectangles preserves their unary bounds (Appendix E). -/
theorem good_mul {r t : MajRect σ} {d : ℕ} (hr : r.Good d) (ht : t.Good d) :
    (r.mul t).Good d := by
  rcases hr with ⟨hl, hr, hld, hrd⟩
  rcases ht with ⟨tl, tr, tld, trd⟩
  simp [Good, mul, Form.pnpFree, Form.depth, hl, hr, tl, tr, hld, hrd, tld, trd]

variable [DecidableEq σ]

/-- The integer indicator of a rectangle (Appendix E). -/
def value (r : MajRect σ) (w : List σ) (i j : ℕ) : ℤ :=
  if r.left.sat w i && r.right.sat w j && r.region i j then 1 else 0

/-- The universal rectangle has indicator one (Appendix E). -/
@[simp] theorem value_one (w : List σ) (i j : ℕ) :
    (one : MajRect σ).value w i j = 1 := by
  simp [value, one, region]

/-- A unary rectangle has the original formula's indicator (Appendix E). -/
theorem value_unary (v : Var) (φ : Form σ) (w : List σ) (ξ : Var → ℕ) :
    (unary v φ).value w (ξ .x) (ξ .y) = if φ.sat w (ξ v) then 1 else 0 := by
  cases v <;> simp [value, unary, region]

/-- The order rectangle has the corresponding comparison's indicator (Appendix E). -/
theorem value_order (v u : Var) (w : List σ) (ξ : Var → ℕ) :
    (order (σ := σ) v u).value w (ξ .x) (ξ .y) = if ξ v < ξ u then 1 else 0 := by
  cases v <;> cases u <;> simp [value, order, region]
  split_ifs <;> omega

/-- Intersections multiply the two indicators (Appendix E). -/
theorem value_mul (r t : MajRect σ) (w : List σ) (i j : ℕ) :
    (r.mul t).value w i j = r.value w i j * t.value w i j := by
  have hreg : (r.mul t).region i j = (r.region i j && t.region i j) := by
    simp only [region, mul]
    split_ifs <;> rfl
  simp only [value, hreg]
  simp only [mul, Form.sat]
  cases r.left.sat w i <;> cases r.right.sat w j <;> cases r.region i j <;>
    cases t.left.sat w i <;> cases t.right.sat w j <;> cases t.region i j <;> rfl

end MajRect

/-- Positive and negative rectangle lists for a binary formula (Appendix E). -/
structure MajRects (σ : Type u) where
  positive : List (MajRect σ)
  negative : List (MajRect σ)

namespace MajRects

/-- A single rectangle, with positive coefficient (Appendix E). -/
def single (r : MajRect σ) : MajRects σ := ⟨[r], []⟩

/-- Complementation subtracts the representation from the constant one. -/
def neg (R : MajRects σ) : MajRects σ := ⟨MajRect.one :: R.negative, R.positive⟩

/-- Products of rectangle lists, for conjunction (Appendix E). -/
def products (A B : List (MajRect σ)) : List (MajRect σ) :=
  A.flatMap fun r => B.map r.mul

/-- Multiplication of the two signed sums (Appendix E). -/
def mul (R S : MajRects σ) : MajRects σ :=
  ⟨products R.positive S.positive ++ products R.negative S.negative,
    products R.positive S.negative ++ products R.negative S.positive⟩

/-- Every unary test in the representation has the indicated depth bound. -/
def Good (R : MajRects σ) (d : ℕ) : Prop :=
  ∀ r ∈ R.positive ++ R.negative, r.Good d

variable [DecidableEq σ]

/-- Evaluation of the signed sum (Appendix E). -/
def value (R : MajRects σ) (w : List σ) (i j : ℕ) : ℤ :=
  (R.positive.map fun r => r.value w i j).sum -
    (R.negative.map fun r => r.value w i j).sum

/-- A singleton representation has its rectangle's value (Appendix E). -/
@[simp] theorem value_single (r : MajRect σ) (w : List σ) (i j : ℕ) :
    (single r).value w i j = r.value w i j := by simp [value, single]

/-- The normalized complement subtracts the indicator from one (Appendix E). -/
@[simp] theorem value_neg (R : MajRects σ) (w : List σ) (i j : ℕ) :
    R.neg.value w i j = 1 - R.value w i j := by simp [value, neg]; ring

/-- The product list evaluates to the product of the sums (Appendix E). -/
theorem value_products (A B : List (MajRect σ)) (w : List σ) (i j : ℕ) :
    ((products A B).map fun r => r.value w i j).sum =
      (A.map fun r => r.value w i j).sum * (B.map fun r => r.value w i j).sum := by
  induction A with
  | nil => simp [products]
  | cons r A ih =>
      have hB : ((B.map r.mul).map fun t => t.value w i j).sum =
          r.value w i j * (B.map fun t => t.value w i j).sum := by
        simp only [List.map_map, Function.comp_def, MajRect.value_mul]
        exact List.sum_map_mul_left B (fun t => t.value w i j) (r.value w i j)
      simp only [products, List.flatMap_cons, List.map_append, List.sum_append]
      change ((B.map r.mul).map fun t => t.value w i j).sum +
          ((products A B).map fun t => t.value w i j).sum = _
      rw [hB, ih, List.map_cons, List.sum_cons, add_mul]

/-- Normalized conjunction multiplies the signed sums (Appendix E). -/
@[simp] theorem value_mul (R S : MajRects σ) (w : List σ) (i j : ℕ) :
    (R.mul S).value w i j = R.value w i j * S.value w i j := by
  simp only [value, mul, List.map_append, List.sum_append, value_products]
  ring

end MajRects

/-- The bounds used by the rectangle operations are satisfiable (Appendix E). -/
example : (MajRect.unary .x (Form.sym true)).Good 0 ∧
    ((MajRect.one : MajRect Bool).mul MajRect.one).Good 0 :=
  ⟨MajRect.good_unary rfl le_rfl .x, MajRect.good_mul (MajRect.good_one 0) (MajRect.good_one 0)⟩

end Transformer.CRASP
