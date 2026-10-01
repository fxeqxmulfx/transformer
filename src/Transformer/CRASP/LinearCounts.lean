/-
# Signed linear combinations of past counts

arXiv:2506.16055v3, Appendix B.2, proof of `thm:rtfr_to_TLCl`.
Attention numerators and denominators are integer-weighted counts of
finite activation states. Moving negative coefficients to the other side
expresses their comparisons in the original, subtraction-free syntax.
-/

import Transformer.CRASP.Indicator
import Transformer.CRASP.Conjunctions

namespace Transformer.CRASP

universe u
variable {σ : Type u}

/-- A signed constant plus a finite list of weighted past counts (B.2). -/
structure LinearCount (σ : Type u) where
  constant : ℤ
  weights : List (ℤ × Form σ)

namespace LinearCount

/-- Multiplying an affine count by an integer (Appendix B.2). -/
def scale (z : ℤ) (A : LinearCount σ) : LinearCount σ :=
  ⟨z * A.constant, A.weights.map fun x => (z * x.1, x.2)⟩

/-- Adding affine counts (Appendix B.2). -/
def add (A B : LinearCount σ) : LinearCount σ :=
  ⟨A.constant + B.constant, A.weights ++ B.weights⟩

/-- Subtracting affine counts before eliminating signed coefficients (B.2). -/
def sub (A B : LinearCount σ) : LinearCount σ := A.add (B.scale (-1))

/-- The formulas beneath the counts stay in depth `k` (Appendix B.2). -/
def Good (A : LinearCount σ) (k : ℕ) : Prop :=
  ∀ x ∈ A.weights, x.2 ∈ TLCl σ k

/-- The nonnegative part of the weighted terms (Appendix B.2). -/
def positive (A : LinearCount σ) : List (Term σ) :=
  A.weights.flatMap fun x => List.replicate x.1.toNat (.countL x.2)

/-- The comparison `A < 0`, after moving negative terms to the right (B.2). -/
def ltZero (A : LinearCount σ) : Form σ :=
  Form.ltSum A.constant.toNat A.positive [] (-A.constant).toNat (A.scale (-1)).positive []

/-- Linear operations preserve the subformula bound (Appendix B.2). -/
theorem good_scale {A : LinearCount σ} {k : ℕ} (hA : A.Good k) (z : ℤ) :
    (A.scale z).Good k := by
  rintro x hx
  obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
  exact hA y hy

/-- Concatenating weighted counts preserves their bound (Appendix B.2). -/
theorem good_add {A B : LinearCount σ} {k : ℕ} (hA : A.Good k) (hB : B.Good k) :
    (A.add B).Good k := by
  intro x hx
  rcases List.mem_append.mp hx with hx | hx
  · exact hA x hx
  · exact hB x hx

/-- Subtraction preserves the subformula bound (Appendix B.2). -/
theorem good_sub {A B : LinearCount σ} {k : ℕ} (hA : A.Good k) (hB : B.Good k) :
    (A.sub B).Good k := good_add hA (good_scale hB (-1))

/-- One layer of counting suffices for a signed comparison (Appendix B.2). -/
theorem ltZero_mem {A : LinearCount σ} {k : ℕ} (hA : A.Good k) :
    A.ltZero ∈ TLCl σ (k + 1) := by
  have hp {B : LinearCount σ} (hB : B.Good k) :
      ∀ t ∈ B.positive, t.past = true ∧ t.pnpFree = true ∧ t.depth ≤ k + 1 := by
    intro t ht
    obtain ⟨x, hx, ht⟩ := List.mem_flatMap.mp ht
    have heq : t = Term.countL x.2 := List.eq_of_mem_replicate ht
    subst t
    obtain ⟨h₁, h₂, h₃⟩ := hB x hx
    exact ⟨h₁, h₂, Nat.add_le_add_right h₃ 1⟩
  exact Form.ltSum_mem_TLCl _ _ (hp hA) (by simp)
    (hp (good_scale hA (-1))) (by simp)

variable [DecidableEq σ]

/-- The intended integer evaluation of a signed count (Appendix B.2). -/
def val (A : LinearCount σ) (w : List σ) (i : ℕ) : ℤ :=
  A.constant + (A.weights.map fun x => x.1 * (Term.countL x.2 |>.val w i : ℤ)).sum

/-- Scaling agrees with integer multiplication (Appendix B.2). -/
@[simp] theorem val_scale (A : LinearCount σ) (z : ℤ) (w : List σ) (i : ℕ) :
    (A.scale z).val w i = z * A.val w i := by
  simp only [val, scale, List.map_map, Function.comp_def, mul_assoc]
  rw [List.sum_map_mul_left]
  ring

/-- Concatenation agrees with integer addition (Appendix B.2). -/
@[simp] theorem val_add (A B : LinearCount σ) (w : List σ) (i : ℕ) :
    (A.add B).val w i = A.val w i + B.val w i := by
  simp [val, add, List.map_append, List.sum_append]
  ring

/-- Subtraction agrees with integer subtraction (Appendix B.2). -/
@[simp] theorem val_sub (A B : LinearCount σ) (w : List σ) (i : ℕ) :
    (A.sub B).val w i = A.val w i - B.val w i := by
  simp [sub]; ring

/-- The formula computes the strict signed comparison (Appendix B.2). -/
theorem sat_ltZero (A : LinearCount σ) (w : List σ) (i : ℕ) :
    A.ltZero.sat w i = decide (A.val w i < 0) := by
  have hpos (B : LinearCount σ) :
      ((B.positive.map (·.val w i)).sum : ℤ) =
        (B.weights.map fun x => (x.1.toNat : ℤ) * (Term.countL x.2 |>.val w i : ℤ)).sum := by
    unfold positive
    induction B.weights with
    | nil => simp
    | cons x L ih =>
        simp only [List.flatMap_cons, List.map_append, List.sum_append,
          List.map_replicate, List.sum_replicate, List.map_cons, List.sum_cons]
        rw [ih]
        simp
  have hdiff : ((A.positive.map (·.val w i)).sum : ℤ) -
      (((A.scale (-1)).positive.map (·.val w i)).sum : ℤ) =
        (A.weights.map fun x => x.1 * (Term.countL x.2 |>.val w i : ℤ)).sum := by
    rw [hpos, hpos]
    simp only [scale, List.map_map, Function.comp_def, neg_one_mul]
    induction A.weights with
    | nil => simp
    | cons x L ih =>
        simp only [List.map_cons, List.sum_cons]
        have hx : (x.1.toNat : ℤ) * (Term.countL x.2 |>.val w i : ℤ) -
            ((-x.1).toNat : ℤ) * (Term.countL x.2 |>.val w i : ℤ) =
            x.1 * (Term.countL x.2 |>.val w i : ℤ) := by
          rw [← sub_mul, Int.toNat_sub_toNat_neg]
        linarith
  rw [ltZero, Form.sat_ltSum]
  simp only [List.countP_nil, Nat.add_zero]
  apply decide_eq_decide.mpr
  have hcast (a b : ℕ) : a < b ↔ (a : ℤ) < (b : ℤ) := by omega
  rw [hcast]
  push_cast
  simp only [List.map_map, Function.comp_def]
  have hc := Int.toNat_sub_toNat_neg A.constant
  unfold val
  constructor <;> intro h <;> linarith

/-- The weighted-count bound is satisfiable (Appendix B.2). -/
example : (LinearCount.mk 3 [(-2, Form.sym true)]).Good 0 := by
  simp [Good, TLCl, Form.past, Form.pnpFree, Form.depth]

end LinearCount
end Transformer.CRASP
