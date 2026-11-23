module

public import Mathlib.Algebra.Order.Group.Unbundled.Int
public import Mathlib.Tactic
public import Transformer.MRCR.Ordinal

/-!
# Error margins for occurrence counts learned from a causal prefix

Phasor ordinal study, adapting cumulative sums in Percepta's *From programs into
transformer weights*, “Cumulative sums via uniform averages”, and selection from
the start in MRCR, arXiv:2409.12640v2. A token may contribute a fractional write
to the address of a preceding request. The analysis applies independently to
each key and prefix. Only the outer scorer uses the true response boundaries.
-/

@[expose] public section
namespace Transformer.Phasor.OrdinalCounter

variable {A : Type*}

/-- Write mass accumulated through a prefix at a chosen address. -/
def mass (weight : A → ℝ) (xs : List A) : ℝ := (xs.map weight).sum

/-- Number of relevant writes through the same prefix, counted from its start. -/
def count (relevant : A → Bool) (xs : List A) : ℕ := (xs.filter relevant).length

/-- Missed mass at true writes and spurious mass at other tokens have distinct
costs. The bound grows with the actual numbers of the two kinds of tokens. -/
theorem mass_error (weight : A → ℝ) (relevant : A → Bool) (xs : List A) (miss stray : ℝ)
    (localError : ∀ x ∈ xs, if relevant x then |weight x - 1| ≤ miss else |weight x| ≤ stray) :
    |mass weight xs - count relevant xs| ≤
      (count relevant xs : ℝ)*miss + (count (fun x => !(relevant x)) xs : ℝ)*stray := by
  induction xs with
  | nil => simp [mass, count]
  | cons x xs ih =>
    have tail := ih (fun y hy => localError y (List.mem_cons_of_mem x hy))
    have top := localError x List.mem_cons_self
    have m : mass weight (x::xs) = weight x + mass weight xs := rfl
    cases hx : relevant x with
    | true =>
      simp [hx] at top
      have yes : count relevant (x::xs) = count relevant xs + 1 := by simp [count, hx]
      have no : count (fun t => !(relevant t)) (x::xs) = count (fun t => !(relevant t)) xs := by
        simp [count, hx]
      rw [m, yes, no, Nat.cast_add, Nat.cast_one]
      calc
        _ = |(weight x-1) + (mass weight xs - count relevant xs)| := by congr 1; ring
        _ ≤ |weight x-1| + |mass weight xs - count relevant xs| := abs_add_le _ _
        _ ≤ _ := (add_le_add top tail).trans_eq (by ring)
    | false =>
      simp [hx] at top
      have no : count relevant (x::xs) = count relevant xs := by simp [count, hx]
      have yes : count (fun t => !(relevant t)) (x::xs) = count (fun t => !(relevant t)) xs + 1 := by
        simp [count, hx]
      rw [m, no, yes, Nat.cast_add, Nat.cast_one]
      calc
        _ = |weight x + (mass weight xs - count relevant xs)| := by congr 1; ring
        _ ≤ |weight x| + |mass weight xs - count relevant xs| := abs_add_le _ _
        _ ≤ _ := (add_le_add top tail).trans_eq (by ring)

/-- If the sum of the ordinal-query error and the worst stored-count error is
below half an occurrence, every record with a different true ordinal is
strictly farther from the query than the intended record. -/
theorem ordinal_margin (r s : ℕ) (different : r ≠ s) (q a b eps noise : ℝ)
    (query : |q - r| ≤ eps) (correct : |a - r| ≤ noise) (other : |b - s| ≤ noise)
    (room : eps + noise < 1/2) : |q-a| < |q-b| := by
  have nz : (r : ℤ) - (s : ℤ) ≠ 0 := by omega
  have ints := Int.one_le_abs nz
  have sep : (1:ℝ) ≤ |(r:ℝ) - s| := by exact_mod_cast ints
  have up := abs_add_le (q-r) ((r:ℝ)-a)
  have lo := abs_add_le ((r:ℝ)-q) (q-b)
  have lo' := abs_add_le ((r:ℝ)-b) (b-s)
  rw [sub_add_sub_cancel] at up lo lo'
  rw [abs_sub_comm (r:ℝ) a] at up
  rw [abs_sub_comm (r:ℝ) q] at lo
  linarith

example : |mass (fun b => if b then 0.95 else 0.01) [true,false,true,false,false] -
    count id [true,false,true,false,false]| ≤ 2*0.05+3*0.01 := by
  apply mass_error (fun b => if b then 0.95 else 0.01) id _ 0.05 0.01
  intro b _
  cases b <;> norm_num

example : |(2.1:ℝ)-2.2| < |(2.1:ℝ)-2.8| :=
  ordinal_margin 2 3 (by decide) _ _ _ 0.1 0.2 (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Phasor.OrdinalCounter
