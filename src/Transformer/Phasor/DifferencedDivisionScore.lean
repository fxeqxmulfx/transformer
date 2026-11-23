module

public import Transformer.Phasor.ModularChainScore
public import Mathlib.FieldTheory.Finite.Basic

/-!
# First- and second-difference division on the GPT catalog

Source: DiffGPT `div97.token_table`, `make_tensors` and
`current_grokking.adapters.differenced_division` at 5a37c515. The positive
orders have a unit level grid and a fixed zero prefix. Two encoded inputs
are visible. The loss supervises the final encoded quotient at position one.
-/

@[expose] public section

namespace Transformer.Phasor.DifferencedDivisionScore

inductive Order where | first | second

def Order.offset : Order → ℤ
  | .first => 128
  | .second => 256

/-- Differences of successive levels, starting at the supplied prior level. -/
def difference (prior : ℤ) : List ℤ → List ℤ
  | [] => []
  | x :: tail => (x - prior) :: difference x tail

def inputs (o : Order) (a b : ℤ) : List ℤ :=
  [o.offset + a, o.offset + (match o with | .first => b-a | .second => b-2*a)]

def target (o : Order) (a b q : ℤ) : ℤ :=
  o.offset + (match o with | .first => q-b | .second => q-2*b+a)

def decode (o : Order) (a b token : ℤ) : ℤ :=
  token - o.offset + (match o with | .first => b | .second => 2*b-a)

/-- Applying the source difference operator once or twice to the equation gives
exactly the two prompt tokens and the masked target used by the native port. -/
theorem codec (o : Order) (a b q : ℤ) :
    ((match o with | .first => difference 0 [a,b,q]
                   | .second => difference 0 (difference 0 [a,b,q])).map (o.offset + ·)) =
      inputs o a b ++ [target o a b q] := by
  cases o <;> simp [difference, inputs, target, Order.offset]
  constructor <;> ring

/-- All three tokens lie in the full source vocabulary for both orders.
Its unrestricted argmax can predict any token in that vocabulary. -/
theorem in_vocabulary (o : Order) (a b q : ℤ)
    (ha : 0 ≤ a ∧ a ≤ 96) (hb : 1 ≤ b ∧ b ≤ 96) (hq : 0 ≤ q ∧ q ≤ 96) :
    ∀ x ∈ inputs o a b ++ [target o a b q], 0 ≤ x ∧ x < 2 * o.offset := by
  cases o <;> simp only [inputs, target, Order.offset, List.cons_append, List.nil_append,
    List.mem_cons, List.not_mem_nil, or_false] <;> intro x h <;> rcases h with rfl|rfl|rfl <;> omega

/-- A context determines the two operands before seeing an answer, even when
some encoded differences are negative before the vocabulary shift. -/
theorem prefix_injective (o : Order) (a b c d : ℤ) :
    inputs o a b = inputs o c d ↔ a = c ∧ b = d := by
  cases o <;> simp [inputs, Order.offset] <;> omega

/-- For an arbitrary causal prefix predictor, the original masked dense score
is exact precisely when undoing the difference recovers the quotient. -/
theorem exact_iff (o : Order) (predict : List ℤ → ℤ) (a b q : ℤ) :
    (∀ i y, (ModularChainScore.targets 1 (target o a b q))[i]? = some (some y) →
      predict ((inputs o a b).take (i+1)) = y) ↔ decode o a b (predict (inputs o a b)) = q := by
  have decoded (z : ℤ) : z = target o a b q ↔ decode o a b z = q := by
    cases o <;> simp [target, decode, Order.offset] <;> omega
  rw [← decoded]
  have full : (inputs o a b).take 2 = inputs o a b := by simp [inputs]
  constructor
  · intro h
    simpa only [full] using h 1 _ ((ModularChainScore.supervised_iff ..).2 ⟨rfl,rfl⟩)
  · intro h i y hy
    obtain ⟨rfl,rfl⟩ := (ModularChainScore.supervised_iff ..).1 hy
    simpa only [full] using h

local instance : Fact (Nat.Prime 97) := ⟨by decide⟩

/-- The native data generator enumerates every nonzero divisor and implements
the source modular inverse using its Fermat exponent. -/
theorem quotient (a b : ZMod 97) (nonzero : b ≠ 0) : a * b^95 = a / b := by
  have unit : b * b^95 = 1 := by
    rw [mul_comm, ← pow_succ]
    exact ZMod.pow_card_sub_one_eq_one nonzero
  rw [div_eq_mul_inv, ZMod.inv_eq_of_mul_eq_one _ _ _ unit]

example : (7 : ZMod 97) * 3^95 = 7 / 3 := quotient _ _ (by decide)

example : ∀ x ∈ inputs .second 7 3 ++ [target .second 7 3 67], 0 ≤ x ∧ x < 512 :=
  in_vocabulary _ _ _ _ (by decide) (by decide) (by decide)

example : inputs .second 7 3 = [263,245] ∧ target .second 7 3 67 = 324 ∧
    decode .second 7 3 324 = 67 := by decide

example : ∀ (i : ℕ) (y : ℤ), (ModularChainScore.targets 1 (target .first 7 3 67))[i]? = some (some y) →
    (192 : ℤ) = y :=
  (exact_iff .first (fun _ => 192) 7 3 67).2 (by decide)

end Transformer.Phasor.DifferencedDivisionScore
