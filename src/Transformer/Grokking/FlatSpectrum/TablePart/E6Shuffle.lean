module

public import Mathlib.Logic.Function.Iterate

/-!
# PT6: the shuffle that splits E6

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6, and
`experiments/convex_stress/src/vendor/arithmetic_data.rs`. The experiment E6 shuffles the 9409 cells
`a · 97 + b` by SplitMix64 seeded with 1 and trains on the first 4704.

This file states the shuffle over `ℕ`, whose arithmetic the kernel runs on big numbers. `word t` is
the word number `t` of the generator (`word`). The array is one number, 14 bits per entry, each
stored as its difference from the identity permutation, so that the number `0` is the array
`0, 1, 2, …` (`get`, `swap`). `tick` is one draw of the inner loop of the shuffle.

The kernel evaluates arguments lazily, so a plain iteration of 9408 steps would stack up 9408
unevaluated calls. `iterFast` forces the state after every step and recurses on the exponent, with
depth `k` for `2^k` steps, and `iterFast_eq` says it computes the plain iteration.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

/-- The output function of SplitMix64 on a 64-bit state. -/
def mix (z : ℕ) : ℕ :=
  let z₁ := (z ^^^ (z >>> 30)) * 0xBF58476D1CE4E5B9 % 2 ^ 64
  let z₂ := (z₁ ^^^ (z₁ >>> 27)) * 0x94D049BB133111EB % 2 ^ 64
  z₂ ^^^ (z₂ >>> 31)

/-- Word number `t` of SplitMix64 seeded with 1, counting from 0. The state after `t + 1` steps is
`1 + (t + 1) γ` modulo `2^64`. -/
def word (t : ℕ) : ℕ := mix ((1 + (t + 1) * 0x9E3779B97F4A7C15) % 2 ^ 64)

/-- Entry `i` of the array packed into `A`: 14 bits, stored as the difference from `i`. -/
def get (A i : ℕ) : ℕ := ((A >>> (14 * i)) &&& 16383) ^^^ i

/-- The array `A` with its entries `i` and `j` swapped. -/
def swap (A i j : ℕ) : ℕ :=
  let d := get A i ^^^ get A j
  A ^^^ (d <<< (14 * i)) ^^^ (d <<< (14 * j))

/-- One draw of the shuffle on `(end, words drawn, array)`. The word is rejected below the threshold
`2^64 mod n`, and the loop is over at `end = 0`. -/
def tick (s : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  if s.1 = 0 then s else
    let x := word s.2.1
    if (2 ^ 64 - (s.1 + 1)) % (s.1 + 1) ≤ x then
      (s.1 - 1, s.2.1 + 1, swap s.2.2 s.1 (x % (s.1 + 1)))
    else (s.1, s.2.1 + 1, s.2.2)

/-- Evaluates `x` before going on: matching on a number makes the kernel compute it. -/
def strict {α : Type} (x : ℕ) (k : ℕ → α) : α :=
  match x with
  | 0 => k 0
  | n + 1 => k (n + 1)

lemma strict_eq {α : Type} (x : ℕ) (k : ℕ → α) : strict x k = k x := by
  cases x <;> rfl

/-- Evaluates the three fields of a state before going on. -/
def force (s : ℕ × ℕ × ℕ) (k : ℕ × ℕ × ℕ → ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  strict s.1 fun a => strict s.2.1 fun b => strict s.2.2 fun c => k (a, b, c)

lemma force_eq (s : ℕ × ℕ × ℕ) (k : ℕ × ℕ × ℕ → ℕ × ℕ × ℕ) : force s k = k s := by
  simp [force, strict_eq]

/-- `2^k` steps of `f`, with the state forced after each step. -/
def iterFast (f : ℕ × ℕ × ℕ → ℕ × ℕ × ℕ) : ℕ → ℕ × ℕ × ℕ → ℕ × ℕ × ℕ
  | 0, s => force (f s) id
  | k + 1, s => force (iterFast f k s) (iterFast f k)

/-- `iterFast` computes the plain iteration. -/
theorem iterFast_eq (f : ℕ × ℕ × ℕ → ℕ × ℕ × ℕ) (k : ℕ) (s : ℕ × ℕ × ℕ) :
    iterFast f k s = f^[2 ^ k] s := by
  induction k generalizing s with
  | zero => simp [iterFast, force_eq]
  | succ k ih =>
    rw [iterFast, force_eq, ih, ih, Nat.pow_succ, Nat.mul_two, Function.iterate_add_apply]

/-- The state after the whole shuffle, computed through `iterFast`: `9408 = 2^13 + 2^10 + 2^7 + 2^6`
draws, if no word is rejected. -/
def shuffleFast : ℕ × ℕ × ℕ :=
  iterFast tick 6 (iterFast tick 7 (iterFast tick 10 (iterFast tick 13 (9408, 0, 0))))

/-- One step of FNV-1a over the entries of the array, on `(entries read, hash, array)`. -/
def fnvStep (s : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  (s.1 + 1, (s.2.1 ^^^ get s.2.2 s.1) * 0x100000001b3 % 2 ^ 64, s.2.2)

/-- The FNV-1a hash of all `9409 = 2^13 + 2^10 + 2^7 + 2^6 + 2^0` entries after the shuffle. -/
def fnvFast : ℕ × ℕ × ℕ :=
  iterFast fnvStep 0 (iterFast fnvStep 6 (iterFast fnvStep 7 (iterFast fnvStep 10
    (iterFast fnvStep 13 (0, 0xcbf29ce484222325, shuffleFast.2.2)))))

end Transformer.Grokking.FlatSpectrum.TablePart.E6
