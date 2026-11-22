module

public import Transformer.Phasor.SharedValues

/-!
# Causal geometric memory with a learned vocabulary table

Source: the geometric-memory trial in `experiments/phasor_value_transport/README.md`.
Each hidden coordinate advances one scalar recurrence over token values.
The forward scan begins at zero at every sequence boundary. The reverse
scan carries gradients back to every earlier occurrence of a token.
-/

@[expose] public section

namespace Transformer.Phasor

variable {α : Type*}

/-- The state after reading the first `n` values. A signed retention
coefficient can distinguish the order of inputs with identical counts. -/
def geometricState (retention : ℝ) (value : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n+1 => retention * geometricState retention value n + value n

/-- Segments compose at arbitrary lengths. Only the final state of the
first segment enters the second through a power of the retention factor. -/
theorem geometricState_split (r : ℝ) (value : ℕ → ℝ) (m n : ℕ) :
    geometricState r value (m+n) =
      r^n * geometricState r value m + geometricState r (fun i => value (m+i)) n := by
  induction n with
  | zero => simp [geometricState]
  | succ n ih =>
    rw [Nat.add_succ, geometricState, ih, geometricState, pow_succ]
    ring

/-- A zero-initialized vocabulary table gives zero at every position,
including after arbitrarily many repetitions of any token. -/
theorem geometricState_zero (r : ℝ) (n : ℕ) : geometricState r (fun _ => 0) n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [geometricState, ih]

/-- Retentions of magnitude at most one give at most linear growth for
bounded token values. The bound includes cumulative sums at unit retention. -/
theorem abs_geometricState_le (r M : ℝ) (value : ℕ → ℝ) (n : ℕ) (hr : |r| ≤ 1)
    (bounded : ∀ i < n, |value i| ≤ M) : |geometricState r value n| ≤ n * M := by
  induction n with
  | zero => simp [geometricState]
  | succ n ih =>
    have prior := ih (fun i hi => bounded i (by omega))
    have scale := mul_le_mul_of_nonneg_right hr (abs_nonneg (geometricState r value n))
    calc
      |geometricState r value (n+1)| ≤ |r| * |geometricState r value n| + |value n| := by
        simpa only [geometricState, abs_mul] using abs_add_le (r * geometricState r value n) (value n)
      _ ≤ (n+1 : ℕ) * M := by rw [Nat.cast_succ]; linarith [bounded n (by omega)]

/-- Starting at `i`, a token contributes to the next `length` output
gradients, with one more factor of retention at each later position. -/
def geometricTail (r : ℝ) (g : ℕ → ℝ) (i : ℕ) : ℕ → ℝ
  | 0 => 0
  | n+1 => g i + r * geometricTail r g (i+1) n

private theorem geometricParts (r : ℝ) (u adj : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ Finset.range n, (adj i - r * adj (i+1)) * geometricState r u (i+1) =
      (∑ i ∈ Finset.range n, adj i * u i) - r * adj n * geometricState r u n := by
  induction n with
  | zero => simp [geometricState]
  | succ n ih =>
    rw [Finset.sum_range_succ, Finset.sum_range_succ, ih, geometricState]
    ring

/-- Pairing an arbitrary change of all vocabulary values with the output
gradient equals the reverse-scan gradient grouped by token. Repeated tokens
contribute once per occurrence and a masked output can have zero gradient. -/
theorem geometricToken_adjoint [Fintype α] [DecidableEq α] (r : ℝ) (xs : ℕ → α)
    (change : α → ℝ) (g : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ Finset.range n, g i * geometricState r (fun j => change (xs j)) (i+1) =
      ∑ a, (∑ i ∈ Finset.range n with xs i = a, geometricTail r g i (n-i)) * change a := by
  let adj := fun i => geometricTail r g i (n-i)
  have stepRule (i : ℕ) (hi : i < n) : adj i - r * adj (i+1) = g i := by
    change geometricTail r g i (n-i) - r * geometricTail r g (i+1) (n-(i+1)) = g i
    rw [show n-i = (n-(i+1))+1 by omega, geometricTail]
    ring
  calc
    _ = ∑ i ∈ Finset.range n, (adj i - r * adj (i+1)) *
        geometricState r (fun j => change (xs j)) (i+1) := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [stepRule i (Finset.mem_range.mp hi)]
    _ = ∑ i ∈ Finset.range n, adj i * change (xs i) := by
      rw [geometricParts]
      simp [adj, geometricTail]
    _ = _ := by
      rw [← Finset.sum_fiberwise (Finset.range n) xs (fun i => adj i * change (xs i))]
      apply Finset.sum_congr rfl
      intro a _
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro i hi
      rw [(Finset.mem_filter.mp hi).2]

example : geometricState (1/2) (fun i => if i = 0 then 1 else 2) 2 = 5/2 ∧
    geometricState (1/2) (fun i => if i = 0 then 2 else 1) 2 = 2 := by
  norm_num [geometricState]

example : |geometricState (-1) (fun i => if i % 2 = 0 then 2 else -2) 8| ≤ (8 : ℕ) * 2 := by
  apply abs_geometricState_le
  · norm_num
  · intro i _
    split <;> norm_num

example (a b : ℝ) : geometricTail (1/2) (fun i => if i = 0 then a else b) 0 2 = a + b/2 := by
  simp [geometricTail]
  ring

end Transformer.Phasor
