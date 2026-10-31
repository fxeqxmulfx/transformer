module

public import Transformer.SequenceModel.Attention.Finite.Model

/-!
# Arbitrary finite arity needs neither infinite width nor infinite depth

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F6.
One source-shaped family has W=2(p+1)^2+6p^N+2 padding and
H=floor((8W+1)/3), with one head and any positive finite depth. The input
protocol appends one fixed vocabulary token for readout. The rotary phase
table is constructed; this does not quantify over every prescribed table.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity

/-- F6: finite width provides both branch reachability and all interpolation channels. -/
theorem sequence_room (p : ℕ) (hp : 0 < p) (N padding : ℕ) :
    9 < modelWidth p (3 * p ^ N + padding) ∧
      3 * p ^ N ≤ (8 * modelWidth p (3 * p ^ N + padding) + 1) / 3 := by
  have hpow : 0 < p ^ N := Nat.pow_pos hp
  constructor
  · unfold modelWidth
    nlinarith
  · rw [Nat.le_div_iff_mul_le (by decide)]
    unfold modelWidth
    nlinarith

/-- Actual source FFN ratio and a context window holding the operands plus readout. -/
abbrev sequenceConfig (p : ℕ) (hp : 0 < p) (N padding extraDepth : ℕ) : Config :=
  { Expressivity.config p hp (3 * p ^ N + padding) extraDepth with
    max_seq_len := N + 1
    max_seq_len_pos := by omega }

/-- F6: every function of any finite number of finite-valued variables has
fixed finite weights and a fixed active rotary phase table giving strict
correct classification on every tuple. This is capacity, not convergence
of an optimizer, and the added readout token is explicit in the conclusion. -/
theorem exists_finite_sequence_classifier (p : ℕ) (hp : 0 < p) (N padding extraDepth : ℕ)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1) (f : (Fin N → Fin p) → Fin p) :
    ∃ angle : Fin (N + 1) → Fin ((sequenceConfig p hp N padding extraDepth).head_dim / 2) → ℝ,
      ∃ weights : Parameters (sequenceConfig p hp N padding extraDepth), ∀ x q, q ≠ f x →
        score (sequenceConfig p hp N padding extraDepth) eps xeps (fun _ => angle) (Fin.last N)
          weights (withReadout ⟨0, hp⟩ x) q <
        score (sequenceConfig p hp N padding extraDepth) eps xeps (fun _ => angle) (Fin.last N)
          weights (withReadout ⟨0, hp⟩ x) (f x) := by
  exact exists_classifier (sequenceConfig p hp N padding extraDepth) rfl
    (sequence_room p hp N padding).1 N (sequence_room p hp N padding).2 heps hx
    (banks p hp (3 * p ^ N + padding) extraDepth) ⟨0, hp⟩ f

example := sequence_room 97 (by decide) 3 0
example := exists_finite_sequence_classifier 3 (by decide) 3 0 0
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun x => x 0 * x 1 + x 2)
example := exists_finite_sequence_classifier 3 (by decide) 4 100 7
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun x => if x 0 = x 2 then x 1 else x 3)
example := exists_finite_sequence_classifier 97 (by decide) 3 0 4
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun x => x 0 ^ 5 + x 1 * x 2)
example := exists_finite_sequence_classifier 193 (by decide) 4 256 2
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun x => (x 0 + x 1) * (x 2 + x 3))
example := exists_finite_sequence_classifier 2 (by decide) 0 0 0
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) (fun _ => 1)

end Transformer.SequenceModel.Attention.Finite
