module

public import Transformer.Phasor.TripleTransport

/-!
# Scaling learned affine reads of a dense prefix

Source: “Unit scales for the context and direct reads” in the triple-transport
experiment. The two scales change the initial forward and first variation.
All affine gates remain representable over the reals for a nonzero new scale.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.GateScale

open scoped BigOperators

/-- A learned affine gate with a unit initial offset. It multiplies the
hidden value by its branch scale after reading the previous dense state. -/
def gain {d : ℕ} (s b : ℝ) (g h : Fin d → ℝ) : ℝ :=
  s * (1 + b + ∑ i, g i * h i)

/-- Replacing the scale of the direct read preserves its complete affine
gate class for arbitrary hidden states and with exactly the same number
of learned parameters. The new coordinates are independent of the input. -/
theorem rescale {d : ℕ} (s t b : ℝ) (ht : t ≠ 0) (g h : Fin d → ℝ) :
    gain t (s / t * (1 + b) - 1) (fun i => s / t * g i) h = gain s b g h := by
  unfold gain
  have expand : (∑ i, s / t * g i * h i) = s / t * ∑ i, g i * h i := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [expand]
  field_simp
  ring

/-- At zero value weights and zero gate weights the first variation of the
read in a value direction is exactly the branch scale times that direction. -/
theorem zero_initial_change {d : ℕ} (s ε : ℝ) (h v : Fin d → ℝ) (i : Fin d) :
    gain s 0 0 h * (0 + ε * v i) - gain s 0 0 h * 0 = ε * (s * v i) := by
  simp [gain]
  ring

example : gain 3 (6 / 3 * (1 + 2) - 1) (fun _ : Fin 2 => 6 / 3 * 4) (fun _ => 5) =
    gain 6 2 (fun _ : Fin 2 => 4) (fun _ => 5) :=
  rescale 6 3 2 (by norm_num) _ _

/-- A unit-scaled contextual and direct read represents every independently
scaled pair of affine prefix gates. The structural and phase contributions
can be arbitrary functions of the same prefix and learned weights, including
discrete graph search. The parameter substitution is independent of that
prefix or the values returned by any read. -/
theorem unit_mix {d : ℕ} (sc sd bc bd : ℝ) (gc gd h : Fin d → ℝ)
    (context direct rest : Fin d → ℝ) (i : Fin d) :
    gain 1 (sc * (1 + bc) - 1) (fun j => sc * gc j) h * context i +
    gain 1 (sd * (1 + bd) - 1) (fun j => sd * gd j) h * direct i + rest i =
    gain sc bc gc h * context i + gain sd bd gd h * direct i + rest i := by
  have c := rescale sc 1 bc (by norm_num) gc h
  have v := rescale sd 1 bd (by norm_num) gd h
  simp only [div_one] at c v
  rw [c, v]

example (h : Fin 2 → ℝ) : gain 1 (2 * (1 + 3) - 1) (fun _ => 2 * 4) h * 5 +
    gain 1 (8 * (1 + 1) - 1) (fun _ => 8 * 2) h * 7 + 9 =
    gain 2 3 (fun _ => 4) h * 5 + gain 8 1 (fun _ => 2) h * 7 + 9 :=
  unit_mix 2 8 3 1 _ _ h (fun _ => 5) (fun _ => 7) (fun _ => 9) 0

end Transformer.Phasor.TripleTransport.GateScale
