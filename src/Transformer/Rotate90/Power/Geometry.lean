module

public import Transformer.Rotate90.Power.Defs

/-!
# Exact geometry of the powered-division weights

Source: `review/flat_spectrum.md`, *Actual network expressivity for every exponent*,
construction and norm balance. C(a) is the faithful three-coordinate answer code,
including its zero flag. The tied embedding holds (1,C(a),C(a^k)) in its even slots.
Its squared length is three even for a=0 or k=0; both operands' raw sum has length
squared six. The branch's faithful-code pairs are (C(t)-C(a),C(t)+C(a)), where
t=(a/b)^k, and its powered-code pairs are (-C(a^k),C(a^k)). The parallelogram
identity makes its squared length six as well. Odd slots remain in normalization
while being invisible to the tied readout. No approximate cancellation is used.
-/

@[expose] public section

namespace Transformer.Rotate90.Power

open Landscape.Neural
open scoped BigOperators

variable {p : ℕ} [Fact p.Prime] {K : Type*} [Fintype K]

/-- The powered quotient code is computed from the powered operand codes, including
the natural exponent zero. Source: construction's four complex multiplication terms. -/
theorem character_div_pow (χ : MulChar (ZMod p) ℂ) (a b : ZMod p) (k : ℕ) :
    χ ((a / b) ^ k) = χ (a ^ k) * star (χ (b ^ k)) := by
  rw [div_pow, Rotate90.character_div]

/-- The powered zero flag is read directly from a's powered embedding coordinates.
The denominator condition remains explicit, including when k=0. -/
theorem zeroFlag_div_pow (a : ZMod p) {b : ZMod p} (hb : b ≠ 0) (k : ℕ) :
    zeroFlag ((a / b) ^ k) = zeroFlag (a ^ k) := by
  rw [div_pow, Rotate90.zeroFlag_div _ (pow_ne_zero k hb)]

/-- Every tied embedding row has squared length three, including the zero row.
Both code blocks have unit length; their coordinates need not individually be nonzero. -/
theorem embedding_sq (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a : ZMod p) :
    ∑ i, (embedding (K := K) χ k a i) ^ 2 = 3 := by
  rw [embedding, sum_sq_pack]
  norm_num [Fin.sum_univ_succ]
  have ha := character_sq χ a
  have hpow := character_sq χ (a ^ k)
  simp only [map_pow] at hpow
  linear_combination ha + hpow

omit [Fintype K] in
/-- The source quarter-turn places b's features in odd slots and a's in even slots. -/
theorem tiedInput_embedding (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a b : ZMod p) :
    tiedInput (embedding (K := K) χ k) a b = raw χ k a b := by
  funext ⟨i, j⟩
  rcases i with i | i
  · fin_cases i <;> fin_cases j <;> simp [tiedInput, embedding, rotate90, raw, pack]
  · fin_cases j <;> simp [tiedInput, embedding, rotate90, raw, pack]

/-- Both embedding rows contribute their full norm to the raw residual input. -/
theorem raw_sq (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a b : ZMod p) :
    ∑ i, (raw (K := K) χ k a b i) ^ 2 = 6 := by
  rw [raw, sum_sq_pack]
  norm_num [Fin.sum_univ_succ]
  have ha := character_sq χ a
  have hb := character_sq χ b
  have hpa := character_sq χ (a ^ k)
  have hpb := character_sq χ (b ^ k)
  simp only [map_pow] at hpa hpb
  linear_combination ha + hb + hpa + hpb

/-- **Exact branch norm balance**. The branch has the same squared norm as the raw
input, allowing its normalized even slots to cancel the input at the same scale. -/
theorem branchVector_sq (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a b : ZMod p) :
    ∑ i, (branchVector (K := K) χ k a b i) ^ 2 = 6 := by
  rw [branchVector, sum_sq_pack]
  norm_num [Fin.sum_univ_succ]
  have ha := character_sq χ a
  have ht := character_sq χ ((a / b) ^ k)
  have hpa := character_sq χ (a ^ k)
  simp only [map_pow] at ht hpa
  linear_combination 2 * ht + 2 * ha + 2 * hpa

/-- The powered lookup channels cancel out of the final tied readout. The remaining
class-dependent term is the faithful code of the powered quotient, source readout equation. -/
theorem readout_raw_add_branchVector (χ : MulChar (ZMod p) ℂ) (k : ℕ)
    (a b q : ZMod p) :
    tiedReadout (embedding (K := K) χ k)
      (fun i => raw χ k a b i + branchVector χ k a b i) q =
        1 + characterScore χ ((a / b) ^ k) q := by
  simp only [tiedReadout, Fintype.sum_prod_type, Fintype.sum_sum_type, embedding,
    raw, branchVector, pack, zero_mul, Finset.sum_const_zero, add_zero]
  norm_num [Fin.sum_univ_succ, characterScore]
  ring

omit [Fintype K] in
/-- Odd slots of the embedding are exactly zero. They can balance intermediate norms
without adding unwanted class-dependent terms to the tied readout. -/
theorem embedding_odd (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a : ZMod p) (i : PairIndex K) :
    embedding χ k a (i, 1) = 0 := by
  rcases i with i | i
  · fin_cases i <;> simp [embedding, pack]
  · simp [embedding, pack]

omit [Fintype K] in
/-- At exponent zero the powered code in the lookup table is always C(1), even at a=0.
This is the stored feature, rather than a special case in the network's forward pass. -/
theorem embedding_zero_exponent (χ : MulChar (ZMod p) ℂ) (a : ZMod p) :
    embedding (K := K) χ 0 a =
      pack ![![1, 0], ![(χ a).re, 0], ![(χ a).im, 0], ![zeroFlag a, 0],
        ![1, 0], ![0, 0], ![0, 0]] := by
  simp [embedding, zeroFlag]

omit [Fintype K] in
/-- After cancellation, the even residual slots hold only the constant and answer
code. The powered lookup features leave no extra class-dependent readout terms.
The common positive normalization factor is supplied later by `state_weights`. -/
theorem residual_even (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a b : ZMod p) (i : Fin 7) :
    raw (K := K) χ k a b (.inl i, 0) +
      branchVector (K := K) χ k a b (.inl i, 0) =
      ![1, (χ ((a / b) ^ k)).re, (χ ((a / b) ^ k)).im,
        zeroFlag ((a / b) ^ k), 0, 0, 0] i := by
  fin_cases i <;> simp [raw, branchVector, pack]

omit [Fintype K] in
/-- All padding stays zero in the raw residual sum. Its size affects normalization,
but cannot introduce spurious output features into the tied readout. -/
theorem residual_padding (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a b : ZMod p)
    (i : K) (j : Fin 2) :
    raw χ k a b (.inr i, j) + branchVector χ k a b (.inr i, j) = 0 := by
  simp [raw, branchVector, pack]

example := character_div_pow (p := 3) 1 0 2 2
example := character_div_pow (p := 3) 1 0 2 0
example := zeroFlag_div_pow (p := 3) 0 (b := 2) (by decide) 0
example := zeroFlag_div_pow (p := 3) 0 (b := 2) (by decide) 3
example := embedding_sq (K := Fin 1) (p := 3) 1 0 0
example := embedding_sq (K := Fin 1) (p := 3) 1 2 0
example := tiedInput_embedding (K := Fin 1) (p := 3) 1 2 0 2
example := raw_sq (K := Fin 1) (p := 3) 1 0 0 2
example := raw_sq (K := Fin 1) (p := 3) 1 3 0 2
example := branchVector_sq (K := Fin 1) (p := 3) 1 0 0 2
example := branchVector_sq (K := Fin 1) (p := 3) 1 3 0 2
example := readout_raw_add_branchVector (K := Fin 1) (p := 3) 1 0 0 2 1
example := readout_raw_add_branchVector (K := Fin 1) (p := 3) 1 3 0 2 0
example := embedding_odd (K := Fin 1) (p := 3) 1 2 0 (.inl 4)
example := embedding_zero_exponent (K := Fin 1) (p := 3) 1 0
example := residual_even (K := Fin 1) (p := 3) 1 2 0 2 4
example := residual_padding (K := Fin 1) (p := 3) 1 2 0 2 0 1

end Transformer.Rotate90.Power
