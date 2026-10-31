module

public import Transformer.SequenceModel.Attention.Expressivity.Geometry

/-!
# Exact two-token contextual separation by the supplied XSA head

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E3.
Zero query/key matrices make the causal weights uniform at any supplied
RoPE angles. Unit values remain active: XSA deletes the current value's
component and retains an orthogonal earlier value with coefficient 1/2.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini
open scoped BigOperators RealInnerProductSpace

/-- E3: arbitrary rotations preserve zero queries/keys. -/
theorem rotateHead_zero (d : ℕ) (angle : Fin (d / 2) → ℝ) : rotateHead d angle 0 = 0 := by
  ext i
  simp only [rotateHead, WithLp.ofLp_toLp]
  rcases ropeSplit d i with (k | k) | k <;> simp [rotate, splitPairs]

/-- E3: zero Q/K have equal scores with every positive-epsilon normalizer retained. -/
theorem headScore_zero (d : ℕ) (eps : ℝ) (a b : Fin (d / 2) → ℝ) :
    headScore d eps a b 0 0 = 0 := by
  simp [headScore, rmsNormEps, rotateHead_zero]

/-- E3: the last row of a two-token uniform causal head gives both tokens weight 1/2. -/
theorem causalWeight_two (j : Fin 2) : causalWeight (fun _ _ : Fin 2 => 0) 1 j = 1 / 2 := by
  have hi (i : Fin 2) : i ≤ 1 := by omega
  have hI : Finset.Iic (1 : Fin 2) = Finset.univ := by ext i; simp [hi]
  simp [causalWeight, hI, hi]

/-- E3: unit own values are above clipping when the source epsilon is at most one. -/
theorem normL2_unit {d : ℕ} (i : Fin d) (xeps : ℝ) (h : xeps ≤ 1) :
    normL2 xeps (unit i) = unit i := by
  simp [normL2, norm_unit, max_eq_left h]

/-- E3: actual normalized rotary XSA distinguishes equal tokens from different
ones, for arbitrary phase tables. This is active attention, not a disabled branch. -/
theorem headOutput_pair {d : ℕ} (eps xeps : ℝ) (hx : xeps ≤ 1)
    (angle : Fin 2 → Fin (d / 2) → ℝ) (a b : Fin d) :
    headOutput eps xeps angle (fun _ => 0) (fun _ => 0) ![unit a, unit b] 1 =
      if a = b then 0 else (1 / 2 : ℝ) • unit a := by
  simp only [headOutput, headScore_zero, causalWeight_two, Fin.sum_univ_two,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one, Fin.val_one,
    one_ne_zero, ↓reduceIte, normL2_unit b xeps hx]
  rw [inner_add_left, real_inner_smul_left, real_inner_smul_left, inner_unit, inner_unit]
  by_cases hab : a = b
  · subst b
    simp only [↓reduceIte]
    module
  · simp only [hab, ↓reduceIte]
    module

example := normL2_unit (0 : Fin 4) (1 / 1000000000000) (by norm_num)
example := headOutput_pair (d := 4) (1 / 8388608) (1 / 1000000000000) (by norm_num)
  (fun j k => j.val * ((k.val : ℝ) + 1) / 7) 0 1
example := headOutput_pair (d := 4) (1 / 8388608) (1 / 1000000000000) (by norm_num)
  (fun j k => j.val * ((k.val : ℝ) + 1) / 7) 1 1

end Transformer.SequenceModel.Attention.Expressivity
