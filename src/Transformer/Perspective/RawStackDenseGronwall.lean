/-
# A discrete Gronwall bound for the dense stack

**Not a statement of any paper.**  In the gauge, the dense stack of
`Perspective.RawStackDense` is the skip-free gauge stack plus the rescaled skips
it has taken (`ungauged_sub_gaugeStack_dense`), and the skips read the stream's
own states: the displacement of the ungauged stream feeds back into itself, and
what bounds it is a discrete Gronwall inequality.  Write

  `W_{k,i} = Σ_{j ≤ k} |s_{k,j,i} Λ_{j,i} / Λ_{k+1,i}|`   (`skipMass`)

for the weight of the skips of step `k` as the directions see it, and
`P_{k,i} = Π_{j<k} (1 + W_{j,i})` (`skipGrowth`).  If the gauge stack stays
within `D` of the start up to depth `k`, then

  `‖y_{k,i} - x_{0,i}‖ ≤ D + (‖x_{0,i}‖ + D) (P_{k,i} - 1)`

(`norm_ungauged_sub_le_dense`), and the direction of token `i` is within twice
that, divided by `‖x_{0,i}‖`, of where it started
(`norm_normalize_sub_le_dense`).  Without skips `P = 1` and the bound is `D`;
each step multiplies the growth by one plus what its skips weigh.
-/

import Transformer.Perspective.RawStackDense

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The weight of the skips of one step, as the directions see it:**
`W_{k,i} = Σ_{j ≤ k} |s_{k,j,i} Λ_{j,i} / Λ_{k+1,i}|`, each gate divided by the
gain accumulated over the depth it jumps (`ungauged_rec_dense`). -/
noncomputable def skipMass (lam : ℕ → Idx n → ℝ) (s : ℕ → ℕ → Idx n → ℝ) (k : ℕ)
    (i : Idx n) : ℝ :=
  ∑ j ∈ Finset.range (k + 1), |s k j i * gainProd lam j i / gainProd lam (k + 1) i|

/-- **The growth the skips allow up to depth `k`:**
`P_{k,i} = Π_{j<k} (1 + W_{j,i})`. -/
noncomputable def skipGrowth (lam : ℕ → Idx n → ℝ) (s : ℕ → ℕ → Idx n → ℝ) (k : ℕ)
    (i : Idx n) : ℝ :=
  ∏ j ∈ Finset.range k, (1 + skipMass lam s j i)

/-- **Skip weights are nonnegative.**

Source: none — posed here; a sum of absolute values. -/
theorem skipMass_nonneg (lam : ℕ → Idx n → ℝ) (s : ℕ → ℕ → Idx n → ℝ) (k : ℕ) (i : Idx n) :
    0 ≤ skipMass lam s k i :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- **One step of the growth:** `P_{k+1,i} = P_{k,i} (1 + W_{k,i})`.

Source: none — posed here; `Finset.prod_range_succ`. -/
theorem skipGrowth_succ (lam : ℕ → Idx n → ℝ) (s : ℕ → ℕ → Idx n → ℝ) (k : ℕ) (i : Idx n) :
    skipGrowth lam s (k + 1) i = skipGrowth lam s k i * (1 + skipMass lam s k i) :=
  Finset.prod_range_succ _ k

/-- **The growth is at least `1`.**

Source: none — posed here; a product of factors at least `1`. -/
theorem one_le_skipGrowth (lam : ℕ → Idx n → ℝ) (s : ℕ → ℕ → Idx n → ℝ) (k : ℕ) (i : Idx n) :
    1 ≤ skipGrowth lam s k i := by
  induction k with
  | zero => simp [skipGrowth]
  | succ k ih =>
      rw [skipGrowth_succ]
      exact one_le_mul_of_one_le_of_one_le ih
        (le_add_of_nonneg_right (skipMass_nonneg lam s k i))

/-- **The growth is monotone in the depth.**

Source: none — posed here; `skipGrowth_succ` and `one_le_skipGrowth`. -/
theorem skipGrowth_mono (lam : ℕ → Idx n → ℝ) (s : ℕ → ℕ → Idx n → ℝ) (i : Idx n) :
    Monotone fun k => skipGrowth lam s k i := by
  refine monotone_nat_of_le_succ fun k => ?_
  rw [skipGrowth_succ]
  have h := one_le_skipGrowth lam s k i
  have hW := skipMass_nonneg lam s k i
  nlinarith

/-- **The growth telescopes:** `Σ_{j<k} W_{j,i} P_{j,i} = P_{k,i} - 1`, the
identity the Gronwall bound closes on.

Source: none — posed here; `skipGrowth_succ`, summed over the depths. -/
theorem sum_skipMass_mul_skipGrowth (lam : ℕ → Idx n → ℝ) (s : ℕ → ℕ → Idx n → ℝ) (k : ℕ)
    (i : Idx n) :
    ∑ j ∈ Finset.range k, skipMass lam s j i * skipGrowth lam s j i
      = skipGrowth lam s k i - 1 := by
  induction k with
  | zero => simp [skipGrowth]
  | succ k ih =>
      rw [Finset.sum_range_succ, ih, skipGrowth_succ]
      ring

/-- **A discrete Gronwall bound for the dense stack.**  If the skip-free gauge
stack stays within `D` of the start up to depth `k`, then

  `‖y_{k,i} - x_{0,i}‖ ≤ D + (‖x_{0,i}‖ + D) (P_{k,i} - 1)`,

with `y` the ungauged stream and `P = skipGrowth`.  The gains are any positive
numbers, and the skips any number per step, onto any depth: each state a skip
reads is at most `(‖x_{0,i}‖ + D) P` long, so the skips of step `j` move the
stream by at most `(‖x_{0,i}‖ + D) W_{j,i} P_{j,i}`, and these telescope.

Source: none — posed here; `ungauged_sub_gaugeStack_dense`, by strong induction
on the depth, with `sum_skipMass_mul_skipGrowth`. -/
theorem norm_ungauged_sub_le_dense {lam : ℕ → Idx n → ℝ} {s : ℕ → ℕ → Idx n → ℝ}
    {x g : ℕ → Idx n → EucSpace d} {D : ℝ} (hlam : ∀ j i, 0 < lam j i)
    (hx : ∀ k i, x (k + 1) i
      = lam k i • x k i + ∑ j ∈ Finset.range (k + 1), s k j i • x j i + g k i)
    {k : ℕ} (i : Idx n) (hD : ∀ j ≤ k, ‖gaugeStack lam (x 0) g j i - x 0 i‖ ≤ D) :
    ‖ungauged lam x k i - x 0 i‖ ≤ D + (‖x 0 i‖ + D) * (skipGrowth lam s k i - 1) := by
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    have hD0 : 0 ≤ ‖x 0 i‖ + D :=
      add_nonneg (norm_nonneg _) ((norm_nonneg _).trans (hD 0 (Nat.zero_le k)))
    have hy : ∀ l < k, ‖ungauged lam x l i‖ ≤ (‖x 0 i‖ + D) * skipGrowth lam s l i := by
      intro l hl
      have h := ih l hl fun j hj => hD j (hj.trans hl.le)
      have h' := norm_le_norm_add_norm_sub' (ungauged lam x l i) (x 0 i)
      linarith
    have hinner : ∀ j ∈ Finset.range k,
        ‖∑ l ∈ Finset.range (j + 1),
          (s j l i * gainProd lam l i / gainProd lam (j + 1) i) • ungauged lam x l i‖
        ≤ (‖x 0 i‖ + D) * (skipMass lam s j i * skipGrowth lam s j i) := by
      intro j hj
      rw [Finset.mem_range] at hj
      calc _ ≤ ∑ l ∈ Finset.range (j + 1),
              ‖(s j l i * gainProd lam l i / gainProd lam (j + 1) i) • ungauged lam x l i‖ :=
            norm_sum_le _ _
        _ ≤ ∑ l ∈ Finset.range (j + 1),
              |s j l i * gainProd lam l i / gainProd lam (j + 1) i|
                * ((‖x 0 i‖ + D) * skipGrowth lam s j i) := by
            refine Finset.sum_le_sum fun l hl => ?_
            rw [Finset.mem_range] at hl
            rw [norm_smul, Real.norm_eq_abs]
            refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
            exact (hy l (by omega)).trans
              (mul_le_mul_of_nonneg_left (skipGrowth_mono lam s i (by omega)) hD0)
        _ = (‖x 0 i‖ + D) * (skipMass lam s j i * skipGrowth lam s j i) := by
            rw [skipMass, ← Finset.sum_mul]
            ring
    calc ‖ungauged lam x k i - x 0 i‖
        = ‖(gaugeStack lam (x 0) g k i - x 0 i)
            + (ungauged lam x k i - gaugeStack lam (x 0) g k i)‖ := by
          congr 1
          abel
      _ ≤ ‖gaugeStack lam (x 0) g k i - x 0 i‖
            + ‖ungauged lam x k i - gaugeStack lam (x 0) g k i‖ := norm_add_le _ _
      _ ≤ D + (‖x 0 i‖ + D) * (skipGrowth lam s k i - 1) := by
          refine add_le_add (hD k le_rfl) ?_
          rw [ungauged_sub_gaugeStack_dense hlam hx k i, ← sum_skipMass_mul_skipGrowth,
            Finset.mul_sum]
          exact (norm_sum_le _ _).trans (Finset.sum_le_sum hinner)

/-- **Every direction stays near its start, in `D` and the skips.**  If the
skip-free gauge stack stays within `D` of the start up to depth `k`, the
direction of token `i` at depth `k` is within
`2 (D + (‖x_{0,i}‖ + D) (P_{k,i} - 1)) / ‖x_{0,i}‖` of the one it started with.

Source: none — posed here; `norm_ungauged_sub_le_dense` through
`normalize_ungauged` and `norm_normalize_sub_le_div`, as in `rawStack_frozen`. -/
theorem norm_normalize_sub_le_dense {lam : ℕ → Idx n → ℝ} {s : ℕ → ℕ → Idx n → ℝ}
    {x g : ℕ → Idx n → EucSpace d} {D : ℝ} (hlam : ∀ j i, 0 < lam j i)
    (hx : ∀ k i, x (k + 1) i
      = lam k i • x k i + ∑ j ∈ Finset.range (k + 1), s k j i • x j i + g k i)
    {k : ℕ} (i : Idx n) (hD : ∀ j ≤ k, ‖gaugeStack lam (x 0) g j i - x 0 i‖ ≤ D)
    (h0 : x 0 i ≠ 0) :
    ‖‖x k i‖⁻¹ • x k i - ‖x 0 i‖⁻¹ • x 0 i‖
      ≤ 2 * (D + (‖x 0 i‖ + D) * (skipGrowth lam s k i - 1)) / ‖x 0 i‖ := by
  rw [← normalize_ungauged hlam x k i, norm_sub_rev]
  calc ‖‖x 0 i‖⁻¹ • x 0 i - ‖ungauged lam x k i‖⁻¹ • ungauged lam x k i‖
      ≤ 2 * ‖x 0 i - ungauged lam x k i‖ / ‖x 0 i‖ := norm_normalize_sub_le_div h0
    _ ≤ 2 * (D + (‖x 0 i‖ + D) * (skipGrowth lam s k i - 1)) / ‖x 0 i‖ := by
        gcongr
        rw [norm_sub_rev]
        exact norm_ungauged_sub_le_dense hlam hx i hD

/-- The hypotheses of `norm_ungauged_sub_le_dense` and
`norm_normalize_sub_le_dense` are satisfiable, with skips that do something:
gains `1`, no block output, a skip of gate `1` onto the start at every step, the
stack it forces, `x_k = (k + 1) e₀`, and `D = 0`, the gauge stack of no outputs
staying where it starts. -/
example : (∀ (_ : ℕ) (_ : Idx 1), (0 : ℝ) < 1) ∧
    (∀ (k : ℕ) (i : Idx 1),
      (fun k (_ : Idx 1) => ((k : ℝ) + 1) • (basePoint 0 : EucSpace 1)) (k + 1) i
        = (1 : ℝ) • (fun k (_ : Idx 1) => ((k : ℝ) + 1) • (basePoint 0 : EucSpace 1)) k i
          + ∑ j ∈ Finset.range (k + 1), (if j = 0 then (1 : ℝ) else 0)
            • (fun k (_ : Idx 1) => ((k : ℝ) + 1) • (basePoint 0 : EucSpace 1)) j i
          + 0) ∧
    (∀ (k : ℕ) (i : Idx 1), ∀ j ≤ k,
      ‖gaugeStack (fun _ _ => (1 : ℝ))
          ((fun k (_ : Idx 1) => ((k : ℝ) + 1) • (basePoint 0 : EucSpace 1)) 0)
          (fun _ _ => (0 : EucSpace 1)) j i
        - (fun k (_ : Idx 1) => ((k : ℝ) + 1) • (basePoint 0 : EucSpace 1)) 0 i‖ ≤ 0) ∧
    ∀ i : Idx 1, (fun k (_ : Idx 1) => ((k : ℝ) + 1) • (basePoint 0 : EucSpace 1)) 0 i ≠ 0 := by
  refine ⟨fun _ _ => one_pos, fun k _ => ?_, fun _ _ _ _ => by simp [gaugeStack],
    fun _ => by simp [basePoint]⟩
  simp only [ite_smul, one_smul, zero_smul, Finset.sum_ite_eq', Finset.mem_range,
    Nat.zero_lt_succ, ite_true, add_zero]
  push_cast
  module

end Perspective
end Transformer
