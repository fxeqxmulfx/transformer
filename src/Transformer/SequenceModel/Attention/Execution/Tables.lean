module

public import Transformer.SequenceModel.Attention.Stability.Local

/-!
# Independent cosine/sine table errors

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C2;
diff_gpt `b98c665b`, `rope.py`. Generated cosine and sine entries can move
independently and need not lie on the unit circle. This arithmetic map keeps
the actual half-split layout. A separate execution contract accounts for
rounding in the multiplies and additions which apply the table.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini Stability Landscape.Neural
open scoped BigOperators

/-- The source half-split multiply/add formula, with arbitrary stored table entries. -/
def tableRotate (d : ℕ) (c s : Fin (d / 2) → ℝ) (x : EucSpace d) : EucSpace d :=
  WithLp.toLp 2 (fun i => match ropeSplit d i with
    | .inl (.inl k) => c k * x (ropeFst d k) - s k * x (ropeSnd d k)
    | .inl (.inr k) => s k * x (ropeFst d k) + c k * x (ropeSnd d k)
    | .inr j => x ((ropeSplit d).symm (.inr j)))

/-- Arbitrary finite table coefficients still send the zero vector to zero. -/
theorem tableRotate_zero (d : ℕ) (c s : Fin (d / 2) → ℝ) : tableRotate d c s 0 = 0 := by
  ext i
  simp only [tableRotate, PiLp.toLp_apply, PiLp.zero_apply]
  split <;> simp

/-- C2: exact tables recover the existing real source rotation. -/
theorem tableRotate_exact (d : ℕ) (a : Fin (d / 2) → ℝ) (x : EucSpace d) :
    tableRotate d (fun k => Real.cos (a k)) (fun k => Real.sin (a k)) x = rotateHead d a x := by
  ext i
  simp only [tableRotate, rotateHead, PiLp.toLp_apply]
  rcases ropeSplit d i with (k | k) | j <;> simp [rotate, splitPairs] <;> ring

/-- C2: independent component errors have a joint operator bound of 2*tau.
No isometry or exact-phase representation of the stored pair is assumed. -/
theorem table_pair_error (c s a x y τ : ℝ) (hτ : 0 ≤ τ)
    (hc : |c - Real.cos a| ≤ τ) (hs : |s - Real.sin a| ≤ τ) :
    ((c * x - s * y) - (Real.cos a * x - Real.sin a * y)) ^ 2 +
      ((s * x + c * y) - (Real.sin a * x + Real.cos a * y)) ^ 2 ≤
        (2 * τ) ^ 2 * (x ^ 2 + y ^ 2) := by
  have hc2 : (c - Real.cos a) ^ 2 ≤ τ ^ 2 := by
    simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) hτ).mpr hc
  have hs2 : (s - Real.sin a) ^ 2 ≤ τ ^ 2 := by
    simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) hτ).mpr hs
  calc
    _ = ((c - Real.cos a) ^ 2 + (s - Real.sin a) ^ 2) * (x ^ 2 + y ^ 2) := by ring
    _ ≤ (2 * τ) ^ 2 * (x ^ 2 + y ^ 2) :=
      mul_le_mul_of_nonneg_right (by nlinarith [sq_nonneg τ]) (by positivity)

/-- C2: the complete half-split table has error at most 2*tau times input
norm. Odd leftover coordinates are unchanged on both sides. -/
theorem tableRotate_error (d : ℕ) (c s a : Fin (d / 2) → ℝ) (x : EucSpace d)
    (τ : ℝ) (hτ : 0 ≤ τ) (hc : ∀ k, |c k - Real.cos (a k)| ≤ τ)
    (hs : ∀ k, |s k - Real.sin (a k)| ≤ τ) :
    ‖tableRotate d c s x - rotateHead d a x‖ ≤ (2 * τ) * ‖x‖ := by
  rw [← tableRotate_exact]
  apply (sq_le_sq₀ (norm_nonneg _) (by positivity : 0 ≤ (2 * τ) * ‖x‖)).mp
  rw [mul_pow, EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  simp only [PiLp.sub_apply]
  rw [sum_split d (fun i => (tableRotate d c s x i -
      tableRotate d (fun k => Real.cos (a k)) (fun k => Real.sin (a k)) x i) ^ 2),
    sum_split d (fun i => x i ^ 2)]
  have hf (b t : Fin (d / 2) → ℝ) (k : Fin (d / 2)) :
      tableRotate d b t x (ropeFst d k) = b k * x (ropeFst d k) - t k * x (ropeSnd d k) := by
    simp only [tableRotate, ropeFst, PiLp.toLp_apply, Equiv.apply_symm_apply]
  have ht (b t : Fin (d / 2) → ℝ) (k : Fin (d / 2)) :
      tableRotate d b t x (ropeSnd d k) = t k * x (ropeFst d k) + b k * x (ropeSnd d k) := by
    simp only [tableRotate, ropeSnd, PiLp.toLp_apply, Equiv.apply_symm_apply]
  have hl (b t : Fin (d / 2) → ℝ) (j : Fin (d - 2 * (d / 2))) :
      tableRotate d b t x ((ropeSplit d).symm (.inr j)) = x ((ropeSplit d).symm (.inr j)) := by
    simp only [tableRotate, PiLp.toLp_apply, Equiv.apply_symm_apply]
  simp only [hf, ht, hl, sub_self, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
    zero_pow, Finset.sum_const_zero, add_zero]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib, mul_add, Finset.mul_sum]
  exact (Finset.sum_le_sum (fun k _ => table_pair_error (c k) (s k) (a k)
    (x (ropeFst d k)) (x (ropeSnd d k)) τ hτ (hc k) (hs k))).trans
      (le_add_of_nonneg_right (by positivity))

example := table_pair_error (1 + 1 / 1024) (1 / 1024) 0 2 3 (1 / 1024)
  (by norm_num) (by norm_num) (by norm_num)
example := tableRotate_error 4 (fun _ => 1 + 1 / 1024) (fun _ => 1 / 1024)
  (fun _ => 0) (WithLp.toLp 2 ![1, 2, 3, 4]) (1 / 1024)
  (by norm_num) (by intro k; norm_num) (by intro k; norm_num)

end Transformer.SequenceModel.Attention.Execution
