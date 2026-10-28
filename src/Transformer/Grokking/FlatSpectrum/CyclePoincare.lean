module

public import Transformer.Grokking.FlatSpectrum.Basic
public import Mathlib.Algebra.Order.Chebyshev

/-!
# A Poincaré inequality on the cycle `ℤ/N`

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel* (added). Along the
cycle, `u k − u j` is a sum of `m = (k − j).val ≤ N − 1` distinct consecutive increments, so
`|u k − u j|² ≤ (N − 1) D` with `D = ∑_l |u (l + 1) − u l|²` (`norm_sq_sub_le_cycle`). For a
centered `u`, `N u j = ∑_k (u j − u k)`, hence `N² |u j|² ≤ N ∑_k |u j − u k|²` and
`∑_j |u j|² ≤ N (N − 1) D` (`sum_norm_sq_le_cycle`). Two companions: a matrix is a contraction
once it shortens every vector (`isContraction_of_forall_sum_norm_sq_le`), and adding a
constant to a centered vector adds its squared length (`sum_norm_sq_add_const`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

/-- A matrix that shortens every vector is a contraction. -/
lemma isContraction_of_forall_sum_norm_sq_le {m n : Type*} [Fintype m] [Fintype n]
    [DecidableEq n] {Z : Matrix m n ℂ} (h : ∀ v, ∑ i, ‖(Z *ᵥ v) i‖ ^ 2 ≤ ∑ j, ‖v j‖ ^ 2) :
    IsContraction Z := by
  have hsq (w : m → ℂ) : star w ⬝ᵥ w = ((∑ i, ‖w i‖ ^ 2 : ℝ) : ℂ) := by
    simp only [dotProduct, Pi.star_apply, Complex.star_def, Complex.conj_mul',
      Complex.ofReal_sum, Complex.ofReal_pow]
  have hsq' (w : n → ℂ) : star w ⬝ᵥ w = ((∑ i, ‖w i‖ ^ 2 : ℝ) : ℂ) := by
    simp only [dotProduct, Pi.star_apply, Complex.star_def, Complex.conj_mul',
      Complex.ofReal_sum, Complex.ofReal_pow]
  refine posSemidef_iff_dotProduct_mulVec.mpr
    ⟨isHermitian_one.sub (isHermitian_conjTranspose_mul_self Z), fun x => ?_⟩
  rw [sub_mulVec, one_mulVec, dotProduct_sub, ← mulVec_mulVec, dotProduct_mulVec,
    ← star_mulVec, hsq, hsq', ← Complex.ofReal_sub, Complex.zero_le_real, sub_nonneg]
  exact h x

example := isContraction_of_forall_sum_norm_sq_le (Z := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  fun v => by rw [one_mulVec]

/-- Adding a constant to a centered vector adds its squared length:
`∑ |w j + m|² = ∑ |w j|² + N |m|²` when `∑ w = 0`. -/
lemma sum_norm_sq_add_const {ι : Type*} [Fintype ι] {w : ι → ℂ} (hw : ∑ j, w j = 0) (m : ℂ) :
    ∑ j, ‖w j + m‖ ^ 2 = ∑ j, ‖w j‖ ^ 2 + Fintype.card ι * ‖m‖ ^ 2 := by
  simp only [norm_add_sq_real, sum_add_distrib, ← mul_sum, ← sum_inner, hw, inner_zero_left,
    mul_zero, add_zero, sum_const, card_univ, nsmul_eq_mul]

example := sum_norm_sq_add_const (w := ![(1 : ℂ), -1]) (by simp) 2

variable {N : ℕ} [NeZero N]

/-- Along the cycle `ℤ/N`, `|u k − u j|² ≤ (N − 1) ∑_l |u (l + 1) − u l|²`. -/
lemma norm_sq_sub_le_cycle (u : ZMod N → ℂ) (j k : ZMod N) :
    ‖u k - u j‖ ^ 2 ≤ ((N : ℝ) - 1) * ∑ l, ‖u (l + 1) - u l‖ ^ 2 := by
  set m := (k - j).val with hm
  have hmN : m < N := ZMod.val_lt _
  set e : ℕ → ℝ := fun t => ‖u (j + t + 1) - u (j + t)‖ ^ 2 with he
  have htel : u k - u j = ∑ t ∈ range m, (u (j + t + 1) - u (j + t)) := by
    have h := sum_range_sub (fun t : ℕ => u (j + t)) m
    simp only [Nat.cast_succ, ← add_assoc, Nat.cast_zero, add_zero] at h
    rw [h, hm, ZMod.natCast_zmod_val, add_sub_cancel]
  have hcs : ‖u k - u j‖ ^ 2 ≤ m * ∑ t ∈ range m, e t := by
    rw [htel]
    calc ‖∑ t ∈ range m, (u (j + t + 1) - u (j + t))‖ ^ 2
        ≤ (∑ t ∈ range m, ‖u (j + t + 1) - u (j + t)‖) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) (norm_sum_le _ _) 2
      _ ≤ #(range m) * ∑ t ∈ range m, ‖u (j + t + 1) - u (j + t)‖ ^ 2 :=
          sq_sum_le_card_mul_sum_sq
      _ = m * ∑ t ∈ range m, e t := by rw [card_range]
  have hinj : Set.InjOn (fun t : ℕ => j + (t : ZMod N)) (range m) := by
    intro a ha b hb hab
    have h := add_left_cancel hab
    rw [← ZMod.val_cast_of_lt ((mem_range.mp ha).trans hmN), h,
      ZMod.val_cast_of_lt ((mem_range.mp hb).trans hmN)]
  have hsub : ∑ t ∈ range m, e t ≤ ∑ l, ‖u (l + 1) - u l‖ ^ 2 := by
    rw [he, ← sum_image (f := fun l => ‖u (l + 1) - u l‖ ^ 2) hinj]
    exact sum_le_sum_of_subset_of_nonneg (subset_univ _) fun _ _ _ => sq_nonneg _
  have hm1 : (m : ℝ) ≤ N - 1 := by
    have h1 : m + 1 ≤ N := hmN
    have h2 : ((m + 1 : ℕ) : ℝ) ≤ N := by exact_mod_cast h1
    push_cast at h2
    linarith
  calc ‖u k - u j‖ ^ 2 ≤ m * ∑ t ∈ range m, e t := hcs
    _ ≤ ((N : ℝ) - 1) * ∑ l, ‖u (l + 1) - u l‖ ^ 2 :=
        mul_le_mul hm1 hsub (sum_nonneg fun _ _ => sq_nonneg _)
          (by linarith [Nat.cast_nonneg (α := ℝ) m])

example := @norm_sq_sub_le_cycle 3 _ (fun l => (l.val : ℂ)) 0 2

/-- **A Poincaré inequality on the cycle `ℤ/N`**: a centered `u` has
`∑_l |u l|² ≤ N (N − 1) ∑_l |u (l + 1) − u l|²`. -/
theorem sum_norm_sq_le_cycle {u : ZMod N → ℂ} (hu : ∑ l, u l = 0) :
    ∑ l, ‖u l‖ ^ 2 ≤ N * ((N : ℝ) - 1) * ∑ l, ‖u (l + 1) - u l‖ ^ 2 := by
  set D := ∑ l, ‖u (l + 1) - u l‖ ^ 2
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hj (j : ZMod N) : ‖u j‖ ^ 2 ≤ ((N : ℝ) - 1) * D := by
    have hsum : (N : ℂ) * u j = ∑ k, (u j - u k) := by
      rw [sum_sub_distrib, hu, sub_zero, sum_const, card_univ, ZMod.card, nsmul_eq_mul]
    have h1 : ((N : ℝ) * ‖u j‖) ^ 2 ≤ N * ∑ k, ‖u j - u k‖ ^ 2 := by
      have h := sq_sum_le_card_mul_sum_sq (s := univ) (f := fun k => ‖u j - u k‖)
      rw [card_univ, ZMod.card] at h
      calc ((N : ℝ) * ‖u j‖) ^ 2 = ‖(N : ℂ) * u j‖ ^ 2 := by
            rw [norm_mul, Complex.norm_natCast]
        _ ≤ (∑ k, ‖u j - u k‖) ^ 2 := by
            rw [hsum]
            exact pow_le_pow_left₀ (norm_nonneg _) (norm_sum_le _ _) 2
        _ ≤ N * ∑ k, ‖u j - u k‖ ^ 2 := h
    have h2 : ∑ k, ‖u j - u k‖ ^ 2 ≤ N * (((N : ℝ) - 1) * D) := by
      have h := sum_le_card_nsmul univ (fun k => ‖u j - u k‖ ^ 2) (((N : ℝ) - 1) * D)
        fun k _ => norm_sq_sub_le_cycle u k j
      rwa [card_univ, ZMod.card, nsmul_eq_mul] at h
    have h3 : (N : ℝ) ^ 2 * ‖u j‖ ^ 2 ≤ (N : ℝ) ^ 2 * (((N : ℝ) - 1) * D) := by
      nlinarith
    exact le_of_mul_le_mul_left h3 (by positivity)
  calc ∑ l, ‖u l‖ ^ 2 ≤ ∑ _l : ZMod N, ((N : ℝ) - 1) * D := sum_le_sum fun l _ => hj l
    _ = N * ((N : ℝ) - 1) * D := by
        rw [sum_const, card_univ, ZMod.card, nsmul_eq_mul, mul_assoc]

example := @sum_norm_sq_le_cycle 2 _ (fun l => if l = 0 then 1 else -1) (by
  change ∑ l : Fin 2, (if l = 0 then (1 : ℂ) else -1) = 0
  simp [Fin.sum_univ_two])

end Transformer.Grokking.FlatSpectrum
