module

public import Transformer.Grokking.FlatSpectrum.Training.AlmostEvery

/-!
# Width below `min(m, n)`: the `2 × 2` identity at width one

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD10, its
counterexample. The width `min(m, n)` of `Training.AlmostEvery` cannot be lowered. For the
squared loss on all cells of the `2 × 2` identity and a weight decay `0 ≤ λ < 1`:

* every factorization `(U, V)` of width one has weight-decayed loss at least
  `1 − (1 − λ)² / 2` (`le_decayedLoss_width_one`): with `q = ⟨U, V⟩`,
  `F = (‖U‖² ‖V‖² − 2 q + 2) / 2 + λ (‖U‖² + ‖V‖²) / 2 ≥ |q|² / 2 − (1 − λ) |q| + 1`;
* the factorization `√(1 − λ) · √(1 − λ)` of width two has weight-decayed loss `2 λ − λ²`
  (`decayedLoss_sqrt_smul_one`), smaller by `(1 − λ)² / 2`.

So no factorization of width one is a decayed minimum (`not_isDecayedMin_width_one`): a run of
width one converges (WD9), and its limit is stationary, but it is never a decayed minimum. At
width two decayed minima exist and almost every run reaches one.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

/-- `‖1‖_S2² = 2` for the `2 × 2` identity. -/
lemma frobeniusSq_one_fin_two : frobeniusSq (1 : Matrix (Fin 2) (Fin 2) ℂ) = 2 := by
  norm_num [frobeniusSq, Fin.sum_univ_two, Matrix.one_apply]

/-- **Width one costs at least `1 − (1 − λ)² / 2`** (`review/flat_spectrum.md`, item WD10):
the squared loss on all cells of the `2 × 2` identity, for `λ ≥ 0`. -/
theorem le_decayedLoss_width_one {lam : ℝ} (h0 : 0 ≤ lam) (U V : Matrix (Fin 2) (Fin 1) ℂ) :
    1 - (1 - lam) ^ 2 / 2 ≤
      decayedLoss (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) lam U V := by
  have hmask : mask Set.univ (U * Vᴴ - 1) = U * Vᴴ - 1 := by
    ext i j
    simp [mask]
  have hprod : frobeniusSq (U * Vᴴ) = frobeniusSq V * frobeniusSq U := by
    rw [← pairing_self, pairing_mul_conjTranspose_unit, pairing_self]
  have hpair : pairing (U * Vᴴ) 1 = pairing U V := by
    rw [pairing_mul_right, conjTranspose_conjTranspose, Matrix.one_mul]
  have ha := frobeniusSq_nonneg U
  have hb := frobeniusSq_nonneg V
  have hp := pairing_sq_le U V
  have hq : |pairing U V| ≤ (frobeniusSq U + frobeniusSq V) / 2 :=
    abs_le_of_sq_le_sq (hp.trans (by nlinarith [sq_nonneg (frobeniusSq U - frobeniusSq V)]))
      (by positivity)
  have ht := abs_nonneg (pairing U V)
  have ht2 := sq_abs (pairing U V)
  have hle := le_abs_self (pairing U V)
  rw [decayedLoss, fitLoss, hmask, frobeniusSq_sub, hprod, hpair, frobeniusSq_one_fin_two]
  nlinarith [sq_nonneg (|pairing U V| - (1 - lam)), mul_le_mul_of_nonneg_left hq h0]

example := le_decayedLoss_width_one (lam := 1 / 2) (by norm_num) 0 0

/-- **Width two costs `2 λ − λ²`** (`review/flat_spectrum.md`, item WD10): the factorization
`√(1 − λ) · √(1 − λ)` of `(1 − λ) · 1` for the squared loss on all cells of the `2 × 2`
identity, for `λ ≤ 1`. -/
theorem decayedLoss_sqrt_smul_one {lam : ℝ} (h1 : lam ≤ 1) :
    decayedLoss (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) lam
      (√(1 - lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) (√(1 - lam) • 1) = 2 * lam - lam ^ 2 := by
  have hs : √(1 - lam) * √(1 - lam) = 1 - lam := Real.mul_self_sqrt (by linarith)
  have hmul : (√(1 - lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) * (√(1 - lam) • 1)ᴴ - 1 =
      (-lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
    rw [conjTranspose_smul, conjTranspose_one, star_trivial, Matrix.smul_mul, Matrix.mul_smul,
      Matrix.one_mul, smul_smul, hs]
    module
  have hmask : mask Set.univ ((-lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) = (-lam) • 1 := by
    ext i j
    simp [mask]
  rw [decayedLoss, fitLoss, hmul, hmask, frobeniusSq_smul, frobeniusSq_smul,
    frobeniusSq_one_fin_two, sq √(1 - lam), hs]
  ring

example := decayedLoss_sqrt_smul_one (lam := 1 / 2) (by norm_num)

/-- **Width one is too narrow for the `2 × 2` identity** (`review/flat_spectrum.md`, item
WD10): for the squared loss on all cells of the `2 × 2` identity and a weight decay
`0 ≤ λ < 1`, no factorization of width one has the least weight-decayed loss among the
factorizations of every width; the width `min(m, n) = 2` of
`exists_pos_ae_tendsto_decayRun_isDecayedMin` cannot be lowered. -/
theorem not_isDecayedMin_width_one {lam : ℝ} (h0 : 0 ≤ lam) (h1 : lam < 1)
    (U V : Matrix (Fin 2) (Fin 1) ℂ) :
    ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) lam U V := fun h => by
  have h2 := h 2 (√(1 - lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) (√(1 - lam) • 1)
  rw [decayedLoss_sqrt_smul_one h1.le] at h2
  have h3 := le_decayedLoss_width_one h0 U V
  nlinarith [sq_pos_of_pos (sub_pos.mpr h1)]

example := not_isDecayedMin_width_one (lam := 1 / 2) (by norm_num) (by norm_num) 0 0

/-- At width two the same kernel has decayed minima for every `λ > 0`. -/
example := exists_isDecayedMin_of_card_le (k := Fin 2)
  (fitLoss_first_order (Set.univ : Set (Fin 2 × Fin 2)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) (lam := 1 / 2) (by norm_num) (fitLoss_nonneg Set.univ 1)
  (Or.inl le_rfl)

/-- At width one its runs converge all the same, to stationary points. -/
example := exists_pos_tendsto_decayRun (k := Fin 1)
  (fitLoss_first_order (Set.univ : Set (Fin 2 × Fin 2)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) (lam := 1 / 2) (by norm_num) (fitLoss_nonneg Set.univ 1) 2

end Transformer.Grokking.FlatSpectrum.Training
