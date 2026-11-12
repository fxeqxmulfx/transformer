module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Basic
public import Transformer.Grokking.FlatSpectrum.Blocks

/-!
# A nuclear budget recovers a rank-one missing entry

Source: `review/flat_spectrum.md`, CM6. With three observed `+1` entries in a 2x2 matrix,
the nuclear-norm budget two forces the fourth entry to be `+1`. This complements SB2's
full-rank example. The proof uses the actual 2x2 nuclear-norm formula, not a rank assumption
in the feasible set. It is not a theorem for arbitrary partially observed tables.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Matrix

/-- The three observed entries are fixed; only the fourth is free (CM6). -/
def rankOnePatch (z : ℂ) : Matrix (Fin 2) (Fin 2) ℂ := !![1, 1; 1, z]

/-- The sum-of-singular-values budget singles out the rank-one completion (CM6). -/
theorem rankOnePatch_budget_iff (z : ℂ) : nuclearNorm (rankOnePatch z) ≤ 2 ↔ z = 1 := by
  constructor
  · intro h
    rw [nuclearNorm_fin_two] at h
    have hs := (sq_le_sq₀ (Real.sqrt_nonneg _) (by norm_num : (0 : ℝ) ≤ 2)).2 h
    rw [Real.sq_sqrt (add_nonneg (frobeniusSq_nonneg _) (by positivity))] at hs
    simp [rankOnePatch, frobeniusSq, Fin.sum_univ_two, Matrix.det_fin_two] at hs
    have hr : 1 - z.re ≤ ‖z - 1‖ := calc
      1 - z.re = (1 - z).re := by simp
      _ ≤ ‖1 - z‖ := Complex.re_le_norm _
      _ = ‖z - 1‖ := norm_sub_rev _ _
    rw [Complex.sq_norm, Complex.normSq_apply] at hs
    apply Complex.ext
    · simp only [Complex.one_re]
      nlinarith [sq_nonneg z.im, sq_nonneg (z.re - 1)]
    · simp only [Complex.one_im]
      nlinarith [sq_nonneg z.im, sq_nonneg (z.re - 1)]
  · rintro rfl
    norm_num [rankOnePatch, nuclearNorm_fin_two, frobeniusSq, Fin.sum_univ_two,
      Matrix.det_fin_two]

/-- Any complex matrix matching these three observations and the nuclear budget must
also match the unseen entry (CM6). No rank or factorization constraint is imposed. -/
theorem rankOne_missing {X : Matrix (Fin 2) (Fin 2) ℂ}
    (h00 : X 0 0 = 1) (h01 : X 0 1 = 1) (h10 : X 1 0 = 1)
    (hB : nuclearNorm X ≤ 2) : X 1 1 = 1 := by
  have he : X = rankOnePatch (X 1 1) := by
    ext i j
    fin_cases i <;> fin_cases j <;> simp [rankOnePatch, h00, h01, h10]
  rw [he] at hB
  exact (rankOnePatch_budget_iff _).1 hB

example := rankOne_missing (X := rankOnePatch 1) rfl rfl rfl
  ((rankOnePatch_budget_iff _).2 rfl)

/-- The nonzero Fourier reading of a binary probability vector (CM6). -/
def signCode (c : Bool) : ℂ := if c then 1 else -1

private theorem binary_moment_re {P : Table (Fin 2) (Fin 2) Bool} (hP : Rows P)
    (i j : Fin 2) : (moment signCode P i j).re = 2 * P i j true - 1 := by
  have hs := hP.2 i j
  simp at hs
  simp [moment, signCode]
  linarith

private theorem binary_moment_one {P : Table (Fin 2) (Fin 2) Bool} (hP : Rows P)
    (i j : Fin 2) (hp : P i j true = 1) : moment signCode P i j = 1 := by
  apply Complex.ext
  · rw [binary_moment_re hP, hp]
    norm_num
  · simp [moment, signCode]

/-- A shared probability table fitting the three observed one-hot labels also recovers
the unobserved class under the same nuclear budget (CM6). -/
theorem rankOne_missing_probability {P : Table (Fin 2) (Fin 2) Bool} (hP : Rows P)
    (h00 : P 0 0 true = 1) (h01 : P 0 1 true = 1) (h10 : P 1 0 true = 1)
    (hB : nuclearNorm (moment signCode P) ≤ 2) : P 1 1 true = 1 := by
  have hm := rankOne_missing (binary_moment_one hP 0 0 h00)
    (binary_moment_one hP 0 1 h01) (binary_moment_one hP 1 0 h10) hB
  have hr := congrArg Complex.re hm
  rw [binary_moment_re hP] at hr
  simp only [Complex.one_re] at hr
  linarith

example := binary_moment_re
  (rows_pointMass (fun (_ : Fin 2) (_ : Fin 2) => true)) 0 0

example := binary_moment_one
  (rows_pointMass (fun (_ : Fin 2) (_ : Fin 2) => true)) 0 0 (by simp [pointMass])

example := rankOne_missing_probability
  (rows_pointMass (fun (_ : Fin 2) (_ : Fin 2) => true))
  (by simp [pointMass]) (by simp [pointMass]) (by simp [pointMass])
  (by
    rw [moment_pointMass]
    have he : (fun (_ : Fin 2) (_ : Fin 2) => signCode true) = rankOnePatch 1 := by
      ext i j
      fin_cases i <;> fin_cases j <;> norm_num [rankOnePatch, signCode]
    rw [he]
    exact (rankOnePatch_budget_iff _).2 rfl)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
