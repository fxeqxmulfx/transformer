module

public import Transformer.Grokking.FlatSpectrum.TwoPermutations
public import Transformer.Grokking.FlatSpectrum.AddReadoutSpectrum

/-!
# The spectrum of the certificate of two neighbouring diagonals

Source: `review/flat_spectrum.md`, Theorem D, item 3, and its *Numerical check*: "The
certificate of item 3 for the diagonals `{0, 1}` has `‖Y − Q‖_op = cos(π/p)`". On `ℤ/N` let
`Y = (P₀ + P₁)/2`, so `(Y v)(l) = (v l + v (l + 1))/2`, and `Q = J/N`. The columns
`l ↦ ψ(l k)` of the Fourier matrix `F` are eigenvectors of `Y − Q`:
`(Y − Q) F = F diag(λ)` with `λ_k = (1 + ψ(k))/2 − [k = 0]` (`cycleAvg_mul_fourierMatrix`).
As `F* F = F F* = N I`, `‖(Y − Q) v‖ ≤ max_k |λ_k| ‖v‖`. For `k ≠ 0`,
`|λ_k| = |cos(π k/N)|` (`norm_sq_one_add_stdAddChar_div_two`), at most `|cos(π/N)|`
(`cos_sq_mul_div_le`), so `‖(Y − Q) v‖² ≤ cos²(π/N) ‖v‖²`
(`sum_norm_sq_mulVec_cycleAvg_le`); for `N ≥ 2` the character `ψ` reaches it
(`sum_norm_sq_mulVec_cycleAvg_stdAddChar`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset Real

/-- `‖U x‖² = c ‖x‖²` when `U* U = c I`. -/
lemma sum_norm_sq_mulVec_of_conjTranspose_mul {m n : Type*} [Fintype m] [Fintype n]
    [DecidableEq n] {U : Matrix m n ℂ} {c : ℝ} (hU : Uᴴ * U = (c : ℂ) • 1) (x : n → ℂ) :
    ∑ i, ‖(U *ᵥ x) i‖ ^ 2 = c * ∑ j, ‖x j‖ ^ 2 := by
  have hsq (w : m → ℂ) : star w ⬝ᵥ w = ((∑ i, ‖w i‖ ^ 2 : ℝ) : ℂ) := by
    simp only [dotProduct, Pi.star_apply, Complex.star_def, Complex.conj_mul',
      Complex.ofReal_sum, Complex.ofReal_pow]
  have hsq' (w : n → ℂ) : star w ⬝ᵥ w = ((∑ i, ‖w i‖ ^ 2 : ℝ) : ℂ) := by
    simp only [dotProduct, Pi.star_apply, Complex.star_def, Complex.conj_mul',
      Complex.ofReal_sum, Complex.ofReal_pow]
  have h : star (U *ᵥ x) ⬝ᵥ (U *ᵥ x) = (c : ℂ) * (star x ⬝ᵥ x) := by
    rw [star_mulVec, ← dotProduct_mulVec, mulVec_mulVec, hU, smul_mulVec, one_mulVec,
      dotProduct_smul, smul_eq_mul]
  rw [hsq, hsq'] at h
  exact_mod_cast h

example : ∑ i, ‖((1 : Matrix (Fin 2) (Fin 2) ℂ) *ᵥ ![1, 0]) i‖ ^ 2 =
    (1 : ℝ) * ∑ j, ‖(![1, 0] : Fin 2 → ℂ) j‖ ^ 2 :=
  sum_norm_sq_mulVec_of_conjTranspose_mul (by simp) _

variable {N : ℕ} [NeZero N]

/-- **The characters diagonalize the certificate**: `(Y − Q) F = F diag(λ)` on `ℤ/N`, with
`Y = (P₀ + P₁)/2`, `Q = J/N` and `λ_k = (1 + ψ(k))/2 − [k = 0]`. -/
theorem cycleAvg_mul_fourierMatrix :
    (permAvg (Equiv.refl (ZMod N)) (Equiv.addRight 1) - meanProj (ZMod N)) * fourierMatrix N =
      fourierMatrix N * diagonal fun k =>
        (1 + ZMod.stdAddChar k) / 2 - if k = 0 then 1 else 0 := by
  ext i k
  rw [mul_diagonal, Matrix.sub_mul, Matrix.sub_apply]
  have h1 : (permAvg (Equiv.refl (ZMod N)) (Equiv.addRight 1) * fourierMatrix N) i k =
      (permAvg (Equiv.refl (ZMod N)) (Equiv.addRight 1) *ᵥ fun j => fourierMatrix N j k) i :=
    rfl
  have h2 : (meanProj (ZMod N) * fourierMatrix N) i k = if k = 0 then 1 else 0 := by
    simp only [meanProj, fourierMatrix, mul_apply, of_apply, ← mul_sum, sum_stdAddChar_mul,
      ZMod.card]
    split_ifs
    · exact inv_mul_cancel₀ (by exact_mod_cast NeZero.ne N)
    · exact mul_zero _
  rw [h1, h2, permAvg_mulVec]
  simp only [fourierMatrix, of_apply, Equiv.refl_apply, Equiv.coe_addRight, add_mul, one_mul,
    AddChar.map_add_eq_mul]
  split_ifs with hk
  · simp [hk]
  · ring

/-- `|(1 + ψ(k))/2|² = cos²(π k/N)`. -/
lemma norm_sq_one_add_stdAddChar_div_two (k : ZMod N) :
    ‖(1 + ZMod.stdAddChar k) / 2‖ ^ 2 = cos (π * k.val / N) ^ 2 := by
  set x := π * k.val / N
  have h : (1 + ZMod.stdAddChar k) / 2 =
      (((1 + cos (2 * x)) / 2 : ℝ) : ℂ) + ((sin (2 * x) / 2 : ℝ) : ℂ) * Complex.I := by
    rw [ZMod.stdAddChar_apply, ZMod.toCircle_apply,
      show 2 * (π : ℂ) * Complex.I * (k.val : ℂ) / (N : ℂ) = ((2 * x : ℝ) : ℂ) * Complex.I by
        simp only [x]; push_cast; ring,
      Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
    push_cast
    ring
  rw [h, Complex.sq_norm, Complex.normSq_add_mul_I, cos_sq x]
  linear_combination (1 / 4 : ℝ) * sin_sq_add_cos_sq (2 * x)

example : ‖(1 + ZMod.stdAddChar (0 : ZMod 3)) / 2‖ ^ 2 = cos (π * (0 : ZMod 3).val / 3) ^ 2 :=
  mod_cast norm_sq_one_add_stdAddChar_div_two (N := 3) 0

omit [NeZero N] in
/-- `cos²(π j/N) ≤ cos²(π/N)` for `0 < j < N`. -/
lemma cos_sq_mul_div_le {j : ℕ} (hj : 0 < j) (hjN : j < N) :
    cos (π * j / N) ^ 2 ≤ cos (π / N) ^ 2 := by
  have hN : (0 : ℝ) < N := by exact_mod_cast hj.trans hjN
  have h1 : (1 : ℝ) ≤ j := by exact_mod_cast hj
  have h2 : (j : ℝ) + 1 ≤ N := by exact_mod_cast hjN
  have hx : π / N ≤ π * j / N := div_le_div_of_nonneg_right (by nlinarith [pi_pos]) hN.le
  have hx' : π * j / N ≤ π - π / N := by
    rw [show π - π / N = π * (N - 1) / N by field_simp]
    exact div_le_div_of_nonneg_right (by nlinarith [pi_pos]) hN.le
  have h0 : 0 ≤ π / N := by positivity
  have hc1 : cos (π * j / N) ≤ cos (π / N) :=
    cos_le_cos_of_nonneg_of_le_pi h0 (by linarith) hx
  have hc2 : cos (π - π * j / N) ≤ cos (π / N) :=
    cos_le_cos_of_nonneg_of_le_pi h0 (by linarith) (by linarith)
  rw [cos_pi_sub] at hc2
  exact sq_le_sq' (by linarith) hc1

example : cos (π * (1 : ℕ) / 3) ^ 2 ≤ cos (π / 3) ^ 2 :=
  cos_sq_mul_div_le (N := 3) (by norm_num) (by norm_num)

omit [NeZero N] in
/-- `cos(π/N) ≥ 0` for `N ≥ 2`. -/
lemma cos_pi_div_nonneg (hN : 2 ≤ N) : 0 ≤ cos (π / N) := by
  have hN' : (2 : ℝ) ≤ N := by exact_mod_cast hN
  have h0 : 0 ≤ π / N := by positivity
  exact cos_nonneg_of_neg_pi_div_two_le_of_le (by linarith [pi_pos])
    (div_le_div_of_nonneg_left pi_pos.le two_pos hN')

example : 0 ≤ cos (π / (3 : ℕ)) := cos_pi_div_nonneg (by norm_num)

/-- Every eigenvalue has `|λ_k|² ≤ cos²(π/N)`. -/
lemma norm_sq_cycleEigen_le (k : ZMod N) :
    ‖(1 + ZMod.stdAddChar k) / 2 - (if k = 0 then 1 else 0 : ℂ)‖ ^ 2 ≤ cos (π / N) ^ 2 := by
  split_ifs with hk
  · simp [hk, sq_nonneg]
  · rw [sub_zero, norm_sq_one_add_stdAddChar_div_two]
    exact cos_sq_mul_div_le (Nat.pos_of_ne_zero ((ZMod.val_ne_zero k).mpr hk)) (ZMod.val_lt k)

/-- **The bound**: `‖(Y − Q) v‖² ≤ cos²(π/N) ‖v‖²` on `ℤ/N`, `Y = (P₀ + P₁)/2`, `Q = J/N`. -/
theorem sum_norm_sq_mulVec_cycleAvg_le (v : ZMod N → ℂ) :
    ∑ l, ‖((permAvg (Equiv.refl (ZMod N)) (Equiv.addRight 1) - meanProj (ZMod N)) *ᵥ v) l‖ ^ 2 ≤
      cos (π / N) ^ 2 * ∑ l, ‖v l‖ ^ 2 := by
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have key : (N : ℂ) • (permAvg (Equiv.refl (ZMod N)) (Equiv.addRight 1) - meanProj (ZMod N)) =
      fourierMatrix N * (diagonal fun k =>
        (1 + ZMod.stdAddChar k) / 2 - if k = 0 then 1 else 0) * (fourierMatrix N)ᴴ := by
    rw [← cycleAvg_mul_fourierMatrix, Matrix.mul_assoc, fourierMatrix_mul_conjTranspose,
      Matrix.mul_smul, Matrix.mul_one]
  have hF := sum_norm_sq_mulVec_of_conjTranspose_mul (U := fourierMatrix N) (c := N)
    (by exact_mod_cast conjTranspose_fourierMatrix_mul)
  have hF' := sum_norm_sq_mulVec_of_conjTranspose_mul (U := (fourierMatrix N)ᴴ) (c := N)
    (by rw [conjTranspose_conjTranspose]; exact_mod_cast fourierMatrix_mul_conjTranspose)
  have h1 := congrArg (fun w : ZMod N → ℂ => ∑ l, ‖w l‖ ^ 2) (congrArg (· *ᵥ v) key)
  simp only [smul_mulVec, Pi.smul_apply, smul_eq_mul, norm_mul, Complex.norm_natCast, mul_pow,
    ← mul_sum, ← mulVec_mulVec, hF] at h1
  have h2 := (sum_le_sum fun k (_ : k ∈ univ) => mul_le_mul_of_nonneg_right
    (norm_sq_cycleEigen_le k) (sq_nonneg ‖((fourierMatrix N)ᴴ *ᵥ v) k‖))
  rw [← mul_sum, hF'] at h2
  simp only [mulVec_diagonal, norm_mul, mul_pow] at h1
  have h3 : (N : ℝ) ^ 2 * ∑ l, ‖((permAvg (Equiv.refl (ZMod N)) (Equiv.addRight 1) -
      meanProj (ZMod N)) *ᵥ v) l‖ ^ 2 ≤ (N : ℝ) ^ 2 * (cos (π / N) ^ 2 * ∑ l, ‖v l‖ ^ 2) := by
    rw [h1]
    nlinarith
  exact le_of_mul_le_mul_left h3 (by positivity)

example := sum_norm_sq_mulVec_cycleAvg_le (N := 3) ![1, 0, 0]

/-- The character `ψ` is an eigenvector: `(Y − Q) ψ = ((1 + ψ(1))/2) ψ` for `N ≥ 2`. -/
lemma cycleAvg_mulVec_stdAddChar (hN : 2 ≤ N) :
    (permAvg (Equiv.refl (ZMod N)) (Equiv.addRight 1) - meanProj (ZMod N)) *ᵥ
        (fun l => ZMod.stdAddChar l) =
      ((1 + ZMod.stdAddChar (1 : ZMod N)) / 2 : ℂ) • fun l => ZMod.stdAddChar l := by
  have : Fact (1 < N) := ⟨hN⟩
  ext l
  have h := congrFun (congrFun cycleAvg_mul_fourierMatrix l) 1
  simp only [mul_diagonal, one_ne_zero, ite_false, sub_zero] at h
  simpa [fourierMatrix, mul_apply, mulVec, dotProduct, mul_comm] using h

/-- **The bound is reached**: for `N ≥ 2`, `‖(Y − Q) ψ‖² = cos²(π/N) ‖ψ‖²`. -/
theorem sum_norm_sq_mulVec_cycleAvg_stdAddChar (hN : 2 ≤ N) :
    ∑ l, ‖((permAvg (Equiv.refl (ZMod N)) (Equiv.addRight 1) - meanProj (ZMod N)) *ᵥ
        fun l => ZMod.stdAddChar l) l‖ ^ 2 =
      cos (π / N) ^ 2 * ∑ l : ZMod N, ‖(ZMod.stdAddChar l : ℂ)‖ ^ 2 := by
  have : Fact (1 < N) := ⟨hN⟩
  rw [cycleAvg_mulVec_stdAddChar hN]
  simp only [Pi.smul_apply, smul_eq_mul, norm_mul, mul_pow, ← mul_sum,
    norm_sq_one_add_stdAddChar_div_two, ZMod.val_one, Nat.cast_one, mul_one]

example := sum_norm_sq_mulVec_cycleAvg_stdAddChar (N := 3) (by norm_num)

end Transformer.Grokking.FlatSpectrum
