module

public import Transformer.Grokking.FlatSpectrum.CrossCertificate

/-!
# The cross determines `a bᵀ`, yet the least nuclear norm misses it

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel* (added). Let
`m² = N − 1`, `α² = (m + 1)/2`, `β² = (m − 1)/2`, and `f = e_{x₀} + 1'/m`, `g = e_{x₀} − 1'/m`
with `1'` the indicator of the indices other than `x₀`. The matrix `X₀ = α² f fᵀ − β² g gᵀ`
(with `f, g` built at `i₀` on the left and at `j₀` on the right) is `1` on the cross of row `i₀`
and column `j₀` and `1/(N − 1)` elsewhere (`crossWitness_eq_mul`). Its factors `[α f, β g]`
and `[α f, −β g]` have `‖·‖²_F = 2m` (`frobeniusSq_crossFactor`), so `‖X₀‖_S1 ≤ 2m`. With the
certificate bound of `CrossCertificate`, `2√(N − 1)` is the least nuclear norm of the
completions of `a bᵀ` from the cross when `|a i| = |b j| = 1`
(`isLeast_nuclearNorm_vecMulVec_cross`). For `N ≥ 3` it is below `N = ‖a bᵀ‖_S1`: the cross
determines `a bᵀ` among rank-one matrices, yet the least nuclear norm does not return it
(`vecMulVec_cross_determined_not_isMinOn`). For `ψ(a + b)`, the values on column `0` and row `0`
are `ψ(a)` and `ψ(b)` and fix `ψ(a + b) = ψ(a) ψ(b)`, yet their completions reach
`2√(p − 1) < p` (`isLeast_nuclearNorm_addKernel_cross`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The witness on the cross: `1` on row `i₀` and column `j₀`, `1/(N − 1)` elsewhere. -/
noncomputable def crossWitness (i₀ j₀ : n) : Matrix n n ℂ :=
  of fun i j => ((if i = i₀ ∨ j = j₀ then 1 else ((Fintype.card n : ℝ) - 1)⁻¹ : ℝ) : ℂ)

/-- The factor `[α f, β g]` with `f = e_{x₀} + 1'/m` and `g = e_{x₀} − 1'/m`. -/
noncomputable def crossFactor (x₀ : n) (m α β : ℝ) : Matrix n (Fin 2) ℂ :=
  of fun x => ![((α * if x = x₀ then 1 else m⁻¹ : ℝ) : ℂ),
    ((β * if x = x₀ then 1 else -m⁻¹ : ℝ) : ℂ)]

/-- `X₀ = [α f, β g] · [α f, −β g]ᴴ` when `m² = N − 1`, `α² = (m + 1)/2`, `β² = (m − 1)/2`. -/
lemma crossWitness_eq_mul {m α β : ℝ} (hm : m * m = (Fintype.card n : ℝ) - 1) (hm0 : m ≠ 0)
    (hα : α * α = (m + 1) / 2) (hβ : β * β = (m - 1) / 2) (i₀ j₀ : n) :
    crossWitness i₀ j₀ = crossFactor i₀ m α β * (crossFactor j₀ m α (-β))ᴴ := by
  have hinv : m * m⁻¹ = 1 := mul_inv_cancel₀ hm0
  ext i j
  rw [mul_apply, Fin.sum_univ_two]
  simp only [crossWitness, crossFactor, conjTranspose_apply, of_apply, cons_val_zero,
    cons_val_one, Complex.star_def, Complex.conj_ofReal, ← Complex.ofReal_mul,
    ← Complex.ofReal_add, Complex.ofReal_inj]
  by_cases hi : i = i₀ <;> by_cases hj : j = j₀ <;>
    simp only [hi, hj, true_or, or_true, or_self, ↓reduceIte, mul_one]
  · linear_combination -hα + hβ
  · linear_combination (-m⁻¹) * hα + (-m⁻¹) * hβ - hinv
  · linear_combination (-m⁻¹) * hα + (-m⁻¹) * hβ - hinv
  · rw [← hm, mul_inv]
    linear_combination (-(m⁻¹ * m⁻¹)) * hα + (m⁻¹ * m⁻¹) * hβ

example := crossWitness_eq_mul (n := Fin 2) (m := 1) (α := 1) (β := 0) (by norm_num) one_ne_zero
  (by norm_num) (by norm_num) 0 0

/-- `‖[α f, β g]‖²_F = 2m` when `m² = N − 1` and `α² + β² = m`. -/
lemma frobeniusSq_crossFactor {m α β : ℝ} (hN : 1 ≤ Fintype.card n)
    (hm : m * m = (Fintype.card n : ℝ) - 1) (hm0 : m ≠ 0) (hα : α * α = (m + 1) / 2)
    (hβ : β * β = (m - 1) / 2) (x₀ : n) : frobeniusSq (crossFactor x₀ m α β) = 2 * m := by
  have hinv : m * m⁻¹ = 1 := mul_inv_cancel₀ hm0
  have hx (x : n) (hx : x ∈ ({x₀}ᶜ : Finset n)) : ∑ k, ‖crossFactor x₀ m α β x k‖ ^ 2 =
      (α * α + β * β) * (m⁻¹ * m⁻¹) := by
    have hx' : x ≠ x₀ := by simpa using hx
    simp only [Fin.sum_univ_two, crossFactor, of_apply, cons_val_zero, cons_val_one, hx',
      ↓reduceIte, Complex.norm_real, Real.norm_eq_abs, sq_abs]
    ring
  rw [frobeniusSq, Fintype.sum_eq_add_sum_compl x₀, sum_congr rfl hx, sum_const, nsmul_eq_mul,
    card_compl_singleton_cast hN]
  simp only [Fin.sum_univ_two, crossFactor, of_apply, cons_val_zero, cons_val_one, ↓reduceIte,
    mul_one, Complex.norm_real, Real.norm_eq_abs, sq_abs]
  rw [← hm]
  linear_combination (1 + m * m * (m⁻¹ * m⁻¹)) * hα + (1 + m * m * (m⁻¹ * m⁻¹)) * hβ +
    m * (m * m⁻¹ + 1) * hinv

example := frobeniusSq_crossFactor (n := Fin 2) (m := 1) (α := 1) (β := 0) (by simp)
  (by norm_num) one_ne_zero (by norm_num) (by norm_num) 0

/-- `2√(N − 1) < N` for `N ≥ 3`. -/
lemma two_mul_sqrt_lt {N : ℕ} (hN : 3 ≤ N) : 2 * √((N : ℝ) - 1) < N := by
  have h3 : (3 : ℝ) ≤ N := by exact_mod_cast hN
  have h := (Real.sqrt_lt' (x := (N : ℝ) - 1) (show (0 : ℝ) < N / 2 by positivity)).mpr
    (by nlinarith)
  linarith

example := two_mul_sqrt_lt (N := 3) le_rfl

/-- **The least nuclear norm on the cross**, `review/flat_spectrum.md`, *Exact recovery of a
rank-one kernel*: for `N ≥ 2` and `|a i| = |b j| = 1`, the least nuclear norm of the matrices
equal to `a bᵀ` on row `i₀` and column `j₀` is `2√(N − 1)`. -/
theorem isLeast_nuclearNorm_vecMulVec_cross {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) (hN : 2 ≤ Fintype.card n) (i₀ j₀ : n) :
    IsLeast (nuclearNorm '' {X | ∀ ij ∈ {ij : n × n | ij.1 = i₀ ∨ ij.2 = j₀},
      X ij.1 ij.2 = vecMulVec a b ij.1 ij.2}) (2 * √((Fintype.card n : ℝ) - 1)) := by
  have hN' : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hN
  set m := √((Fintype.card n : ℝ) - 1) with hmdef
  have hm : m * m = (Fintype.card n : ℝ) - 1 := Real.mul_self_sqrt (by linarith)
  have hm1 : 1 ≤ m := Real.one_le_sqrt.mpr (by linarith)
  have hα := Real.mul_self_sqrt (show (0 : ℝ) ≤ (m + 1) / 2 by linarith)
  have hβ := Real.mul_self_sqrt (show (0 : ℝ) ≤ (m - 1) / 2 by linarith)
  have hup : nuclearNorm (crossWitness i₀ j₀) ≤ 2 * m := by
    have h := nuclearNorm_le_of_eq_mul _ _ (crossWitness_eq_mul hm (by linarith) hα hβ i₀ j₀)
    rw [frobeniusSq_crossFactor (by omega) hm (by linarith) hα hβ,
      frobeniusSq_crossFactor (by omega) hm (by linarith) hα (by rw [neg_mul_neg]; exact hβ)] at h
    linarith
  have hJ : ∀ ij ∈ {ij : n × n | ij.1 = i₀ ∨ ij.2 = j₀},
      crossWitness i₀ j₀ ij.1 ij.2 = onesMatrix n ij.1 ij.2 := fun ij hij => by
    simp only [crossWitness, onesMatrix, of_apply, Set.mem_ofPred_eq.mp hij, ↓reduceIte,
      Complex.ofReal_one]
  have hsa (i : n) : star (a i) * a i = 1 := by
    rw [Complex.star_def, Complex.conj_mul', ha i]
    norm_num
  have hsb (j : n) : star (b j) * b j = 1 := by
    rw [Complex.star_def, Complex.conj_mul', hb j]
    norm_num
  refine ⟨⟨diagonal a * crossWitness i₀ j₀ * diagonal b, fun ij hij => ?_, ?_⟩, ?_⟩
  · rw [mul_diagonal, diagonal_mul, hJ ij hij, vecMulVec_apply, onesMatrix, of_apply, mul_one]
  · rw [nuclearNorm_diagonal_mul_mul_diagonal ha hb]
    exact le_antisymm hup (two_mul_sqrt_le_nuclearNorm_of_cross hN hJ)
  · rintro t ⟨X, hX, rfl⟩
    rw [← nuclearNorm_diagonal_mul_mul_diagonal (a := star a) (b := star b)
      (fun i => by rw [Pi.star_apply, norm_star, ha i]) (fun j => by rw [Pi.star_apply,
        norm_star, hb j]) X]
    refine two_mul_sqrt_le_nuclearNorm_of_cross (i₀ := i₀) (j₀ := j₀) hN fun ij hij => ?_
    rw [mul_diagonal, diagonal_mul, hX ij hij, vecMulVec_apply, onesMatrix, of_apply,
      Pi.star_apply, Pi.star_apply]
    linear_combination (star (b ij.2) * b ij.2) * hsa ij.1 + hsb ij.2

example := isLeast_nuclearNorm_vecMulVec_cross (a := fun _ : Fin 2 => 1) (b := fun _ => 1)
  (by simp) (by simp) (by simp) 0 0

/-- **The cross determines `a bᵀ`, yet the least nuclear norm misses it**,
`review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*: for `N ≥ 3` and
`|a i| = |b j| = 1`, every rank-one `a' b'ᵀ` equal to `a bᵀ` on row `i₀` and column `j₀` is
`a bᵀ`, while `a bᵀ` does not minimize the nuclear norm among the matrices equal to it there. -/
theorem vecMulVec_cross_determined_not_isMinOn {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) (hN : 3 ≤ Fintype.card n) (i₀ j₀ : n) :
    (∀ a' b' : n → ℂ, (∀ ij ∈ {ij : n × n | ij.1 = i₀ ∨ ij.2 = j₀},
      vecMulVec a' b' ij.1 ij.2 = vecMulVec a b ij.1 ij.2) → vecMulVec a' b' = vecMulVec a b) ∧
      ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ {ij : n × n | ij.1 = i₀ ∨ ij.2 = j₀},
        X ij.1 ij.2 = vecMulVec a b ij.1 ij.2} (vecMulVec a b) := by
  have h₀ : a i₀ * b j₀ ≠ 0 := fun h => by
    have h1 : ‖a i₀ * b j₀‖ = 1 := by rw [norm_mul, ha, hb, one_mul]
    rw [h, norm_zero] at h1
    exact zero_ne_one h1
  refine ⟨fun a' b' h => vecMulVec_eq_of_cross h₀ h, fun hmin => ?_⟩
  obtain ⟨⟨X, hX, hXv⟩, -⟩ := isLeast_nuclearNorm_vecMulVec_cross ha hb (by omega) i₀ j₀
  have h := isMinOn_iff.mp hmin X hX
  rw [nuclearNorm_vecMulVec_of_norm_eq_one ha hb, hXv] at h
  linarith [two_mul_sqrt_lt hN]

example := vecMulVec_cross_determined_not_isMinOn (a := fun _ : Fin 3 => 1) (b := fun _ => 1)
  (by simp) (by simp) (by simp) 0 0

/-- **`ψ(a)` and `ψ(b)` do not give `ψ(a + b)` by least nuclear norm**,
`review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*: for `p ≥ 3`, the least
nuclear norm of the matrices equal to the `a + b` kernel on row `0` and column `0` is
`2√(p − 1)`, below its own `p`. -/
theorem isLeast_nuclearNorm_addKernel_cross {p : ℕ} [NeZero p] (hp : 3 ≤ p) :
    IsLeast (nuclearNorm '' {X | ∀ ij ∈ {ij : ZMod p × ZMod p | ij.1 = 0 ∨ ij.2 = 0},
      X ij.1 ij.2 = addKernel p ij.1 ij.2}) (2 * √((p : ℝ) - 1)) ∧
      2 * √((p : ℝ) - 1) < nuclearNorm (addKernel p) := by
  have h := isLeast_nuclearNorm_vecMulVec_cross (a := (ZMod.stdAddChar : ZMod p → ℂ))
    (b := ZMod.stdAddChar) (by simp) (by simp) (by rw [ZMod.card]; omega) 0 0
  rw [ZMod.card] at h
  rw [addKernel_eq_vecMulVec, nuclearNorm_vecMulVec_of_norm_eq_one (by simp) (by simp),
    ZMod.card]
  exact ⟨h, two_mul_sqrt_lt hp⟩

example := isLeast_nuclearNorm_addKernel_cross (p := 3) le_rfl

end Transformer.Grokking.FlatSpectrum
