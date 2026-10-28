module

public import Transformer.Grokking.FlatSpectrum.Rigidity
public import Transformer.Grokking.FlatSpectrum.Roots

/-!
# Without rigidity a rank-one kernel of any moduli is not recovered

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem G, item 3,
first half, which contains Theorem F's. A rank-one matrix has nuclear norm `‖a bᵀ‖_S1 = ‖a‖ ‖b‖`
(`nuclearNorm_vecMulVec`): the positive semidefinite `P = (‖a‖ / ‖b‖) b̄ bᵀ` has
`P² = (a bᵀ)ᴴ (a bᵀ)` and trace `‖a‖ ‖b‖`. So the norm depends only on the moduli `|a_i|` and
`|b_j|`. Let `r i + s j = 0` on the cells of `Ω` inside the supports of `a` and `b`, and
`w = r i₀ + s j₀ ≠ 0`. The phases of item D.7, `u i = exp (π i Re (r i / w))` and
`v j = exp (π i Re (s j / w))`, have `u i v j = 1` on these cells and `u i₀ v j₀ = −1`. On the
other cells of `Ω` both `(u a)(v b)ᵀ` and `a bᵀ` vanish. So `(u a)(v b)ᵀ` equals `a bᵀ` on `Ω`,
has the same moduli, and differs from `a bᵀ` at `(i₀, j₀)` when `a i₀ b j₀ ≠ 0`
(`exists_vecMulVec_ne_of_not_rigid_of_ne_zero`). Both
have the least norm if one has, so `a bᵀ` is not the unique minimum
(`not_forall_isMinOn_vecMulVec_of_not_rigid`). Item D.7 is the case `|a_i| = |b_j| = 1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

omit [Fintype m] in
/-- `star w ⬝ᵥ w = ∑ |w_i|²`. -/
lemma star_dotProduct_self_eq_sum_norm_sq [Fintype m] (w : m → ℂ) :
    star w ⬝ᵥ w = ((∑ i, ‖w i‖ ^ 2 : ℝ) : ℂ) := by
  simp only [dotProduct, Pi.star_apply, Complex.star_def, Complex.conj_mul',
    Complex.ofReal_sum, Complex.ofReal_pow]

/-- **The nuclear norm of a rank-one matrix**: `‖a bᵀ‖_S1 = ‖a‖ ‖b‖`. -/
theorem nuclearNorm_vecMulVec (a : m → ℂ) (b : n → ℂ) :
    nuclearNorm (vecMulVec a b) = √(∑ i, ‖a i‖ ^ 2) * √(∑ j, ‖b j‖ ^ 2) := by
  set A := ∑ i, ‖a i‖ ^ 2
  set B := ∑ j, ‖b j‖ ^ 2
  have hA : 0 ≤ A := sum_nonneg fun _ _ => sq_nonneg _
  have hB : 0 ≤ B := sum_nonneg fun _ _ => sq_nonneg _
  set c : ℝ := √A / √B
  set V : Matrix n n ℂ := vecMulVec (star b) b
  have hbb : b ⬝ᵥ star b = (B : ℂ) := by
    rw [dotProduct_comm, star_dotProduct_self_eq_sum_norm_sq]
  have hX : (vecMulVec a b)ᴴ * vecMulVec a b = (A : ℂ) • V := by
    rw [conjTranspose_vecMulVec, vecMulVec_mul_vecMulVec, star_dotProduct_self_eq_sum_norm_sq,
      vecMulVec_smul]
  -- `c² B = A` unless `b = 0`, and then `V = 0`.
  have hcV : ((c ^ 2 * B : ℝ) : ℂ) • V = (A : ℂ) • V := by
    by_cases hB0 : B = 0
    · have hb : b = 0 := (dotProduct_star_self_eq_zero (v := b)).mp (by
        rw [dotProduct_comm, hbb, hB0, Complex.ofReal_zero])
      simp [V, hb]
    · rw [div_pow, Real.sq_sqrt hA, Real.sq_sqrt hB, div_mul_cancel₀ A hB0]
  have hP : ((c : ℂ) • V).PosSemidef :=
    (posSemidef_vecMulVec_star_self b).smul (Complex.zero_le_real.mpr (by positivity))
  have hVV : V * V = (B : ℂ) • V := by
    rw [vecMulVec_mul_vecMulVec, hbb, vecMulVec_smul]
  have hPP : (c : ℂ) • V * ((c : ℂ) • V) = (vecMulVec a b)ᴴ * vecMulVec a b := by
    rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, hVV, smul_smul, hX, ← hcV]
    congr 1
    push_cast
    ring
  have h := nuclearNorm_eq_trace_of_mul_self hP hPP
  rw [trace_smul, trace_vecMulVec, star_dotProduct_self_eq_sum_norm_sq, smul_eq_mul,
    ← Complex.ofReal_mul, Complex.ofReal_inj] at h
  rw [h]
  by_cases hB0 : √B = 0
  · simp [c, hB0]
  · rw [div_mul_eq_mul_div, mul_div_assoc, Real.div_sqrt]

example := nuclearNorm_vecMulVec (m := Fin 1) (n := Fin 1) (fun _ => 1) fun _ => 1

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- **Without rigidity on the supports a second matrix of the same moduli fits**,
`review/flat_spectrum.md`, Theorem G, item 3: if `r i + s j = 0` on the cells of `Ω` inside the
supports of `a` and `b`, `r i₀ + s j₀ ≠ 0` and `a i₀ b j₀ ≠ 0`, some `a' b'ᵀ ≠ a bᵀ` with
`|a' i| = |a i|` and `|b' j| = |b j|` equals `a bᵀ` on `Ω`. -/
theorem exists_vecMulVec_ne_of_not_rigid_of_ne_zero {a : m → ℂ} {b : n → ℂ} {Ω : Set (m × n)}
    {r : m → ℂ} {s : n → ℂ} (hrs : ∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0)
    {i₀ : m} {j₀ : n} (h₀ : r i₀ + s j₀ ≠ 0) (ha : a i₀ ≠ 0) (hb : b j₀ ≠ 0) :
    ∃ (a' : m → ℂ) (b' : n → ℂ), (∀ i, ‖a' i‖ = ‖a i‖) ∧ (∀ j, ‖b' j‖ = ‖b j‖) ∧
      (∀ ij ∈ Ω, vecMulVec a' b' ij.1 ij.2 = vecMulVec a b ij.1 ij.2) ∧
      vecMulVec a' b' ≠ vecMulVec a b := by
  set w := r i₀ + s j₀
  set u : m → ℂ := fun i => Complex.exp (↑(Real.pi * (r i / w).re) * Complex.I) with hu
  set v : n → ℂ := fun j => Complex.exp (↑(Real.pi * (s j / w).re) * Complex.I) with hv
  have huv (i : m) (j : n) : vecMulVec (u * a) (v * b) i j =
      Complex.exp (↑(Real.pi * ((r i + s j) / w).re) * Complex.I) * vecMulVec a b i j := by
    simp only [vecMulVec_apply, Pi.mul_apply, hu, hv, add_div, Complex.add_re, mul_add,
      Complex.ofReal_add, add_mul, Complex.exp_add]
    ring
  refine ⟨u * a, v * b, fun i => ?_, fun j => ?_, fun ij hij => ?_, fun h => ?_⟩
  · rw [Pi.mul_apply, norm_mul, hu, Complex.norm_exp_ofReal_mul_I, one_mul]
  · rw [Pi.mul_apply, norm_mul, hv, Complex.norm_exp_ofReal_mul_I, one_mul]
  · by_cases hai : a ij.1 = 0
    · simp [vecMulVec_apply, hai]
    by_cases hbj : b ij.2 = 0
    · simp [vecMulVec_apply, hbj]
    rw [huv, hrs ij hij hai hbj, zero_div, Complex.zero_re, mul_zero, Complex.ofReal_zero,
      zero_mul, Complex.exp_zero, one_mul]
  · have h1 := congrFun (congrFun h i₀) j₀
    rw [huv, div_self h₀, Complex.one_re, mul_one, Complex.exp_pi_mul_I, vecMulVec_apply] at h1
    have h2 : (2 : ℂ) * (a i₀ * b j₀) = 0 := by linear_combination -h1
    exact mul_ne_zero two_ne_zero (mul_ne_zero ha hb) h2

example := exists_vecMulVec_ne_of_not_rigid_of_ne_zero (a := ![2, 1]) (b := ![1, 3])
  (Ω := {ij | ij.2 = ij.1}) (r := ![1, 0]) (s := ![-1, 0])
  (fun ij hij _ _ => by
    obtain ⟨i, j⟩ := ij
    simp only [Set.mem_ofPred_eq] at hij
    subst hij
    fin_cases j <;> simp)
  (i₀ := 0) (j₀ := 1) (by simp) (by simp) (by simp)

/-- **Without rigidity on the supports M does not recover `a bᵀ`**, `review/flat_spectrum.md`,
Theorem G, item 3: if `r i + s j = 0` on the cells of `Ω` inside the supports of `a` and `b`,
`r i₀ + s j₀ ≠ 0` and `a i₀ b j₀ ≠ 0`, then `a bᵀ` is not the unique minimum of the nuclear
norm among the matrices equal to it on `Ω`. -/
theorem not_forall_isMinOn_vecMulVec_of_not_rigid {a : m → ℂ} {b : n → ℂ} {Ω : Set (m × n)}
    {r : m → ℂ} {s : n → ℂ} (hrs : ∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0)
    {i₀ : m} {j₀ : n} (h₀ : r i₀ + s j₀ ≠ 0) (ha : a i₀ ≠ 0) (hb : b j₀ ≠ 0) :
    ¬ ∀ X ∈ {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
        X = vecMulVec a b) := by
  intro H
  obtain ⟨a', b', ha', hb', hmem, hne⟩ :=
    exists_vecMulVec_ne_of_not_rigid_of_ne_zero hrs h₀ ha hb
  have hK := isMinOn_iff.mp ((H _ fun _ _ => rfl).mpr rfl)
  refine hne ((H _ hmem).mp (isMinOn_iff.mpr fun X hX => ?_))
  rw [nuclearNorm_vecMulVec]
  simp only [ha', hb']
  rw [← nuclearNorm_vecMulVec]
  exact hK X hX

example := not_forall_isMinOn_vecMulVec_of_not_rigid (a := ![2, 1]) (b := ![1, 3])
  (Ω := {ij | ij.2 = ij.1}) (r := ![1, 0]) (s := ![-1, 0])
  (fun ij hij _ _ => by
    obtain ⟨i, j⟩ := ij
    simp only [Set.mem_ofPred_eq] at hij
    subst hij
    fin_cases j <;> simp)
  (i₀ := 0) (j₀ := 1) (by simp) (by simp) (by simp)

/-- In the `2 × 3` table, `(1, 1)(1, 1, 0)ᵀ` is not recovered from `{(0, 0), (1, 2)}`: the cell
`(1, 2)` lies off the support of `b`, so row `1` meets no cell inside the supports. -/
example := not_forall_isMinOn_vecMulVec_of_not_rigid (a := ![1, 1]) (b := ![1, 1, 0])
  (Ω := {(0, 0), (1, 2)}) (r := ![0, 1]) (s := 0)
  (fun ij hij _ hb => by
    rcases hij with rfl | rfl
    · simp
    · simp at hb)
  (i₀ := 1) (j₀ := 0) (by simp) (by simp) (by simp)

end Transformer.Grokking.FlatSpectrum
