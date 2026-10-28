module

public import Transformer.Grokking.FlatSpectrum.Rigidity
public import Transformer.Grokking.FlatSpectrum.Roots

/-!
# Without rigidity a rank-one kernel of any moduli is not recovered

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem F, item 3,
first half (added). A rank-one matrix has nuclear norm `‖a bᵀ‖_S1 = ‖a‖ ‖b‖`
(`nuclearNorm_vecMulVec`): the positive semidefinite `P = (‖a‖ / ‖b‖) b̄ bᵀ` has
`P² = (a bᵀ)ᴴ (a bᵀ)` and trace `‖a‖ ‖b‖`. So the norm depends only on the moduli `|a_i|` and
`|b_j|`. If `r i + s j = 0` on `Ω` and `w = r i₀ + s j₀ ≠ 0`, the phases of item D.7,
`u i = exp (π i Re (r i / w))` and `v j = exp (π i Re (s j / w))`, have `u i v j = 1` on `Ω` and
`u i₀ v j₀ = −1`. So `(u a)(v b)ᵀ` equals `a bᵀ` on `Ω`, has the same moduli, and differs from
`a bᵀ` at `(i₀, j₀)` when `a i₀ b j₀ ≠ 0` (`exists_vecMulVec_ne_of_not_rigid_of_ne_zero`). Both
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

omit [Fintype n] [DecidableEq n] in
/-- **Without rigidity a second matrix of the same moduli fits**, `review/flat_spectrum.md`,
Theorem F, item 3: if `r i + s j = 0` on `Ω`, `r i₀ + s j₀ ≠ 0` and `a i₀ b j₀ ≠ 0`, some
`a' b'ᵀ ≠ a bᵀ` with `|a' i| = |a i|` and `|b' j| = |b j|` equals `a bᵀ` on `Ω`. -/
theorem exists_vecMulVec_ne_of_not_rigid_of_ne_zero {a b : n → ℂ} {Ω : Set (n × n)}
    {r s : n → ℂ} (hrs : ∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) {i₀ j₀ : n} (h₀ : r i₀ + s j₀ ≠ 0)
    (ha : a i₀ ≠ 0) (hb : b j₀ ≠ 0) :
    ∃ a' b' : n → ℂ, (∀ i, ‖a' i‖ = ‖a i‖) ∧ (∀ j, ‖b' j‖ = ‖b j‖) ∧
      (∀ ij ∈ Ω, vecMulVec a' b' ij.1 ij.2 = vecMulVec a b ij.1 ij.2) ∧
      vecMulVec a' b' ≠ vecMulVec a b := by
  set w := r i₀ + s j₀
  set u : n → ℂ := fun i => Complex.exp (↑(Real.pi * (r i / w).re) * Complex.I) with hu
  set v : n → ℂ := fun j => Complex.exp (↑(Real.pi * (s j / w).re) * Complex.I) with hv
  have huv (i j : n) : vecMulVec (u * a) (v * b) i j =
      Complex.exp (↑(Real.pi * ((r i + s j) / w).re) * Complex.I) * vecMulVec a b i j := by
    simp only [vecMulVec_apply, Pi.mul_apply, hu, hv, add_div, Complex.add_re, mul_add,
      Complex.ofReal_add, add_mul, Complex.exp_add]
    ring
  refine ⟨u * a, v * b, fun i => ?_, fun j => ?_, fun ij hij => ?_, fun h => ?_⟩
  · rw [Pi.mul_apply, norm_mul, hu, Complex.norm_exp_ofReal_mul_I, one_mul]
  · rw [Pi.mul_apply, norm_mul, hv, Complex.norm_exp_ofReal_mul_I, one_mul]
  · rw [huv, hrs ij hij, zero_div, Complex.zero_re, mul_zero, Complex.ofReal_zero, zero_mul,
      Complex.exp_zero, one_mul]
  · have h1 := congrFun (congrFun h i₀) j₀
    rw [huv, div_self h₀, Complex.one_re, mul_one, Complex.exp_pi_mul_I, vecMulVec_apply] at h1
    have h2 : (2 : ℂ) * (a i₀ * b j₀) = 0 := by linear_combination -h1
    exact mul_ne_zero two_ne_zero (mul_ne_zero ha hb) h2

example := exists_vecMulVec_ne_of_not_rigid_of_ne_zero (a := ![2, 1]) (b := ![1, 3])
  (Ω := {ij | ij.2 = ij.1}) (r := ![1, 0]) (s := ![-1, 0])
  (fun ij hij => by
    obtain ⟨i, j⟩ := ij
    simp only [Set.mem_ofPred_eq] at hij
    subst hij
    fin_cases j <;> simp)
  (i₀ := 0) (j₀ := 1) (by simp) (by simp) (by simp)

/-- **Without rigidity M does not recover `a bᵀ`**, `review/flat_spectrum.md`, Theorem F,
item 3: if `r i + s j = 0` on `Ω`, `r i₀ + s j₀ ≠ 0` and `a i₀ b j₀ ≠ 0`, then `a bᵀ` is not
the unique minimum of the nuclear norm among the matrices equal to it on `Ω`. -/
theorem not_forall_isMinOn_vecMulVec_of_not_rigid {a b : n → ℂ} {Ω : Set (n × n)}
    {r s : n → ℂ} (hrs : ∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) {i₀ j₀ : n} (h₀ : r i₀ + s j₀ ≠ 0)
    (ha : a i₀ ≠ 0) (hb : b j₀ ≠ 0) :
    ¬ ∀ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
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
  (fun ij hij => by
    obtain ⟨i, j⟩ := ij
    simp only [Set.mem_ofPred_eq] at hij
    subst hij
    fin_cases j <;> simp)
  (i₀ := 0) (j₀ := 1) (by simp) (by simp) (by simp)

end Transformer.Grokking.FlatSpectrum
