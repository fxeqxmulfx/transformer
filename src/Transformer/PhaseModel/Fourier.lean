/-
# Phase model — a Fourier bound on the full division table

For phases `ψ`, let `û_j = Σ_x e^{i(ψ_x - 2πjx/n)}` be the Fourier
coefficients of `x ↦ e^{iψ_x}`, so that `Re û_j = Σ_x cos (ψ_x - 2πjx/n)`.
The orthogonality of the characters gives Parseval's identity
`Σ_j |û_j|² = n²` (`sum_normSq_fourier`) and the cubic identity
`Σ_j |û_j|² Re û_j = n Σ_{y,z} cos (ψ_{y+z} - ψ_y - ψ_z)`
(`sum_normSq_mul_re_fourier`), where the double sum is `n² - V_full(ψ)`
(`divisionEnergy_univ_eq`).  Under the weights `|û_j|² / n²` the `Re û_j`
have the mean `n - V_full(ψ)/n`, so some `j` has
`n Σ_x (1 - cos (ψ_x - 2πjx/n)) ≤ V_full(ψ)` (`exists_mul_sum_one_sub_cos_le`).
-/

import Transformer.PhaseModel.Orthogonality

open Real ComplexConjugate

namespace Transformer.PhaseModel

variable {n : ℕ} [NeZero n]

/-- **Parseval's identity**: `Σ_j |û_j|² = n²` for
`û_j = Σ_x e^{i(ψ_x - 2πjx/n)}`. -/
theorem sum_normSq_fourier (ψ : EucSpace n) :
    ∑ j : Fin n, Complex.normSq (∑ x, Complex.exp
      (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)) = n ^ 2 := by
  apply Complex.ofReal_injective
  rw [Complex.ofReal_sum, Complex.ofReal_pow, Complex.ofReal_natCast]
  simp only [← Complex.mul_conj]
  set X : Fin n → Fin n → ℂ := fun a b => Complex.exp (↑(ψ.ofLp a - ψ.ofLp b) * Complex.I)
    with hX
  have h1 : ∀ j : Fin n,
      (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)) *
        conj (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)) =
      ∑ a, ∑ b, X a b * Complex.exp (↑((characterPhase n (j : ℕ)).ofLp b -
        (characterPhase n (j : ℕ)).ofLp a) * Complex.I) := by
    intro j
    rw [map_sum, Finset.sum_mul_sum]
    refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
    rw [conj_exp_ofReal_mul_I, hX]
    simp only [← Complex.exp_add]
    congr 1
    push_cast
    ring
  rw [Finset.sum_congr rfl fun j _ => h1 j, Finset.sum_comm]
  calc ∑ a, ∑ j : Fin n, ∑ b, X a b * Complex.exp (↑((characterPhase n (j : ℕ)).ofLp b -
        (characterPhase n (j : ℕ)).ofLp a) * Complex.I)
      = ∑ a, ∑ b, X a b * (if b = a then (n : ℂ) else 0) := by
        refine Finset.sum_congr rfl fun a _ => ?_
        rw [Finset.sum_comm]
        simp only [← Finset.mul_sum, sum_exp_characterPhase_sub]
    _ = (n : ℂ) ^ 2 := by simp [hX, sq]

/-- **The cubic identity**:
`Σ_j |û_j|² Re û_j = n Σ_{y,z} cos (ψ_{y+z} - ψ_y - ψ_z)`, the real part of
`Σ_j conj(û_j)² û_j = n Σ_{y,z} e^{i(ψ_{y+z} - ψ_y - ψ_z)}`. -/
theorem sum_normSq_mul_re_fourier (ψ : EucSpace n) :
    ∑ j : Fin n, Complex.normSq (∑ x, Complex.exp
        (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)) *
      (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)).re =
      n * ∑ y, ∑ z, cos (ψ.ofLp (y + z) - ψ.ofLp y - ψ.ofLp z) := by
  have hre : ∀ w : ℂ, Complex.normSq w * w.re = (conj w * conj w * w).re := fun w => by
    rw [mul_comm (conj w * conj w) w, ← mul_assoc, Complex.mul_conj, Complex.re_ofReal_mul,
      Complex.conj_re]
  simp only [hre, ← Complex.re_sum]
  set X : Fin n → Fin n → Fin n → ℂ := fun a b c =>
    Complex.exp (↑(ψ.ofLp a - ψ.ofLp b - ψ.ofLp c) * Complex.I) with hX
  set Y : Fin n → Fin n → Fin n → Fin n → ℂ := fun j a b c =>
    Complex.exp (↑((characterPhase n (j : ℕ)).ofLp b + (characterPhase n (j : ℕ)).ofLp c -
      (characterPhase n (j : ℕ)).ofLp a) * Complex.I) with hY
  have h1 : ∀ j : Fin n,
      conj (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)) *
        conj (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)) *
          ∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I) =
      ∑ b, ∑ c, ∑ a, X a b c * Y j a b c := by
    intro j
    rw [map_sum, Finset.sum_mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [Finset.sum_mul_sum]
    refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun a _ => ?_
    rw [conj_exp_ofReal_mul_I, conj_exp_ofReal_mul_I, hX, hY]
    simp only [← Complex.exp_add]
    congr 1
    push_cast
    ring
  have h2 : ∑ j : Fin n, ∑ b, ∑ c, ∑ a, X a b c * Y j a b c =
      ∑ b, ∑ c, ∑ a, X a b c * (if a = b + c then (n : ℂ) else 0) := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, hY, sum_exp_characterPhase_add_sub]
  rw [Finset.sum_congr rfl fun j _ => h1 j, h2]
  simp only [mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true, hX,
    Complex.re_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun b _ => Finset.sum_congr rfl fun c _ => ?_
  rw [mul_comm, ← Complex.ofReal_natCast, Complex.re_ofReal_mul, Complex.exp_ofReal_mul_I_re]

/-- **The energy of the full table as a double sum**:
`V_full(ψ) = n² - Σ_{y,z} cos (ψ_{y+z} - ψ_y - ψ_z)`, with the pair
`(y + z, y)` for `x / y`. -/
theorem divisionEnergy_univ_eq (ψ : EucSpace n) :
    divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ =
      n ^ 2 - ∑ y, ∑ z, cos (ψ.ofLp (y + z) - ψ.ofLp y - ψ.ofLp z) := by
  rw [divisionEnergy_apply, Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
    nsmul_eq_mul, mul_one]
  congr 1
  · push_cast
    ring
  · rw [Fintype.sum_prod_type, Finset.sum_comm]
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [← Equiv.sum_comp (Equiv.addLeft y) fun x => cos (ψ.ofLp x - ψ.ofLp y - ψ.ofLp (x - y))]
    simp only [Equiv.coe_addLeft, add_sub_cancel_left]

/-- **Some character is within `V_full(ψ)/n` of the phases**: there is `j`
with `n Σ_x (1 - cos (ψ_x - 2πjx/n)) ≤ V_full(ψ)`.  Otherwise every
`n Re û_j < n² - V_full(ψ)`, and the weights `|û_j|²`, of sum `n²`, would
give `n Σ_j |û_j|² Re û_j < n² (n² - V_full(ψ))`, against the cubic
identity. -/
theorem exists_mul_sum_one_sub_cos_le (ψ : EucSpace n) : ∃ j : ℤ,
    n * ∑ x, (1 - cos (ψ.ofLp x - (characterPhase n j).ofLp x)) ≤
      divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ := by
  set w : Fin n → ℝ := fun j => Complex.normSq (∑ x, Complex.exp
    (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)) with hw
  set r : Fin n → ℝ := fun j => (∑ x, Complex.exp
    (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)).re with hr
  have hr' : ∀ j : Fin n, r j = ∑ x, cos (ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) :=
    fun j => by simp only [hr, Complex.re_sum, Complex.exp_ofReal_mul_I_re]
  have hP : ∑ j, w j = n ^ 2 := sum_normSq_fourier ψ
  have hT : ∑ j, w j * r j = n * ∑ y, ∑ z, cos (ψ.ofLp (y + z) - ψ.ofLp y - ψ.ofLp z) :=
    sum_normSq_mul_re_fourier ψ
  have hV := divisionEnergy_univ_eq ψ
  by_contra h
  push Not at h
  have hlt : ∀ j : Fin n, n * r j <
      n ^ 2 - divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ := fun j => by
    have h1 := h (j : ℕ)
    rw [Finset.sum_sub_distrib] at h1
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      mul_one] at h1
    rw [hr' j]
    nlinarith
  have hn2 : ∑ j, w j ≠ 0 := by
    rw [hP]
    exact pow_ne_zero 2 (Nat.cast_ne_zero.2 (NeZero.ne n))
  obtain ⟨j₀, -, hj₀⟩ := Finset.exists_ne_zero_of_sum_ne_zero hn2
  have hsum : ∑ j, w j * (n * r j) <
      ∑ j, w j * (n ^ 2 - divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ) :=
    Finset.sum_lt_sum (fun j _ => mul_le_mul_of_nonneg_left (hlt j).le (Complex.normSq_nonneg _))
      ⟨j₀, Finset.mem_univ _, mul_lt_mul_of_pos_left (hlt j₀)
        (lt_of_le_of_ne (Complex.normSq_nonneg _) (Ne.symm hj₀))⟩
  rw [← Finset.sum_mul, hP] at hsum
  have h3 : ∑ j, w j * (n * r j) = n * ∑ j, w j * r j := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [h3, hT, hV] at hsum
  nlinarith

/-- The hypothesis `NeZero n` of the theorems above holds for `n = 4`; at the
phases `0`, a zero of `V_full`, Parseval's identity reads
`Σ_j |Σ_x e^{-2πijx/4}|² = 16`. -/
example : ∑ j : Fin 4, Complex.normSq (∑ x, Complex.exp
    (↑((0 : EucSpace 4).ofLp x - (characterPhase 4 (j : ℕ)).ofLp x) * Complex.I)) = 16 := by
  rw [sum_normSq_fourier]
  norm_num

end Transformer.PhaseModel
