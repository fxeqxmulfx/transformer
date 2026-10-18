/-
# Phase model over a finite abelian group — a Fourier bound on the full table

For phases `ψ` and a character `χ` of `G`, the Fourier coefficient of
`x ↦ e^{iψ_x}` is `û_χ = Σ_x e^{i(ψ_x - arg χ(x))}`, with
`Re û_χ = Σ_x cos (ψ_x - arg χ(x))`.  From `Σ_χ χ(a) = |G|` for `a = 0` and
`0` otherwise (`AddChar.sum_apply_eq_ite`) come the orthogonality relations
`sum_exp_characterPhase_sub` and `sum_exp_characterPhase_add_sub`, and, as
for `ℤ/n` (`Landscape.PhaseModel.Fourier`), Parseval's identity `Σ_χ |û_χ|² = |G|²`
(`sum_normSq_fourier`) and the cubic identity
`Σ_χ |û_χ|² Re û_χ = |G| Σ_{y,z} cos (ψ_{y+z} - ψ_y - ψ_z)`
(`sum_normSq_mul_re_fourier`), where the double sum is `|G|² - V_full(ψ)`
(`divisionEnergy_univ_eq`).  Under the weights `|û_χ|² / |G|²` the `Re û_χ`
have the mean `|G| - V_full(ψ)/|G|`, so some character `χ` has
`|G| Σ_x (1 - cos (ψ_x - arg χ(x))) ≤ V_full(ψ)`
(`exists_mul_sum_one_sub_cos_le`).
-/

import Transformer.Landscape.PhaseModel.Abelian.Basic

open Real ComplexConjugate

namespace Transformer.Landscape.PhaseModel.Abelian

variable {G : Type*} [AddCommGroup G] [Fintype G] [DecidableEq G]

/-- **Orthogonality of the characters at two points**:
`Σ_χ e^{i(arg χ(a) - arg χ(b))} = Σ_χ χ(a - b)` is `|G|` for `a = b` and `0`
otherwise. -/
theorem sum_exp_characterPhase_sub (a b : G) :
    ∑ χ : AddChar G ℂ, Complex.exp (↑((characterPhase χ).ofLp a - (characterPhase χ).ofLp b) *
      Complex.I) = if a = b then (Fintype.card G : ℂ) else 0 := by
  have h : ∀ χ : AddChar G ℂ, Complex.exp (↑((characterPhase χ).ofLp a -
      (characterPhase χ).ofLp b) * Complex.I) = χ (a - b) := fun χ => by
    rw [Complex.ofReal_sub, sub_mul, Complex.exp_sub, exp_characterPhase, exp_characterPhase,
      AddChar.map_sub_eq_div]
  simp only [h, AddChar.sum_apply_eq_ite, sub_eq_zero]

/-- **Orthogonality for the relations of division**:
`Σ_χ e^{i(arg χ(b) + arg χ(c) - arg χ(a))} = Σ_χ χ(b + c - a)` is `|G|` for
`a = b + c` and `0` otherwise. -/
theorem sum_exp_characterPhase_add_sub (a b c : G) :
    ∑ χ : AddChar G ℂ, Complex.exp (↑((characterPhase χ).ofLp b + (characterPhase χ).ofLp c -
      (characterPhase χ).ofLp a) * Complex.I) = if a = b + c then (Fintype.card G : ℂ) else 0 := by
  have h : ∀ χ : AddChar G ℂ, Complex.exp (↑((characterPhase χ).ofLp b +
      (characterPhase χ).ofLp c - (characterPhase χ).ofLp a) * Complex.I) = χ (b + c - a) :=
    fun χ => by
      rw [Complex.ofReal_sub, Complex.ofReal_add, sub_mul, add_mul, Complex.exp_sub,
        Complex.exp_add, exp_characterPhase, exp_characterPhase, exp_characterPhase,
        AddChar.map_sub_eq_div, AddChar.map_add_eq_mul]
  simp only [h, AddChar.sum_apply_eq_ite, sub_eq_zero, @eq_comm _ (b + c) a]

/-- **Parseval's identity**: `Σ_χ |û_χ|² = |G|²` for
`û_χ = Σ_x e^{i(ψ_x - arg χ(x))}`. -/
theorem sum_normSq_fourier (ψ : EuclideanSpace ℝ G) :
    ∑ χ : AddChar G ℂ, Complex.normSq (∑ x, Complex.exp
      (↑(ψ.ofLp x - (characterPhase χ).ofLp x) * Complex.I)) = Fintype.card G ^ 2 := by
  apply Complex.ofReal_injective
  rw [Complex.ofReal_sum, Complex.ofReal_pow, Complex.ofReal_natCast]
  simp only [← Complex.mul_conj]
  set X : G → G → ℂ := fun a b => Complex.exp (↑(ψ.ofLp a - ψ.ofLp b) * Complex.I) with hX
  have h1 : ∀ χ : AddChar G ℂ,
      (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase χ).ofLp x) * Complex.I)) *
        conj (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase χ).ofLp x) * Complex.I)) =
      ∑ a, ∑ b, X a b * Complex.exp (↑((characterPhase χ).ofLp b -
        (characterPhase χ).ofLp a) * Complex.I) := by
    intro χ
    rw [map_sum, Finset.sum_mul_sum]
    refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
    rw [conj_exp_ofReal_mul_I, hX]
    simp only [← Complex.exp_add]
    congr 1
    push_cast
    ring
  rw [Finset.sum_congr rfl fun χ _ => h1 χ, Finset.sum_comm]
  calc ∑ a, ∑ χ : AddChar G ℂ, ∑ b, X a b * Complex.exp (↑((characterPhase χ).ofLp b -
        (characterPhase χ).ofLp a) * Complex.I)
      = ∑ a, ∑ b, X a b * (if b = a then (Fintype.card G : ℂ) else 0) := by
        refine Finset.sum_congr rfl fun a _ => ?_
        rw [Finset.sum_comm]
        simp only [← Finset.mul_sum, sum_exp_characterPhase_sub]
    _ = (Fintype.card G : ℂ) ^ 2 := by simp [hX, sq]

/-- **The cubic identity**:
`Σ_χ |û_χ|² Re û_χ = |G| Σ_{y,z} cos (ψ_{y+z} - ψ_y - ψ_z)`, the real part
of `Σ_χ conj(û_χ)² û_χ = |G| Σ_{y,z} e^{i(ψ_{y+z} - ψ_y - ψ_z)}`. -/
theorem sum_normSq_mul_re_fourier (ψ : EuclideanSpace ℝ G) :
    ∑ χ : AddChar G ℂ, Complex.normSq (∑ x, Complex.exp
        (↑(ψ.ofLp x - (characterPhase χ).ofLp x) * Complex.I)) *
      (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase χ).ofLp x) * Complex.I)).re =
      Fintype.card G * ∑ y, ∑ z, cos (ψ.ofLp (y + z) - ψ.ofLp y - ψ.ofLp z) := by
  have hre : ∀ w : ℂ, Complex.normSq w * w.re = (conj w * conj w * w).re := fun w => by
    rw [mul_comm (conj w * conj w) w, ← mul_assoc, Complex.mul_conj, Complex.re_ofReal_mul,
      Complex.conj_re]
  simp only [hre, ← Complex.re_sum]
  set X : G → G → G → ℂ := fun a b c =>
    Complex.exp (↑(ψ.ofLp a - ψ.ofLp b - ψ.ofLp c) * Complex.I) with hX
  set Y : AddChar G ℂ → G → G → G → ℂ := fun χ a b c =>
    Complex.exp (↑((characterPhase χ).ofLp b + (characterPhase χ).ofLp c -
      (characterPhase χ).ofLp a) * Complex.I) with hY
  have h1 : ∀ χ : AddChar G ℂ,
      conj (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase χ).ofLp x) * Complex.I)) *
        conj (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase χ).ofLp x) * Complex.I)) *
          ∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase χ).ofLp x) * Complex.I) =
      ∑ b, ∑ c, ∑ a, X a b c * Y χ a b c := by
    intro χ
    rw [map_sum, Finset.sum_mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [Finset.sum_mul_sum]
    refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun a _ => ?_
    rw [conj_exp_ofReal_mul_I, conj_exp_ofReal_mul_I, hX, hY]
    simp only [← Complex.exp_add]
    congr 1
    push_cast
    ring
  have h2 : ∑ χ : AddChar G ℂ, ∑ b, ∑ c, ∑ a, X a b c * Y χ a b c =
      ∑ b, ∑ c, ∑ a, X a b c * (if a = b + c then (Fintype.card G : ℂ) else 0) := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, hY, sum_exp_characterPhase_add_sub]
  rw [Finset.sum_congr rfl fun χ _ => h1 χ, h2]
  simp only [mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true, hX,
    Complex.re_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun b _ => Finset.sum_congr rfl fun c _ => ?_
  rw [mul_comm, ← Complex.ofReal_natCast, Complex.re_ofReal_mul, Complex.exp_ofReal_mul_I_re]

/-- **The energy of the full table as a double sum**:
`V_full(ψ) = |G|² - Σ_{y,z} cos (ψ_{y+z} - ψ_y - ψ_z)`, with the pair
`(y + z, y)` for `x / y`. -/
theorem divisionEnergy_univ_eq (ψ : EuclideanSpace ℝ G) :
    divisionEnergy (Finset.univ : Finset (G × G)) ψ =
      Fintype.card G ^ 2 - ∑ y, ∑ z, cos (ψ.ofLp (y + z) - ψ.ofLp y - ψ.ofLp z) := by
  rw [divisionEnergy_apply, Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_prod, nsmul_eq_mul, mul_one]
  congr 1
  · push_cast
    ring
  · rw [Fintype.sum_prod_type, Finset.sum_comm]
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [← Equiv.sum_comp (Equiv.addLeft y) fun x => cos (ψ.ofLp x - ψ.ofLp y - ψ.ofLp (x - y))]
    simp only [Equiv.coe_addLeft, add_sub_cancel_left]

/-- **Some character is within `V_full(ψ)/|G|` of the phases**: there is a
character `χ` with `|G| Σ_x (1 - cos (ψ_x - arg χ(x))) ≤ V_full(ψ)`.
Otherwise every `|G| Re û_χ < |G|² - V_full(ψ)`, and the weights `|û_χ|²`, of
sum `|G|²`, would give `|G| Σ_χ |û_χ|² Re û_χ < |G|² (|G|² - V_full(ψ))`,
against the cubic identity. -/
theorem exists_mul_sum_one_sub_cos_le (ψ : EuclideanSpace ℝ G) : ∃ χ : AddChar G ℂ,
    Fintype.card G * ∑ x, (1 - cos (ψ.ofLp x - (characterPhase χ).ofLp x)) ≤
      divisionEnergy (Finset.univ : Finset (G × G)) ψ := by
  set w : AddChar G ℂ → ℝ := fun χ => Complex.normSq (∑ x, Complex.exp
    (↑(ψ.ofLp x - (characterPhase χ).ofLp x) * Complex.I)) with hw
  set r : AddChar G ℂ → ℝ := fun χ => (∑ x, Complex.exp
    (↑(ψ.ofLp x - (characterPhase χ).ofLp x) * Complex.I)).re with hr
  have hr' : ∀ χ, r χ = ∑ x, cos (ψ.ofLp x - (characterPhase χ).ofLp x) :=
    fun χ => by simp only [hr, Complex.re_sum, Complex.exp_ofReal_mul_I_re]
  have hP : ∑ χ, w χ = Fintype.card G ^ 2 := sum_normSq_fourier ψ
  have hT : ∑ χ, w χ * r χ =
      Fintype.card G * ∑ y, ∑ z, cos (ψ.ofLp (y + z) - ψ.ofLp y - ψ.ofLp z) :=
    sum_normSq_mul_re_fourier ψ
  have hV := divisionEnergy_univ_eq ψ
  by_contra h
  push Not at h
  have hlt : ∀ χ, Fintype.card G * r χ <
      Fintype.card G ^ 2 - divisionEnergy (Finset.univ : Finset (G × G)) ψ := fun χ => by
    have h1 := h χ
    rw [Finset.sum_sub_distrib] at h1
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one] at h1
    rw [hr' χ]
    nlinarith
  have hn2 : ∑ χ, w χ ≠ 0 := by
    rw [hP]
    exact pow_ne_zero 2 (Nat.cast_ne_zero.2 Fintype.card_ne_zero)
  obtain ⟨χ₀, -, hχ₀⟩ := Finset.exists_ne_zero_of_sum_ne_zero hn2
  have hsum : ∑ χ, w χ * (Fintype.card G * r χ) <
      ∑ χ, w χ * (Fintype.card G ^ 2 - divisionEnergy (Finset.univ : Finset (G × G)) ψ) :=
    Finset.sum_lt_sum (fun χ _ => mul_le_mul_of_nonneg_left (hlt χ).le (Complex.normSq_nonneg _))
      ⟨χ₀, Finset.mem_univ _, mul_lt_mul_of_pos_left (hlt χ₀)
        (lt_of_le_of_ne (Complex.normSq_nonneg _) (Ne.symm hχ₀))⟩
  rw [← Finset.sum_mul, hP] at hsum
  have h3 : ∑ χ, w χ * (Fintype.card G * r χ) = Fintype.card G * ∑ χ, w χ * r χ := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun χ _ => by ring
  rw [h3, hT, hV] at hsum
  nlinarith

/-- In the Klein four-group `ℤ/2 × ℤ/2`, at the phases `0`, Parseval's
identity reads `Σ_χ |Σ_x conj χ(x)|² = 16`. -/
example : ∑ χ : AddChar (ZMod 2 × ZMod 2) ℂ, Complex.normSq (∑ x, Complex.exp
    (↑((0 : EuclideanSpace ℝ (ZMod 2 × ZMod 2)).ofLp x - (characterPhase χ).ofLp x) *
      Complex.I)) = 16 := by
  rw [sum_normSq_fourier]
  norm_num [Fintype.card_prod, ZMod.card]

end Transformer.Landscape.PhaseModel.Abelian
