module

public import Transformer.Grokking.FlatSpectrum.WeightedUniqueness

/-!
# A transport plan inside a rigid pattern makes `a bᵀ` the unique minimum

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem G, item 2:
Theorem F, item 2, for any `a`, `b`. Under a plan from `|a_i|²` to `|b_j|²`, the lower bound
`sum_norm_sq_le_nuclearNorm_of_transport` meets the upper bound `nuclearNorm_vecMulVec_le`, so
`‖a bᵀ‖_S1 = ∑ |a_i|²` (`nuclearNorm_vecMulVec_of_transport`). Every matrix equal to `a bᵀ` on
`Ω` has at least this norm, and when `Ω` is rigid on the supports of `a` and `b`,
`eq_vecMulVec_of_transport` pins the one that attains it: `a bᵀ` is the unique minimum
(`isMinOn_vecMulVec_iff_of_transport`). The zero entries of `a` and `b` need no cells: from its
entry `1` alone, M completes the row `(1, ?)` to `(1, 0)`.

A plan from `‖b‖² |a_i|²` to `‖a‖² |b_j|²` serves as well: with `q² ‖a‖² = ‖b‖²` and `c² = q`,
the plan scaled by `q / ‖b‖²` goes from `|c a_i|²` to `|c⁻¹ b_j|²`, and `(c a)(c⁻¹ b)ᵀ = a bᵀ`
(`isMinOn_vecMulVec_iff_of_transport_mul`). If `a = 0` or `b = 0`, the kernel is `0 0ᵀ`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- Under a plan from `|a_i|²` to `|b_j|²`, `‖a bᵀ‖_S1 = ∑ |a_i|²`. -/
theorem nuclearNorm_vecMulVec_of_transport {a : m → ℂ} {b : n → ℂ} {Ω : Set (m × n)}
    {T : m → n → ℝ} (hT : IsTransportPlan Ω (fun i => ‖a i‖ ^ 2) (fun j => ‖b j‖ ^ 2) T) :
    nuclearNorm (vecMulVec a b) = ∑ i, ‖a i‖ ^ 2 := by
  have hab : ∑ i, ‖a i‖ ^ 2 = ∑ j, ‖b j‖ ^ 2 := hT.sum_eq
  refine le_antisymm ?_ (sum_norm_sq_le_nuclearNorm_of_transport hT fun _ _ => rfl)
  have h := nuclearNorm_vecMulVec_le a b
  linarith

example := nuclearNorm_vecMulVec_of_transport (a := ![2, 1]) (b := ![2, 1])
  (Ω := {ij : Fin 2 × Fin 2 | ij ≠ (1, 0)})
  (isTransportPlan_perm (σ := 1) (Fin.forall_fin_two.mpr ⟨by simp, by simp⟩)
    fun _ => sq_nonneg _)

/-- **Theorem G, item 2**, `review/flat_spectrum.md`: if `Ω` carries a plan from `|a_i|²` to
`|b_j|²` and is rigid on the supports of `a` and `b`, then `a bᵀ` is the unique minimum of the
nuclear norm among the matrices equal to it on `Ω`. -/
theorem isMinOn_vecMulVec_iff_of_transport {a : m → ℂ} {b : n → ℂ} {Ω : Set (m × n)}
    {T : m → n → ℝ} (hT : IsTransportPlan Ω (fun i => ‖a i‖ ^ 2) (fun j => ‖b j‖ ^ 2) T)
    (hΩ : ∀ (r : m → ℂ) (s : n → ℂ),
      (∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0) →
        ∀ i j, a i ≠ 0 → b j ≠ 0 → r i + s j = 0)
    {X : Matrix m n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
      X = vecMulVec a b := by
  have hK := nuclearNorm_vecMulVec_of_transport hT
  rw [isMinOn_iff]
  refine ⟨fun hmin => eq_vecMulVec_of_transport hT hΩ hX
    ((hmin _ fun _ _ => rfl).trans hK.le), ?_⟩
  rintro rfl X' hX'
  rw [hK]
  exact sum_norm_sq_le_nuclearNorm_of_transport hT hX'

/-- The rigid pattern `{(0, 0), (0, 1), (1, 1)}` carries the plan `[[1, 3], [0, 1]]` from
`|a_i|² = (4, 1)` to `|b_j|² = (1, 4)`, which no permutation gives, so it recovers
`(2, 1)(1, 2)ᵀ`. -/
example := (isMinOn_vecMulVec_iff_of_transport (a := ![2, 1]) (b := ![1, 2])
  (Ω := {ij : Fin 2 × Fin 2 | ij ≠ (1, 0)}) (T := fun i j => ![![1, 3], ![0, 1]] i j)
  ⟨by norm_num [Fin.forall_fin_two], by simp,
    by norm_num [Fin.forall_fin_two, Fin.sum_univ_two],
    by norm_num [Fin.forall_fin_two, Fin.sum_univ_two]⟩
  (rigid_support_of_ne_zero (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two])
    rigid_ne_one_zero) (X := vecMulVec ![2, 1] ![1, 2]) fun _ _ => rfl).mpr rfl

/-- From its entry `1` alone, M completes the row `(1, ?)` to `(1, 0)`: the zero entry of `b`
needs no cell. -/
example := (isMinOn_vecMulVec_iff_of_transport (a := ![(1 : ℂ)]) (b := ![1, 0])
  (Ω := {((0 : Fin 1), (0 : Fin 2))}) (T := fun _ j => ![1, 0] j)
  ⟨by simp [Fin.forall_fin_two], by simp [Fin.forall_fin_two], by simp [Fin.sum_univ_two],
    by simp [Fin.forall_fin_two]⟩
  (fun r s h i j _ hj => by
    fin_cases i
    fin_cases j
    · exact h _ rfl (by simp) (by simp)
    · simp at hj) (X := vecMulVec ![1] ![1, 0]) fun _ _ => rfl).mpr rfl

/-- **Theorem G, item 2, without the scaling**, `review/flat_spectrum.md`: if `Ω` is rigid on
the supports of `a` and `b` and carries a plan from `‖b‖² |a_i|²` to `‖a‖² |b_j|²`, then
`a bᵀ` is the unique minimum of the nuclear norm among the matrices equal to it on `Ω`. -/
theorem isMinOn_vecMulVec_iff_of_transport_mul {a : m → ℂ} {b : n → ℂ} {Ω : Set (m × n)}
    {T : m → n → ℝ}
    (hT : IsTransportPlan Ω (fun i => (∑ j, ‖b j‖ ^ 2) * ‖a i‖ ^ 2)
      (fun j => (∑ i, ‖a i‖ ^ 2) * ‖b j‖ ^ 2) T)
    (hΩ : ∀ (r : m → ℂ) (s : n → ℂ),
      (∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0) →
        ∀ i j, a i ≠ 0 → b j ≠ 0 → r i + s j = 0)
    {X : Matrix m n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
      X = vecMulVec a b := by
  -- A zero kernel is `0 0ᵀ`, which the plan `0` recovers.
  have hzero (h0 : vecMulVec a b = 0) :
      IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
        X = vecMulVec a b := by
    have h0' : vecMulVec a b = vecMulVec (0 : m → ℂ) (0 : n → ℂ) := by rw [h0, zero_vecMulVec]
    rw [h0'] at hX ⊢
    exact isMinOn_vecMulVec_iff_of_transport (a := 0) (b := 0) (T := 0)
      ⟨fun _ _ => le_rfl, fun _ _ _ => rfl, fun _ => by simp, fun _ => by simp⟩
      (fun _ _ _ _ _ hi => (hi rfl).elim) hX
  set A := ∑ i, ‖a i‖ ^ 2 with hA
  set B := ∑ j, ‖b j‖ ^ 2 with hB
  by_cases hA0 : A = 0
  · have ha : a = 0 := funext fun i => by
      simpa using (sum_eq_zero_iff_of_nonneg fun i _ => sq_nonneg ‖a i‖).mp
        (hA.symm.trans hA0) i (mem_univ i)
    exact hzero (by rw [ha, zero_vecMulVec])
  by_cases hB0 : B = 0
  · have hb : b = 0 := funext fun j => by
      simpa using (sum_eq_zero_iff_of_nonneg fun j _ => sq_nonneg ‖b j‖).mp
        (hB.symm.trans hB0) j (mem_univ j)
    -- `vecMulVec_zero` is stated for square matrices only.
    exact hzero (by ext i j; simp [vecMulVec_apply, hb])
  have hA0' := hA0
  have hB0' := hB0
  replace hA0 : 0 < A := lt_of_le_of_ne (sum_nonneg fun _ _ => sq_nonneg _) (Ne.symm hA0)
  replace hB0 : 0 < B := lt_of_le_of_ne (sum_nonneg fun _ _ => sq_nonneg _) (Ne.symm hB0)
  -- `c² = q` with `q² A = B` balances the masses.
  set q := √(B / A) with hq
  have hq0 : 0 < q := Real.sqrt_pos.mpr (div_pos hB0 hA0)
  have hq0' := hq0.ne'
  have hqB : q ^ 2 * A = B := by
    rw [hq, Real.sq_sqrt (div_pos hB0 hA0).le, div_mul_cancel₀ B hA0']
  set c := √q with hc
  have hc0 : 0 < c := Real.sqrt_pos.mpr hq0
  have hc2 : c ^ 2 = q := Real.sq_sqrt hq0.le
  have hc' : (c : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hc0.ne'
  have hc'' : ((c⁻¹ : ℝ) : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr (inv_pos.mpr hc0).ne'
  have hK : vecMulVec (fun i => (c : ℂ) * a i) (fun j => ((c⁻¹ : ℝ) : ℂ) * b j) =
      vecMulVec a b := by
    ext i j
    simp only [vecMulVec_apply, Complex.ofReal_inv]
    field_simp
  have hna (i : m) : ‖(c : ℂ) * a i‖ ^ 2 = q * ‖a i‖ ^ 2 := by
    rw [norm_mul, Complex.norm_real, Real.norm_of_nonneg hc0.le, mul_pow, hc2]
  have hnb (j : n) : ‖((c⁻¹ : ℝ) : ℂ) * b j‖ ^ 2 = q⁻¹ * ‖b j‖ ^ 2 := by
    rw [norm_mul, Complex.norm_real, Real.norm_of_nonneg (inv_nonneg.mpr hc0.le), mul_pow,
      inv_pow, hc2]
  -- The plan scaled by `q / B`.
  obtain ⟨h0, hΩT, hrow, hcol⟩ := hT
  dsimp only at hrow hcol
  have hT' : IsTransportPlan Ω (fun i => ‖(c : ℂ) * a i‖ ^ 2)
      (fun j => ‖((c⁻¹ : ℝ) : ℂ) * b j‖ ^ 2) fun i j => q / B * T i j := by
    refine ⟨fun i j => mul_nonneg (div_nonneg hq0.le hB0.le) (h0 i j),
      fun i j hij => by dsimp only; rw [hΩT i j hij, mul_zero], fun i => ?_, fun j => ?_⟩
    · dsimp only
      rw [← mul_sum, hrow i, hna]
      field_simp
    · dsimp only
      rw [← mul_sum, hcol j, hnb, ← hqB]
      field_simp
  -- Scaling by `c ≠ 0` keeps the supports.
  have := isMinOn_vecMulVec_iff_of_transport hT'
    (fun r s h i j hi hj => hΩ r s (fun ij hij hai hbj => h ij hij (mul_ne_zero hc' hai)
      (mul_ne_zero hc'' hbj)) i j (right_ne_zero_of_mul hi) (right_ne_zero_of_mul hj))
    (X := X) (by rw [hK]; exact hX)
  rwa [hK] at this

example := (isMinOn_vecMulVec_iff_of_transport_mul (a := ![2, 1]) (b := ![2, 1])
  (Ω := {ij : Fin 2 × Fin 2 | ij ≠ (1, 0)})
  (isTransportPlan_perm (σ := 1) (Fin.forall_fin_two.mpr ⟨by simp, by simp⟩)
    fun _ => mul_nonneg (sum_nonneg fun _ _ => sq_nonneg _) (sq_nonneg _))
  (rigid_support_of_ne_zero (by simp [Fin.forall_fin_two]) (by simp [Fin.forall_fin_two])
    rigid_ne_one_zero) (X := vecMulVec ![2, 1] ![2, 1]) fun _ _ => rfl).mpr rfl

/-- The scaled masses of `(1)(1, 0)ᵀ` are `1` and `(1, 0)`: the cell `(0, 0)` carries them. -/
example := (isMinOn_vecMulVec_iff_of_transport_mul (a := ![(1 : ℂ)]) (b := ![1, 0])
  (Ω := {((0 : Fin 1), (0 : Fin 2))}) (T := fun _ j => ![1, 0] j)
  ⟨by simp [Fin.forall_fin_two], by simp [Fin.forall_fin_two], by simp [Fin.sum_univ_two],
    by simp [Fin.forall_fin_two]⟩
  (fun r s h i j _ hj => by
    fin_cases i
    fin_cases j
    · exact h _ rfl (by simp) (by simp)
    · simp at hj) (X := vecMulVec ![1] ![1, 0]) fun _ _ => rfl).mpr rfl

end Transformer.Grokking.FlatSpectrum
