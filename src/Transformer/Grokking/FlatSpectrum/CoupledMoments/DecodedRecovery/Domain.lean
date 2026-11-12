module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.PairedLimits

/-!
# The spectral domain at the budgets `N`, by rows and nuclear norms alone

Source: `review/flat_spectrum.md`, CM18. A moment of probability rows under a code of modulus
at most one has entries of modulus at most one (CM1), hence operator norm at most the side `N`
(`moment_norm_le_card`). The constant moment of the Fourier bank is the all-ones matrix, of
nuclear norm `N` (CM13). So the domain of CM14 to CM16 with both budgets `N` is the set of
probability tables whose nonconstant moments have nuclear norm at most `N`
(`budget_iff_nuclear`): the operator bounds and the constant frequency are implied.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Matrix Finset Schatten
open scoped Matrix.Norms.L2Operator

variable {n C : Type*} [Fintype n] [DecidableEq n]

/-- A square matrix with entries of modulus at most one has operator norm at most its side. -/
theorem norm_le_card_of_entry_le_one {X : Matrix n n ℂ} (h : ∀ i j, ‖X i j‖ ≤ 1) :
    ‖X‖ ≤ Fintype.card n := by
  rw [Matrix.l2_opNorm_def]
  apply ContinuousLinearMap.opNorm_le_bound _ (Nat.cast_nonneg _)
  intro v
  have hv : 0 ≤ ∑ j, ‖v.ofLp j‖ ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
  have hrow (i : n) : ‖(X *ᵥ v.ofLp) i‖ ^ 2 ≤ Fintype.card n * ∑ j, ‖v.ofLp j‖ ^ 2 := by
    have hc := norm_sum_star_mul_le univ (fun j => star (X i j)) v.ofLp
    simp only [star_star, norm_star] at hc
    have hs := (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Real.sqrt_nonneg _)
      (Real.sqrt_nonneg _))).mpr hc
    rw [mul_pow, Real.sq_sqrt (sum_nonneg fun _ _ => sq_nonneg _), Real.sq_sqrt hv] at hs
    have hX : ∑ j, ‖X i j‖ ^ 2 ≤ Fintype.card n := by
      calc ∑ j, ‖X i j‖ ^ 2 ≤ ∑ _j : n, (1 : ℝ) :=
            sum_le_sum fun j _ => pow_le_one₀ (norm_nonneg _) (h i j)
        _ = Fintype.card n := by simp
    have hmul : (X *ᵥ v.ofLp) i = ∑ j, X i j * v.ofLp j := rfl
    rw [hmul]
    exact hs.trans (mul_le_mul_of_nonneg_right hX hv)
  change ‖(WithLp.toLp 2 (X *ᵥ v.ofLp) : EuclideanSpace ℂ n)‖ ≤ _
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Nat.cast_nonneg _) (norm_nonneg _))).mp
  rw [mul_pow, EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq]
  calc ∑ i, ‖(X *ᵥ v.ofLp) i‖ ^ 2
      ≤ ∑ _i : n, (Fintype.card n : ℝ) * ∑ j, ‖v.ofLp j‖ ^ 2 := sum_le_sum fun i _ => hrow i
    _ = (Fintype.card n : ℝ) ^ 2 * ∑ j, ‖v.ofLp j‖ ^ 2 := by
      rw [sum_const, card_univ, nsmul_eq_mul]
      ring

variable [Fintype C]

/-- A moment of probability rows under a code of modulus at most one has operator norm at
most the side: with such codes the radius `N` of CM14 to CM16 excludes no probability table. -/
theorem moment_norm_le_card {φ : C → ℂ} (hφ : ∀ c, ‖φ c‖ ≤ 1) {P : Table n n C}
    (hP : Rows P) : ‖moment φ P‖ ≤ Fintype.card n :=
  norm_le_card_of_entry_le_one fun i j => moment_norm_le_one hφ hP i j

/-- **The spectral domain at the budgets `N`** (CM18): probability rows whose nonconstant
Fourier moments have nuclear norm at most `N`. -/
theorem budget_iff_nuclear {q : ℕ} [NeZero q] (e : C ≃ ZMod q) (P : Table n n C) :
    BudgetFeasible (phaseCode e) (fun _ => Fintype.card n) (fun _ => Fintype.card n) P ↔
      Rows P ∧ ∀ f : ZMod q, f ≠ 0 →
        nuclearNorm (moment (phaseCode e f) P) ≤ Fintype.card n := by
  refine ⟨fun h => ⟨h.1.1, fun f _ => h.2 f⟩, fun h => ⟨⟨h.1, fun f => ?_⟩, fun f => ?_⟩⟩
  · refine moment_norm_le_card (fun c => ?_) h.1
    change ‖ZMod.stdAddChar (f * e c)‖ ≤ 1
    rw [ZMod.stdAddChar_apply, Circle.norm_coe]
  · by_cases hf : f = 0
    · subst hf
      have hz : phaseCode e 0 = fun _ : C => (1 : ℂ) := by
        funext c
        change ZMod.stdAddChar ((0 : ZMod q) * e c) = 1
        rw [zero_mul, AddChar.map_zero_eq_one]
      rw [hz, constant_moment_nuclear h.1]
    · exact h.2 f hf

example := norm_le_card_of_entry_le_one (X := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (by intro i j; fin_cases i <;> fin_cases j <;> simp)
example := moment_norm_le_card (φ := fun b : Bool => if b then (1 : ℂ) else -1)
  (by intro b; cases b <;> simp) (rows_pointMass (fun (_ _ : Fin 2) => true))
example := (budget_iff_nuclear (Equiv.refl (ZMod 3))
  (pointMass (phaseLabels (Equiv.refl (ZMod 3)) cornerPhase cornerPhase))).1
  (phaseCandidate_budget (Equiv.refl (ZMod 3)) cornerPhase cornerPhase)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
