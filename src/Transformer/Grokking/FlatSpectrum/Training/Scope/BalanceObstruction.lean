module

public import Transformer.Grokking.FlatSpectrum.BalancedFactorization
public import Transformer.Grokking.FlatSpectrum.Training.Scope.Balanced
public import Mathlib.Analysis.Convex.Basic

/-!
# Balance does not make factor training convex or certify optimality

Source: `review/flat_spectrum.md`, *Balancedness as a capacity certificate*, B3.
The real scalar balance constraint `u² = v²` fails at the midpoint of `(1,1)`
and `(1,-1)`. Imposing balance is therefore not a convex parameter constraint.

For an explicit nonzero stationary obstruction, use the convex squared loss
`L(X) = ½ ‖X - (3/2) I₂‖²`, decay `λ = 1/2`, and factors
`U = V = diag(1,0)`. They are balanced and stationary, but their decayed loss
is `7/4`, whereas `U = V = I₂` gives `5/4`. The target is flat with `c = 9/4`.
This is a counterexample to a loss-independent optimality claim, not a change
to the ordinary cross-entropy used by the convex-channel experiments.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix

/-- **The scalar balance constraint is not convex**, B3. Both endpoints are
balanced, but their midpoint has first square `1` and second square `0`. -/
theorem not_convex_scalar_balance :
    ¬ Convex ℝ {p : ℝ × ℝ | p.1 ^ 2 = p.2 ^ 2} := by
  intro h
  have hmid := h (x := (1, 1)) (by norm_num) (y := (1, -1)) (by norm_num)
    (a := 1 / 2) (b := 1 / 2) (by norm_num) (by norm_num) (by norm_num)
  norm_num at hmid

/-- The nonzero rank-deficient factors in B3; the allocated width is two. -/
def balanceWitness : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, 0]

/-- The flat target `(3/2) I₂` in B3. -/
noncomputable def balanceTarget : Matrix (Fin 2) (Fin 2) ℂ := (3 / 2 : ℂ) • 1

/-- The B3 witness is nonzero, so this obstruction is not only the zero start. -/
theorem balanceWitness_ne_zero : balanceWitness ≠ 0 := by
  intro h
  have h00 := congrArg (fun X : Matrix (Fin 2) (Fin 2) ℂ => X 0 0) h
  norm_num [balanceWitness] at h00

/-- The B3 target has two equal singular values `3/2`. -/
theorem isFlat_balanceTarget : IsFlat balanceTarget (9 / 4) := by
  left
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [balanceTarget, Matrix.mul_apply, Fin.sum_univ_two, conjTranspose_apply]

private lemma balanceWitness_product : balanceWitness * balanceWitnessᴴ = balanceWitness := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [balanceWitness, Matrix.mul_apply, Fin.sum_univ_two, conjTranspose_apply]

/-- The B3 factors have actual rank one despite having two columns. -/
theorem rank_balanceWitness : balanceWitness.rank = 1 := by
  classical
  have hdiag : balanceWitness = diagonal ![(1 : ℂ), 0] := by
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [balanceWitness]
  rw [hdiag, rank_diagonal, Fintype.card_subtype]
  norm_num [Finset.univ_fin2, Finset.filter_insert, Finset.filter_singleton]

/-- B2 detects the missing rank in B3: balancing alone leaves at least `9/4`
squared error against the flat target. -/
theorem balanceWitness_flat_error :
    (9 / 4 : ℝ) ≤ frobeniusSq (balanceTarget - balanceWitness * balanceWitnessᴴ) := by
  have h := isFlat_balanceTarget.mul_sub_factor_rank_le_frobeniusSq_sub (by norm_num)
    (U := balanceWitness) (V := balanceWitness) rfl
  norm_num [rank_balanceWitness] at h
  exact h

/-- **The B3 factors are stationary for their actual squared-loss gradient**,
not for an independently supplied matrix field. -/
theorem stationary_balanceWitness :
    lift (1 / 2) (mask Set.univ (balanceWitness * balanceWitnessᴴ - balanceTarget)) *
      stack balanceWitness balanceWitness = 0 := by
  rw [lift_mul_stack, balanceWitness_product]
  have hleft : mask Set.univ (balanceWitness - balanceTarget) * balanceWitness +
      (1 / 2 : ℝ) • balanceWitness = 0 := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      norm_num [mask, balanceWitness, balanceTarget, Matrix.mul_apply, Fin.sum_univ_two]
  have hright : (mask Set.univ (balanceWitness - balanceTarget))ᴴ * balanceWitness +
      (1 / 2 : ℝ) • balanceWitness = 0 := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      norm_num [mask, balanceWitness, balanceTarget, Matrix.mul_apply, Fin.sum_univ_two,
        conjTranspose_apply]
  rw [hleft, hright, stack_zero]

/-- The loss of the stationary B3 factors, including positive weight decay. -/
theorem decayedLoss_balanceWitness :
    decayedLoss (fitLoss Set.univ balanceTarget) (1 / 2) balanceWitness balanceWitness =
      7 / 4 := by
  rw [decayedLoss, balanceWitness_product]
  norm_num [fitLoss, mask, frobeniusSq, Fin.sum_univ_two, balanceWitness, balanceTarget,
    Complex.norm_def, Complex.normSq]

/-- A concrete balanced competitor has strictly lower loss in B3. -/
theorem decayedLoss_balanceIdentity :
    decayedLoss (fitLoss Set.univ balanceTarget) (1 / 2)
      (1 : Matrix (Fin 2) (Fin 2) ℂ) 1 = 5 / 4 := by
  norm_num [decayedLoss, fitLoss, mask, frobeniusSq, Fin.sum_univ_two, balanceTarget,
    Complex.norm_def, Complex.normSq]

/-- **Nonzero balanced stationarity does not imply a decayed minimum**, B3.
The competitor has the same allocated width as the stationary factors. -/
theorem not_isDecayedMin_balanceWitness :
    ¬ IsDecayedMin (fitLoss Set.univ balanceTarget) (1 / 2)
      balanceWitness balanceWitness := by
  intro hmin
  have hle := hmin 2 (1 : Matrix (Fin 2) (Fin 2) ℂ) 1
  rw [decayedLoss_balanceWitness, decayedLoss_balanceIdentity] at hle
  norm_num at hle

/-- **A flat-target balanced stationary point with a strictly better balanced
competitor**, B3. Every property is checked on explicit matrices. -/
theorem exists_balanced_stationary_not_minimal :
    ∃ U V : Matrix (Fin 2) (Fin 2) ℂ,
      U ≠ 0 ∧ V ≠ 0 ∧ Uᴴ * U = Vᴴ * V ∧
      lift (1 / 2) (mask Set.univ (U * Vᴴ - balanceTarget)) * stack U V = 0 ∧
      decayedLoss (fitLoss Set.univ balanceTarget) (1 / 2)
        (1 : Matrix (Fin 2) (Fin 2) ℂ) 1 <
        decayedLoss (fitLoss Set.univ balanceTarget) (1 / 2) U V := by
  refine ⟨balanceWitness, balanceWitness, balanceWitness_ne_zero, balanceWitness_ne_zero,
    rfl, stationary_balanceWitness, ?_⟩
  rw [decayedLoss_balanceIdentity, decayedLoss_balanceWitness]
  norm_num

/-! The general identities of B1 also apply to this nonoptimal nonzero witness. -/

example := nuclearNorm_mul_conjTranspose_of_balanced
  (U := balanceWitness) (V := balanceWitness) rfl

example := rank_fromRows_of_balanced
  (U := balanceWitness) (V := balanceWitness) rfl

example := isFlat_balanceTarget.mul_sub_factor_rank_le_frobeniusSq_sub (by norm_num)
  (U := balanceWitness) (V := balanceWitness) rfl

end Transformer.Grokking.FlatSpectrum.Training.Scope
