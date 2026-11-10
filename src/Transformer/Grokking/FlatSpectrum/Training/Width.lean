module

public import Transformer.Grokking.FlatSpectrum.Training.Deficient
public import Transformer.Grokking.FlatSpectrum.Training.FullColumn

/-!
# Width `min(m, n)`: a critical point is a decayed minimum or a strict saddle

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD10.
Let `L` be real-analytic in the entries with the first-order inequality and a continuous `G`,
and let `W = [U; V]` be a critical point of the weight-decayed loss `F` on the factor space of
width `k` that is not a strict saddle.

* If the columns of `W` are dependent, `(U, V)` is a decayed minimum
  (`isDecayedMin_of_not_isStrictSaddle`): for `W z = 0`, a negative direction `S` of `M` would
  give the negative direction `S zᴴ` of the Hessian (`inner_fderiv_gradient_flatLoss`).
* If `λ > 0` and `k ≥ min(m, n)`, `(U, V)` is a decayed minimum
  (`isDecayedMin_of_not_isStrictSaddle_of_card_le`): independent columns give `M ⪰ 0` at this
  width (`lift_posSemidef_of_injective`).

A local minimum is a critical point and not a strict saddle. So a local minimum of the
weight-decayed loss with dependent columns (`isDecayedMin_of_isLocalMin`), and at width
`k ≥ min(m, n)` every local minimum (`isDecayedMin_of_isLocalMin_of_card_le`), has the least
weight-decayed loss among the factorizations of every width: no local minimum is spurious.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology
open Transformer.Landscape.SaddleAvoidance (IsStrictSaddle)

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] {L : Matrix m n ℂ → ℝ}
  {G : Matrix m n ℂ → Matrix m n ℂ} {lam : ℝ}

/-- A decayed minimum of width `r` minimizes the weight-decayed loss among the pairs of width
`r`. -/
lemma IsDecayedMin.isLocalMin {r : ℕ} {U : Matrix m (Fin r) ℂ} {V : Matrix n (Fin r) ℂ}
    (h : IsDecayedMin L lam U V) :
    IsLocalMin (fun p : Matrix m (Fin r) ℂ × Matrix n (Fin r) ℂ => decayedLoss L lam p.1 p.2)
      (U, V) :=
  Filter.Eventually.of_forall fun p => h r p.1 p.2

/-- A local minimum of the weight-decayed loss is a critical point on the factor space and not
a strict saddle. -/
lemma gradient_eq_zero_and_not_isStrictSaddle_of_isLocalMin
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    {U : Matrix m k ℂ} {V : Matrix n k ℂ}
    (hmin : IsLocalMin (fun p : Matrix m k ℂ × Matrix n k ℂ => decayedLoss L lam p.1 p.2)
      (U, V)) :
    gradient (flatLoss L lam) (flatten (U, V)) = 0 ∧
      ¬ IsStrictSaddle (flatLoss L lam) (flatten (U, V)) := by
  have hf : AnalyticOnNhd ℝ (flatLoss (k := k) L lam) Set.univ :=
    fun y _ => analyticAt_flatLoss hL lam y
  have h : IsLocalMin (fun p : Matrix m k ℂ × Matrix n k ℂ => decayedLoss L lam p.1 p.2)
      (factors (flatten (U, V))) := by rwa [factors_flatten]
  have hmin' : IsLocalMin (flatLoss L lam) (flatten (U, V)) :=
    h.comp_continuous continuous_factors.continuousAt
  exact ⟨Landscape.SaddleAvoidance.gradient_eq_zero_of_isLocalMin hmin',
    Landscape.SaddleAvoidance.not_isStrictSaddle_of_isLocalMin hf.contDiff hmin'⟩

variable [DecidableEq m] [DecidableEq n]

/-- **A critical point with dependent columns is a decayed minimum or a strict saddle**
(`review/flat_spectrum.md`, item WD10): let `L` be real-analytic in the entries with the
first-order inequality and a continuous `G`. A critical point of the weight-decayed loss on the
factor space whose stacked factors `[U; V]` have dependent columns, and which is not a strict
saddle, has the least weight-decayed loss among the factorizations of every width. -/
theorem isDecayedMin_of_not_isStrictSaddle (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    {x : EuclideanSpace ℂ ((m ⊕ n) × k)} (hgrad : gradient (flatLoss L lam) x = 0)
    (hns : ¬ IsStrictSaddle (flatLoss L lam) x)
    (hker : ¬ Function.Injective (stack (factors x).1 (factors x).2).mulVec) :
    IsDecayedMin L lam (factors x).1 (factors x).2 := by
  have hf : AnalyticOnNhd ℝ (flatLoss (k := k) L lam) Set.univ :=
    fun y _ => analyticAt_flatLoss hL lam y
  obtain ⟨z, hz, hz0⟩ : ∃ z : k → ℂ, stack (factors x).1 (factors x).2 *ᵥ z = 0 ∧ z ≠ 0 := by
    by_contra hall
    push Not at hall
    refine hker fun a b hab => sub_eq_zero.mp (hall (a - b) ?_)
    rw [mulVec_sub, sub_eq_zero]
    exact hab
  have hWZ : stack (factors x).1 (factors x).2 * replicateCol Unit z = 0 := by
    rw [← replicateCol_mulVec, hz, replicateCol_zero]
  rw [stack, fromRows_mul] at hWZ
  obtain ⟨hU, hV⟩ := fromRows_inj (hWZ.trans (fromRows_zero (R := ℂ)).symm)
  have hZ : 0 < frobeniusSq (replicateCol Unit z) :=
    lt_of_le_of_ne (frobeniusSq_nonneg _) fun h0 => hz0 <| funext fun i =>
      congrFun (congrFun (eq_zero_of_frobeniusSq_eq_zero h0.symm) i) ()
  refine isDecayedMin_of_posSemidef hconv
    (lift_mul_stack_eq_zero_of_gradient_eq_zero hconv hG hL hgrad)
    (posSemidef_of_pairing_nonneg (lift_isHermitian lam _) fun W => ?_)
  obtain ⟨A, B, rfl⟩ := exists_eq_stack W
  by_contra hneg
  push Not at hneg
  refine hns (Landscape.SaddleAvoidance.isStrictSaddle_of_inner_neg hf.contDiff hgrad
    (h := flatten (A * (replicateCol Unit z)ᴴ, B * (replicateCol Unit z)ᴴ)) ?_)
  rw [real_inner_comm, inner_fderiv_gradient_flatLoss hconv hG hL lam x hU hV A B]
  exact mul_neg_of_pos_of_neg hZ hneg

/-- **Width `min(m, n)`: a critical point is a decayed minimum or a strict saddle**
(`review/flat_spectrum.md`, item WD10): let `L` be real-analytic in the entries with the
first-order inequality and a continuous `G`, let `λ > 0` and let the width be at least the
number of rows of `U` or of `V`. A critical point of the weight-decayed loss on the factor
space that is not a strict saddle has the least weight-decayed loss among the factorizations of
every width. -/
theorem isDecayedMin_of_not_isStrictSaddle_of_card_le
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) {x : EuclideanSpace ℂ ((m ⊕ n) × k)}
    (hgrad : gradient (flatLoss L lam) x = 0) (hns : ¬ IsStrictSaddle (flatLoss L lam) x)
    (hcard : Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k) :
    IsDecayedMin L lam (factors x).1 (factors x).2 := by
  by_cases hinj : Function.Injective (stack (factors x).1 (factors x).2).mulVec
  · have hstat := lift_mul_stack_eq_zero_of_gradient_eq_zero hconv hG hL hgrad
    exact isDecayedMin_of_posSemidef hconv hstat
      (lift_posSemidef_of_injective hlam hstat hinj hcard)
  · exact isDecayedMin_of_not_isStrictSaddle hconv hG hL hgrad hns hinj

/-- **A local minimum with dependent columns is a decayed minimum** (`review/flat_spectrum.md`,
item WD10): for `L` real-analytic in the entries with the first-order inequality and a
continuous `G`, a local minimum `(U, V)` of the weight-decayed loss among the pairs of its width
whose stacked factors have dependent columns has the least weight-decayed loss among the
factorizations of every width. -/
theorem isDecayedMin_of_isLocalMin (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    {U : Matrix m k ℂ} {V : Matrix n k ℂ}
    (hmin : IsLocalMin (fun p : Matrix m k ℂ × Matrix n k ℂ => decayedLoss L lam p.1 p.2)
      (U, V))
    (hker : ¬ Function.Injective (stack U V).mulVec) : IsDecayedMin L lam U V := by
  obtain ⟨hgrad, hns⟩ := gradient_eq_zero_and_not_isStrictSaddle_of_isLocalMin hL hmin
  have h := isDecayedMin_of_not_isStrictSaddle hconv hG hL hgrad hns (by rwa [factors_flatten])
  rwa [factors_flatten] at h

/-- **At width `min(m, n)` every local minimum is a decayed minimum**
(`review/flat_spectrum.md`, item WD10): for `L` real-analytic in the entries with the
first-order inequality and a continuous `G`, `λ > 0` and a width at least the number of rows of
`U` or of `V`, a local minimum of the weight-decayed loss among the pairs of that width has the
least weight-decayed loss among the factorizations of every width. -/
theorem isDecayedMin_of_isLocalMin_of_card_le
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) {U : Matrix m k ℂ} {V : Matrix n k ℂ}
    (hmin : IsLocalMin (fun p : Matrix m k ℂ × Matrix n k ℂ => decayedLoss L lam p.1 p.2)
      (U, V))
    (hcard : Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k) :
    IsDecayedMin L lam U V := by
  obtain ⟨hgrad, hns⟩ := gradient_eq_zero_and_not_isStrictSaddle_of_isLocalMin hL hmin
  have h := isDecayedMin_of_not_isStrictSaddle_of_card_le hconv hG hL hlam hgrad hns hcard
  rwa [factors_flatten] at h

/-! The examples: the `1 × 1` kernel `1` with the trained factors `(1 − s) · (1 − s)` of
`isDecayedMin_one_sub`, a local minimum at width one. For `s = 1` the factors vanish, so the
one column is dependent; for `s = 1/2` it is independent and the width is `1 = min(m, n)`. -/

example := (isDecayedMin_one_sub zero_le_one le_rfl).isLocalMin

example := gradient_eq_zero_and_not_isStrictSaddle_of_isLocalMin
  (analyticAt_fitLoss Set.univ 1) (isDecayedMin_one_sub zero_le_one le_rfl).isLocalMin

example :=
  have h := gradient_eq_zero_and_not_isStrictSaddle_of_isLocalMin
    (analyticAt_fitLoss Set.univ 1) (isDecayedMin_one_sub zero_le_one le_rfl).isLocalMin
  isDecayedMin_of_not_isStrictSaddle (fitLoss_first_order Set.univ 1)
    (continuous_mask_sub Set.univ 1) (analyticAt_fitLoss Set.univ 1) h.1 h.2 (by
      rw [factors_flatten]
      exact fun h => one_ne_zero (h (a₁ := 1) (a₂ := 0) (by simp [stack])))

example :=
  have h := gradient_eq_zero_and_not_isStrictSaddle_of_isLocalMin (analyticAt_fitLoss Set.univ 1)
    (isDecayedMin_one_sub (s := 1 / 2) (by norm_num) (by norm_num)).isLocalMin
  isDecayedMin_of_not_isStrictSaddle_of_card_le (fitLoss_first_order Set.univ 1)
    (continuous_mask_sub Set.univ 1) (analyticAt_fitLoss Set.univ 1) (by norm_num) h.1 h.2
    (Or.inl le_rfl)

example := isDecayedMin_of_isLocalMin (fitLoss_first_order Set.univ 1)
  (continuous_mask_sub Set.univ 1) (analyticAt_fitLoss Set.univ 1)
  (isDecayedMin_one_sub zero_le_one le_rfl).isLocalMin
  fun h => one_ne_zero (h (a₁ := 1) (a₂ := 0) (by simp [stack]))

example := isDecayedMin_of_isLocalMin_of_card_le (fitLoss_first_order Set.univ 1)
  (continuous_mask_sub Set.univ 1) (analyticAt_fitLoss Set.univ 1) (by norm_num)
  (isDecayedMin_one_sub (s := 1 / 2) (by norm_num) (by norm_num)).isLocalMin (Or.inl le_rfl)

end Transformer.Grokking.FlatSpectrum.Training
