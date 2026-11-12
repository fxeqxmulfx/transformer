module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.GraphRecovery

/-!
# A training-compatible phase completion gives a feasible CE minimizer

Source: `review/flat_spectrum.md`, CM14. A categorical class bijection and
training-compatible row/column phases construct a one-hot table in the same
convex operator/nuclear budget family. The observed equations alone certify
ordinary CE optimality. They do not certify unseen ground-truth labels or
convexity of the discrete coding search. Input indices carry no arithmetic.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Matrix Finset
open scoped Matrix.Norms.L2Operator

variable {q : ℕ} [NeZero q] {C : Type*}

/-- All Fourier readings after a categorical class bijection (CM14). -/
noncomputable def phaseCode (e : C ≃ ZMod q) : Matrix (ZMod q) C ℂ :=
  fun f c => ZMod.stdAddChar (f * e c)

/-- The compatible model completion, without assuming any unseen target (CM14). -/
def phaseLabels {n : Type*} (e : C ≃ ZMod q) (r s : n → ZMod q) : n → n → C :=
  fun i j => e.symm (r i + s j)

variable [Fintype C] [DecidableEq C]

omit [Fintype C] in
/-- Renaming the full vocabulary preserves the complete Fourier Gram matrix (CM14). -/
theorem phaseCode_gram (e : C ≃ ZMod q) :
    (phaseCode e)ᴴ * phaseCode e = (q : ℂ) • 1 := by
  ext c d
  simp only [phaseCode, mul_apply, conjTranspose_apply, Matrix.smul_apply,
    one_apply, smul_eq_mul]
  rw [sum_congr rfl fun f _ => (by
    rw [star_stdAddChar_mul, show f * e d - f * e c = f * (e d - e c) by ring] :
      star (ZMod.stdAddChar (f * e c)) * ZMod.stdAddChar (f * e d) =
        ZMod.stdAddChar (f * (e d - e c))), sum_stdAddChar_mul]
  by_cases h : c = d
  · simp [h]
  · have he : e d ≠ e c := fun hd => h (e.injective hd).symm
    simp [h, he, sub_eq_zero]

variable {n : Type*} [Fintype n] [DecidableEq n]

omit [Fintype n] [DecidableEq n] in
/-- The constructed table's moments are actual unit-phase outer products (CM14). -/
theorem phaseCandidate_moment (e : C ≃ ZMod q) (r s : n → ZMod q) (f : ZMod q) :
    moment (phaseCode e f) (pointMass (phaseLabels e r s)) =
      vecMulVec (fun i => ZMod.stdAddChar (f * r i))
        (fun j => ZMod.stdAddChar (f * s j)) := by
  rw [moment_pointMass]
  ext i j
  change ZMod.stdAddChar (f * e (e.symm (r i + s j))) = _
  rw [e.apply_symm_apply, mul_add, AddChar.map_add_eq_mul]
  rfl

/-- Unit row and column phases give the radius-N operator upper bound (CM14). -/
private theorem phase_outer_operator {a b : n → ℂ}
    (ha : ∀ i, ‖a i‖ = 1) (hb : ∀ j, ‖b j‖ = 1) :
    ‖vecMulVec a b‖ ≤ Fintype.card n := by
  rw [Matrix.l2_opNorm_def]
  apply ContinuousLinearMap.opNorm_le_bound _ (Nat.cast_nonneg _)
  intro v
  have hc : ‖b ⬝ᵥ v.ofLp‖ ≤ √(Fintype.card n : ℝ) * √(∑ j, ‖v.ofLp j‖ ^ 2) := by
    simpa [dotProduct, hb] using norm_sum_star_mul_le univ (star b) v.ofLp
  have hs := (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Real.sqrt_nonneg _)
    (Real.sqrt_nonneg _))).mpr hc
  rw [mul_pow, Real.sq_sqrt (Nat.cast_nonneg _),
    Real.sq_sqrt (sum_nonneg fun _ _ => sq_nonneg _)] at hs
  change ‖(WithLp.toLp 2 ((vecMulVec a b) *ᵥ v.ofLp) : EuclideanSpace ℂ n)‖ ≤ _
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Nat.cast_nonneg _) (norm_nonneg _))).mp
  rw [mul_pow, EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq, vecMulVec_mulVec]
  change (∑ i, ‖a i * (b ⬝ᵥ v.ofLp)‖ ^ 2) ≤ _
  simp only [norm_mul, ha, one_mul, sum_const, card_univ, nsmul_eq_mul]
  calc
    _ ≤ (Fintype.card n : ℝ) * ((Fintype.card n : ℝ) * ∑ j, ‖v.ofLp j‖ ^ 2) :=
      mul_le_mul_of_nonneg_left hs (Nat.cast_nonneg _)
    _ = _ := by ring

/-- The constructed probabilities satisfy both original rank-one upper budgets (CM14).
The probability table is learned directly; no rank equality is a domain constraint. -/
theorem phaseCandidate_budget (e : C ≃ ZMod q) (r s : n → ZMod q) :
    BudgetFeasible (phaseCode e) (fun _ => Fintype.card n) (fun _ => Fintype.card n)
      (pointMass (phaseLabels e r s)) := by
  have hu (f x : ZMod q) : ‖ZMod.stdAddChar (f * x)‖ = 1 := by
    rw [ZMod.stdAddChar_apply, Circle.norm_coe]
  refine ⟨⟨rows_pointMass _, ?_⟩, ?_⟩
  · intro f
    rw [phaseCandidate_moment]
    exact phase_outer_operator (fun i => hu f (r i)) (fun j => hu f (s j))
  · intro f
    rw [phaseCandidate_moment,
      nuclearNorm_vecMulVec_of_norm_eq_one (fun i => hu f (r i)) (fun j => hu f (s j))]

variable {S : Type*} [Fintype S]

/-- Training phase consistency constructs a feasible global minimizer of ordinary
observed CE, with no iterative-convergence hypothesis (CM14). -/
theorem phaseCandidate_minimum (e : C ≃ ZMod q) (r s : n → ZMod q)
    (cell : S → n × n) (labels : S → C)
    (hfit : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) :
    BudgetFeasible (phaseCode e) (fun _ => Fintype.card n) (fun _ => Fintype.card n)
        (pointMass (phaseLabels e r s)) ∧
      IsMinOn (loss cell labels)
        {P | BudgetFeasible (phaseCode e) (fun _ => Fintype.card n)
          (fun _ => Fintype.card n) P} (pointMass (phaseLabels e r s)) := by
  apply budget_fit_minimum _ _ _ cell labels (phaseCandidate_budget e r s)
  intro t
  have he : phaseLabels e r s (cell t).1 (cell t).2 = labels t := by
    apply e.injective
    simpa [phaseLabels] using (hfit t).symm
  funext c
  simp [pointMass, he, ConvexRecall.targetLogits]

/-- Connected matching coverage identifies the constructed optimum using only
the nonconstant nuclear budgets (CM14). This identifies a model completion,
without asserting that unseen ground truth follows its phase structure. -/
theorem phaseCandidate_unique [Nonempty n] [Nontrivial C]
    (e : C ≃ ZMod q) (r s : n → ZMod q) (cell : S → n × n) (labels : S → C)
    (hfit : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2)
    (hgraph : (patternGraph (Set.range cell)).Connected)
    (hmatch : ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Set.range cell)
    {P : Table n n C} (hP : Rows P)
    (hB : ∀ f : ZMod q, f ≠ 0 → nuclearNorm (moment (phaseCode e f) P) ≤ Fintype.card n)
    (hL : loss cell labels P ≤ idealLoss labels) : P = pointMass (phaseLabels e r s) := by
  have he : (fun t => phaseLabels e r s (cell t).1 (cell t).2) = labels := by
    funext t
    apply e.injective
    simpa [phaseLabels] using (hfit t).symm
  apply graph_probability_recovery (phaseCode e) (Nat.cast_ne_zero.mpr (NeZero.ne q))
    (phaseCode_gram e) (fun f i => ZMod.stdAddChar (f * r i))
    (fun f j => ZMod.stdAddChar (f * s j))
    (by intro f i; rw [ZMod.stdAddChar_apply, Circle.norm_coe])
    (by intro f j; rw [ZMod.stdAddChar_apply, Circle.norm_coe])
    (phaseLabels e r s) ?_ cell hgraph hmatch hP ?_ ?_
  · intro f i j
    simp [phaseCode, phaseLabels, mul_add, AddChar.map_add_eq_mul]
  · intro f
    by_cases hf : f = 0
    · subst f
      have hz : phaseCode e 0 = fun _ : C => (1 : ℂ) := by
        funext c
        change ZMod.stdAddChar ((0 : ZMod q) * e c) = 1
        rw [zero_mul, AddChar.map_zero_eq_one]
      rw [hz, constant_moment_nuclear hP]
    · exact hB f hf
  · simpa only [he] using hL

/-! A nonconstant two-class witness observes three of four input cells. -/

private def witnessPhase (i : Fin 2) : ZMod 2 := i.val
private def witnessLabels : Fin 3 → ZMod 2 := ![0, 1, 1]

private theorem witness_fit (t : Fin 3) :
    witnessLabels t = witnessPhase (threeCells t).1 + witnessPhase (threeCells t).2 := by
  fin_cases t <;> norm_num [witnessLabels, witnessPhase, threeCells]

example := phaseCode_gram (Equiv.refl (ZMod 2))
example := phaseCandidate_moment (Equiv.refl (ZMod 2)) witnessPhase witnessPhase 1
example := phase_outer_operator (a := fun _ : Fin 2 => (1 : ℂ))
  (b := fun _ : Fin 2 => (-1 : ℂ)) (by intro i; norm_num) (by intro j; norm_num)
example := phaseCandidate_budget (Equiv.refl (ZMod 2)) witnessPhase witnessPhase
example := phaseCandidate_minimum (Equiv.refl (ZMod 2)) witnessPhase witnessPhase
  threeCells witnessLabels witness_fit
example := witness_fit 0

example : pointMass (phaseLabels (Equiv.refl (ZMod 2)) witnessPhase witnessPhase) =
    pointMass (phaseLabels (Equiv.refl (ZMod 2)) witnessPhase witnessPhase) := by
  apply phaseCandidate_unique (Equiv.refl (ZMod 2)) witnessPhase witnessPhase
    (id : Fin 2 × Fin 2 → Fin 2 × Fin 2)
    (fun ij => phaseLabels (Equiv.refl (ZMod 2)) witnessPhase witnessPhase ij.1 ij.2)
  · intro ij
    simp [phaseLabels]
  · apply (rigid_iff_connected _).mp
    intro a b h i j
    exact h (i, j) ⟨(i, j), rfl⟩
  · exact ⟨Equiv.refl _, fun i => ⟨(i, i), rfl⟩⟩
  · exact rows_pointMass _
  · intro f _
    exact (phaseCandidate_budget (Equiv.refl (ZMod 2)) witnessPhase witnessPhase).2 f
  · apply (loss_eq_ideal_of_fit _ _ ?_).le
    intro ij
    rfl

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
