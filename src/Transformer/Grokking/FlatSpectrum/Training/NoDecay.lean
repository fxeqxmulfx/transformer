module

public import Transformer.Grokking.FlatSpectrum.Training.AnswerCode
public import Transformer.Grokking.FlatSpectrum.Training.NuclearMin

/-!
# Without weight decay: no minimum and no trained fit

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE6 and XE13.
Let some product have a positive margin. The cross-entropy is positive at every product
(`ceLoss_pos`) and tends to `0` along the multiples of that product
(`tendsto_ceLoss_smul_atTop`). So its infimum is `0` and is not attained
(`isGLB_range_ceLoss`): the loss has no minimum (`not_exists_forall_ceLoss_le`). Hence for
`λ = 0` there is no trained product (`not_isNuclearMin_ceLoss_zero`) and no trained fit of any
width (`not_isDecayedMin_ceLoss_zero`); the statements of item XE5 at `λ = 0` are about
samples that no product classifies.

For every loss, a sublevel set of the loss on the factor space without weight decay that
contains a factorization of `0` contains `(U, 0)` for every `U`, so it is not compact
(`not_isCompact_setOf_flatLoss_zero_le`): the compactness of item WD9 needs `λ > 0`.

The witness is `signProbe`, where `1` has the margin `2`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [Fintype S] [Fintype C]
  {A : S → C → Matrix m n ℂ} {y : S → C} {X₀ : Matrix m n ℂ}

omit [Fintype k] in
/-- The cross-entropy is positive at every product, for at least two classes. -/
lemma ceLoss_pos [Nonempty S] [Nontrivial C] (A : S → C → Matrix m n ℂ) (y : S → C)
    (X : Matrix m n ℂ) : 0 < ceLoss A y X := by
  obtain ⟨⟨⟨s, c⟩, hc⟩⟩ := nonempty_wrong y
  exact (Real.log_pos (by linarith [Real.exp_pos (-(gap A y X s c))])).trans_le
    (log_le_ceLoss A y X hc)

omit [Fintype k] in
/-- **The cross-entropy tends to `0` along the multiples of a product of positive margin**
(`review/flat_spectrum.md`, item XE6, statement 1). -/
theorem tendsto_ceLoss_smul_atTop (hsep : 0 < margin A y X₀) :
    Tendsto (fun t : ℝ => ceLoss A y (t • X₀)) atTop (𝓝 0) := by
  have h1 : Tendsto (fun t : ℝ => (Fintype.card S : ℝ) * (Fintype.card C - 1) *
      Real.exp (-(t * margin A y X₀))) atTop (𝓝 (Fintype.card S * (Fintype.card C - 1) * 0)) :=
    (Real.tendsto_exp_neg_atTop_nhds_zero.comp (tendsto_id.atTop_mul_const hsep)).const_mul _
  rw [mul_zero] at h1
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds h1
    (Eventually.of_forall fun t => ceLoss_nonneg A y _) ?_
  filter_upwards [eventually_ge_atTop 0] with t ht
  exact ceLoss_le_mul_exp fun s c hc => by
    rw [gap_smul]
    exact mul_le_mul_of_nonneg_left (margin_le_gap X₀ hc) ht

omit [Fintype k] in
/-- **The infimum of the cross-entropy of separable samples is `0`, and it is not attained**
(`review/flat_spectrum.md`, items XE6 and XE13). -/
theorem isGLB_range_ceLoss [Nonempty S] [Nontrivial C] (hsep : 0 < margin A y X₀) :
    IsGLB (Set.range (ceLoss A y)) 0 ∧ (0 : ℝ) ∉ Set.range (ceLoss A y) := by
  refine ⟨⟨?_, fun b hb => ?_⟩, fun ⟨X, hX⟩ => (ceLoss_pos A y X).ne' hX⟩
  · rintro _ ⟨X, rfl⟩
    exact ceLoss_nonneg A y X
  · exact ge_of_tendsto' (tendsto_ceLoss_smul_atTop hsep) fun t => hb ⟨_, rfl⟩

omit [Fintype k] in
/-- **The cross-entropy of separable samples has no minimum** (`review/flat_spectrum.md`,
items XE6 and XE13). -/
theorem not_exists_forall_ceLoss_le [Nonempty S] [Nontrivial C] (hsep : 0 < margin A y X₀) :
    ¬ ∃ X, ∀ Y, ceLoss A y X ≤ ceLoss A y Y := by
  rintro ⟨X, hX⟩
  exact (ceLoss_pos A y X).not_ge
    (ge_of_tendsto' (tendsto_ceLoss_smul_atTop hsep) fun t => hX (t • X₀))

section

variable [DecidableEq n]

omit [Fintype k] in
/-- **Without weight decay separable samples have no trained product**
(`review/flat_spectrum.md`, item XE13): nothing minimizes `L + 0 · ‖·‖_S1`. -/
theorem not_isNuclearMin_ceLoss_zero [Nonempty S] [Nontrivial C] (hsep : 0 < margin A y X₀)
    (X : Matrix m n ℂ) : ¬ IsNuclearMin (ceLoss A y) 0 X := fun h =>
  not_exists_forall_ceLoss_le hsep ⟨X, fun Y => by simpa [IsNuclearMin] using h Y⟩

/-- **Without weight decay separable samples have no trained fit**, of any width
(`review/flat_spectrum.md`, item XE13). -/
theorem not_isDecayedMin_ceLoss_zero [Nonempty S] [Nontrivial C] (hsep : 0 < margin A y X₀)
    (U : Matrix m k ℂ) (V : Matrix n k ℂ) : ¬ IsDecayedMin (ceLoss A y) 0 U V := fun h =>
  not_isNuclearMin_ceLoss_zero hsep _ (h.isNuclearMin le_rfl)

end

omit [Fintype S] [Fintype C] in
/-- **Without weight decay the sublevel sets on the factor space are not compact**
(`review/flat_spectrum.md`, items WD9 and XE13): one that contains a factorization of `0`
contains `(U, 0)` for every `U`. -/
theorem not_isCompact_setOf_flatLoss_zero_le [DecidableEq m] [DecidableEq k] [Nonempty m]
    [Nonempty k] {L : Matrix m n ℂ → ℝ} {c : ℝ} (hc : L 0 ≤ c) :
    ¬ IsCompact {x : EuclideanSpace ℂ ((m ⊕ n) × k) | flatLoss L 0 x ≤ c} := by
  intro hcomp
  obtain ⟨R, hR⟩ := hcomp.isBounded.exists_norm_le
  obtain ⟨i⟩ := ‹Nonempty m›
  obtain ⟨l⟩ := ‹Nonempty k›
  have hf := factors_flatten
    ((|R| + 1) • Matrix.single i l (1 : ℂ), (0 : Matrix n k ℂ))
  have h1 := hR (flatten ((|R| + 1) • Matrix.single i l (1 : ℂ), (0 : Matrix n k ℂ))) (by
    change flatLoss L 0 _ ≤ c
    rw [flatLoss, decayedLoss, hf, conjTranspose_zero, Matrix.mul_zero, zero_mul, add_zero]
    exact hc)
  have h2 : ‖flatten ((|R| + 1) • Matrix.single i l (1 : ℂ), (0 : Matrix n k ℂ))‖ ^ 2
      = (|R| + 1) ^ 2 := by
    rw [norm_sq_eq_frobeniusSq_factors, hf, frobeniusSq_smul, frobeniusSq_single, norm_one]
    simp [frobeniusSq]
  rw [(sq_eq_sq₀ (norm_nonneg _) (by positivity)).mp h2] at h1
  linarith [le_abs_self R]

/-! The witness: `signProbe`, where the product `1` has the margin `2`; its loss at `0` is
`log 2`. -/

example := ceLoss_pos signProbe (fun _ => true) 0

example := tendsto_ceLoss_smul_atTop (A := signProbe) (y := fun _ => true) (X₀ := 1)
  (two_pos.trans_le (le_margin le_gap_signProbe_one))

example := isGLB_range_ceLoss (A := signProbe) (y := fun _ => true) (X₀ := 1)
  (two_pos.trans_le (le_margin le_gap_signProbe_one))

example := not_exists_forall_ceLoss_le (A := signProbe) (y := fun _ => true) (X₀ := 1)
  (two_pos.trans_le (le_margin le_gap_signProbe_one))

/-- No product is a trained product of the witness without weight decay. -/
example := not_isNuclearMin_ceLoss_zero (A := signProbe) (y := fun _ => true) (X₀ := 1)
  (two_pos.trans_le (le_margin le_gap_signProbe_one)) 1

/-- No factorization of width one is a trained fit of the witness without weight decay. -/
example := not_isDecayedMin_ceLoss_zero (A := signProbe) (y := fun _ => true) (X₀ := 1)
  (two_pos.trans_le (le_margin le_gap_signProbe_one)) (1 : Matrix (Fin 1) (Fin 1) ℂ) 1

/-- The sublevel set of the witness at the level of its loss at `0` is not compact. -/
example := not_isCompact_setOf_flatLoss_zero_le (k := Fin 1)
  (L := ceLoss signProbe fun _ => true) (c := ceLoss signProbe (fun _ => true) 0) le_rfl

end Transformer.Grokking.FlatSpectrum.Training
