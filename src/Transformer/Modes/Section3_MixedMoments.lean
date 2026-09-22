import Transformer.Modes.Section3_ExpMoments

/-
# The number of modes of a Gaussian KDE — the mixed moments of a law on `ℝ²`

`expMoment μ i j u v = ∫ x₁^i x₂^j e^{u x₁ + v x₂} dμ`, the quantity every
derivative of the cumulant generating function of arXiv:2412.09080v3, §3.1, is
built from: the moment generating function is `expMoment μ 0 0`, a derivative
in `u` raises `i` by one and a derivative in `v` raises `j` by one
(`hasDerivAt_expMoment_fst`, `hasDerivAt_expMoment_snd`).  Both come from the
general lemmas of `Section3_ExpMoments.lean`, the second by exchanging the two
coordinates.

`HasExpMomentsOn μ ε` names the hypothesis those lemmas need — exponential
moments on the open square of half-side `ε` — of which `HasExpMoments` of
`Section3_Cumulants.lean` is the existential closure.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`.
-/

open Real MeasureTheory Filter
open scoped Topology

namespace Transformer
namespace Modes

variable {μ : Measure (ℝ × ℝ)} {ε u v : ℝ}

/-- `μ` has exponential moments on the open square of half-side `ε`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
def HasExpMomentsOn (μ : Measure (ℝ × ℝ)) (ε : ℝ) : Prop :=
  ∀ u v : ℝ, |u| < ε → |v| < ε → Integrable (fun x : ℝ × ℝ => exp (u * x.1 + v * x.2)) μ

/-- `HasExpMoments` is the existential closure of `HasExpMomentsOn`. -/
theorem hasExpMoments_iff_exists : HasExpMoments μ ↔ ∃ ε > 0, HasExpMomentsOn μ ε := Iff.rfl

/-- `N(0, I₂)` has exponential moments on every square.

Source: arXiv:2412.09080v3, §3.1. -/
theorem hasExpMomentsOn_stdGauss2 : HasExpMomentsOn stdGauss2 1 := fun u v _ _ => by
  simpa [Real.exp_add] using
    (ProbabilityTheory.integrable_exp_mul_gaussianReal (μ := 0) (v := 1) u).mul_prod
      (ProbabilityTheory.integrable_exp_mul_gaussianReal (μ := 0) (v := 1) v)

/-- The mixed moment `∫ x₁^i x₂^j e^{u x₁ + v x₂} dμ`.  For `i = j = 0` this is
the moment generating function whose logarithm `cumulantOf` differentiates.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
noncomputable def expMoment (μ : Measure (ℝ × ℝ)) (i j : ℕ) (u v : ℝ) : ℝ :=
  ∫ x, x.1 ^ i * x.2 ^ j * exp (u * x.1 + v * x.2) ∂μ

/-- `expMoment μ 0 0` is the moment generating function. -/
@[simp] theorem expMoment_zero_zero (μ : Measure (ℝ × ℝ)) (u v : ℝ) :
    expMoment μ 0 0 u v = ∫ x, exp (u * x.1 + v * x.2) ∂μ := by
  simp [expMoment]

/-- At the origin the mixed moment is the plain moment. -/
theorem expMoment_origin (μ : Measure (ℝ × ℝ)) (i j : ℕ) :
    expMoment μ i j 0 0 = ∫ x, x.1 ^ i * x.2 ^ j ∂μ := by
  simp [expMoment]

/-- The mixed moments exist on the square where the exponential moments do. -/
theorem integrable_expMoment (hE : HasExpMomentsOn μ ε) (hu : |u| < ε) (hv : |v| < ε) (i j : ℕ) :
    Integrable (fun x : ℝ × ℝ => x.1 ^ i * x.2 ^ j * exp (u * x.1 + v * x.2)) μ :=
  integrable_pow_mul_exp measurable_fst measurable_snd hE hu hv i j

/-- The hypotheses of `integrable_expMoment` are satisfiable: `N(0, I₂)` at the
origin. -/
example : HasExpMomentsOn stdGauss2 1 ∧ |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨hasExpMomentsOn_stdGauss2, by norm_num, by norm_num⟩

/-- Every plain moment exists when the exponential moments do. -/
theorem integrable_pow_mul_pow (hE : HasExpMomentsOn μ ε) (hε : 0 < ε) (i j : ℕ) :
    Integrable (fun x : ℝ × ℝ => x.1 ^ i * x.2 ^ j) μ := by
  simpa using integrable_expMoment hE (u := 0) (v := 0) (by simpa using hε) (by simpa using hε) i j

/-- The hypotheses of `integrable_pow_mul_pow` are satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 ∧ (0 : ℝ) < 1 := ⟨hasExpMomentsOn_stdGauss2, one_pos⟩

/-- **A derivative in `u` raises the power of `x₁`.**

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem hasDerivAt_expMoment_fst (hE : HasExpMomentsOn μ ε) (hu : |u| < ε) (hv : |v| < ε)
    (i j : ℕ) :
    HasDerivAt (fun s => expMoment μ i j s v) (expMoment μ (i + 1) j u v) u :=
  hasDerivAt_integral_pow_mul_exp measurable_fst measurable_snd hE hu hv i j

/-- The hypotheses of `hasDerivAt_expMoment_fst` are satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 ∧ |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨hasExpMomentsOn_stdGauss2, by norm_num, by norm_num⟩

/-- **A derivative in `v` raises the power of `x₂`**, by the same lemma with the
two coordinates exchanged.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem hasDerivAt_expMoment_snd (hE : HasExpMomentsOn μ ε) (hu : |u| < ε) (hv : |v| < ε)
    (i j : ℕ) :
    HasDerivAt (fun t => expMoment μ i j u t) (expMoment μ i (j + 1) u v) v := by
  have hE' : ∀ a b : ℝ, |a| < ε → |b| < ε →
      Integrable (fun x : ℝ × ℝ => exp (a * x.2 + b * x.1)) μ := by
    intro a b ha hb
    refine (hE b a hb ha).congr (Eventually.of_forall fun x => ?_)
    show exp (b * x.1 + a * x.2) = exp (a * x.2 + b * x.1)
    rw [add_comm]
  have hswap : ∀ (p q : ℕ) (a b : ℝ),
      (∫ x : ℝ × ℝ, x.2 ^ p * x.1 ^ q * exp (b * x.2 + a * x.1) ∂μ) = expMoment μ q p a b := by
    intro p q a b
    rw [expMoment]
    refine integral_congr_ae (Eventually.of_forall fun x => ?_)
    show x.2 ^ p * x.1 ^ q * exp (b * x.2 + a * x.1)
        = x.1 ^ q * x.2 ^ p * exp (a * x.1 + b * x.2)
    rw [add_comm (b * x.2) (a * x.1)]
    ring
  simpa only [hswap] using
    hasDerivAt_integral_pow_mul_exp (X := Prod.snd) (Y := Prod.fst) measurable_snd measurable_fst
      hE' hv hu j i

/-- The hypotheses of `hasDerivAt_expMoment_snd` are satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 ∧ |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨hasExpMomentsOn_stdGauss2, by norm_num, by norm_num⟩

/-- The moment generating function is positive, so its logarithm is defined. -/
theorem expMoment_zero_zero_pos [NeZero μ] (hE : HasExpMomentsOn μ ε) (hu : |u| < ε)
    (hv : |v| < ε) : 0 < expMoment μ 0 0 u v := by
  rw [expMoment_zero_zero]
  exact integral_exp_pos (hE u v hu hv)

/-- The hypotheses of `expMoment_zero_zero_pos` are satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 ∧ |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨hasExpMomentsOn_stdGauss2, by norm_num, by norm_num⟩

/-- A probability measure has total mass one. -/
@[simp] theorem expMoment_origin_zero_zero (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] :
    expMoment μ 0 0 0 0 = 1 := by
  simp [expMoment]

/-- A standardized law has mean zero in the first coordinate. -/
theorem expMoment_origin_one_zero (hμ : IsStandardized μ) : expMoment μ 1 0 0 0 = 0 := by
  rw [expMoment_origin]
  simpa using hμ.mean_fst

/-- The hypothesis of `expMoment_origin_one_zero` is satisfiable. -/
example : IsStandardized stdGauss2 := isStandardized_stdGauss2

/-- A standardized law has mean zero in the second coordinate. -/
theorem expMoment_origin_zero_one (hμ : IsStandardized μ) : expMoment μ 0 1 0 0 = 0 := by
  rw [expMoment_origin]
  simpa using hμ.mean_snd

/-- The hypothesis of `expMoment_origin_zero_one` is satisfiable. -/
example : IsStandardized stdGauss2 := isStandardized_stdGauss2

end Modes
end Transformer
