import Transformer.Modes.Section3_ScalarMGF
import Transformer.Modes.Section3_MixedMoments
import Mathlib.MeasureTheory.Integral.Pi

/-
# The number of modes of a Gaussian KDE — the law of the normalized sum

arXiv:2412.09080v3, §3.1, reads the moment identity on `q_t`, the density of
`S_n = n^{-1/2} Σ Yᵢ(t)`.  Everything about `S_n` used there follows from one
identity between moment generating functions: for a linear form
`W(x) = s x₁ + t x₂`,

  `𝔼 e^{θ W(S_n)} = (𝔼 e^{θ n^{-1/2} W(Y)})^n`,

because `W(S_n) = Σᵢ n^{-1/2} W(Yᵢ)` is a sum of independent copies and the
exponential of a sum is a product.  Differentiating three times at `θ = 0`
with `iteratedDeriv_three_pow` — legitimate because `Y` has exponential
moments and `𝔼 W(Y) = 0` — gives `𝔼 W(S_n)³ = n · n^{-3/2} 𝔼 W(Y)³`, the
factor `n^{-1/2}` of the moment identity.

The linear form is a device: `s` and `t` are free, and the four mixed third
moments are recovered from the four values of `𝔼 W³` in
`Section3_ScaledSumMoments.lean`.  It replaces differentiating a
two-parameter moment generating function in a mixed order.

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`.
-/

open Real MeasureTheory Filter
open scoped Topology

namespace Transformer
namespace Modes

variable {μ : Measure (ℝ × ℝ)} {ε : ℝ} {n : ℕ}

/-- The normalized sum is a measurable function of the sample. -/
theorem measurable_scaledSum (n : ℕ) : Measurable (scaledSum n) := by
  unfold scaledSum; fun_prop

/-- **A linear form of the normalized sum is a sum of scaled linear forms**,
`s (S_n)₁ + t (S_n)₂ = Σᵢ n^{-1/2} (s (Xᵢ)₁ + t (Xᵢ)₂)`.

Source: arXiv:2412.09080v3, §3. -/
theorem scaledSum_dot (n : ℕ) (s t : ℝ) (X : Fin n → ℝ × ℝ) :
    s * (scaledSum n X).1 + t * (scaledSum n X).2
      = ∑ i, (Real.sqrt n)⁻¹ * (s * (X i).1 + t * (X i).2) := by
  simp only [scaledSum, Prod.smul_fst, Prod.smul_snd, smul_eq_mul, Prod.fst_sum, Prod.snd_sum,
    Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- **A linear form has exponential moments where the coordinates do.**
`|θ| < ε / (1 + |a| + |b|)` forces both `|θa|` and `|θb|` below `ε`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem hasScalarExpMomentsOn_linear (hE : HasExpMomentsOn μ ε) (a b : ℝ) :
    HasScalarExpMomentsOn μ (fun x => a * x.1 + b * x.2) (ε / (1 + |a| + |b|)) := by
  have hpos : (0 : ℝ) < 1 + |a| + |b| := by positivity
  intro θ hθ
  have key : ∀ r : ℝ, |r| ≤ 1 + |a| + |b| → |θ * r| < ε := by
    intro r hr
    have h1 : |θ| * |r| ≤ |θ| * (1 + |a| + |b|) := by
      exact mul_le_mul_of_nonneg_left hr (abs_nonneg θ)
    have h2 : |θ| * (1 + |a| + |b|) < ε / (1 + |a| + |b|) * (1 + |a| + |b|) :=
      mul_lt_mul_of_pos_right hθ hpos
    rw [div_mul_cancel₀ _ hpos.ne'] at h2
    rw [abs_mul]
    exact h1.trans_lt h2
  have ha : |θ * a| < ε := key a (by linarith [abs_nonneg a, abs_nonneg b])
  have hb : |θ * b| < ε := key b (by linarith [abs_nonneg a, abs_nonneg b])
  refine (hE _ _ ha hb).congr (Eventually.of_forall fun x => ?_)
  show exp (θ * a * x.1 + θ * b * x.2) = exp (θ * (a * x.1 + b * x.2))
  ring_nf

/-- The hypothesis of `hasScalarExpMomentsOn_linear` is satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 := hasExpMomentsOn_stdGauss2

/-- **A sum of independent copies has exponential moments on the same
interval**, because the exponential of the sum is a product over the
coordinates.

Source: arXiv:2412.09080v3, §3. -/
theorem hasScalarExpMomentsOn_pi_sum [IsProbabilityMeasure μ] {g : ℝ × ℝ → ℝ}
    (hg : HasScalarExpMomentsOn μ g ε) (n : ℕ) :
    HasScalarExpMomentsOn (Measure.pi fun _ : Fin n => μ) (fun X => ∑ i, g (X i)) ε := by
  intro θ hθ
  have hprod : Integrable (fun X : Fin n → ℝ × ℝ => ∏ i, exp (θ * g (X i)))
      (Measure.pi fun _ : Fin n => μ) :=
    Integrable.fintype_prod (f := fun _ x => exp (θ * g x)) fun _ => hg θ hθ
  refine hprod.congr (Eventually.of_forall fun X => ?_)
  show (∏ i, exp (θ * g (X i))) = exp (θ * ∑ i, g (X i))
  rw [Finset.mul_sum, Real.exp_sum]

/-- The hypothesis of `hasScalarExpMomentsOn_pi_sum` is satisfiable. -/
example : HasScalarExpMomentsOn stdGauss2 (fun x => 1 * x.1 + 0 * x.2) (1 / (1 + |(1 : ℝ)| + |(0 : ℝ)|)) :=
  hasScalarExpMomentsOn_linear hasExpMomentsOn_stdGauss2 1 0

/-- **The moment generating function of a sum of `n` independent copies is the
`n`-th power**, `𝔼 e^{θ Σ g(Xᵢ)} = (𝔼 e^{θ g(X)})^n`.

Source: arXiv:2412.09080v3, §3. -/
theorem scalarMoment_pi_sum [IsProbabilityMeasure μ] {g : ℝ × ℝ → ℝ} (n : ℕ) (θ : ℝ) :
    scalarMoment (Measure.pi fun _ : Fin n => μ) (fun X => ∑ i, g (X i)) 0 θ
      = scalarMoment μ g 0 θ ^ n := by
  have hrw : ∀ X : Fin n → ℝ × ℝ,
      (∑ i, g (X i)) ^ 0 * exp (θ * ∑ i, g (X i)) = ∏ i, exp (θ * g (X i)) := by
    intro X
    rw [pow_zero, one_mul, Finset.mul_sum, Real.exp_sum]
  simp only [scalarMoment, hrw]
  rw [MeasureTheory.integral_fintype_prod_eq_pow (f := fun x => exp (θ * g x))]
  simp [Fintype.card_fin]

/-- A coordinate of the product measure is integrable when the factor is. -/
theorem integrable_eval_pi [IsProbabilityMeasure μ] {F : ℝ × ℝ → ℝ} (hF : Integrable F μ)
    (n : ℕ) (i : Fin n) :
    Integrable (fun X : Fin n → ℝ × ℝ => F (X i)) (Measure.pi fun _ : Fin n => μ) := by
  have hmap := (measurePreserving_eval (μ := fun _ : Fin n => μ) i).map_eq
  have := (integrable_map_measure (μ := Measure.pi fun _ : Fin n => μ) (g := F)
    (f := Function.eval i) (by rw [hmap]; exact hF.aestronglyMeasurable)
    (measurable_pi_apply i).aemeasurable).1 (by rw [hmap]; exact hF)
  exact this

/-- A coordinate of the product measure has the same integral as the factor. -/
theorem integral_eval_pi [IsProbabilityMeasure μ] {F : ℝ × ℝ → ℝ} (hF : Integrable F μ)
    (n : ℕ) (i : Fin n) :
    ∫ X : Fin n → ℝ × ℝ, F (X i) ∂(Measure.pi fun _ : Fin n => μ) = ∫ x, F x ∂μ := by
  have hmap := (measurePreserving_eval (μ := fun _ : Fin n => μ) i).map_eq
  have := integral_map (μ := Measure.pi fun _ : Fin n => μ) (φ := Function.eval i) (f := F)
    (measurable_pi_apply i).aemeasurable (by rw [hmap]; exact hF.aestronglyMeasurable)
  rw [hmap] at this
  exact this.symm

end Modes
end Transformer
