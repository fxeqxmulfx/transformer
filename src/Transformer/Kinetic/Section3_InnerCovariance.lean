/-
# The inner covariance

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 4, `eq:est-kap3-th`: the
third cumulant is the covariance of `F̄Ḡ` with `H`, and the swap identity writes
it through the cross terms `δ_l G · Δ_l H` of the two copies. Integrating a
centred `F` against such a term is again a covariance: the one of `F` with the
cross term averaged over the copy (`abs_integral_centered_mul_le`). The cross term
moves with the coordinate `m` of `x` by at most
`s²_{l,m}(G) s_l(H) + s_l(G) s²_{l,m}(H)` (`abs_crossTerm_sub_le`), where `s²` bounds
the mixed second differences.
-/

import Transformer.Kinetic.Section3_SwapCovariance

open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {N : ℕ} (μs : Idx N → Measure ℝ) [∀ i, IsProbabilityMeasure (μs i)]

/-- **Integrating a centred observable against a function of both copies.** -/
theorem abs_integral_centered_mul_le {F : (Idx N → ℝ) → ℝ}
    {Φ : (Idx N → ℝ) × (Idx N → ℝ) → ℝ} (hF : Measurable F) (hΦ : Measurable Φ) {CF CΦ : ℝ}
    (hFb : ∀ x, |F x| ≤ CF) (hΦb : ∀ p, |Φ p| ≤ CΦ) {sF sΦ : Idx N → ℝ}
    (hsF : ∀ x l z, |F x - F (Function.update x l z)| ≤ sF l)
    (hsΦ : ∀ x x' l z, |Φ (x, x') - Φ (Function.update x l z, x')| ≤ sΦ l) :
    |∫ p, (F p.1 - ∫ x, F x ∂(Measure.pi μs)) * Φ p ∂((Measure.pi μs).prod (Measure.pi μs))| ≤
      1 / 2 * ∑ l, sF l * sΦ l := by
  set μ := Measure.pi μs
  set c := ∫ x, F x ∂μ
  set Ψ : (Idx N → ℝ) → ℝ := fun x => ∫ x', Φ (x, x') ∂μ
  have hΦx : ∀ x, Integrable (fun x' => Φ (x, x')) μ := fun x =>
    integrable_of_abs_le (hΦ.comp measurable_prodMk_left) fun x' => hΦb (x, x')
  have hΨm : Measurable Ψ :=
    (StronglyMeasurable.integral_prod_right (f := fun x x' => Φ (x, x'))
      hΦ.stronglyMeasurable).measurable
  have hΨb : ∀ x, |Ψ x| ≤ CΦ := fun x => by
    simpa using norm_integral_le_of_norm_le_const (μ := μ) (f := fun x' => Φ (x, x'))
      (ae_of_all _ fun x' => by simpa using hΦb (x, x'))
  have hsΨ : ∀ x l z, |Ψ x - Ψ (Function.update x l z)| ≤ sΦ l := fun x l z => by
    have := norm_integral_le_of_norm_le_const (μ := μ)
      (f := fun x' => Φ (x, x') - Φ (Function.update x l z, x'))
      (ae_of_all _ fun x' => by simpa using hsΦ x x' l z)
    rw [integral_sub (hΦx x) (hΦx _)] at this
    simpa using this
  have hc : |c| ≤ CF := by
    simpa using norm_integral_le_of_norm_le_const (μ := μ) (f := F)
      (ae_of_all _ fun x => by simpa using hFb x)
  have hCΦ : 0 ≤ CΦ := (abs_nonneg _).trans (hΦb (0, 0))
  have hFc : ∀ x, |F x - c| ≤ 2 * CF := fun x =>
    (abs_sub _ _).trans (by linarith [hFb x])
  have hint : Integrable (fun p : (Idx N → ℝ) × (Idx N → ℝ) => (F p.1 - c) * Φ p) (μ.prod μ) :=
    integrable_of_abs_le (((hF.comp measurable_fst).sub measurable_const).mul hΦ) fun p => by
      rw [abs_mul]
      exact mul_le_mul (hFc _) (hΦb p) (abs_nonneg _) (by linarith [abs_nonneg (F 0 - c), hFc 0])
  have hFint : Integrable F μ := integrable_of_abs_le hF hFb
  have heq : ∫ p, (F p.1 - c) * Φ p ∂(μ.prod μ) = cov[F, Ψ; μ] := by
    rw [integral_prod _ hint]
    simp only [integral_const_mul]
    unfold covariance
    simp_rw [mul_sub]
    rw [integral_sub, integral_mul_const, integral_sub hFint (integrable_const c), integral_const]
    · simp [c, Ψ]
    · exact integrable_of_abs_le ((hF.sub measurable_const).mul hΨm) fun x => by
        rw [abs_mul]
        exact mul_le_mul (hFc x) (hΨb x) (abs_nonneg _) (by linarith [abs_nonneg (F 0 - c), hFc 0])
    · exact (hFint.sub (integrable_const c)).mul_const _
  rw [heq]
  exact abs_covariance_le_sum μs hF hΨm hFb hΨb hsF hsΨ

/-- The cross term `δ_l G · Δ_l H` of the swap identity. -/
noncomputable def crossTerm (G H : (Idx N → ℝ) → ℝ) (l : Idx N)
    (p : (Idx N → ℝ) × (Idx N → ℝ)) : ℝ :=
  (G p.1 - G (Function.update p.1 l (p.2 l))) *
    (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2))

theorem hybrid_update_of_lt {l : ℕ} {m : Idx N} (hm : (m : ℕ) < l) (x x' : Idx N → ℝ) (z : ℝ) :
    hybrid l (Function.update x m z) x' = hybrid l x x' := by
  funext k
  unfold hybrid
  by_cases hk : k = m
  · subst hk
    simp [hm]
  · simp [hk]

theorem hybrid_update_of_le {l : ℕ} {m : Idx N} (hm : l ≤ (m : ℕ)) (x x' : Idx N → ℝ) (z : ℝ) :
    hybrid l (Function.update x m z) x' = Function.update (hybrid l x x') m z := by
  funext k
  unfold hybrid
  by_cases hk : k = m
  · subst hk
    simp [show ¬ (k : ℕ) < l by omega]
  · simp [hk]

/-- The hypotheses of `hybrid_update_of_lt` are satisfiable. -/
example : ((0 : Idx 2) : ℕ) < 1 := by decide

end Kinetic
end Transformer
