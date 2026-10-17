/-
# The covariance of two observables of the flow

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 2, `eq:est-cov-phi`: under a
product law of the initial angles, for `j < i`,
`|Cov(φ(θ_i(t)), φ(θ_j(t')))| ≤ ½ (Kπe^{L₁T})² τ(1 + 2τ) q_γ(i, j)/(i+1)`
(`abs_covariance_flow_le`): the swap formula against the first differences of the
flow, `|∂_{θ_l} θ_x| ≤ π e^{L₁T}([l = x] + τ F(x, l))`
(`abs_obs_update_sub_le_env`).
-/

import Transformer.Kinetic.Section3_Observables
import Transformer.Kinetic.Section3_SwapCovariance
import Transformer.Kinetic.Section3_TokenSums

open MeasureTheory ProbabilityTheory Real

namespace Transformer
namespace Kinetic

/-- A first difference of an observable under the envelope of its profile. -/
theorem abs_obs_update_sub_le_env (lam β : ℝ) (N : ℕ) {L₁ α γ T κ : ℝ}
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (hα : 2 * Real.exp (2 * |lam|) * L₁ ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    {φ dφ : ℝ → ℝ} (hper : Function.Periodic φ (2 * π)) (hφ : ∀ x, HasDerivAt φ (dφ x) x)
    (hκ : ∀ x, |dφ x| ≤ κ) {t : ℝ} (ht : t ∈ Set.Icc 0 T) (a : Idx N)
    (x : Idx N → ℝ) (l : Idx N) (z : ℝ) :
    |φ (flowMap lam β N t x a) - φ (flowMap lam β N t (Function.update x l z) a)| ≤
      κ * π * Real.exp (L₁ * T) * profileEnv γ (profileRate α γ T) a l := by
  have hκ0 : 0 ≤ κ := (abs_nonneg _).trans (hκ 0)
  exact (abs_obs_update_sub_le lam β N hL hL₁ hα hγ0 hγ1 hper hφ hκ ht a x l z).trans
    (mul_le_mul_of_nonneg_left (profile_le_env hγ0 a l) (by positivity))

variable {N : ℕ} (μs : Idx N → Measure ℝ) [∀ i, IsProbabilityMeasure (μs i)]

/-- **The covariance under a product law.** -/
theorem abs_covariance_flow_le (lam β : ℝ) {L₁ α γ T K : ℝ}
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (hα : 2 * Real.exp (2 * |lam|) * L₁ ≤ α) (hγ0 : 0 < γ) (hγ : γ ≤ 1 / 4)
    {φ dφ : ℝ → ℝ} (hper : Function.Periodic φ (2 * π)) (hφ : ∀ x, HasDerivAt φ (dφ x) x)
    (hK : ∀ x, |dφ x| ≤ K) {t t' : ℝ} (ht : t ∈ Set.Icc 0 T) (ht' : t' ∈ Set.Icc 0 T)
    {i j : Idx N} (hji : j < i) :
    |cov[fun x => φ (flowMap lam β N t x i), fun x => φ (flowMap lam β N t' x j);
        Measure.pi μs]| ≤
      1 / 2 * (K * π * Real.exp (L₁ * T)) ^ 2 * (profileRate α γ T *
        (1 + 2 * profileRate α γ T) * (ratioPow γ i j / ((i : ℝ) + 1))) := by
  have hγ1 : γ ≤ 1 := by linarith
  have hT : 0 ≤ T := ht.1.trans ht.2
  have hα0 : 0 ≤ α := le_trans (by positivity) hα
  obtain ⟨C, hC⟩ := exists_abs_le_of_periodic hper hφ
  refine (abs_covariance_le_sum μs (measurable_obs lam β N hφ ht.1 i)
    (measurable_obs lam β N hφ ht'.1 j) (fun x => hC _) (fun x => hC _)
    (abs_obs_update_sub_le_env lam β N hL hL₁ hα hγ0 hγ1 hper hφ hK ht i)
    (abs_obs_update_sub_le_env lam β N hL hL₁ hα hγ0 hγ1 hper hφ hK ht' j)).trans ?_
  have h := sum_profileEnv_mul_two_le hγ0 hγ (profileRate_nonneg (γ := γ) hα0 hT) N
    (show (j : ℕ) < i from hji)
  calc 1 / 2 * ∑ l : Idx N, K * π * Real.exp (L₁ * T) * profileEnv γ (profileRate α γ T) i l *
        (K * π * Real.exp (L₁ * T) * profileEnv γ (profileRate α γ T) j l)
      = 1 / 2 * (K * π * Real.exp (L₁ * T)) ^ 2 * ∑ l ∈ Finset.range N,
          profileEnv γ (profileRate α γ T) i l * profileEnv γ (profileRate α γ T) j l := by
        rw [Finset.sum_range, Finset.mul_sum, Finset.mul_sum]
        exact Finset.sum_congr rfl fun l _ => by ring
    _ ≤ _ := mul_le_mul_of_nonneg_left h (by positivity)

/-- The hypotheses of `abs_obs_update_sub_le_env` and `abs_covariance_flow_le` hold for
`sin`, the kernel's own constant `L₁`, `α = 2e^{2|λ|}L₁`, `γ = 1/16`, the times
`0 ≤ 1 ≤ T = 1`, Dirac masses and the tokens `0 < 1`. -/
example (lam β : ℝ) (x : Idx 2 → ℝ) :=
  have hw := (exists_lipschitz_wBetaDeriv β).choose_spec.choose_spec
  And.intro (abs_obs_update_sub_le_env lam β 2 hw.2.2.2.1 hw.1 le_rfl (γ := 1 / 16)
    (by norm_num) (by norm_num) Real.sin_periodic Real.hasDerivAt_sin Real.abs_cos_le_one
    (T := 1) (t := 0) ⟨le_rfl, zero_le_one⟩ 1 x 0 0)
  (abs_covariance_flow_le (fun _ : Idx 2 => Measure.dirac (0 : ℝ)) lam β hw.2.2.2.1 hw.1
    le_rfl (γ := 1 / 16) (by norm_num) (by norm_num) Real.sin_periodic Real.hasDerivAt_sin
    Real.abs_cos_le_one (T := 1) (t := 0) (t' := 1) ⟨le_rfl, zero_le_one⟩
    ⟨zero_le_one, le_rfl⟩ (i := 1) (j := 0) (by decide))

end Kinetic
end Transformer
