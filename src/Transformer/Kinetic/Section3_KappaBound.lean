module

public import Transformer.Kinetic.Section3_CovarianceBound
public import Transformer.Kinetic.Section3_ThirdCumulantSplit
public import Transformer.Kinetic.Section3_PairTotal

/-!
# The third cumulant of three observables of the flow

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 4, `eq:est-kap3-th`: under a
product law of the initial angles, for `k < j < i`, the third cumulant of
`φ(θ_i(t)), φ(θ_j(t')), φ(θ_k(t''))` is at most
`(3/2) A³ τ²(1 + 2τ) q_γ(i, k) q_γ(j, k)/((i+1)(j+1)) + 2A²B Λ q_γ(i, k)⁶/((i+1)(j+1))`
(`abs_thirdCumulant_flow_le`), with `A = κπe^{L₁T}` from the first differences and
`B = K'π²e^{2L₁T}(2e^{2|λ|}L₂ T(1 + 2τ/γ)³ + τ + τ²)` from the mixed second differences
(`abs_obs_second_diff_le_env`).
-/

@[expose] public section

open MeasureTheory ProbabilityTheory Real

namespace Transformer
namespace Kinetic

/-- A mixed second difference of an observable under the pair envelope. -/
theorem abs_obs_second_diff_le_env (lam β : ℝ) (N : ℕ) {L₁ L₂ α γ T K' : ℝ}
    (hd : ∀ x, HasDerivAt (wBetaDeriv β) (deriv (wBetaDeriv β) x) x)
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (hL₂ : ∀ x y, |deriv (wBetaDeriv β) x - deriv (wBetaDeriv β) y| ≤ L₂ * |x - y|)
    (hL₂0 : 0 ≤ L₂) (hα : 2 * Real.exp (2 * |lam|) * L₁ ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    {φ dφ ddφ : ℝ → ℝ} (hper : Function.Periodic φ (2 * π)) (hφ : ∀ x, HasDerivAt φ (dφ x) x)
    (hdφ : ∀ x, HasDerivAt dφ (ddφ x) x) (hK₁ : ∀ x, |dφ x| ≤ K') (hK₂ : ∀ x, |ddφ x| ≤ K')
    {t : ℝ} (ht : t ∈ Set.Icc 0 T) (a : Idx N) (x : Idx N → ℝ) (l m : Idx N) (z z' : ℝ)
    (hml : m ≠ l) :
    |φ (flowMap lam β N t x a) - φ (flowMap lam β N t (Function.update x l z) a) -
        φ (flowMap lam β N t (Function.update x m z') a) +
        φ (flowMap lam β N t (Function.update (Function.update x l z) m z') a)| ≤
      K' * π ^ 2 * Real.exp (2 * L₁ * T) * (2 * Real.exp (2 * |lam|) * L₂ * T *
        (1 + 2 * (profileRate α γ T / γ)) ^ 3 + profileRate α γ T + profileRate α γ T ^ 2) *
        (pairEnv γ a l m + pairEnv γ a m l) := by
  have hK0 : 0 ≤ K' := (abs_nonneg _).trans (hK₁ 0)
  have hT : 0 ≤ T := ht.1.trans ht.2
  have hα0 : 0 ≤ α := le_trans (by positivity) hα
  rw [mul_assoc _ (_ + _ + _)]
  exact (abs_obs_second_diff_le lam β N hd hL hL₁ hL₂ hα hγ0 hγ1 hper hφ hdφ hK₁ hK₂ ht a x
    hml z z').trans (mul_le_mul_of_nonneg_left (pair_le_pairEnv hα0 hγ0 hγ1 hT
      (by positivity) N (fun h => hml (Fin.ext h).symm) a) (by positivity))

variable {N : ℕ} (μs : Idx N → Measure ℝ) [∀ i, IsProbabilityMeasure (μs i)]

/-- **The third cumulant under a product law.** -/
theorem abs_thirdCumulant_flow_le (lam β : ℝ) {L₁ L₂ α γ T κ K' : ℝ}
    (hd : ∀ x, HasDerivAt (wBetaDeriv β) (deriv (wBetaDeriv β) x) x)
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (hL₂ : ∀ x y, |deriv (wBetaDeriv β) x - deriv (wBetaDeriv β) y| ≤ L₂ * |x - y|)
    (hL₂0 : 0 ≤ L₂) (hα : 2 * Real.exp (2 * |lam|) * L₁ ≤ α) (hγ0 : 0 < γ)
    (hγ : γ ≤ 1 / 12) {φ dφ ddφ : ℝ → ℝ} (hper : Function.Periodic φ (2 * π))
    (hφ : ∀ x, HasDerivAt φ (dφ x) x) (hdφ : ∀ x, HasDerivAt dφ (ddφ x) x)
    (hκ : ∀ x, |dφ x| ≤ κ) (hK₁ : ∀ x, |dφ x| ≤ K') (hK₂ : ∀ x, |ddφ x| ≤ K')
    {t t' t'' : ℝ} (ht : t ∈ Set.Icc 0 T) (ht' : t' ∈ Set.Icc 0 T) (ht'' : t'' ∈ Set.Icc 0 T)
    {i j k : Idx N} (hkj : k < j) (hji : j < i) :
    |∫ x, (φ (flowMap lam β N t x i) - ∫ y, φ (flowMap lam β N t y i) ∂(Measure.pi μs)) *
        (φ (flowMap lam β N t' x j) - ∫ y, φ (flowMap lam β N t' y j) ∂(Measure.pi μs)) *
        (φ (flowMap lam β N t'' x k) - ∫ y, φ (flowMap lam β N t'' y k) ∂(Measure.pi μs))
        ∂(Measure.pi μs)| ≤
      3 / 2 * (κ * π * Real.exp (L₁ * T)) ^ 3 * (profileRate α γ T ^ 2 *
          (1 + 2 * profileRate α γ T) *
          (ratioPow γ i k * ratioPow γ j k / (((i : ℝ) + 1) * ((j : ℝ) + 1)))) +
        2 * (κ * π * Real.exp (L₁ * T)) ^ 2 * (K' * π ^ 2 * Real.exp (2 * L₁ * T) *
          (2 * Real.exp (2 * |lam|) * L₂ * T * (1 + 2 * (profileRate α γ T / γ)) ^ 3 +
            profileRate α γ T + profileRate α γ T ^ 2)) * pairConst γ (profileRate α γ T) *
          (ratioPow γ i k ^ 6 / (((i : ℝ) + 1) * ((j : ℝ) + 1))) := by
  have hγ1 : γ ≤ 1 := by linarith
  have hT : 0 ≤ T := ht.1.trans ht.2
  have hα0 : 0 ≤ α := le_trans (by positivity) hα
  have hτ := profileRate_nonneg (γ := γ) hα0 hT
  have hκ0 : 0 ≤ κ := (abs_nonneg _).trans (hκ 0)
  have hK0 : 0 ≤ K' := (abs_nonneg _).trans (hK₁ 0)
  obtain ⟨C, hC⟩ := exists_abs_le_of_periodic hper hφ
  have hw0 : ∀ (a l : Idx N), 0 ≤ K' * π ^ 2 * Real.exp (2 * L₁ * T) *
      (2 * Real.exp (2 * |lam|) * L₂ * T * (1 + 2 * (profileRate α γ T / γ)) ^ 3 +
        profileRate α γ T + profileRate α γ T ^ 2) * (pairEnv γ a l l + pairEnv γ a l l) :=
    fun a l => mul_nonneg (by positivity)
      (add_nonneg (pairEnv_nonneg _ _ _ _) (pairEnv_nonneg _ _ _ _))
  refine (abs_thirdCumulant_le_split μs (measurable_obs lam β N hφ ht.1 i)
    (measurable_obs lam β N hφ ht'.1 j) (measurable_obs lam β N hφ ht''.1 k)
    (fun x => hC _) (fun x => hC _) (fun x => hC _)
    (abs_obs_update_sub_le_env lam β N hL hL₁ hα hγ0 hγ1 hper hφ hκ ht i)
    (abs_obs_update_sub_le_env lam β N hL hL₁ hα hγ0 hγ1 hper hφ hκ ht' j)
    (abs_obs_update_sub_le_env lam β N hL hL₁ hα hγ0 hγ1 hper hφ hκ ht'' k)
    (abs_obs_second_diff_le_env lam β N hd hL hL₁ hL₂ hL₂0 hα hγ0 hγ1 hper hφ hdφ hK₁ hK₂ ht i)
    (abs_obs_second_diff_le_env lam β N hd hL hL₁ hL₂ hL₂0 hα hγ0 hγ1 hper hφ hdφ hK₁ hK₂ ht' j)
    (abs_obs_second_diff_le_env lam β N hd hL hL₁ hL₂ hL₂0 hα hγ0 hγ1 hper hφ hdφ hK₁ hK₂
      ht'' k) (hw0 i) (hw0 j) (hw0 k)).trans ?_
  have hD : ∀ (c : ℝ) (f g h : Idx N → ℝ),
      ∑ l, c * f l * (c * g l) * (c * h l) = c ^ 3 * ∑ l, f l * g l * h l :=
    fun c f g h => by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun l _ => by ring
  have hW : ∀ (c b : ℝ) (fi fj fk : Idx N → ℝ) (ei ej ek : Idx N → Idx N → ℝ),
      ∑ l, ∑ m, (c * fi m * (b * ej l m * (c * fk l) + c * fj l * (b * ek l m)) +
        c * fj m * (b * ei l m * (c * fk l) + c * fi l * (b * ek l m))) =
      c ^ 2 * b * ∑ l, ∑ m, (fk l * fi m * ej l m + fj l * fi m * ek l m +
        fk l * fj m * ei l m + fi l * fj m * ek l m) :=
    fun c b fi fj fk ei ej ek => by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun l _ => ?_
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun m _ => by ring
  rw [hD, hW]
  have hkj' : (k : ℕ) < j := hkj
  have hji' : (j : ℕ) < i := hji
  have h3 := sum_profileEnv_mul_three_le hγ0 (by linarith) hτ N hkj' hji'
  have h4 := sum_pairEnv_four_le hγ0 hγ hτ N hkj' hji'
  simp only [Finset.sum_range] at h3 h4
  calc _ ≤ 3 / 2 * ((κ * π * Real.exp (L₁ * T)) ^ 3 * (profileRate α γ T ^ 2 *
          (1 + 2 * profileRate α γ T) *
          (ratioPow γ i k * ratioPow γ j k / (((i : ℝ) + 1) * ((j : ℝ) + 1))))) +
        1 / 4 * ((κ * π * Real.exp (L₁ * T)) ^ 2 * (K' * π ^ 2 * Real.exp (2 * L₁ * T) *
          (2 * Real.exp (2 * |lam|) * L₂ * T * (1 + 2 * (profileRate α γ T / γ)) ^ 3 +
            profileRate α γ T + profileRate α γ T ^ 2)) *
          (8 * pairConst γ (profileRate α γ T) *
            (ratioPow γ i k ^ 6 / (((i : ℝ) + 1) * ((j : ℝ) + 1))))) := by
        gcongr
    _ = _ := by ring

/-- The hypotheses of `abs_obs_second_diff_le_env` hold for `sin`, the kernel's own
constants, `α = 2e^{2|λ|}L₁`, `γ = 1/16`, `t = 0 ≤ T = 1` and the tokens `0 < 1`. -/
example (lam β : ℝ) (x : Idx 2 → ℝ) :=
  have hw := (exists_lipschitz_wBetaDeriv β).choose_spec.choose_spec
  abs_obs_second_diff_le_env lam β 2 hw.2.2.1 hw.2.2.2.1 hw.1 hw.2.2.2.2 hw.2.1 le_rfl
    (γ := 1 / 16) (by norm_num) (by norm_num) Real.sin_periodic Real.hasDerivAt_sin
    Real.hasDerivAt_cos Real.abs_cos_le_one (fun x => by rw [abs_neg]; exact Real.abs_sin_le_one x)
    (T := 1) (t := 0) ⟨le_rfl, zero_le_one⟩ 1 x 0 1 0 0 (by decide)

/-- The hypotheses of `abs_thirdCumulant_flow_le` hold for `sin`, the kernel's own
constants, `α = 2e^{2|λ|}L₁`, `γ = 1/16`, the times `0, 0, 1 ≤ T = 1`, Dirac masses and
the tokens `0 < 1 < 2`. -/
example (lam β : ℝ) :=
  have hw := (exists_lipschitz_wBetaDeriv β).choose_spec.choose_spec
  abs_thirdCumulant_flow_le (fun _ : Idx 3 => Measure.dirac (0 : ℝ)) lam β hw.2.2.1 hw.2.2.2.1
    hw.1 hw.2.2.2.2 hw.2.1 le_rfl (γ := 1 / 16) (by norm_num) (by norm_num) Real.sin_periodic
    Real.hasDerivAt_sin Real.hasDerivAt_cos Real.abs_cos_le_one Real.abs_cos_le_one
    (fun x => by rw [abs_neg]; exact Real.abs_sin_le_one x) (T := 1) (t := 0) (t' := 0)
    (t'' := 1) ⟨le_rfl, zero_le_one⟩ ⟨le_rfl, zero_le_one⟩ ⟨zero_le_one, le_rfl⟩ (i := 2)
    (j := 1) (k := 0) (by decide) (by decide)

end Kinetic
end Transformer
