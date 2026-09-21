import Transformer.AMSGrad.Section5_Theorem
import Transformer.AMSGrad.Section5_Sums
import Transformer.AMSGrad.Section4_Rate

/-
# AdamX — the convergence of the average regret

§5 of arXiv:1904.03590v4: Corollary 5.5 (`ge_cor`) and Corollary 5.6
(`bound`), from Theorem 5.1.

**What the source says and what is carried here.**

* Corollaries 5.5 and 5.6 assert `lim R(T)/T = 0`; as for Corollary 4.5 only
  the upper half follows from an upper bound and is stated.  The lower half is
  false: with `β_{1,t} = 0` AdamX is AMSGrad (`adamX_eq_amsgrad`), and
  `not_cor_lower` is a run with `β_{1,t} = 0` and `R(T) ≤ -T/2` infinitely often.

* Corollary 5.6's second setting is `β_{1,t} = 1/t`, which gives
  `β_{1,1} = 1`, against the standing `β₁ = β_{1,1} < 1`, and makes AdamX
  divide by `1 - β_{1,1} = 0`.  It is corrected to `β_{1,t} = β₁/t`, the
  setting of Theorem 4.1; its proof (last2) is unchanged.

Source: arXiv:1904.03590v4, §5, Corollaries 5.5, 5.6.
-/

open Finset Filter Topology

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- **Corollary 5.5, the upper half.**  Under the assumptions of `mainthm2`, if
`Σ_{t=2}^T β_{1,t}√(t-1)/T → 0`, then for every `ε > 0`, eventually
`R(T)/T ≤ ε` for all `x* ∈ F`.
Source: arXiv:1904.03590v4, §5, Corollary 5.5. -/
theorem ge_cor {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    (hlim : Tendsto (fun T : ℕ => (∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1)) / T)
      atTop (𝓝 0))
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ T : ℕ in atTop, ∀ xstar ∈ F, S.regret (adamXRule S.β₁) xstar T / T ≤ ε := by
  have hB : 0 < 1 - S.β₁ 1 := by linarith
  obtain ⟨A, hA⟩ : ∃ A : ℝ, A = d * D ^ 2 * G / (2 * α * (1 - S.β₁ 1)) := ⟨_, rfl⟩
  obtain ⟨C, hC⟩ : ∃ C : ℝ, C = d * D ^ 2 * G / (2 * α * (1 - S.β₁ 1) ^ 2) := ⟨_, rfl⟩
  obtain ⟨k, hk⟩ : ∃ k : ℝ, k = α /
      ((1 - S.β₁ 1) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - S.β₁ 1 / Real.sqrt S.β₂)) := ⟨_, rfl⟩
  have hk0 : 0 ≤ k := hk ▸ div_nonneg hα.le
    (mul_nonneg (mul_nonneg (sq_nonneg _) (Real.sqrt_nonneg _)) (by linarith))
  obtain ⟨M, hM⟩ : ∃ M : ℝ, M = |A| + |k * (d * G)| := ⟨_, rfl⟩
  have hM0 : 0 ≤ M := hM ▸ add_nonneg (abs_nonneg _) (abs_nonneg _)
  have hC0 : 0 ≤ |C| := abs_nonneg _
  filter_upwards [tendsto_rate.eventually
      (eventually_le_nhds (show (0 : ℝ) < ε / (2 * (M + 1)) by positivity)),
    Metric.tendsto_nhds.1 hlim _ (show (0 : ℝ) < ε / (2 * (|C| + 1)) by positivity),
    eventually_ge_atTop 1] with T hq hσ hT xstar hx
  rw [Real.dist_eq, sub_zero] at hσ
  have hT0 : (0 : ℝ) < T := by exact_mod_cast hT
  have hmain := mainthm2 hS hα hαt hβ₁ hβ₁' hβ₂ hβ₂' hγ hT hx
  rw [← hA, ← hC] at hmain
  set L := Real.log T + 1
  set σ := (∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1)) / T
  have e : α * Real.sqrt L /
      ((1 - S.β₁ 1) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - S.β₁ 1 / Real.sqrt S.β₂)) =
      k * Real.sqrt L := by rw [hk, div_mul_eq_mul_div]
  rw [e] at hmain
  have hg : ∑ i, S.gnorm (adamXRule S.β₁) T i ≤ d * (G * Real.sqrt T) :=
    (sum_le_sum fun i _ => gnorm_le hS _ T i).trans_eq
      (by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul])
  have h3 := mul_le_mul_of_nonneg_left hg (mul_nonneg hk0 (Real.sqrt_nonneg L))
  have hR : S.regret (adamXRule S.β₁) xstar T / T ≤
      (A * Real.sqrt T + k * (d * G) * Real.sqrt L * Real.sqrt T) / T + C * σ := by
    refine (div_le_div_of_nonneg_right (b := A * Real.sqrt T +
      C * ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1) +
      k * Real.sqrt L * (d * (G * Real.sqrt T))) (by linarith) hT0.le).trans (le_of_eq ?_)
    simp only [σ]
    ring
  have hr := rate_le 0 A (k * (d * G)) hT
  rw [abs_zero, zero_add, zero_add, ← hM] at hr
  have h1 : M * Real.sqrt (L / T) ≤ ε / 2 := by
    refine (mul_le_mul_of_nonneg_left hq hM0).trans ?_
    rw [mul_div_assoc', div_le_div_iff₀ (by positivity) two_pos]
    nlinarith
  have h2 : C * σ ≤ ε / 2 := by
    refine (le_abs_self _).trans ?_
    rw [abs_mul]
    refine (mul_le_mul_of_nonneg_left hσ.le hC0).trans ?_
    rw [mul_div_assoc', div_le_div_iff₀ (by positivity) two_pos]
    nlinarith
  linarith

/-- **Corollary 5.6, `β_{1,t} = β₁λ^{t-1}`, the upper half.**  AdamX with
`α_t = α/√t`, `0 ≤ β₁ < 1`, `0 < λ < 1`, `0 < β₂ < 1` and `γ < 1`: for every
`ε > 0`, eventually `R(T)/T ≤ ε` for all `x* ∈ F`.
Source: arXiv:1904.03590v4, §5, Corollary 5.6. -/
theorem bound_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ lam : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hl : 0 < lam) (hl' : lam < 1)
    (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ T : ℕ in atTop, ∀ xstar ∈ F, S.regret (adamXRule S.β₁) xstar T / T ≤ ε := by
  have hb1 : S.β₁ 1 = β₁ := by rw [hb]; simp
  have hβt : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1 := fun t _ => by
    rw [hb, hb1]
    exact ⟨by positivity, mul_le_of_le_one_right hβ₁ (pow_le_one₀ hl.le hl'.le)⟩
  have hlim : Tendsto (fun T : ℕ => (∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1)) / T)
      atTop (𝓝 0) := by
    simp only [hb]
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
      (tendsto_const_div_atTop_nhds_zero_nat (1 / (1 - lam) ^ 2)) (fun T => ?_) (fun T => ?_)
    · exact div_nonneg (sum_nonneg fun t _ => by positivity) (Nat.cast_nonneg _)
    · exact div_le_div_of_nonneg_right (sum_lambda_le hβ₁'.le hl.le hl' T) (Nat.cast_nonneg _)
  exact ge_cor hS hα hαt hβt (hb1 ▸ hβ₁') hβ₂ hβ₂' (hb1 ▸ hγ) hlim hε

/-- **Corollary 5.6, `β_{1,t} = β₁/t`, the upper half.**  As `bound_lambda`
with `β_{1,t} = β₁/t`; the source has `β_{1,t} = 1/t`, see the module
docstring.
Source: arXiv:1904.03590v4, §5, Corollary 5.6. -/
theorem bound_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hb : ∀ t, S.β₁ t = β₁ / t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ T : ℕ in atTop, ∀ xstar ∈ F, S.regret (adamXRule S.β₁) xstar T / T ≤ ε := by
  have hb1 : S.β₁ 1 = β₁ := by rw [hb]; simp
  have hβt : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1 := fun t ht => by
    rw [hb, hb1]
    exact ⟨by positivity, div_le_self hβ₁ (by exact_mod_cast ht)⟩
  have hlim : Tendsto (fun T : ℕ => (∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1)) / T)
      atTop (𝓝 0) := by
    simp only [hb]
    have h2 := tendsto_rate.const_mul 2
    rw [mul_zero] at h2
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds h2
      (Eventually.of_forall fun T => ?_) ?_
    · exact div_nonneg (sum_nonneg fun t _ => by positivity) (Nat.cast_nonneg _)
    · filter_upwards [eventually_ge_atTop 1] with T hT
      refine (div_le_div_of_nonneg_right (sum_inv_le hβ₁'.le T) (Nat.cast_nonneg _)).trans ?_
      have h := rate_le 0 2 0 hT
      simp only [zero_mul, zero_add, add_zero, abs_zero, abs_two] at h
      exact h
  exact ge_cor hS hα hαt hβt (hb1 ▸ hβ₁') hβ₂ hβ₂' (hb1 ▸ hγ) hlim hε

/-- The hypotheses of Corollaries 5.5 and 5.6 are satisfiable:
the zero cost on `[-1, 1]`, `α = 1`, `β₁ = 0`, `λ = 1/2`, `β₂ = 1/2`, `ε = 1`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧
      S.β₁ 1 < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ S.β₁ 1 / Real.sqrt S.β₂ < 1 ∧
      Tendsto (fun T : ℕ => (∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1)) / T)
        atTop (𝓝 0) ∧
      (∀ t, S.β₁ t = 0 * (1 / 2 : ℝ) ^ (t - 1)) ∧ (∀ t, S.β₁ t = 0 / t) ∧ (0 : ℝ) < 1 :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [zeroSetup], by norm_num [zeroSetup], by norm_num [zeroSetup],
    by simp [zeroSetup], by simp [zeroSetup],
    fun _ => by simp [zeroSetup], fun _ => by simp [zeroSetup], one_pos⟩

end AMSGrad
end Transformer
