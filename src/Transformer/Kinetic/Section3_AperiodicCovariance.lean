module

public import Transformer.Kinetic.Fluctuations
public import Transformer.Kinetic.Section3_ThreeTokens
public import Transformer.Homogenized.RademacherLaw

/-!
# A test that is not periodic — `lem:correl`

ArXiv:2605.09213v1, §3, `lem:correl`. The source's test is `φ ∈ C²(𝕋)`.
The angles are carried on `ℝ`, and the dynamics sees them only modulo `2π`.
A test on `ℝ` that is not periodic also reads the winding numbers, and these
can be correlated with the residues the dynamics does see.

Three tokens on a fair coin show this. The first token sits at `0` and the
third starts at `π/2`. The second token sits at `0` on one side of the coin
and at `π + 2πM` on the other; both are equilibria of its field. The third
token feels a different field on the two sides, so at some time `t₀ > 0` its
two positions differ by some `Δ ≠ 0`. With `φ = id`, which has `φ' = 1` and
`φ'' = 0`, the covariance of the third and second tokens is
`Δ (π + 2πM) / 4`. That is unbounded in `M`, while the bound of
`eq:est-cov-th` does not depend on `M` (`correlation_bounds_aperiodic_false`).
`correlation_bounds` therefore takes `φ` periodic, as the source does.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter Topology

namespace Transformer
namespace Kinetic

/-- The covariance under the fair coin is a quarter of the product of the two
jumps. -/
theorem covariance_fairCoin (X Y : Bool → ℝ) :
    covariance X Y Homogenized.fairCoin = (X true - X false) * (Y true - Y false) / 4 := by
  simp only [covariance, Homogenized.integral_fairCoin, smul_eq_mul]
  ring

/-- **The former statement of `lem:correl` is false.** It took every
`φ : ℝ → ℝ` with bounded `φ'` and `φ''`, while the source takes
`φ ∈ C²(𝕋)`. For every `λ` and every `β ≠ 0` no constant bounds the
covariance of `eq:est-cov-th` for `φ = id`. Three tokens on a fair coin
give a covariance `Δ (π + 2πM) / 4` with `Δ ≠ 0` independent of the
winding number `M` of the second token, and the bound does not depend on `M`.

`correlation_bounds` now takes `φ` periodic, as the source does.

Source: arXiv:2605.09213v1, §3, `lem:correl`, `eq:est-cov-th`. -/
theorem correlation_bounds_aperiodic_false (lam β : ℝ) (hβ : β ≠ 0) :
    ¬ ∃ C : ℝ, 0 < C ∧ ∀ (N : ℕ) (P : Measure Bool), IsProbabilityMeasure P →
      ∀ ϑ : Bool → ℝ → Idx N → ℝ,
        (∀ ω, IsGPTFlow lam β N (ϑ ω)) →
        iIndepFun (fun (j : Idx N) (ω : Bool) => ϑ ω 0 j) P →
      ∀ (φ dφ ddφ : ℝ → ℝ), (∀ x, HasDerivAt φ (dφ x) x) → (∀ x, HasDerivAt dφ (ddφ x) x) →
      ∀ K K' : ℝ, (∀ x, |dφ x| ≤ K) → (∀ x, |dφ x| ≤ K') → (∀ x, |ddφ x| ≤ K') →
      ∀ T : ℝ, 0 ≤ T → ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ t' ∈ Set.Icc (0 : ℝ) T,
        ∀ t'' ∈ Set.Icc (0 : ℝ) T, ∀ i j k : Idx N, k < j → j < i →
          |covariance (fun ω => φ (ϑ ω t i)) (fun ω => φ (ϑ ω t' j)) P|
              ≤ C * ((i : ℝ) + 1)⁻¹ *
                Real.exp (Real.sqrt (C * T * Real.log (((i : ℝ) + 1) / ((j : ℝ) + 1)))) *
                Real.exp (C * T) * K ^ 2 ∧
            |thirdCumulant P (fun ω => φ (ϑ ω t i)) (fun ω => φ (ϑ ω t' j))
                (fun ω => φ (ϑ ω t'' k))|
              ≤ C * (((i : ℝ) + 1) * ((j : ℝ) + 1))⁻¹ *
                Real.exp (Real.sqrt (C * T * Real.log (((i : ℝ) + 1) / ((k : ℝ) + 1)))) *
                Real.exp (C * T) * K ^ 2 * K' := by
  rintro ⟨C, -, hC⟩
  obtain ⟨xA, hA0, hA⟩ := exists_three_flow lam β 0 Real.sin_zero
  obtain ⟨xB, hB0, hB⟩ := exists_three_flow lam β π Real.sin_pi
  have hd : HasDerivAt (fun t => xB t - xA t)
      (gptField lam β 3 ![0, π, π / 2] 2 - gptField lam β 3 ![0, 0, π / 2] 2) 0 := by
    have h := (hB 0).sub (hA 0)
    rwa [hA0, hB0] at h
  have hev : ∀ᶠ t in 𝓝[>] (0 : ℝ), 0 < t ∧ xB t - xA t ≠ 0 :=
    eventually_mem_nhdsWithin.and ((hd.eventually_ne (gptField_three_pi_sub_zero lam β hβ)).filter_mono
      (nhdsWithin_mono _ fun t (ht : 0 < t) => ht.ne'))
  obtain ⟨t₀, ht₀, hΔ⟩ := hev.exists
  set R : ℝ := C * (((2 : Idx 3) : ℝ) + 1)⁻¹ *
    Real.exp (Real.sqrt (C * t₀ * Real.log ((((2 : Idx 3) : ℝ) + 1) / (((1 : Idx 3) : ℝ) + 1)))) *
    Real.exp (C * t₀) * 1 ^ 2
  have hR : ∀ M : ℕ, |(xB t₀ - xA t₀) * (π + 2 * π * M - 0) / 4| ≤ R := by
    intro M
    let ϑ : Bool → ℝ → Idx 3 → ℝ := fun ω t =>
      bif ω then ![0, π + 2 * π * M, xB t] else ![0, 0, xA t]
    have hsin : Real.sin (π + 2 * π * M) = 0 := by
      rw [show π + 2 * π * M = π + M * (2 * π) by ring, Real.sin_add_nat_mul_two_pi,
        Real.sin_pi]
    have hflow : ∀ ω, IsGPTFlow lam β 3 (ϑ ω) := by
      intro ω
      cases ω
      · exact isGPTFlow_three lam β 0 Real.sin_zero xA hA
      · exact isGPTFlow_three lam β _ hsin xB fun t => by
          rw [gptField_three_two_add_turns]
          exact hB t
    have hind : iIndepFun (fun (j : Idx 3) (ω : Bool) => ϑ ω 0 j) Homogenized.fairCoin := by
      refine Homogenized.iIndepFun_of_const_of_ne _ 1 fun j hj => ?_
      fin_cases j
      · exact ⟨0, funext fun ω => by cases ω <;> rfl⟩
      · exact absurd rfl hj
      · exact ⟨π / 2, funext fun ω => by cases ω <;> simp [ϑ, hA0, hB0]⟩
    have h := (hC 3 Homogenized.fairCoin inferInstance ϑ hflow hind (fun x => x) (fun _ => 1)
      (fun _ => 0) (fun x => hasDerivAt_id' x) (fun x => hasDerivAt_const x 1) 1 1
      (by simp) (by simp) (by simp) t₀ ht₀.le t₀ ⟨ht₀.le, le_rfl⟩ t₀ ⟨ht₀.le, le_rfl⟩
      t₀ ⟨ht₀.le, le_rfl⟩ 2 1 0 (by decide) (by decide)).1
    rw [covariance_fairCoin] at h
    exact h
  obtain ⟨M, hM⟩ := exists_nat_gt (4 * R / (2 * π * |xB t₀ - xA t₀|))
  have hpos : 0 < 2 * π * |xB t₀ - xA t₀| := by
    have := abs_pos.mpr hΔ
    positivity
  have h1 := hR M
  rw [div_lt_iff₀ hpos] at hM
  rw [sub_zero, abs_div, abs_mul, abs_of_pos (by positivity : 0 < π + 2 * π * M),
    abs_of_pos (by norm_num : (0 : ℝ) < 4)] at h1
  nlinarith [Real.pi_pos, abs_nonneg (xB t₀ - xA t₀)]

/-- The hypothesis of `correlation_bounds_aperiodic_false` holds at `β = 1`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

end Kinetic
end Transformer
